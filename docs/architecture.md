# 架构设计

> 本文档是接口契约、决策口径与评测门槛的唯一事实来源。与代码冲突时以本文档为准，并修正代码或本文档之一。
> 全部数字的来源与消融过程见 [experiments.md](./experiments.md)。

## 1. 总览

```mermaid
flowchart TD
    B[浏览器] -->|POST /api/chat · SSE| GW

    subgraph GW["Spring Boot 网关（Java 17 / Boot 3.3.5）"]
        GF["GuardFilter<br/>认证 · 限流 · 长度上限"]
        SS["SessionStore<br/>InMemory | Redis · TTL 淘汰"]
        CP["CapabilityPlanner<br/>调 plan 取能力集 · 门控在模型服务侧<br/>order 需 orderId 或会话上下文"]
        FO["FanOut<br/>并行能力 · 有界池 · 满即降级"]
        CA["ContextAssembler<br/>去重 · 裁剪 · 来源标注 · 冲突提示"]
        SC["SseChannel<br/>事件封装 · 多线程心跳 · 断连取消"]
    end

    GW -->|一次原子调用 POST /api/plan| MS
    GW --> OT["OrderTool<br/>orderId → 状态 / 物流"]

    subgraph MS["模型服务（FastAPI）· BERT + bge，不常驻生成模型"]
        IC["IntentClassifier<br/>L0 RuleClassifier：正则 转人工 / 订单号<br/>L1 MultiTaskBert：一次前向 15–20ms<br/>route head 3 × sigmoid · topic head 41 × softmax"]
        RT["Retriever<br/>InMemory | PgVector"]
        RR["Reranker<br/>Noop（默认）| FineTuned（可选）"]
    end

    MS -->|生成| LLM["LlmProvider<br/>DeepSeek（唯一实现）"]
```

三处结构性特征：

1. **`switch(route)` → `FanOut`**：单标签硬分支改为多能力并行，由 `ContextAssembler` 合并。
2. **生成只走 DeepSeek**：不常驻本地生成模型，缺 key 启动失败；常驻内存只有 BERT + bge（精排默认不加载）。
3. **意图一次前向出两个决策**：同一次 BERT 前向同时产出能力集与主题，不再分两次编码。

## 2. 模块划分

### 2.1 模型服务（`apps/inference/`）

```
model_service.py            # FastAPI 入口，只做 HTTP 编解码
intent/
  base.py                   # IntentClassifier 抽象
  rule.py                   # L0 规则
  multitask.py              # L1 多任务 BERT
  gate.py                   # 前置门控（纯函数）
  capability_map.py         # 读 intent_capability_map.yaml
  metrics.py                # intent_plan_total
retrieval/
  base.py                   # Retriever 抽象 + Retrieved
  in_memory.py              # 进程内暴力点积（默认）
  pgvector.py               # PostgreSQL + pgvector，DATABASE_URL 存在时启用（§7）
  reranker.py               # Reranker 抽象 + Noop / 微调 / 零样本
llm/
  base.py                   # LlmProvider 抽象
  deepseek.py               # 唯一实现
```

意图由规则 + 多任务 BERT 组成。Reranker 保留 Noop（默认）、微调、零样本三实现；LLM 只有 DeepSeek；Retriever 有 InMemory / PgVector 两实现（`DATABASE_URL` 存在时用 PgVector，见 §7），上层接口不变。

### 2.2 网关（`apps/backend/`）

```
chat/ChatController.java        # SSE 端点，只做协议转换
chat/GuardFilter.java           # 认证 / 限流 / 长度
chat/SessionStore.java          # InMemory 实现：turnId 归档意图 + 空闲淘汰
chat/CapabilityPlanner.java     # 调 /api/plan，消费门控结果并做契约校验
chat/AnswerabilityGate.java     # 可答性三档判定（§5.4）
chat/FanOut.java                # 并行调用，独立超时，有界池满即降级
chat/ContextAssembler.java      # 合并、去重、裁剪
chat/SseChannel.java            # 事件封装、心跳、断连取消
chat/ModelServiceClient.java    # HTTP + 错误码映射
chat/ErrorCode.java             # 枚举
ticket/Ticket.java              # 工单记录，context 存转人工时的证据快照
ticket/TicketStatus.java        # 状态机：PENDING→IN_PROGRESS→RESOLVED / CLOSED
ticket/TicketStore.java         # 存储接口：create / find / findOpen / list / transition
ticket/JdbcTicketStore.java     # H2/JDBC 实现，条件更新保证流转原子性
ticket/TicketController.java    # REST：查询 + accept/resolve/close
ticket/TicketNotFoundException.java
ticket/TicketTransitionException.java
ticket/InvalidTicketStatusException.java
```

网关使用 H2 文件库（`TICKET_DB_URL`，默认 `jdbc:h2:file:./data/tickets`），启动时由 `schema.sql` 建表。存储接口与实现分离，换 PostgreSQL 只改数据源。

## 3. 接口契约

### 3.1 `POST /api/plan`

请求：

```json
{ "query": "我的订单 SO123456 什么时候到", "session_id": "abc", "history": [] }
```

响应：

```json
{
  "capabilities": ["knowledge", "order"],
  "route_scores": { "knowledge": 0.91, "order": 0.78, "human": 0.02 },
  "topic": { "label": "物流全程跟踪", "confidence": 0.83 },
  "entities": { "orderId": "SO123456" },
  "gate": {
    "order": { "allowed": true, "reason": "order_id_present" }
  },
  "source": "bert",
  "elapsed_ms": 18
}
```

- `capabilities` 是最终可执行集合，已经过门控。
- `route_scores` 保留原始概率，便于可观测面板展示与阈值调优。
- `source` ∈ `rule | bert`，用于统计规则短路与模型命中的比例。

### 3.2 `POST /api/retrieve`

```json
{ "query": "...", "top_k": 10, "intent": "物流全程跟踪" }
```

```json
{
  "candidates": [ { "text": "...", "intent": "物流全程跟踪", "score": 0.81 } ],
  "intent": "物流全程跟踪",
  "elapsed": 0.42
}
```

`intent` 为知识主题，非空时对该分区候选加权（不硬过滤，分类错误时只损失排序）。

### 3.3 `POST /api/rerank`

```json
{ "query": "...", "candidates": ["...", "..."], "top_k": 2 }
```

```json
{ "selected": [ { "index": 0, "text": "...", "score": 0.88 } ], "elapsed": 0.31 }
```

`index` 回指请求里 `candidates` 的下标；`text` 原样回传，省得网关再拼一次。线上默认 `NoopReranker`，`score` 统一 0，顺序即召回序。

### 3.4 `POST /api/generate`

请求 `{ "messages": [{ "role": "user", "content": "..." }], "max_new_tokens": 128 }`，
响应为 `text/event-stream`：

| 事件 | 载荷 | 说明 |
|---|---|---|
| `token` | `{ "type": "token", "text": "..." }` | 文本增量 |
| `done` | `{ "type": "done", "reply": "...", "elapsed": 2.1 }` | 完整回答 |
| `error` | `{ "type": "error", "code": "...", "message": "...", "partial": "..." }` | 生成失败，`partial` 是已吐出的内容 |

### 3.5 统一错误封装

模型服务所有非流式接口出错时返回：

```json
{ "code": "UPSTREAM_TIMEOUT", "message": "生成服务超时", "retriable": true }
```

错误码定义见 §9。缺省 HTTP 状态：`4xx` 客户端可修正，`5xx` 服务端可重试。

### 3.6 工单接口

转人工时网关自动建单，客服侧通过 REST 流转：

| 方法 | 路径 | 语义 |
|---|---|---|
| `GET` | `/api/tickets/{id}` | 单条；不存在返回 404 |
| `GET` | `/api/tickets?status=PENDING` | 列表，省略 `status` 返回全部（创建时间倒序） |
| `POST` | `/api/tickets/{id}/accept` | `PENDING → IN_PROGRESS` |
| `POST` | `/api/tickets/{id}/resolve` | `IN_PROGRESS → RESOLVED`（终态） |
| `POST` | `/api/tickets/{id}/close` | `PENDING / IN_PROGRESS → CLOSED`（终态） |

工单体：

```json
{
  "id": 42, "sessionId": "abc", "userId": "u1",
  "status": "PENDING",
  "query": "我要退款",
  "context": "订单 SO123456 已签收；知识库命中「七天无理由退货」",
  "createdAt": 1762732800000, "updatedAt": 1762732800000
}
```

- `context` 是转人工那一刻的证据快照，接手客服无需重放会话。
- 同一会话存在未结束工单（`PENDING`/`IN_PROGRESS`）时复用，不重复建单。
- 非法流转返回 409 `TICKET_TRANSITION_INVALID`；错误体与 §3.5 同形。
- chat 链路在转人工时额外推送一条 SSE 事件：
  `{"type":"ticket","ticketId":42,"status":"PENDING"}`，回复文本带工单号。

## 4. 意图层设计

### 4.1 两层职责

| 层 | 实现 | 命中条件 | 耗时目标 |
|---|---|---|---|
| L0 | 正则（转人工 / 订单号） | 明确模式 | <1ms |
| L1 | 多任务 BERT | 其余全部 | ~15ms（预热后） |

关键原则：**意图决策在 L1 结束，不引入第三层**。

第三层曾以 LLM 兜底的形式存在：`max(route_scores) < 0.60` 时调一次 DeepSeek 补判。A/B 量化后删除，三条理由：

- 300 条评测里只触发 **3 次（1%）**，稳定收益集中在 1 条问法
- 同类问法跨轮在 helpful / no_label 之间翻转，无法当稳定信号
- 机制上只能加标签不能减：兜底命中时 `probs[decided] = 1.0`，实测一次 order 假阳性把 order precision 从 0.889 压到 0.877，跌破 CI 门槛

低置信度时直接返回 route 头按各自阈值得到的能力集；能力集为空由网关走澄清追问。多一层异步兜底会把 DeepSeek 的网络抖动塞进一条 15ms 的判定路径，延迟尾部随之变长。

### 4.2 多任务输出

同一次 `bert-base-chinese` 前向，两个头：

```mermaid
flowchart TD
    IN["input（batch=1, len≤64）"] --> BB["bert-base-chinese<br/>12 层 · 768 维"]
    BB --> CLS["[CLS] 向量"]
    CLS --> RH["route head<br/>Linear(768→3) → sigmoid"]
    CLS --> TH["topic head<br/>Linear(768→41) → softmax"]
    RH --> ROUT["p_knowledge · p_order · p_human"]
    TH --> TOP["top-1 主题"]
```

实体（订单号）由 L0 规则抽取，复用同层结果，不额外编码。

- route 多标签：`p >= 每标签阈值` 即进入候选能力集，先取全部命中，再交给门控裁剪。阈值机制支持逐标签（`config.json: route_thresholds`、缺键回落到全局 `route_threshold`），但当前产物三个值都是 0.5——val 上没有任何组合同时满足 order precision ≥ 0.90，训练脚本按设计回退全 0.5。三个能力正例率相差 25 倍，逐标签是后续调优方向；训练日志会打印本次是扫描最优还是回退分支。
- topic 单标签：取 argmax，附带置信度。

### 4.3 标签派生

`ml/experiments/intent_capability_map.yaml`：

```yaml
version: 1
# 主题 → 候选能力集。这是训练标签的唯一来源，必须可审计、可 diff。
topics:
  PLUS会员:          [knowledge]
  什么时间出库:       [knowledge]
  价保条件:           [knowledge]
  价保申请流程:       [knowledge]
  物流全程跟踪:       [order, knowledge]   # 既要查物流，也要讲时效政策
  订单状态解释:       [order]
  修改订单:           [order]
  如何取消订单:       [order]
  订单签收异常:       [order, knowledge]
  服务单查询:         [order]
  联系客服:           [human]
  联系客户:           [human]
  联系配送:           [human]
  # …补齐剩余 28 类
  其他:               [knowledge]           # 兜底走知识库
```

规则：

1. 每个主题必须显式出现，不允许隐式默认。缺失键在 CI 里报错。
2. route 标签由该表批量派生，不做任何 LLM 标注。
3. 能力集变化必须走 PR，评审可 diff。这是「标签怎么来的」可追溯的保证。

运行时能力集来自 route 头，映射表只是训练标签的来源，不做 topic→能力查表。

### 4.4 前置门控

映射表给出的是**候选**能力。门控在模型服务侧 `/api/plan` 内完成（单一事实来源，避免网关与模型服务各写一套规则漂移），网关的 `CapabilityPlanner` 消费已裁剪的能力集并做契约校验（未知能力丢弃，全空降级为 `clarify`）：

| 能力 | 门控条件 | 不满足时 |
|---|---|---|
| `knowledge` | 始终允许 | — |
| `order` | 请求带订单号，**或**会话内已有订单上下文 | 移除 `order`，保留 `knowledge` |
| `human` | `p_human >= route_thresholds["human"]` | 移除 |

`order` 被门控移除时，`gate.order.reason` 填 `no_order_context`，前端可据此提示「请提供订单号」。

### 4.5 评测集与基线

- 位置：`ml/data/processed/route_eval.jsonl`
- 规模：300 条，**与训练集隔离**，`provenance` 字段区分两类来源
- 两个来源的标注口径不同，**必须分层计算**。混在一起既低估模型，又放大误判：

| 来源 | 条数 | gold 含义 | 指标 |
|---|---|---|---|
| `boundary` | 60 | 来自 `tool_eval`，该问法必须触发的单一工具 | **必中率** `gold ⊆ predicted`。多预测一个 `knowledge` 不扣分：映射表本来就允许 `order + knowledge`，最终由门控裁剪 |
| `heldout` | 240 | 来自 `qa_test`，`intent_capability_map.yaml` 派生的完整候选集 | **subset accuracy**（多标签完全匹配），路由主指标 |

- 能力级 P/R/F1 只在 `heldout` 上算：`boundary` 的 gold 不是完整集合，混入会污染 precision。
- `order` precision 同时报「heldout 实测」与「自然分布加权」两个口径。加权用 `qa_test` 上映射表派生的 order 先验（0.364）；两者一致（0.8894），说明分层抽样没有抬高 precision。
- `human` recall 统计全部 human 正例，并按实际命中层拆开：规则层（显式表达）与 BERT 层（灰区）。显式表达 12/12，灰区 11/17——灰区召回才是模型的真实能力，也是改进重点。
- 当前基线：

```text
boundary 必中率 0.8667        heldout subset 0.8125
order P/R 0.889 / 0.736       human R 0.793（rule 12/12, bert 11/17）
单次前向 18.1ms（P95 20.6ms）
source 分布 bert 288 / rule 12
topic top-1 0.6375（153/240，1 条一词多标注跳过）
```

- 门槛分两层：CI **回归线**（`eval_router.py` 退出码）拦住倒退，**产品目标**见 §10.3。

## 5. 编排层设计

### 5.1 流程

```
handle(session, query):
  1. guard.check()                          → 拒绝则 401/429
  2. history = session.load(session)
  3. plan = client.plan(query, history)     → capabilities + entities
  4. emit "plan"
  5. tasks = []
     if "order" in capabilities:
         tasks += [orderTool.lookup(entities.orderId)]     # 有硬超时
     if "knowledge" in capabilities:
         tasks += [retrieve → rerank]                      # 串行链
     results = FanOut.all(tasks, timeout=perTask)
  6. context = assembler.merge(results, capabilities)
  7. emit "context"
  8. verdict = AnswerabilityGate.decide(...)  → emit "verdict"（§5.4）
     if verdict == no_context: emit 固定话术 "reply" + "done"，结束
  9. client.generate(messages(context), sse → emit "token")
 10. session.save(); emit "done"
```

### 5.2 并行与超时

每个能力独立超时，互不阻塞：

| 能力 | 超时 | 失败行为 |
|---|---|---|
| plan（/api/plan） | 3s | 明确返回 `UPSTREAM_TIMEOUT`；规划是链首，卡住必须快速失败，不拖到 JSON 默认 120s |
| order | 300ms（真实 API 场景 800ms） | 丢弃该能力，继续生成，答案里注明「订单信息暂时不可用」 |
| knowledge retrieve | 500ms | 返回空 → 降级为纯知识回答 |
| rerank | 500ms | 跳过重排，直接用召回顺序 |
| generate | 无总超时，但要求首字 3s 内到达 | 首字超时返回错误，已吐出的内容保留；不做本地降级 |

> 精排默认走 `NoopReranker`（直接取召回序，不加载 1.1GB 权重）。是否启用微调精排见
> [experiments.md](./experiments.md)「精排取舍消融」；启用后超时按上表降级为召回序。

`FanOut` 的线程池有界：`max(8, 核数×4)` 个线程、队列 256、`AbortPolicy`。任务全是阻塞网络 I/O，队列满时直接拒绝而不是排队——排队会把能力的超时预算白耗在队列里，还会把上游雪崩传回来。被拒的能力收敛成失败的 `Outcome`，按 §8 的降级路径处理。

### 5.3 ContextAssembler

职责只有四条：

1. **去重**：同一知识片段不重复注入。
2. **裁剪**：知识最多 2 条，每条 ≤ 90 字；订单摘要 ≤ 200 字。
3. **来源标注**：每段上下文前加 `[订单]` / `[知识]` 前缀，让模型知道来源不同。
4. **冲突提示**：订单数据与知识政策不一致时（如承诺 3 天但政策写 5 天），显式加一行「注意：以订单承诺时效为准」。

第 4 条是多能力并行的直接产物——合并会带来来源冲突，这是单标签路由不存在的风险。

### 5.4 可答性闸门

前四段评测都建立在「知识库里有答案」的前提上。真实用户会问知识库没有的问题：天气、股票、闲聊。此时生成模型会拿常识补，答案读起来通顺但与客服业务无关。可答性闸门回答的不是「排得对不对」，而是**知识库里到底有没有能答的内容**：没有就不调生成，用固定话术收口。

#### 接口

网关在 `context` 事件之后、`stage generate start` 之前发一条 `verdict` 事件：

```json
{"type":"verdict","verdict":"answer|weak|no_context",
 "reason":"recall_score|empty|order_hit|order_only|degraded",
 "score":0.66,"thresholds":{"low":0.58,"high":0.64}}
```

`score` 只在本轮有召回时出现（无召回不塞 null，`SseChannel` 的 ObjectMapper 不忽略 null 字段）。

三档语义与决策表（`AnswerabilityGate.decide`）：

| 输入 | answer | weak | no_context |
|---|---|---|---|
| 未规划 knowledge，有订单事实 | ✅ `order_only` | | |
| 未规划 knowledge，无订单 | | ✅ `order_only` | |
| 召回降级，有订单事实 | ✅ `order_hit` | | |
| 召回降级，无订单 | | ✅ `degraded` | |
| 召回为空，有订单事实 | | ✅ `order_hit` | |
| 召回为空，无订单 | | | ✅ `empty` |
| 首位分 ≥ T_HIGH | ✅ `recall_score` | | |
| T_LOW ≤ 首位分 < T_HIGH | | ✅ `recall_score` | |
| 首位分 < T_LOW，有订单事实 | | ✅ `order_hit` | |
| 首位分 < T_LOW，无订单 | | | ✅ `recall_score` |

`no_context` 时写 `reply`（`code=NO_GROUNDED_CONTEXT`，HTTP 200、不可重试）+ `done` 并结束，**不调生成模型**。固定话术 `AnswerabilityGate.NO_CONTEXT_REPLY` 引导用户补充细节或转人工，来源与 `ErrorCode.NO_GROUNDED_CONTEXT` 同值（单一真相源）。

可观测：`cs.chat.verdict{verdict,reason}`；前端追踪面板新增「可答性判定」块，显示三档、依据、首位分与阈值。`no_context` 轮次生成阶段保持 `pending`，不标成已完成。

#### 口径

评测集由 `ml/experiments/build_answerability_eval.py` 构建，共 190 条，全部人工标签、不依赖模型预测：

| 切片 | n | 来源 | 作用 |
|---|---|---|---|
| `qa_test` | 120 | `qa_test.jsonl` 固定 seed 42 抽样 | in_kb 正样本，量误拦（必须放行） |
| `boundary` | 10 | 手写：字面含域外实体（iPhone、猫粮、口红），问的是域内政策 | in_kb 硬例，单独切片 |
| `handwritten`（out_kb） | 60 | 手写：天气、编程、医疗、生活常识等 | 负样本，量拦截 |

信号取线上同源链路：多任务 BERT topic → `retrieve_indexed` top10（topic +0.05）→ `hits[0]` 分数。检索为空按 `no_context` 计。评测脚本 `ml/experiments/eval_answerability.py`，阈值与 `AnswerabilityGate` 双阈值同值，改动必须两边同步（Java 测试里有常量同步断言）。

#### 信号选型（为什么不用 reranker 分数）

| 信号 | AUC（域内 vs 域外） | 能做绝对阈值吗 |
|---|---|---|
| **召回首位 cosine（选中）** | **0.9605** | ✅ 域内 P5 0.614，域外 P95 0.683，可切 |
| rerank 微调分数 | 0.565 | ❌ 分数挤压 0.35–0.97，域外问题同样 0.9+ |

微调后的 reranker 只学相对序，失去绝对校准；召回 cosine 是唯一免费且可校准的信号。这也是不引入新模型的理由：加一个模型不如用好已有分数。

#### 预登记门槛

| 门槛 | 值 | 性质 |
|---|---|---|
| `qa_test in_kb → no_context` | ≤ 5% | 硬门槛：把可答问题变固定话术的代价 |
| `boundary in_kb → no_context` | ≤ 20% | 硬门槛：字面域外、实际域内的问题不能误拦 |
| `qa_test in_kb → answer` | ≥ 75% | 硬门槛：weak 只是提示，answer 才直接生成 |
| `out_kb → no_context` | ≥ 50% | 硬门槛：漏拦靠 weak 兜底，不追求全拦 |

退出码：任一门槛不通过返回 1。**不进 per-commit CI**——依赖模型权重与 MPS，放手动回归。

#### 已知限制

1. `in_kb` 正样本来自 `qa_test`，构造上都有 KB 答案，1.7% 误拦是乐观估计；真实用户会问「域内但知识库没覆盖」的问题，线上误拦可能更高。
2. 漏拦样本（「现在几点了」0.7172、「明天是星期几」0.7319）说明时间/常识类问题偶有高分，靠 weak 提示兜底，不追求全拦。
3. 阈值绑 `bge-small-zh` + topic 加成 + top10 口径。换 embedding 或改 `INTENT_BOOST` 必须重跑评测、重定阈值。
4. 产品取向「宁可漏拦，不可误拦」写进代码注释：只有既没可信知识、也没订单事实才拦截。

#### 当前基线

阈值 T_LOW=0.58 / T_HIGH=0.64：

```
切片              n | no_context    weak   answer
qa_test       120 |      0.017   0.083    0.900
boundary       10 |      0.000   0.000    1.000
handwritten    60 |      0.700   0.217    0.083
in_kb 合计      130 |      0.015   0.077    0.908
out_kb         60 |      0.700   0.217    0.083
召回首位分数 AUC(域内 > 域外) = 0.9605
  in_kb  p5=0.614 p25=0.691 p50=0.744 p75=0.785 p95=0.836
  out_kb p5=0.414 p25=0.469 p50=0.541 p75=0.590 p95=0.683
```

四项预登记门槛全 PASS：qa_test 误拦 1.7%、boundary 误拦 0%、qa_test answer 90.0%、out_kb 拦截 70.0%。阈值网格证明 0.58/0.64 是平衡点：抬到 0.62/0.66 会把 in_kb 误拦推到 6.2%，越过 5% 硬门槛。

## 6. 工单闭环

转人工此前只回一句「已转接」，客服侧没有任何可追踪的东西。工单把它变成一个实体：建单、查询、流转都有状态机和持久化，接手客服拿到的是上下文快照，而不是一段需要重放的会话。

### 6.1 状态机

```mermaid
stateDiagram-v2
    [*] --> PENDING
    PENDING --> IN_PROGRESS : accept
    IN_PROGRESS --> RESOLVED : resolve
    PENDING --> CLOSED : close
    IN_PROGRESS --> CLOSED : close
    RESOLVED --> [*]
    CLOSED --> [*]
```

允许的迁移写死在 `TicketStatus.ALLOWED`，存储层只做条件更新，两边不各判一套。终态不设出口：已解决的工单再被关闭只会掩盖真实结论，宁可让调用方拿到 409。

### 6.2 存储

- H2 文件库（`TICKET_DB_URL`，默认 `jdbc:h2:file:./data/tickets;AUTO_SERVER=TRUE`），`schema.sql` 全部 `IF NOT EXISTS`，重复执行幂等；`status`、`session_id` 各建索引。
- `TicketStore` 接口 + `JdbcTicketStore`；换库只改实现。
- 迁移用 `UPDATE ... WHERE id=? AND status=?` 条件更新：两个客服同时点「接受」只有一个成功，另一个重读最新状态后按结果报 409（流转非法）或 404（已被删）。
- 列表上限 200 条，避免无界查询；接后台管理时再加游标分页。

### 6.3 接入 chat 链路

`ChatOrchestrator.handleTransfer` 先 `findOpen(sessionId)` 复用未结束工单（`PENDING` 或 `IN_PROGRESS`），没有才 `create(sessionId, userId, query, ticketContext(sessionId))`，然后推 SSE：

```json
{"type":"ticket","ticketId":42,"status":"PENDING"}
```

回复话术带工单号。`context` 存转人工那一刻的订单 / 知识证据摘要，接手客服无需重放会话。REST 契约见 §3.6。

## 7. 向量库

InMemory 是进程内暴力点积，万级以内够用；向量库是规模化的必经之路。`Retriever` 接口做实，两个实现用 `DATABASE_URL` 切换，上层无感。

### 7.1 两个实现共用一个接口

```mermaid
flowchart LR
    ENV{"DATABASE_URL 是否为空"}
    ENV -->|非空| PG["PgVectorRetriever<br/>PostgreSQL + pgvector · hnsw 索引"]
    ENV -->|为空| IM["InMemoryRetriever<br/>进程内点积（默认）"]
```

`build_retriever` 是唯一开关点。没有隐藏回退：连库或读表失败会把异常抛给 `load_models`、落进 healthz 的 `error`，而不是悄悄退回 InMemory 让人以为向量库在工作。`ping()` 在启动时执行 `SELECT 1 FROM knowledge LIMIT 1`；healthz 额外报告 `"retriever": "pgvector" | "in_memory"`。

### 7.2 SQL 与打分口径

```sql
SELECT id, intent, answer, 1 - (embedding <=> %s::vector) AS score
FROM knowledge
ORDER BY embedding <=> %s::vector
LIMIT %s
```

两个取舍：

1. **候选池放大**：先按纯向量距离取 `top_k * OVERSAMPLE(5)`，意图分区加成（+0.05）在应用层做。加成若塞进 `ORDER BY` 表达式会让 hnsw 索引失效、退化成全表扫描；应用层重排保住索引，语义也与 InMemory 一致——分区只重排候选池，不改变召回口径。
2. **向量以文本传参**：`[0.1,0.2,...]` 随 SQL 传再 `::vector` 转换，不依赖驱动版本不同的适配器注册方式，也让 fake connection 单测成立。

编码口径与 parity 相关：文档侧编码不加指令前缀（与 `pipeline.py` 一致），query 侧补 `QUERY_INSTRUCTION`，两边对得上才能做重叠对比。

### 7.3 索引构建

`ml/experiments/build_vector_index.py`：

- 读 `qa_train.jsonl`，分区标签复刻运行时（意图出现 ≥100 次保留，其余归「其他」），两边必须落在同一标签空间。
- 文档侧 `answer` 分批（64）编码，`normalize_embeddings=True`，无指令前缀。
- `ON CONFLICT (id) DO UPDATE`，重复灌幂等；`--reset` 先 `TRUNCATE`。
- 灌完再建 `USING hnsw (embedding vector_cosine_ops)`：边插边维护 hnsw 比一次性建慢一个数量级。

### 7.4 部署

`docker-compose.yml` 加 `postgres`（`pgvector/pgvector:pg16`），默认不启动，需要时显式带 profile：

```bash
DATABASE_URL=postgresql://csagent:csagent@postgres:5432/csagent \
  docker compose --profile pgvector up -d
```

不设 `depends_on`：postgres 在 profile 里，未激活时模型服务照常以 InMemory 启动。

### 7.5 Parity 验收

`apps/inference/tests/test_pgvector_parity.py`：同一 query 取 10 条，比较两种实现的 top-10 文本集合，要求交集 ≥ 8。默认 skip；`DATABASE_URL` 指向已灌数据的库时运行。

真实冒烟：

```
灌库     17,855 条写入 knowledge（psycopg 3.x + pgvector，hnsw 索引建完）
parity   1 passed in 25.46s（10 条 query，每条 top-10 交集 ≥ 8）
pytest   apps/inference/tests 88 passed, 1 skipped（无 DATABASE_URL 时 parity skip）
```

### 7.6 为什么客服场景需要向量库

检索本来就靠向量，bge 一直在用。这一节解决的是**向量存在哪**：进程内数组 vs pgvector。万级以内前者更快更简单，十万级以上后者才是正解。参考项目把两条路都做到可切换，而不是断言哪条永远对。

## 8. 失败处理

| 事件 | 用户看到 | 服务端 |
|---|---|---|
| 路由低置信（能力集为空） | 「能否具体描述一下您的问题？」 | 记 `ROUTE_LOW_CONFIDENCE` |
| 订单号不存在（纯订单查询） | 「没有找到订单 SO123456，请核对后重试」 | **禁止回退到其他订单** |
| 订单号不存在（订单+知识并行） | 生成回答，上下文注入「订单号不存在，不要编造」 | 知识结果照常合并，仍**禁止回退到其他订单** |
| 检索空 | 用生成模型的常识回答，并标注「知识库无匹配」 | 记 `RETRIEVAL_EMPTY` |
| DeepSeek 429 | 「服务繁忙，请稍后重试」 | 退避 1 次后返回 `UPSTREAM_RATE_LIMITED` |
| DeepSeek 半截流 | 已输出部分保留 + 「回答中断，请重试」 | 记 `STREAM_INTERRUPTED` |
| 客户端断开 | — | 取消上游请求，停止计费 |

内存会话在进程重启后丢失是已知取舍，接口化后生产切 Redis 即可。进程内实现有两条防膨胀约束：会话与待归档意图都按 30 分钟空闲淘汰，惰性触发（最多一分钟扫一次），避免 map 随会话数无限增长；意图按 turnId 归档，同一会话并发两轮不会互相覆盖。

## 9. 错误码

```
INVALID_REQUEST        400  否   请求格式不正确（空问题、sessionId 非法）
UNAUTHORIZED           401  否   身份校验失败（配置 token 时）
REQUEST_TOO_LARGE      413  否   问题超过 512 字 / 请求体超限
RATE_LIMITED           429  是   单会话请求过频
ROUTE_LOW_CONFIDENCE   400  否   路由无法判定
ORDER_NOT_FOUND        404  否   订单号不存在
ORDER_CONTEXT_MISSING  400  否   需要订单号才能查
RETRIEVAL_EMPTY        200  否   召回为空（非故障）
NO_GROUNDED_CONTEXT    200  否   可答性闸门判 no_context，不调生成（§5.4）
TICKET_NOT_FOUND       404  否   工单不存在（§6）
TICKET_TRANSITION_INVALID 409 否 工单当前状态不支持该操作（§6）
UPSTREAM_TIMEOUT       504  是   下游超时
UPSTREAM_RATE_LIMITED  429  是   下游限流
STREAM_INTERRUPTED     200  是   流式中断
INTERNAL               500  是   未分类异常
```

铁律：**任何异常都不允许把底层 `getMessage()` 直接回给用户**。对外文本一律来自 `ErrorCode` 枚举，底层异常只进日志。

## 10. 测试与评测

### 10.1 覆盖图

```mermaid
flowchart TB
    subgraph CODE["代码路径"]
        direction TB
        CP["CapabilityPlanner<br/>规则早退 · BERT 各标签阈值<br/>低置信 → 返回 route 结果 · order 门控"]
        IC["IntentClassifier<br/>route 多标签 · topic 41 类 · 能力集排序"]
        RP["RetrievalPipeline<br/>InMemory / pgvector 两实现（§7）<br/>intent boost · 零样本 / 微调 rerank · 空结果降级"]
        LP["LlmProvider<br/>DeepSeek 流式（唯一实现）<br/>超时 / 空响应 / 半截流"]
    end

    subgraph FLOW["用户流程"]
        direction TB
        OD["订单问答<br/>有订单号，单能力<br/>无订单号 → 知识降级<br/>[E2E] 订单+知识并行"]
        KB["知识问答<br/>命中 · 低置信 → 追问 · 空召回"]
        HM["转人工<br/>正则 → 终止 + 建单（§6）"]
        EX["异常<br/>模型服务超时 · DeepSeek 429 · 客户端断连"]
    end

    subgraph EVAL["模型评测 [EVAL]"]
        direction TB
        E1["路由 300 条（route_eval.jsonl）"]
        E2["rerank 2,995 对（eval_rerank.py）"]
        E3["检索 recall@10（eval_retrieval.py）"]
        E4["端到端生成 100 条（eval_generation.py，§10.4）"]
        E5["可答性 190 条（eval_answerability.py，§5.4）"]
    end

    CODE ~~~ FLOW ~~~ EVAL
```

### 10.2 分层

| 层 | 内容 | 位置 |
|---|---|---|
| 单测 | LLM / Reranker / 意图各实现一套；门控逻辑；Assembler 去重裁剪；可答性闸门 | `apps/backend/src/test`、`apps/inference/tests` |
| 契约测试 | Java ↔ Python 四个接口的出入参结构、错误码映射 | `apps/backend/src/test/java/.../contract` |
| 评测 | 路由 300 条、rerank、检索、生成、可答性 | `ml/experiments/*` |
| E2E | 三条主流程（订单 / 知识 / 转人工）+ 并行场景，HTTP/SSE 黑盒 | `apps/e2e`（pytest，跑真实栈） |
| CI | PR：单测 + 契约 + 映射校验；夜间：评测 + E2E | `.github/workflows/ci.yml` |

### 10.3 评测门槛

门槛分两层：**产品目标**是收敛方向，**CI 回归线**是当前模型下必须守住的下限。两者不一致是显式的——模型达标前，CI 的职责是拦住倒退，目标差值记在 TODO。

| 指标 | 产品目标 | CI 回归线 | 当前基线 | 口径 / 说明 |
|---|---|---|---|---|
| boundary 必中率 | ≥ 0.90 | ≥ 0.85 | 0.867 | 60 条工具边界问法，`gold ⊆ predicted` |
| heldout subset accuracy | ≥ 0.88 | ≥ 0.80 | 0.813 | 240 条映射表口径，多标签完全匹配 |
| order precision | ≥ 0.90 | ≥ 0.88 | 0.889 | 自然分布加权；误触发比漏触发代价高 |
| human recall | ≥ 0.90 | ≥ 0.75 | 0.793 | 灰区召回 0.647，扩规则与换底座进 TODO |
| rerank pairwise | ≥ 0.774 | 0.774 | 0.848 | 不得低于零样本基线 |
| 检索 recall@10 | ≥ 0.72 | ≥ 0.68 | 0.6962 | 线上口径：KB=`qa_train`，路由 topic 分区 +0.05；纯向量 0.6637 |
| 生成 faithfulness | ≥ 4.0 | 夜间 | 4.71 | 100 条端到端，LLM 判分（§10.4） |
| 生成 cosine | — | 夜间 | 0.622 | 回归线 0.572 |
| 可答性 误拦（qa_test） | ≤ 0.05 | — | 0.017 | 可答问题被判 no_context；手动回归，不进 CI（§5.4） |
| 可答性 拦截（out_kb） | ≥ 0.50 | — | 0.700 | 域外问题被判 no_context；漏拦靠 weak 兜底 |
| 首字延迟 P95 | ≤ 3s | — | 0.86s（DeepSeek 单次样本） | 需持续采样再定 P95 |
| 全链路 P95 | ≤ 8s | — | 1.18s（DeepSeek 单次样本） | E2E 4 条用例合计 4.27s |

**口径**：路由四项基线是 **门控前** 口径——分类器只有规则 + BERT，也不经过 `apply_gate`。这与 CI 的调用方式一致（`python ml/experiments/eval_router.py` 无参数），所以 CI 退出码只由门控前口径决定。评测集没有会话历史，门控后口径等价于会话首轮：`order` 只在问法自带订单号时保留，暴露的是「首轮无上下文」的真实行为，不是模型能力退化。线上完整口径需人工跑 `--gate on`，把 source 分布、topic top-1 与两套指标一起写进 [experiments.md](./experiments.md)；门控后数值不参与 CI 判定。

端到端实测（DeepSeek 模式，`u_chen` 固定样例）：knowledge 首字 0.86s / 全链 1.18s，order 0.62s / 1.01s，并行 0.68s / 1.16s；E2E 4/4 合计 4.27s。模型服务常驻只有 BERT + bge（默认不加载精排，不加载本地大模型）。rerank 默认 NoopReranker 直接取召回序；若启用微调精排（10 条候选预热 193ms），超过 500ms 预算时按 §5.2 降级为召回顺序——降级链路已在 E2E 中实际触发并验证。上述为单次样本，P95 需上线后持续采样。

### 10.4 端到端生成评测

前三段评测各管一截：路由管「能力集对不对」，检索管「召回没召回」，rerank 管「排序对不对」。生成层必须单独补：**同一条 query 走完整链路，判生成结果**。

#### 接口

```bash
PYTHONPATH=apps:ml/runtime .venv/bin/python3 ml/experiments/eval_generation.py \
  --data ml/data/processed/qa_test.jsonl \
  --limit 100 --seed 42 \
  --context pipeline          # pipeline | oracle
  --judge auto                # auto | off
  --concurrency 2 \
  --out /tmp/eval_generation.jsonl
```

| 参数 | 语义 |
|---|---|
| `--context pipeline` | 真实链路：意图 → 检索（top10，topic +0.05）→ rerank（top2）→ 组装 `[知识]` 上下文。**端到端口径**，默认。 |
| `--context oracle` | 把 gold answer 直接当上下文。只测生成与 prompt 的天花板，用来把「检索错」和「生成差」分开。 |
| `--judge off` | 只算余弦与延迟，不调评审模型（省 token，可离线复跑）。 |
| `--limit/--seed` | `Random(seed).sample` 取前 N 条，可复现。 |
| `--concurrency` | 只作用于生成 + 判分两段网络调用；检索串行，避开 MPS 并发。 |

#### 口径（三条必须与线上一致，否则数字不可信）

1. **系统 prompt 不复制**：运行时读 `apps/backend/src/main/resources/prompts/chat-system.txt`，`{today}` 替换为运行日。网关改 prompt，评测自动跟着变。
2. **用户消息格式复刻** `ChatOrchestrator.chatMessages()`：`context + "\n\n用户问题：" + query`；上下文为空时用占位符 `（本轮没有可用的订单或知识库上下文）`。
3. **上下文形状复刻** `ContextAssembler`：知识块 `"[知识] " + 限 90 字`（超出截断为 `…`），最多 2 条，`"\n"` 连接。Python 侧的 `KNOWLEDGE_MAX_ITEMS / KNOWLEDGE_MAX_CHARS` 与 Java 常量同名同值，两处注释互指（Java 无法 import Python）。

生成固定 `DeepSeekProvider.stream`（`max_new_tokens=200`），不另走一条网络路径。

#### 判分

| 指标 | 来源 | 说明 |
|---|---|---|
| `faithfulness` / `relevance` / `style` | LLM-as-judge，1–5，temperature 0，严格 JSON | **判分器自己直连 DeepSeek**：`DeepSeekProvider` 不暴露 temperature，判分确定性不能靠它 |
| `cosine` | 生成回答 vs gold answer，bge-small-zh 句向量 | 机器口径，不受判分器偏好影响 |
| `first_token_ms` / `total_ms` | 生成阶段实测 | 检索阶段单独记 `retrieval_ms` |
| `error_rate` | 生成异常 / 超时占比 | 单条失败不中断，落进 JSONL 的 `error` 字段 |

判分 rubric 的关键一条：**上下文不足时，回答明确说明「暂无相关说明」算正确**（faithfulness 4–5），不按「没答上来」扣分——那是可答性闸门的职责，不是生成评测的职责。判分失败（超时 / JSON 不合法 / 分数越界）记 `judge_error`，不参与均分，单独计数。

#### 预登记门槛

| 门槛 | 值 | 性质 |
|---|---|---|
| `faithfulness` 均值 | ≥ 4.0 | 硬门槛，产品要求：不编造 |
| 三项均值 | ≥ 3.5 | 硬门槛 |
| `error_rate` | ≤ 2% | 硬门槛 |
| `cosine` 均值 | 基线 0.622，回归线 0.572 | 回归门槛 = 基线 − 0.05（绝对）或 −5%（相对），取更宽松者 |
| `total_ms` P95 | ≤ 3000ms，只报告 | 网络抖动大，不用作退出码条件 |

退出码：硬门槛任一不通过返回 1。**不进 per-commit CI**——每次跑要花 DeepSeek token 且依赖外网，放夜间任务或手动回归；`--judge off` 可作无 token 的冒烟。

#### 已知限制

1. 判分器与生成器是同一家模型，存在自我偏好；rubric 用固定 1–5 定义压偏差，cosine 作交叉验证。
2. gold answer 来自真实客服对话，含脱敏占位符（`[数字]` 等），cosine 会被这类噪声拉低，绝对值只看趋势。
3. `oracle` 口径的上下文是「答案本身」，天然抬高 faithfulness/relevance，只能和同口径比。

#### 当前基线

100 条 pipeline + 100 条 oracle（同 seed 42）：

| 指标 | pipeline | oracle |
|---|---|---|
| faithfulness / relevance / style | 4.71 / 4.63 / 4.56 | 4.47 / 4.56 / 4.45 |
| cosine 均值 | 0.622 | 0.713 |
| 全链 avg / P95 | 1545ms / 2216ms | 1246ms / 1630ms |
| `[数字]` 补全 | 2/37（5%） | — |
| error_rate | 0% | 0% |

两条读法：

1. **judge 均分对检索误差不敏感**（oracle 4.49 vs pipeline 4.52），cosine 敏感（−0.099）。judge 看生成能力，cosine 看端到端贴合度，两者不可互相替代。
2. 含 `[数字]` 占位符的上下文曾被模型补成具体数字（修复前 9/37 = 24%）。修复方式是在 `chat-system.txt` 明令「脱敏占位符不要猜测或补全成具体值，用户需要具体数字时以订单页为准」，同口径复测降到 2/37（5%），faithfulness 4.54 → 4.71，P95 +650ms（仍在 3000ms 内）。细节见 [experiments.md](./experiments.md)「占位符 prompt 修复验证」。

## 11. 可观测

每个请求通过 SSE 推送以下阶段，前端右侧面板直接展示：

```
plan      → capabilities / route_scores / topic / source / elapsed_ms
retrieve  → 候选数与分数分布
rerank    → 精筛前后顺序变化
context   → 最终注入的上下文来源与长度
verdict   → 可答性三档 / 依据 / 首位分 / 阈值
generate  → 首字延迟 / token 数 / 总耗时
```

四个观测指标必须进监控：

1. `source` 分布（rule / bert）——bert 占比骤降说明规则正在替模型做决定
2. `ORDER_NOT_FOUND` 率——突然升高说明门控或正则有问题
3. 首字延迟 P95
4. `RETRIEVAL_EMPTY` 率

### 11.1 trace_id

网关为每个请求生成或继承 `X-Trace-Id`（`[A-Za-z0-9._-]{1,64}`，非法值替换而不是拒绝），写入 MDC 与响应头；模型服务继承合法值、非法值自己补造，并在响应头回写。跨线程（`StreamingResponseBody` 的异步线程、`FanOut` 线程池）由代码显式复制 MDC，日志格式 `%X{traceId:-}` 一路带下去。错误事件的 `traceId` 字段让用户反馈能直达日志。

### 11.2 指标

网关（Micrometer → `/actuator/prometheus`）：

| 指标 | 标签 | 用途 |
|---|---|---|
| `cs.chat.plan` | `source`, `route` | 能力规划耗时 + 来源分布 |
| `cs.chat.capability` | `capability=order\|retrieve\|rerank`, `outcome=ok\|empty\|degraded\|missing\|no_context` | 单能力耗时与降级 |
| `cs.chat.generate` | `phase=first_token\|total` | 首字 / 全链生成耗时 |
| `cs.chat.verdict` | `verdict=answer\|weak\|no_context`, `reason` | 可答性闸门判定分布（§5.4） |
| `cs.chat.requests` | `outcome=ok\|error\|disconnect` | 一轮问答的最终结局 |
| `http.server.requests` | 由 Spring 自动生成 | 已用 `MeterFilter.deny` 排除 `/api/chat`（响应体就是生成流，混进去会污染 P95） |

全部 Timer 打开 `publishPercentileHistogram()`，Prometheus 端用 `histogram_quantile` 算 P95。

模型服务（`prometheus_client` → `/metrics`）：

| 指标 | 标签 | 用途 |
|---|---|---|
| `model_service_request_seconds` | `path`, `method`, `status` | 单次原子调用耗时 |
| `model_service_requests_total` | `path`, `method`, `status` | 调用数 |
| `intent_plan_total` | `source=rule\|bert` | 意图识别最终来源分布 |

路径标签白名单化：只有 `/healthz`、`/api/plan`、`/api/retrieve`、`/api/rerank` 保留原值，其余（含 404、扫描器乱打）归到 `other`，防止标签爆炸；`/api/generate`（SSE，生命周期等于整段生成）与 `/metrics` 自身不计入。中间件是纯 ASGI，不拆 `StreamingResponse`——首字延迟不能被中间件拖走。

## 12. 不在范围

| 项 | 理由 |
|---|---|
| 真实订单 / 物流 API 对接 | 参考项目保留 mock，签名对齐真实系统即可 |
| 多租户 / 权限模型 | 需产品定义，非技术选型 |
| 向量库运维（分片、备份、调参） | PgVectorRetriever 已可接入（§7）；分片 / 备份 / 索引调参不进参考范围 |
| 流式 token 计费 | DeepSeek 侧能力，网关只做限流 |
| 前端重写 | 复用现有前端，只调整 SSE 事件字段 |
