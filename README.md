# customer-service-agent

**电商客服 AI Agent 参考实现。** 一次前向识别意图，多能力并行取数/检索，DeepSeek 流式生成，整条决策链在右侧面板实时可见。

> An e-commerce customer-service agent: one forward pass for intent, parallel tool/retrieval fan-out, streaming generation, with every routing decision observable in real time.

[![CI](https://github.com/susamlu/customer-service-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/susamlu/customer-service-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Java](https://img.shields.io/badge/java-17-blue)
![Tests](https://img.shields.io/badge/tests-203%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

![聊天界面](docs/images/ui-chat.png)

![决策链路面板](docs/images/ui-trace.png)

右侧面板是项目的核心主张：**回答不是黑盒**。一次提问走过了什么——能力集判成了什么、门控为什么放行或拦截、订单与知识如何并行、召回了哪些候选、依据分够不够生成、每个环节花了多久——全部实时可见。

## 关于这个项目

全链路独立完成：数据构建、模型训练与评测、Python 模型服务、Spring Boot 网关、React 前端。底座模型与训练语料来自公开资源（见「数据与训练」）。

目标不是教程级 demo，而是把真实客服系统里的工程取舍写清楚：**哪些环节值得训练模型，哪些交给 LLM，哪些应该砍掉**。下面每一节都对应一个做过对照实验的决策，数字取自 `docs/experiments.md`。

| 意图识别（预热后单次前向） | 模型服务常驻权重 | 生成首字延迟 | boundary 必中率 |
|---|---|---|---|
| **15–20 ms** | **0.5 GB** | **~1.0 s** | **0.867** |

## 架构

```mermaid
flowchart LR
    FE[React 前端] --> GW

    subgraph GW["Spring Boot 网关 · 业务编排"]
        direction TB
        CP["调用 plan<br/>取能力集"] --> D{能力集分流}
        D -->|order| OT[订单工具<br/>订单状态 / 物流]
        D -->|knowledge| KR["召回 → 精排<br/>默认 noop"]
        D -->|human| H[转人工]
        D -->|空| C[澄清追问]
        OT --> CA["上下文拼装<br/>去重 · 裁剪 · 来源标注"]
        KR --> CA
        CA --> G[流式生成]
    end

    GW -->|原子 API| MS["FastAPI 模型服务<br/>意图（多任务 BERT）· 召回 · 精排 · 生成"]
    MS --> DS[DeepSeek]
    GW --> MOCK[(mock 业务数据<br/>用户 / 订单 / 商品 / 物流)]
```

一次请求的决策顺序：

```mermaid
flowchart TD
    Q["用户提问"] --> L0["L0 规则（正则）<br/>转人工命中即短路，顺带抽订单号"]
    Q --> L1["L1 多任务 BERT · 15–20ms<br/>能力集 + 主题，一次前向两个 head"]
    L0 --> GATE["门控（模型服务侧）<br/>knowledge 恒放行<br/>order 需订单号或会话上下文<br/>human 需高置信"]
    L1 --> GATE
    GATE --> OT["订单工具"]
    GATE --> RT["检索"]
    OT --> CA["上下文拼装"]
    RT --> CA
    CA --> ANS{"可答性判定<br/>首位召回分三档"}
    ANS -->|no_context| FIX["固定话术，不调生成"]
    ANS -->|answer / weak| GEN["DeepSeek 流式生成"]
```

分层职责：

- **模型服务只提供原子推理能力**：意图识别、召回、精排、生成，不承载业务编排，也不访问业务接口
- **网关负责编排**：能力集调度、订单匹配、上下文合并、prompt 拼装、SSE 事件
- **意图先于数据访问**：能力集决定「要不要查订单、要不要检索」，门控保证工具不被误触发

**生成层只走 API。** 客服问答 QPS 不高，本地部署生成模型要多花 GB 级内存和冷启动时间，而 DeepSeek 流式首字约 1s、全链 1.2–1.5s 已经够用（E2E 单次样本 1.18s，100 条端到端评测均值 1545ms）。所以生成固定走 API，缺 key 时启动直接失败，不做本地降级，模型服务常驻只有 BERT + bge。

## 两个工程取舍

### 1. 意图识别：训练一个小模型，不调 LLM，也不堆层

**不调 LLM。** 最初的版本在规则 + BERT 之上加了一层 LLM 兜底，专门处理模型低置信的情况。量化之后删掉了：

- 300 条评测里只触发 **3 次（1%）**，收益集中在 1 条问法
- 同类问法跨轮表现不稳定，无法当稳定信号
- 会注入 order 假阳性，把 order precision 压到 CI 门槛以下

**不堆层。** 更早的链路是「规则 → 向量检索 → 0.5B → 3B」的零训练启发式组合。问题不是某一层不准，而是**层间错误累积**：上游误判，下游没有否决权；同时线上常驻内存约 10GB，每一层都有独立的延迟尖峰。

现在的做法是 L0 正则 + L1 一个多任务 BERT。一次前向同时输出两个 head：

| head | 输出 | 用途 |
|---|---|---|
| route | 3 × sigmoid（knowledge / order / human） | 决定能力集 |
| topic | 41 × softmax | 检索分区加权 |

订单号等实体由同层的规则抽取，不额外编码。41 类主题到能力集的映射是显式 YAML，CI 校验无缺键，不让模型「顺便」决定调用什么工具。

单次前向 15–20ms，线上常驻只剩 BERT + bge，权重文件合计约 **505 MB**——意图链路从旧方案的百毫秒级降到 15ms 级。

### 2. 精排默认关掉（保留可开启路径）

这里曾有两个口径给出相反结论，值得如实记录：

| 口径 | noop（不精排） | 微调精排 | 零样本精排 |
|---|---|---|---|
| intent-hit top-2 命中率（线上最相关） | **0.5477** | 0.5299 | 0.5026 |
| pairwise 硬负例准确率 | — | **0.848** | 0.774 |

intent-hit 口径下不精排最好，因为这个口径的相关集合就是同主题召回，精排没有可区分的信息；pairwise 口径证明微调精排确实能压过零样本 **+7.44pp**（intent-hit 口径上微调也比零样本高 2.73pp）。结论：

- 当前场景（同主题召回为主）**精排不值得 1.1GB 权重 + 500ms 预算**，线上默认 `NoopReranker`，直接取召回序
- 若场景转向簇内硬负例区分，用微调版本而不是零样本
- 终判需要生成侧评测跑有/无精排两组，脚本保留，路径没有删

## 评测门槛

核心链路有 CI 回归线，不达标直接失败；生成与可答性评测花 token 或依赖外网，手动或夜间执行。

| 环节 | 当前基线 | CI 回归线 | 脚本 |
|---|---|---|---|
| boundary 必中率（60 条） | 0.867 | ≥ 0.85 | `ml/experiments/eval_router.py` |
| heldout subset accuracy（240 条） | 0.813 | ≥ 0.80 | 同上 |
| order precision（自然分布加权） | 0.889 | ≥ 0.88 | 同上 |
| human recall | 0.793 | ≥ 0.75 | 同上 |
| 重排 pairwise | 0.848 | ≥ 0.774（零样本基线） | `ml/experiments/eval_rerank.py` |
| 检索 recall@10（线上口径） | 0.696 | ≥ 0.68 | `ml/experiments/eval_retrieval.py` |
| 生成 faithfulness（100 条端到端） | 4.71 | ≥ 4.0（夜间，不进 per-commit） | `ml/experiments/eval_generation.py` |
| 生成贴合度 cosine | 0.622 | ≥ 0.572（回归线） | 同上 |
| 可答性误拦（域内 130 条） | 0.015 | ≤ 0.05（手动） | `ml/experiments/eval_answerability.py` |
| 可答性拦截（域外 60 条） | 0.700 | ≥ 0.50（手动） | 同上 |

路由基线是**门控前**口径（规则 + BERT，无兜底），也是 CI 退出码的唯一依据；门控后的完整口径手动跑 `eval_router.py --gate on`。生成评测花 token 且依赖外网，只在夜间或手动执行。全部数字与消融过程见 [docs/experiments.md](docs/experiments.md)。

重排门槛保的是「微调路径可开启、且相对零样本不退化」，不是线上指标——线上默认 noop，不加载精排权重。

## 快速开始

```bash
export DEEPSEEK_API_KEY=sk-...   # 必填；缺失时启动脚本直接报错退出
./apps/deploy/run.sh             # 启动并打开 http://127.0.0.1:8090
./apps/deploy/run.sh --stop      # 停止
```

本地常驻 BERT + bge（不加载精排），首次启动要等两个模型权重加载完。空状态的四个建议问题从模板池随机抽取，刷新页面、新建会话、切换用户都会换一批；订单类问题的单号取自当前 mock 用户自己的订单，无订单用户（如 `u_zhou`）只出知识/人工题，不会出现点了查不到的按钮。四种能力组合如下（示例以 `u_chen` 视角），也可以直接输入：

| 能力组合 | 示例问题 |
|---|---|
| 知识检索 | 退款多久到账？ |
| 订单工具 | 我的订单 SO2026102901 现在到哪了？ |
| 订单 + 政策并行 | 我的订单 SO2026102901 现在到哪了，另外退款政策怎么说的？ |
| 转人工（规则层毫秒级命中） | 我要投诉，你们这服务太差了 |

**不下载模型权重也能验证什么：**

```bash
.venv/bin/python3 -m pytest apps/inference/tests -q      # 88 passed，秒级（无 DATABASE_URL 时 1 条 parity 测试跳过）
JAVA_HOME=$(/usr/libexec/java_home -v 17) mvn -f apps/backend/pom.xml test   # 115 条单测 + Java↔Python 契约
```

推理层单测与网关契约测试都不加载模型权重，几分钟内可跑完。完整 Demo 另需约 3GB 权重（ModelScope 下载底座 + 微调产物，见下方环境准备）。

<details>
<summary>环境准备</summary>

- `.venv/`：`uv venv && uv pip install -r apps/inference/requirements.txt`
- `ml/models/models/`：ModelScope 下载的底座（`bert-base-chinese`、`bge-small-zh-v1.5`、`bge-reranker-base`）
- `ml/models/intent-multitask-bert/`：微调产物（训练脚本见 `ml/experiments/`）
- `ml/models/rerank-finetuned/`：可选，仅在显式启用精排时需要
- JDK 17、Node 20+（仅首次构建前端需要）

</details>

<details>
<summary>Docker 部署（Linux 服务器）</summary>

```bash
cd apps/deploy && docker compose up -d --build
```

容器内无 MPS；`DEEPSEEK_API_KEY` 为必填，缺失时 compose 直接报错。模型权重与语料 volume 挂载，不打进镜像。

需要 pgvector 召回时显式带 profile（默认 InMemory）：

```bash
DATABASE_URL=postgresql://csagent:csagent@postgres:5432/csagent \
  docker compose --profile pgvector up -d
```

首次启动后用 `ml/experiments/build_vector_index.py` 灌库。

</details>

## 技术栈

| 层 | 选型 |
|---|---|
| 模型训练 | PyTorch · Transformers · BERT-base-chinese（多任务意图）· bge-small-zh-v1.5（检索）· bge-reranker-base（可选精排） |
| 模型服务 | FastAPI · Uvicorn · SSE 流式 · Prometheus 指标 |
| 网关 | Spring Boot 3.3 · Java 17 · SSE · H2（工单） |
| 前端 | React · TypeScript · Vite |
| 存储 | InMemory / pgvector 双实现（`DATABASE_URL` 切换，无静默回退） |
| 测试 | pytest · JUnit 5 · Java↔Python 共享契约 fixtures · 真实栈 HTTP/SSE E2E |
| 可观测 | `X-Trace-Id` 贯穿网关与模型服务 · Prometheus 分段指标 |
| 部署 | Docker Compose |

## 可观测

```bash
curl -H 'X-Trace-Id: demo-123' -N -X POST 'http://127.0.0.1:8090/api/chat?sessionId=s1' \
  -H 'Content-Type: application/json' -d '{"query":"退款多久到账"}'
curl -s http://127.0.0.1:8090/actuator/prometheus | grep cs_chat
curl -s http://127.0.0.1:8000/metrics | grep model_service
```

网关指标：`cs.chat.plan{source,route}`、`cs.chat.capability{capability,outcome}`、`cs.chat.generate{phase}`、`cs.chat.requests{outcome}`。模型服务指标：`model_service_request_seconds{path,method,status}`。Prometheus 暴露时点号转下划线，即 `cs_chat_plan`、`cs_chat_capability` 等。日志每行带 `traceId`，SSE `error` 事件也带 `traceId`，用户反馈可直达日志。口径见 [architecture.md §11](docs/architecture.md)。

## 数据与训练

**CSDS**（Customer Service Dialogue Summarization，EMNLP 2021）：9,101 组客服对话、18,018 个 QA 对，意图标签 207 类长尾分布（前 80 类覆盖 90.1%）。

- `build_dataset.py`：`AnsSummLong` 是第三人称摘要，不能直接当生成目标——通过 `AnsSummLongUttIDs` 回溯 `Dialogue` 原始客服发言，还原 17,855 组口语问答对
- `build_route_eval.py`：构造 300 条独立路由评测集（能力集多标签标注，与训练集隔离）
- `train_intent_multitask.py`：多任务 BERT（route 3 × sigmoid + topic 41 × softmax）联合训练
- `train_rerank.py` / `build_rerank.py`：精排微调，负例分三级难度（同会话硬负例 > 同意图 > 随机）
- `intent_capability_map.yaml`：41 类主题 → 能力集显式映射，CI 校验无缺键；`check_route_drift.py` 夜间比对模型实际输出与声明，漂移只告警

每个脚本的输入、产物与门槛见 [ml/experiments/README.md](ml/experiments/README.md)。

## 文档

- [docs/architecture.md](docs/architecture.md) — 架构、接口契约、决策台账（唯一事实来源）
- [docs/experiments.md](docs/experiments.md) — 数据构建、训练与全部评测数字
- [docs/openapi.json](docs/openapi.json) — 模型服务 OpenAPI 契约，改接口后必须重跑 `dump_openapi.py` 并提交
- [apps/inference](apps/inference) — 模型服务与 API
- [ml/experiments](ml/experiments) — 可复现的训练与评测脚本

## 目录

```
apps/
  inference/      FastAPI 模型服务（原子推理：plan / retrieve / rerank / generate）
    intent/        规则 · 多任务 BERT · 门控
    retrieval/     召回（InMemory / pgvector）· 精排
    llm/           DeepSeek
  backend/        Spring Boot 网关：会话、能力编排、业务数据、工单、prompt、SSE
  contract/       Java ↔ Python 共享契约 fixtures
  e2e/            真实栈 HTTP/SSE 黑盒测试
  frontend/       React 聊天界面（构建产物打进网关）
  deploy/         docker-compose、Dockerfile、一键启动 run.sh
ml/
  runtime/        检索运行时（语料 + bge，供模型服务复用）
  experiments/    离线实验：数据构建、训练、评测
  data/           raw 原始 CSDS / processed 处理产物
  models/         底座模型、微调产物
docs/             设计、实验记录、截图
```

## License

[MIT](LICENSE)
