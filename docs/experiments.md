# 实验与评测

> 全部数字对应线上当前链路：L0 规则 + L1 多任务 BERT → 门控 → 检索 →（可选）精排 → DeepSeek 生成。
> 已淘汰方案的实验（单任务分类基线、四层路由、本地生成模型、旧拒答策略）不在此保留；相关取舍的结论已写进 [architecture.md](./architecture.md)。
> **基线数据截至 2026-11-09。** 同一指标再次评测后，请更新本页与 architecture.md 的门槛表。

## 评测索引

| 环节 | 指标 | 当前基线 | CI 回归线 | 脚本 |
|---|---|---|---|---|
| 路由 | boundary 必中率（60 条） | 0.8667 | ≥ 0.85 | `ml/experiments/eval_router.py` |
| 路由 | heldout subset（240 条） | 0.8125 | ≥ 0.80 | 同上 |
| 路由 | order precision（自然分布加权） | 0.8894 | ≥ 0.88 | 同上 |
| 路由 | human recall | 0.7931 | ≥ 0.75 | 同上 |
| 检索 | recall@10（线上 served 口径） | 0.6962 | ≥ 0.68 | `ml/experiments/eval_retrieval.py` |
| 精排 | pairwise accuracy | 0.8484 | ≥ 0.7740（零样本基线） | `ml/experiments/eval_rerank.py` |
| 生成 | faithfulness（100 条端到端） | 4.71 | ≥ 4.0（夜间） | `ml/experiments/eval_generation.py` |
| 生成 | cosine | 0.622 | ≥ 0.572（夜间） | 同上 |
| 可答性 | 误拦（in_kb 130 条） | 0.015 | ≤ 0.05（手动） | `ml/experiments/eval_answerability.py` |
| 可答性 | 拦截（out_kb 60 条） | 0.700 | ≥ 0.50（手动） | 同上 |
| 映射 | route 头 vs YAML 声明漂移（41 主题 × 3 能力） | 0 处 | 只告警，不拦截 | `ml/experiments/check_route_drift.py` |
| 向量库 | InMemory ↔ PgVector 一致性 | 10/10 query 通过 | — | `apps/inference/tests/test_pgvector_parity.py` |

路由是**门控前**口径（规则 + BERT，不经过 `apply_gate`），也是 CI 退出码的唯一依据；门控后口径手动跑 `eval_router.py --gate on`。生成与可答性评测花 token 或依赖权重，手动或夜间执行。

## 目录

[1 数据](#1-数据) · [2 意图识别](#2-意图识别) · [3 检索](#3-检索) · [4 精排](#4-精排) · [5 生成质量](#5-生成质量) · [6 可答性闸门](#6-可答性闸门) · [7 复现](#7-复现)

## 1. 数据

**CSDS**（Customer Service Dialogue Summarization，EMNLP 2021）：9,101 组客服对话、18,018 个 QA 对，意图标签 207 类长尾分布（前 80 类覆盖 90.1%）。

`AnsSummLong` 是第三人称摘要，不能直接当生成目标。`build_dataset.py` 通过 `AnsSummLongUttIDs` 回溯 `Dialogue` 中的原始客服发言，还原 17,855 组口语问答对作为 `qa_train`；`qa_test`（1,509 条问题）与训练集隔离。

主题空间收成 41 类（40 个头部意图 + 「其他」），划分 train 15,932 / val 1,410 / test 1,354（非空 intent）。

两个独立评测集：

- `route_eval.jsonl` 300 条：boundary 60 条（工具边界问法，人工多标签）+ heldout 240 条（qa_test 经映射表派生），与训练集隔离。能力级标签由 41 类主题经 `intent_capability_map.yaml` 确定性派生，不额外标注。
- `answerability_eval.jsonl` 190 条：qa_test 抽样 120 + 域内硬例 10 + 域外手写 60，全部人工标签（见 §6）。

## 2. 意图识别

### 2.1 多任务 BERT

一次前向同时输出两个 head：

| head | 输出 | 用途 |
|---|---|---|
| route | 3 × sigmoid（knowledge / order / human） | 决定能力集 |
| topic | 41 × softmax | 检索分区加权 |

- 基座：`google-bert/bert-base-chinese`（112M）。底座选择的依据是单任务对照：同一份数据与超参下，`bge-small-zh` 单任务 test macro-F1 0.4924（「其他」类 recall 0.003），`bert-base-chinese` 0.5560（「其他」类 0.1207）——bge-small 是对比学习检索模型，分类不是它的专长。
- 配置：max_length 64，batch 32，lr 3e-5，6 epoch；route 正例加权用 sqrt 阻尼，topic 用 balanced。
- 阈值：knowledge / order / human 在 val 上网格搜索，约束 order precision ≥ 0.90。val 上无组合达标（order P 最高 0.870），脚本按设计回退全 0.5。
- 环境：Apple M1 / MPS。

| 划分 | topic acc | topic macro-F1 | route subset |
|---|---|---|---|
| val | 0.5511 | 0.5524 | 0.8170 |
| test | 0.5901 | 0.5644 | 0.8272 |

test 能力级：

| 能力 | P | R | F1 | support |
|---|---|---|---|---|
| knowledge | 0.9755 | 0.9507 | 0.9630 | 1258 |
| order | 0.7481 | 0.7840 | 0.7656 | 375 |
| human | 0.7188 | 0.7797 | 0.7480 | 59 |

「其他」类 recall 0.3158（323 条）——长尾仍被头部主题吸收，是 topic 准确率的主要拖累，也会 cascade 到能力集。

### 2.2 独立路由评测（route_eval.jsonl）

| 来源 | 条数 | 指标 | 结果 |
|---|---|---|---|
| boundary | 60 | 必中率 `gold ⊆ predicted` | 0.8667 |
| heldout | 240 | 多标签完全匹配 | 0.8125 |

heldout 能力级：knowledge P/R 0.944/0.971，order P/R 0.889/0.736，human P/R 0.929/0.722。

human 全量召回 0.7931（rule 12/12，bert 11/17）；heldout 的 5 条漏判全是间接表达（「你不能转接?」「能帮忙联系下呀?」「让站点再帮忙派送」）。单次前向 18.1ms，P95 20.6ms。

全量 300 条 topic top-1 0.6375（153/240，qa_test 1 条一词多标注跳过），source 分布 bert 288 / rule 12。

读法：

- route 主指标的瓶颈在 41 类 topic：主题错会 cascade 到能力集。换更强中文编码器（MacBERT / Chinese-RoBERTa）与长尾数据清洗是下一步。
- order precision 在自然分布（先验 0.364）加权后仍是 0.8894，分层抽样没有抬高指标。
- human 灰区召回是明确缺口：规则只覆盖显式表达，间接请求只能靠模型本身。

### 2.3 L2 LLM 兜底为什么删掉

在「规则 + BERT」之上曾有一层 LLM 兜底：`max(route_scores) < 0.60` 时调 DeepSeek 补判，超时 0.8s。预登记规则是「`Δhuman recall ≥ +0.03` 或 `Δboundary ≥ +0.02` 或 `Δheldout ≥ +0.02`，且 `ΔP95 ≤ +150ms` 才保留」。

测量方法：`eval_router.py --gate on --fallback {off,real}` 跑全量 300 条，DeepSeek 有随机性，实跑 4 轮逐轮记录（该参数已随兜底代码移除）。基线是确定性的：

| 指标 | 基线（--fallback off） |
|---|---|
| 来源分布 | bert 288 / rule 12 |
| boundary 必中率 | 0.8667 |
| heldout subset | 0.8125 |
| order precision | 0.8894 |
| human recall | 0.7931 |
| 平均 / P95 | 18.2ms / 20.4ms |

A/B 逐轮：

| 轮次 | 触发 / 结局 | boundary | heldout | order P | human R | ΔP95 | 退出码 |
|---|---|---|---|---|---|---|---|
| R1 | — / hit≥1 | 0.8667 | 0.8125 | **0.877** | 0.793 | +3.1ms | **1** |
| R2 | 3 / hit2 + 超时1 | 0.8833 | 0.8125 | 0.889 | 0.862 | +3.7ms | 0 |
| R3 | 3 / hit2 + 超时1 | 0.8833 | 0.8125 | 0.889 | 0.828 | +3.4ms | 0 |
| R4 | 3 / hit3 | 0.9000 | 0.8125 | 0.889 | 0.862 | +2.3ms | 0 |

结论是**删除**：

- 触发率 3/300 = 1.0%，结论由 1–2 条问法构成，评测集无法统计区分兜底好坏。
- 稳定收益只有一条「已经问了三次了，没人管吗？」（gold human，BERT 漏判）；
  另一条在 R1 判成 order，凭空多一个假阳，把 order precision 从 0.889 压到 0.877，
  跌破 CI 下限 0.88。
- 机制上兜底只能给能力置 1.0，**不能删标签**；而路由的主要错误是过预测，结构上修不了。
  heldout subset 四轮全部无变化即为此。
- 同类问法跨轮不稳定，说明收益来自网络抖动而不是灰区能力。

删除后代码只剩 `rule | bert` 两个来源，`eval_router.py` 不再有 `--fallback` 参数。

### 2.4 route 头与 YAML 声明的一致性（漂移检测）

`intent_capability_map.yaml` 只喂训练标签，运行时能力集由 route 头直接输出，不做 topic→能力查表（architecture.md §4.3）。好处是模型能修正 topic 误判，代价是**表上写的和线上跑的之间没有自动约束**。静态校验只看 YAML 自身是否完整，改表不重训、重训不回写表、或换底座模型，它都不会报错。

`check_route_drift.py` 补行为层：每个主题从 `qa_train` 抽 40 条（seed 42）过一遍真实分类器，统计该主题样本中每个能力被 route 头命中的比例，与 YAML 逐格对比。规则层短路样本（`source=rule`）跳过，它们不经过 route 头。

判定口径留死区：声明为真而命中率 < 0.60 报「漏」，声明为假而命中率 > 0.40 报「多」。40 样本时比例的标准差约 8%，死区避免二项噪声把告警弄抖。

**基线（2026-11-09，41 主题 / 123 格）：漂移 0 处。** 分离度很干净：

- 声明为真的格子，命中率最低 88%（物流全程跟踪的 knowledge，其次服务单查询 90%）
- 声明为假的格子，命中率最高 10%（「其他」的 order），其余 ≤ 5%
- 三个 human 主题命中 98–100%（联系客户 / 联系客服 / 联系配送），规则层之外模型也已学会
- 「其他」top-1 只有 50%（桶内异质），但能力命中率仍与声明一致 —— 这正是「能力集不必等于 topic」的设计意图

因为是训练集口径，这个数字首先证明「训练时的标签派生与 YAML 一致」，并守住后续对 YAML 的事后改动。它不测泛化，泛化看 §2.2。默认退出码 0（只告警），CI 里 `continue-on-error`；`--strict` 才把漂移当失败。

## 3. 检索

### 3.1 线上口径 recall@10

评测口径必须和线上一致，否则门槛数字没有意义：

- 模型：`BAAI/bge-small-zh-v1.5`，query 加 bge 中文 instruction，文档侧不加前缀，`normalize_embeddings=True`。
- 候选库：**只含 train 答案 17,855**。线上 KB 就是 `qa_train.jsonl`，test 答案不进库。
- query：test 的 1,509 个问题，有效相关 query 1,353（其余 intent 在 train 中无同意图答案）。
- 相关口径：query 的原始 intent 在 train 中出现的答案都算相关。客服语料有大量语义等价重复，且线上 KB 不含 test 答案，所以不用「必须是自己的答案」的严格口径。

| 口径 | n | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---|---|---|---|---|---|
| pure（纯向量） | 1353 | 0.2129 | 0.5462 | 0.6637 | 0.7827 | 0.3656 |
| served（路由 topic 分区 +0.05） | 1353 | 0.4597 | 0.6349 | **0.6962** | 0.7746 | 0.5443 |

`served` 是线上真实链路：多任务 BERT 预测的 topic 把候选按 41 类分区，命中分区 +0.05 相似度后再排序（`pipeline.INTENT_BOOST`）。CI 门槛守 served recall@10 ≥ 0.68，留约 2 个点回归余量。

### 3.2 pgvector parity

- InMemory：qa_train 17,855 条全表暴力点积；PgVector：同一批 `answer` 编码后灌库，hnsw 索引（`vector_cosine_ops`）。
- 两边编码口径一致：文档侧无指令前缀，query 侧补 `QUERY_INSTRUCTION`，`normalize_embeddings=True`。
- 脚本取 `qa_test` 前 10 条去重 question，各取 top-10，比较文本集合交集，要求 ≥8。默认 skip，需 `DATABASE_URL`。

```
灌库     17,855 条写入 knowledge
索引     idx_knowledge_embedding USING hnsw (embedding vector_cosine_ops)
parity   10/10 query 通过（每条 top-10 交集 ≥8）
耗时     1 passed in 25.46s
```

允许少量边界交换：pgvector 用 `top_k × 5` 候选池截断后应用层重排，InMemory 全表打分，并列附近的顺序可能不同；整片不同才算失败。

## 4. 精排

### 4.1 微调 vs 零样本（pairwise）

`rerank_test.jsonl` 2,995 条正负对，按负例难度分三级（随机 / 同会话 / 同意图），指标是正例得分是否高于负例。

| 负例类型 | 零样本 | 微调 | 提升 |
|---|---|---|---|
| random | 0.8489 | **0.9105** | +6.16 |
| same_session | 0.7115 | **0.7986** | +8.71 |
| same_intent | 0.6533 | **0.7421** | +8.88 |
| **总体 pairwise** | 0.7740 | **0.8484** | +7.44 |
| MRR@2 | 0.8870 | **0.9242** | +3.72 |

微调配置：`BAAI/bge-reranker-base`，`rerank_train.jsonl` 35,521 条三元组展开为 71,042 对，batch 8，max_length 128，lr 2e-5，1 epoch，MPS，约 2 小时 20 分。收益集中在硬负例，same_intent 从 0.65 提到 0.74。CI 回归线 0.774 是零样本基线，微调不得低于它。

### 4.2 精排取舍消融

真正要回答的问题不是「rerank pairwise 准不准」，而是「重排后的 top-2 比不排的 top-2 更好吗」。`eval_rerank_ablation.py` 复用检索语料与相关口径：

- KB = qa_train 17,855 条答案；query = qa_test 有效 1,353 条；召回走线上 served 口径 top-10。
- 三组：`noop` 直接取召回序前 2；`zero_shot` 用 bge-reranker-base 零样本重排；`finetuned` 用 rerank-finetuned 重排。后两组只对这 10 条候选重排，再取 2。
- 指标：hit@1 / hit@2（top-2 内是否命中相关答案）。

```
有相关集合的 query：1353
  noop       hit@1 0.4597   hit@2 0.5477
  zero_shot  hit@1 0.3681   hit@2 0.5026
  finetuned  hit@1 0.3954   hit@2 0.5299

相对 noop：
  zero_shot  hit@1 -0.0916   hit@2 -0.0451
  finetuned  hit@1 -0.0643   hit@2 -0.0177
```

**为什么这个口径偏向 noop**：相关集合是「同 intent 的全部答案」，召回也是同一 embedding 空间的排序，noop 保留的就是 embedding 序，天然与相关口径同源。reranker 是独立交叉编码器，会把 embedding 眼里的高分候选打散，于是 hit@2 反而降。两个口径不矛盾：§4.1 的 pairwise 测「簇内细粒度区分」，这是 reranker 的主场；本节 intent-hit 测「簇级召回」，embedding 已经做得很好，看不见 reranker 的收益。

取舍：

1. **精排非必需。** 线上默认 `NoopReranker`，直接取召回序，省 1.1GB 权重 + 500ms 预算，簇级召回不掉点。500ms 超时自动降级为召回序的路径仍然保留。
2. **要保留精排就用微调版，不用零样本。** 微调比零样本 hit@2 +2.73pp；维持精排时零样本没有保留价值。
3. **本消融不是「精排无价值」的终判。** 终判需要生成侧跑有 / 无精排两组，对比喂给 DeepSeek 的 2 条答案质量；脚本路径保留。

## 5. 生成质量

### 5.1 端到端评测（pipeline / oracle）

路由、检索、精排各有评测，但都没有回答「整条链路拼起来，用户看到的那句话对不对」。`eval_generation.py` 补这一环，样本 `qa_test.jsonl` 按 seed 42 采 100 条（过滤空问题与答案 < 10 字）。两条口径跑同一批样本：

- **pipeline**：意图 → 检索（top10，topic +0.05）→ 精排（默认 noop，评测可选开）→ `[知识]` 上下文（各限 90 字）→ DeepSeek。端到端数字，对应线上。
- **oracle**：gold answer 直接当上下文（不截断），只测生成与 prompt 的天花板，用来把「检索错」和「生成差」分开。

三条运行时对齐（不复制、随线上自动变）：系统 prompt 读 `apps/backend/src/main/resources/prompts/chat-system.txt`；用户消息复刻 `Context + "\n\n用户问题：" + query`；知识块复刻 `ContextAssembler` 的 `[知识] ` 前缀、2 条上限、90 字截断。

判分用 DeepSeek（temperature 0）打 faithfulness / relevance / style 三项 1–5；另有 bge 句向量余弦（生成回答 vs gold answer）作机器口径交叉验证。

首次基线（旧 prompt）：

| 指标 | pipeline | oracle | 差 |
|---|---|---|---|
| faithfulness | 4.54 | 4.47 | +0.07 |
| relevance | 4.52 | 4.56 | −0.04 |
| style | 4.50 | 4.45 | +0.05 |
| **三项均值** | **4.52** | **4.49** | +0.03 |
| cosine 均值 | **0.614** | **0.713** | −0.099 |
| 检索耗时 avg | 165ms | — | — |
| 首字 avg | 718ms | 729ms | — |
| 全链 avg / P95 | 1213ms / 1566ms | 1246ms / 1630ms | — |
| error_rate | 0% | 0% | — |

读法：

1. **判分均分对检索误差几乎不敏感，cosine 敏感。** 把 gold answer 换成检索结果，三项均分只差 0.03（在噪声内），cosine 掉 0.099。检索到的是同 topic 的高相关答案，通常已足够回答问题，判分器看「有没有编造、答没答到」就给高分。要看「检索拖了多少」必须看 cosine 和检索独立指标；judge 均分度量的是生成能力，不是端到端全量。
2. **脱敏占位符被模型补成具体数字，37 条里 9 条（24%）。** 见 §5.2。
3. **faithfulness ≤ 3 的 4 条里，2 条源于占位符补全**，另 2 条是模型把「今天可能到 / 今天送不到」这种模棱两可的 FAQ 说成确定承诺、把流程顺序说反。三类都指向同一件事：**上下文有歧义时模型倾向于给确定答案**。

pipeline 当前基线（prompt 修复后）是三项 **4.63**、faithfulness **4.71**、cosine **0.622**、首字 1009ms、全链 avg 1545ms / P95 2216ms、零错误，全部达标。oracle 只在旧 prompt 下测过一次（4.49 / 0.713），跨 prompt 比较只看 pipeline。

评测成本：100 条 pipeline + 100 条 oracle ≈ 400 次 DeepSeek 调用，几分钟。不进 per-commit CI；`--judge off` 可做无判分的链路冒烟。

### 5.2 占位符 prompt 修复验证

首轮评测暴露的占位符补全问题已修复：`chat-system.txt` 第 3 行从「属于正常内容」扩为「属于正常内容，但不要猜测或补全成具体值（例如不要把 `[数字]` 小时说成 48 小时）；用户需要具体数字时，说明以订单页或实际规则为准」。prompt 是唯一读取源（网关 `ChatOrchestrator` 与评测脚本运行时读同一个文件），改一处两条路径同时生效。

同口径复测 100 条（`--limit 100 --seed 42 --judge auto --context pipeline`）：

| 指标 | 修复前 | 修复后 | Δ |
|---|---|---|---|
| faithfulness | 4.54 | **4.71** | +0.17 |
| relevance | 4.52 | **4.63** | +0.11 |
| style | 4.50 | **4.56** | +0.06 |
| 三项均值 | 4.52 | **4.63** | +0.11 |
| cosine 均值 | 0.614 | **0.622** | +0.008 |
| 占位符补全 | 9/37（24%） | **2/37（5%）** | −19pt |
| faithfulness ≤ 3 | 4 条 | **2 条** | −2 |
| 首字 avg | 718ms | 1009ms | +291ms |
| 全链 avg / P95 | 1213ms / 1566ms | 1545ms / 2216ms | +332ms / +650ms |
| error_rate | 0% | 0% | — |

- **质量净提升，不靠模型变强。** 三项均分 +0.11、cosine 也小幅上升，说明「如实说以订单页为准」没有牺牲贴合度，反而减少了编造带来的偏差。
- **代价是延迟。** 模型不再直接用先验填数，改为多说一句「以页面为准」，首字与全链各涨约 0.3s；P95 2216ms 仍在 3000ms 预算内。用少量延迟换事实准确，客服场景值得。
- **剩余 2 条属「常识压过指令」**：SD 卡那条仍说 `Class 10`（上下文是 `class [数字]`），耳套那条把 `[数字]` 补成「3 个 / 各 1 个」。这类需要上下文侧脱敏改写或更强约束，prompt 单独已到边界。
- **低分条目换了类型。** 修复后 2 条转为医学判断编造（上下文未提）、流程顺序说反（家电回收答「先下单」，上下文要求「先创建回收单」）。都不是 prompt 能解决，属检索 / 知识组织问题。

cosine 回归线随基线重算为 **0.572**（基线 0.622 − 0.05）。

## 6. 可答性闸门

qa_test 每条问题在 qa_train 里都有答案，天然可答，所以现有评测回答不了「知识库到底有没有能答的内容」：域外问题（天气、股票、闲聊）同样会被检索召回一两条弱相关，生成模型于是拿常识补，答案通顺但业务无关。闸门用召回首位分数做三档判定，`no_context` 时不调生成。

### 评测集

`build_answerability_eval.py` 生成 190 条，全部人工标签：

| 切片 | n | 来源 | 标签 |
|---|---|---|---|
| qa_test | 120 | `qa_test.jsonl` 固定 seed 42 抽样 | in_kb |
| boundary | 10 | 手写：字面含域外实体，问的是域内政策 | in_kb |
| handwritten | 60 | 手写：天气、编程、医疗、生活常识 | out_kb |

boundary 是硬例：iPhone 退货、猫粮拆封退换、口红换色号、大促发货、补差价等。它们字面有域外实体但问的是域内政策，必须放行。

### 方法

信号取线上同源链路：多任务 BERT topic → `retrieve_indexed` top10（topic +0.05）→ `hits[0]` 分数；检索为空记 `no_context`。阈值 T_LOW=0.58 / T_HIGH=0.64 与 `AnswerabilityGate` 同值。

```bash
PYTHONPATH=apps:ml/runtime .venv/bin/python3 ml/experiments/eval_answerability.py
.venv/bin/python3 ml/experiments/eval_answerability.py --sweep --head 10
```

### 结果（阈值 0.58 / 0.64）

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

四项预登记门槛全 PASS（qa_test 误拦 ≤5% 实际 1.7%；boundary 误拦 ≤20% 实际 0%；qa_test answer ≥75% 实际 90.0%；out_kb 拦截 ≥50% 实际 70.0%）。

阈值网格（节选）：

| T_low | T_high | in no_ctx | in answer | out no_ctx | out answer |
|---|---|---|---|---|---|
| 0.58 | 0.62 | 0.015 | 0.938 | 0.700 | 0.117 |
| **0.58** | **0.64** | **0.015** | **0.900** | **0.700** | **0.083** |
| 0.62 | 0.66 | 0.062 | 0.846 | 0.883 | 0.067 |

0.58/0.64 是平衡点：再抬阈值把误拦推过 5% 硬门槛，再降则 out 拦截掉得太快。

### 逐条观察

最危险的误拦（in_kb 判 no_context）：

- 0.5295 `默认几个盘 ?`（主题「其他」，长尾）
- 0.5548 `还是现在 ?`（口语残句）

漏拦（out_kb 判 answer）：

- 0.7319 `明天是星期几` / 0.7172 `现在几点了`（时间类触发日期主题）
- 0.7159 `我的手机屏幕碎了去哪修`（含手机实体）
- 0.6812 `高铁票怎么改签` / 0.6433 `今天股市会涨吗`

时间 / 常识类偶有高分，靠 weak 提示兜底，不追求全拦。

### 结论与限制

1. **召回 cosine 是唯一可校准的免费信号。** 微调 reranker 分数挤压在 0.35–0.97，域外问题同样 0.9+，AUC 仅 0.565，不能做绝对阈值；召回首位 AUC 0.9605。
2. **in_kb 误拦率是乐观下限。** 正样本来自 qa_test，构造上都有 KB 答案；真实用户的「域内但知识库没覆盖」问题会更难，线上误拦可能高于 1.7%。
3. **阈值绑数据形态。** 换 embedding、改 `INTENT_BOOST`、改 top_k 都要重跑本评测重定阈值。
4. 不进 per-commit CI（依赖权重与 MPS），手动回归。

## 7. 复现

```bash
# 路由（CI 口径，门控前）
.venv/bin/python3 ml/experiments/eval_router.py
# 路由（门控后参考口径）
.venv/bin/python3 ml/experiments/eval_router.py --gate on

# 检索（served = 线上口径）
.venv/bin/python3 ml/experiments/eval_retrieval.py

# 精排 pairwise
.venv/bin/python3 ml/experiments/eval_rerank.py
# 精排取舍消融
.venv/bin/python3 ml/experiments/eval_rerank_ablation.py

# 生成（需 DEEPSEEK_API_KEY，90s 级）
PYTHONPATH=apps:ml/runtime .venv/bin/python3 ml/experiments/eval_generation.py \
  --limit 100 --seed 42 --context pipeline --judge auto

# 可答性
PYTHONPATH=apps:ml/runtime .venv/bin/python3 ml/experiments/eval_answerability.py

# 向量库 parity（需 DATABASE_URL 与已灌库的 pgvector）
docker compose -f apps/deploy/docker-compose.yml --profile pgvector up -d postgres
.venv/bin/python3 ml/experiments/build_vector_index.py
DATABASE_URL=postgresql://csagent:csagent@localhost:5432/csagent \
  .venv/bin/python3 -m pytest apps/inference/tests/test_pgvector_parity.py -q
```
