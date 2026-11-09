# ml/experiments

离线实验脚本：数据构建 → 训练 → 评测 → CI 校验。不参与运行时，产物写进
`ml/data/processed/` 和 `ml/models/`。具体数字与消融过程见
[docs/experiments.md](../../docs/experiments.md)。

```
CSDS raw
  └─ build_dataset ──► qa_{train,val,test}.jsonl
                         ├─ build_route_eval ─────────► route_eval.jsonl
                         ├─ build_rerank ──────────────► rerank_{train,val,test}.jsonl
                         ├─ build_answerability_eval ──► answerability_eval.jsonl
                         ├─ build_vector_index ────────► PostgreSQL + pgvector
                         └─ train_intent_multitask ────► models/intent-multitask-bert
                                                             ▲
intent_capability_map.yaml ── check_capability_map ──────────┘
                           └─ check_route_drift（跑真实 route 头比对）
```

## 数据构建

| 脚本 | 做什么 | 产物 |
|---|---|---|
| `build_dataset.py` | 把 CSDS 的 `AnsSummLong`（第三人称摘要）通过 `AnsSummLongUttIDs` 回溯 `Dialogue` 里 `speaker == "A"` 的原文，还原成口语问答对 | `qa_{train,val,test}.jsonl` |
| `build_route_eval.py` | 拼 300 条路由评测集：boundary 60 条取自 `tool_eval.jsonl`（手写边界问法），heldout 从 `qa_test` 按 topic 分层抽样（seed 42，抽样目标 252 条，与 boundary 去重后实际 240 条） | `route_eval.jsonl` |
| `build_rerank.py` | 构造 (query, pos, neg) 三元组，负例按难度三级：同会话 > 同意图 > 随机 | `rerank_{train,val,test}.jsonl` |
| `build_answerability_eval.py` | in_kb 120 条（量误拦）+ out_kb 60 条域外问题 + boundary 10 条（问域内政策但带域外实体，必须放行） | `answerability_eval.jsonl` |
| `build_vector_index.py` | 把 `qa_train` 灌进 PostgreSQL + pgvector。文档侧不加指令前缀（与 `pipeline.py` 的 `corpus_emb` 对齐），hnsw 索引灌完再建 | pgvector 表 |

`build_vector_index.py` 需要先起库：

```bash
DATABASE_URL=postgresql://csagent:csagent@localhost:5432/csagent \
  PYTHONPATH=apps:ml/runtime \
  .venv/bin/python3 ml/experiments/build_vector_index.py --reset
```

## 训练

| 脚本 | 做什么 | 产物 |
|---|---|---|
| `train_intent_multitask.py` | 共享 BERT 编码器 + 两个头：route 3×sigmoid、topic 41×softmax，一次前向同时输出。route 标签由 topic 经 YAML 派生，不人工标。阈值在 val 上逐标签搜（subset accuracy 最优，order precision 不低于 0.90） | `models/intent-multitask-bert/`（`encoder/` + `heads.pt` + `config.json`） |
| `train_rerank.py` | 在 `rerank_train.jsonl` 上微调 bge-reranker-base（cross-encoder）。`CrossEncoder.fit` 没有 evaluator 不会自动存盘，必须显式 `save()` | `models/rerank-finetuned/` |

## 评测

| 脚本 | 评什么 | 门槛 / 用途 | 进 CI |
|---|---|---|---|
| `eval_router.py` | boundary 必中率、heldout 完全匹配、order precision、human recall | 0.85 / 0.80 / 0.88 / 0.75；退出码只由门控前 BERT-only 口径决定 | 是 |
| `eval_retrieval.py` | 召回。`pure`（纯向量）与 `served`（topic 分区 +0.05）两组 | served recall@10 ≥ 0.68 | 是 |
| `eval_rerank.py` | 重排 pairwise accuracy + MRR@2，按负例难度分组 | ≥ 0.774（零样本基线） | 是 |
| `eval_rerank_ablation.py` | noop / zero_shot / finetuned 三组比 top-2 命中 | 无硬门槛，回答「1.1GB + 500ms 值不值」 | 否 |
| `eval_generation.py` | 走完整链路拿回答，独立 httpx 调 judge 三维打分 | faithfulness ≥ 4.0；花 token + 外网 | 否（夜间/手动） |
| `eval_answerability.py` | 可答性闸门。top1 分数三档分流 | 域内误拦 ≤ 5%，域外拦截 ≥ 50% | 否（需权重 + MPS） |

`eval_generation.py` 有三个必须和网关对齐的锚点：系统 prompt 直接读
`apps/backend/src/main/resources/prompts/chat-system.txt`，用户消息格式复刻
`ChatOrchestrator.chatMessages()`，知识块形状复刻 `ContextAssembler`。改了网关
不跑一遍，数字就不可信。

## 校验

| 文件 | 作用 |
|---|---|
| `check_capability_map.py` | 用模型 `config.json` 的 `topic_id2label` 做唯一标签源，校验 YAML 无缺键、无未知主题、能力值合法、无重复。权重缺失直接失败，不隐式回退 |
| `check_route_drift.py` | 行为层校验。静态校验只看 YAML 是否完整，看不出「YAML 改了没重训」或「重训了没同步 YAML」。这里每个主题取 `qa_train` 样本跑真实 route 头，把能力命中率与 YAML 声明逐格对比：声明为真而命中率低于 `--floor`（默认 0.60）报漏，声明为假而高于 `--ceil`（默认 0.40）报多，中间留死区避二项噪声。规则短路的样本不计（不经过 route 头）。**只告警不做正确性断言**，CI 里 `continue-on-error`；`--strict` 时漂移非零退出 |
| `intent_capability_map.yaml` | topic → 候选能力集的唯一真相源。order 是门控能力，订单邻近主题宁可多标：多标会被门控拦，漏标会让订单问题掉进知识库答成泛泛政策 |

## 共享常量

`ml/runtime/topic_labels.py` 定义主题标签空间（`MIN_COUNT = 100`、`OTHER = "其他"`），
训练、检索分区、离线评测共用这一份。单独成文件是为了让不加载模型的脚本拿常量时
不拖入 torch 和 sentence-transformers。

## 运行

```bash
.venv/bin/python3 ml/experiments/<script>.py
```

脚本自己处理 `sys.path`，不需要额外设置 `PYTHONPATH`（`build_vector_index.py`
仍需 `DATABASE_URL`）。完整环境准备见主 [README](../../README.md)。
