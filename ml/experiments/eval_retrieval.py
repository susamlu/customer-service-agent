"""用 bge-small-zh 评估检索召回（口径与线上一致）。

语料 = train 的全部答案（线上 KB 就是 `qa_train.jsonl`）；query = test 的问题。

相关口径：query 的原始 intent 在 train 中存在时，同 intent 的 train 答案都算相关。
strict 口径在客服语料上严重低估（同一个问题有大量语义等价的重复答案），而且线上
KB 根本不含 test 答案，已移除；只看 intent 口径。

线上 `/api/retrieve` 会带着路由模型预测的 topic 调用检索，检索对同意图分区
（41 类归并空间）加 0.05 相似度（`pipeline.INTENT_BOOST`）。所以这里同时输出：
  pure    纯向量召回，不带意图信号（参照基线）
  served  线上口径：路由 topic 分区 +0.05（CI 门槛守这个数）
topic 预测直接调线上的 `BertIntentClassifier`，保证评测和线上是同一个模型。

门槛：served recall@10 ≥ 0.68（当前基线 ~0.70，留 2 个点回归余量），
不达标退出码 1，供 CI 直接拦截。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "apps"))
sys.path.insert(0, str(PROJECT_DIR / "runtime"))

from inference.intent.multitask import build_classifier  # noqa: E402
from topic_labels import MIN_COUNT, OTHER  # noqa: E402

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models" / "models"
MODEL_NAME = "BAAI--bge-small-zh-v1.5"
TOP_KS = (1, 5, 10, 20)
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
INTENT_BOOST = 0.05
INTENT_RECALL10_FLOOR = 0.68


def find_model() -> Path:
    matches = list(MODEL_DIR.glob(f"{MODEL_NAME}/snapshots/*"))
    if not matches:
        raise FileNotFoundError(f"找不到模型 {MODEL_NAME}，先运行 ModelScope 下载")
    return matches[0]


def load(split: str) -> list[dict]:
    path = PROCESSED_DIR / f"qa_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def partition_labels(rows: list[dict], keep: set[str]) -> list[str]:
    """把 208 类原始 intent 归并到线上 41 类分区空间。"""
    return [row["intent"] if row["intent"] in keep else OTHER for row in rows]


def report(name: str, ranks: np.ndarray) -> None:
    print(f"\n[{name}] n={len(ranks)}")
    for k in TOP_KS:
        print(f"  recall@{k:<2} {(ranks <= k).mean():.4f}")
    print(f"  MRR      {(1 / ranks).mean():.4f}")


def first_relevant_ranks(
    scores: np.ndarray, relevant_sets: list[list[int]]
) -> np.ndarray:
    """每条 query 第一个相关答案的排名；无相关集合的 query 跳过。"""
    ranks = []
    for index, relevant in enumerate(relevant_sets):
        if not relevant:
            continue
        relevant_set = set(relevant)
        for rank, candidate in enumerate(np.argsort(-scores[index]), start=1):
            if candidate in relevant_set:
                ranks.append(rank)
                break
    return np.asarray(ranks, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", help="mps / cuda / cpu")
    parser.add_argument("--boost", type=float, default=INTENT_BOOST)
    parser.add_argument("--min-intent-recall10", type=float, default=INTENT_RECALL10_FLOOR)
    args = parser.parse_args()

    train = load("train")
    test = load("test")

    # 线上 KB = qa_train.jsonl 的全部答案；test 答案不进 KB。
    corpus = [row["answer"] for row in train]
    queries = [QUERY_INSTRUCTION + row["question"] for row in test]

    model = SentenceTransformer(str(find_model()), device=args.device)
    print(f"候选库 {len(corpus)} 条，query {len(test)} 条")
    corpus_emb = model.encode(
        corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    query_emb = model.encode(
        queries, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    base_scores = query_emb @ corpus_emb.T

    # 相关口径：208 类原始 intent 在 train 中出现的答案都算相关。
    intent_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train):
        if row["intent"]:
            intent_to_indices[row["intent"]].append(index)
    relevant_sets = [
        intent_to_indices.get(row["intent"], []) if row["intent"] else []
        for row in test
    ]

    # 线上分区掩码：41 类归并；topic 用线上同一个多任务模型预测。
    counts = Counter(row["intent"] for row in train if row["intent"])
    keep = {name for name, count in counts.items() if count >= MIN_COUNT}
    corpus_labels = np.asarray(partition_labels(train, keep), dtype=object)

    print("预测路由 topic…", flush=True)
    classifier = build_classifier(device=args.device)
    predicted_topics = np.asarray(
        [classifier.classify(row["question"]).topic for row in test], dtype=object
    )

    pure = first_relevant_ranks(base_scores, relevant_sets)
    mask = (predicted_topics[:, None] == corpus_labels[None, :]).astype(np.float32)
    served = first_relevant_ranks(base_scores + args.boost * mask, relevant_sets)

    report("pure（纯向量）", pure)
    report(f"served（topic 分区 +{args.boost:.2f}）", served)

    recall10 = float((served <= 10).mean())
    verdict = "PASS" if recall10 >= args.min_intent_recall10 else "FAIL"
    print(
        f"\n门槛 served recall@10 {recall10:.4f} "
        f"{'≥' if verdict == 'PASS' else '<'} {args.min_intent_recall10:.3f}  {verdict}"
    )
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
