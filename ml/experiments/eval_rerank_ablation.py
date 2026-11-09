"""rerank 取舍消融：不排 / 零样本 / 微调 三组，比最终 top-2 命中。

回答「rerank 微调是否值得 1.1GB 内存 + 每次 500ms 预算」。线上真实链路是
`召回 top-10 → 重排 top-2 → DeepSeek`，所以这里不比 pairwise，而比最终喂给
生成模型的 2 条里有没有相关答案。语料与相关口径复用 eval_retrieval.py：
KB = qa_train 全部答案，query = qa_test 问题，同 intent 的 train 答案算相关；
召回走线上 served 口径（路由 topic 分区 +0.05）。

三组：
  noop        直接取召回序前 2
  zero_shot   零样本 bge-reranker-base 对 top-10 重排取 2
  finetuned   rerank-finetuned（微调）对 top-10 重排取 2

指标：hit@2（top-2 是否命中相关答案）、MRR@2。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "apps"))
sys.path.insert(0, str(PROJECT_DIR / "runtime"))

from inference.intent.multitask import build_classifier  # noqa: E402
from topic_labels import MIN_COUNT, OTHER  # noqa: E402

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models" / "models"
EMB_NAME = "BAAI--bge-small-zh-v1.5"
ZS_NAME = "BAAI--bge-reranker-base"
FT_DIR = PROJECT_DIR / "models" / "rerank-finetuned"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOP_K = 10
TOP_N = 2


def find_snapshot(name: str) -> Path:
    matches = list(MODEL_DIR.glob(f"{name}/snapshots/*"))
    if not matches:
        raise FileNotFoundError(f"找不到模型 {name}，先运行 ModelScope 下载")
    return matches[0]


def load(split: str) -> list[dict]:
    path = PROCESSED_DIR / f"qa_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def first_rank(order: list[int], relevant: set[int], top_n: int) -> int:
    """top_n 内第一个相关答案的名次；top_n 内没有相关答案返回 0。"""
    for rank, index in enumerate(order[:top_n], start=1):
        if index in relevant:
            return rank
    return 0


def summarize(name: str, ranks: list[int]) -> tuple[float, float]:
    hit1 = sum(1 for rank in ranks if rank == 1) / len(ranks)
    hit2 = sum(1 for rank in ranks if 0 < rank <= TOP_N) / len(ranks)
    print(f"  {name:<10} hit@1 {hit1:.4f}   hit@{TOP_N} {hit2:.4f}")
    return hit1, hit2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", help="mps / cuda / cpu")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条 query，0 表示全部")
    parser.add_argument("--boost", type=float, default=0.05)
    args = parser.parse_args()

    train = load("train")
    test = load("test")
    if args.limit:
        test = test[: args.limit]

    corpus = [row["answer"] for row in train]
    print(f"候选库 {len(corpus)} 条，query {len(test)} 条", flush=True)

    # 相关口径与 eval_retrieval.py 一致：同 intent 的 train 答案都算相关。
    intent_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train):
        if row["intent"]:
            intent_to_indices[row["intent"]].append(index)
    relevant_sets = [
        set(intent_to_indices.get(row["intent"], [])) if row["intent"] else set()
        for row in test
    ]

    # 召回：线上 served 口径（41 类分区 + boost）。
    embedder = SentenceTransformer(str(find_snapshot(EMB_NAME)), device=args.device)
    corpus_emb = embedder.encode(
        corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    query_emb = embedder.encode(
        [QUERY_INSTRUCTION + row["question"] for row in test],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    counts = Counter(row["intent"] for row in train if row["intent"])
    keep = {name for name, count in counts.items() if count >= MIN_COUNT}
    corpus_labels = np.asarray(
        [row["intent"] if row["intent"] in keep else OTHER for row in train], dtype=object
    )

    print("预测路由 topic…", flush=True)
    classifier = build_classifier(device=args.device)
    predicted_topics = np.asarray(
        [classifier.classify(row["question"]).topic for row in test], dtype=object
    )

    scores = query_emb @ corpus_emb.T
    mask = (predicted_topics[:, None] == corpus_labels[None, :]).astype(np.float32)
    topk_idx = np.argsort(-(scores + args.boost * mask), axis=1)[:, :TOP_K]

    # 摊平所有 (query, 候选) 对，一次预测，避免逐条调用开销。
    pairs: list[tuple[str, str]] = []
    for qi, cand in enumerate(topk_idx):
        for ci in cand:
            pairs.append((test[qi]["question"], corpus[int(ci)]))
    print(f"重排打分 {len(pairs)} 对 …", flush=True)

    zero_shot = CrossEncoder(str(find_snapshot(ZS_NAME)), max_length=128, device=args.device)
    finetuned = CrossEncoder(str(FT_DIR), max_length=128, device=args.device)
    zs_scores = zero_shot.predict(pairs, batch_size=64, show_progress_bar=True)
    ft_scores = finetuned.predict(pairs, batch_size=64, show_progress_bar=True)

    # 按 query 归组：qi -> [(pos, zs, ft)]
    grouped: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    cursor = 0
    for qi, cand in enumerate(topk_idx):
        for position, _ in enumerate(cand):
            grouped[qi].append((position, float(zs_scores[cursor]), float(ft_scores[cursor])))
            cursor += 1

    noop_ranks, zs_ranks, ft_ranks = [], [], []
    for qi, cand in enumerate(topk_idx):
        relevant = relevant_sets[qi]
        if not relevant:
            continue
        cand_list = [int(index) for index in cand]
        ranked = grouped[qi]
        zs_order = [cand_list[position] for position, _, _ in sorted(ranked, key=lambda item: -item[1])]
        ft_order = [cand_list[position] for position, _, _ in sorted(ranked, key=lambda item: -item[2])]

        noop_ranks.append(first_rank(cand_list, relevant, TOP_N))
        zs_ranks.append(first_rank(zs_order, relevant, TOP_N))
        ft_ranks.append(first_rank(ft_order, relevant, TOP_N))

    print(f"\n有相关集合的 query：{len(noop_ranks)}")
    noop_h1, noop_h2 = summarize("noop", noop_ranks)
    zs_h1, zs_h2 = summarize("zero_shot", zs_ranks)
    ft_h1, ft_h2 = summarize("finetuned", ft_ranks)

    print("\n相对 noop 的变化：")
    print(f"  zero_shot  hit@1 {zs_h1 - noop_h1:+.4f}   hit@2 {zs_h2 - noop_h2:+.4f}")
    print(f"  finetuned  hit@1 {ft_h1 - noop_h1:+.4f}   hit@2 {ft_h2 - noop_h2:+.4f}")
    print(f"\n微调相对零样本：hit@1 {ft_h1 - zs_h1:+.4f}   hit@2 {ft_h2 - zs_h2:+.4f}")


if __name__ == "__main__":
    main()
