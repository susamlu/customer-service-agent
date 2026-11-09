"""重排的两个实现（architecture.md §2.1）。

NoopReranker 保留召回顺序，是线上默认实现（取舍见 docs/experiments.md
「rerank 取舍消融」）。CrossEncoderReranker 用微调过的 bge-reranker，簇内区分力
更强但占 1.1GB，按需启用。两个实现也是 D5 契约测试的前提。
"""

import numpy as np

from inference.retrieval.base import Ranked


class CrossEncoderReranker:
    # /api/rerank 原样透传，网关据此判断 selected 的 score 是精排分还是召回分。
    scorer = "cross_encoder"

    def __init__(self, model) -> None:
        self.model = model

    def rank(
        self, query: str, candidates: list[str], top_k: int
    ) -> list[Ranked]:
        if not candidates:
            return []
        scores = self.model.predict([(query, text) for text in candidates])
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [
            Ranked(index=int(i), text=candidates[i], score=float(scores[i]))
            for i in order
        ]


class NoopReranker:
    scorer = "noop"

    def rank(
        self, query: str, candidates: list[str], top_k: int
    ) -> list[Ranked]:
        """保留召回顺序，score 统一为 0：调用方据此知道这一步没有精排。"""
        return [
            Ranked(index=i, text=text, score=0.0)
            for i, text in enumerate(candidates[:top_k])
        ]
