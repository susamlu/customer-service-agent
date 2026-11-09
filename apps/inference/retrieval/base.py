"""检索与重排抽象（architecture.md §2.1）。

Retriever 只负责「问题 → 候选」，Reranker 只负责「候选 → 精选」。拆成两个接口
是因为两者解决的不是同一个问题：召回换的是 recall（pgvector / 暴力点积），
重排换的是 top-N 精度（交叉编码器），故障域和扩容方式都不同。
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Retrieved:
    """一条召回候选。id 是语料下标，topic 是语料的意图分区标签。"""

    id: int
    text: str
    score: float
    topic: str | None = None


@dataclass(frozen=True)
class Ranked:
    """重排结果。index 指回传入的 candidates 列表，不是语料下标。"""

    index: int
    text: str
    score: float


class Retriever(Protocol):
    def search(
        self, query: str, top_k: int, topic: str | None = None
    ) -> list[Retrieved]: ...


class Reranker(Protocol):
    def rank(
        self, query: str, candidates: list[str], top_k: int
    ) -> list[Ranked]: ...
