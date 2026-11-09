"""暴力点积召回（默认实现）。

语料规模在万级以内时，一次矩阵乘法就能全量打分，比维护 pgvector 实例更省心。
超过这个量级再换 pgvector.py（§7），接口不变。
"""

from inference.retrieval.base import Retrieved


class InMemoryRetriever:
    # healthz 用它报告当前召回后端，与 PgVectorRetriever.backend 对齐。
    backend = "in_memory"

    def __init__(self, agent) -> None:
        self.agent = agent

    def search(
        self, query: str, top_k: int, topic: str | None = None
    ) -> list[Retrieved]:
        return [
            Retrieved(
                id=index,
                text=row["answer"],
                score=score,
                topic=row.get("_label"),
            )
            for index, row, score in self.agent.retrieve_indexed(
                query, top_k, intent=topic
            )
        ]
