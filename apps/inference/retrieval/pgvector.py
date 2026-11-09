"""pgvector 召回（architecture.md §2.1 / §7）。

InMemory 是进程内暴力点积，万级以内足够；把语料搬进 PostgreSQL + pgvector 后，
召回变成一次索引查询，接口与 InMemoryRetriever 完全一致，两者可就同一批数据做
parity 对比（tests/test_pgvector_parity.py）。

两个设计取舍：

1. 先用纯向量距离取候选池（top_k * OVERSAMPLE），意图分区加成在应用层做，
   不塞进 ORDER BY 表达式——表达式排序会让 hnsw 索引失效、退化成全表扫描。
   这与 InMemory 的加权语义一致：分区只在候选池内做重排，不改变召回口径。
2. 向量以文本形式随 SQL 传参再 `::vector` 转换，不依赖 psycopg 的适配器注册
   （不同驱动版本注册方式不同，显式转换更稳，也便于用 fake connection 单测）。
"""

from typing import Callable

from inference.retrieval.base import Retrieved

DEFAULT_TABLE = "knowledge"

# 候选池放大倍数：分区加成最多把候选池内的条目推进前 top_k，5x 足以覆盖一次
# 重排的位移空间；再大只是白白拖慢查询。
OVERSAMPLE = 5

# 与 ml/runtime/pipeline.py::INTENT_BOOST 保持一致（0.05 是离线扫描的收益/风险
# 平衡点）。这里默认内联，不 import pipeline，避免为一次检索把 torch 拖进来。
DEFAULT_INTENT_BOOST = 0.05


def _safe_table(table: str) -> str:
    """表名只允许字母数字下划线。表名来自配置，不是请求，但仍不允许拼出 SQL。"""
    if not table.replace("_", "").isalnum():
        raise ValueError(f"非法表名：{table}")
    return table


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.7f}" for value in vector) + "]"


class PgVectorRetriever:
    # healthz 用它报告当前召回后端，避免运维靠猜环境变量。
    backend = "pgvector"

    def __init__(
        self,
        conninfo: str,
        encode_query: Callable[[str], list[float]],
        *,
        table: str = DEFAULT_TABLE,
        intent_boost: float = DEFAULT_INTENT_BOOST,
        connect=None,
    ) -> None:
        self.conninfo = conninfo
        self.encode_query = encode_query
        self.table = _safe_table(table)
        self.intent_boost = intent_boost
        self.connect = connect  # 缺省时在 ping/search 里惰性 import psycopg

    def ping(self) -> None:
        """启动探活：连库并确认表可读。失败直接抛，让 healthz 暴露问题，
        而不是等到首个用户请求才发现向量库没灌数据。"""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM {self.table} LIMIT 1")

    def search(
        self, query: str, top_k: int, topic: str | None = None
    ) -> list[Retrieved]:
        vector = _vector_literal(self.encode_query(query))
        # 候选池必须 >= top_k，否则分区重排没有足够的位移空间。
        candidate_k = max(top_k, top_k * OVERSAMPLE)
        sql = (
            f"SELECT id, intent, answer, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.table} "
            f"ORDER BY embedding <=> %s::vector "
            f"LIMIT %s"
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector, vector, candidate_k))
                rows = cur.fetchall()

        scored = [
            Retrieved(
                id=int(row[0]),
                text=row[2],
                # topic 命中时把分区加成叠到余弦相似度上，再统一排序。
                score=float(row[3]) + (
                    self.intent_boost if topic and row[1] == topic else 0.0
                ),
                topic=row[1],
            )
            for row in rows
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def _connection(self):
        if self.connect is None:
            import psycopg

            self.connect = psycopg.connect
        return self.connect(self.conninfo)
