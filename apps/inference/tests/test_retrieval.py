"""召回与重排的具体实现（§2.1）：InMemory / PgVector / CrossEncoder / Noop。"""

import numpy as np
import pytest

from inference.retrieval.in_memory import InMemoryRetriever
from inference.retrieval.pgvector import OVERSAMPLE, PgVectorRetriever
from inference.retrieval.reranker import CrossEncoderReranker, NoopReranker


class FakeAgent:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def retrieve_indexed(self, query, top_k, intent=None):
        self.calls.append((query, top_k, intent))
        return self.rows


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return np.asarray(self.scores, dtype=float)


def test_in_memory_maps_rows_to_retrieved():
    agent = FakeAgent([
        (3, {"answer": "七天无理由退货", "_label": "售后"}, 0.9),
        (7, {"answer": "运费说明", "_label": "物流"}, 0.5),
    ])

    hits = InMemoryRetriever(agent).search("怎么退", 10, topic="售后")

    assert agent.calls == [("怎么退", 10, "售后")]
    assert [(h.id, h.text, h.score, h.topic) for h in hits] == [
        (3, "七天无理由退货", 0.9, "售后"),
        (7, "运费说明", 0.5, "物流"),
    ]


def test_cross_encoder_orders_by_score_desc():
    model = FakeCrossEncoder([0.2, 0.9, 0.5])
    reranker = CrossEncoderReranker(model)

    ranked = reranker.rank("q", ["a", "b", "c"], 2)

    assert [(r.index, r.text, r.score) for r in ranked] == [
        (1, "b", 0.9),
        (2, "c", 0.5),
    ]
    assert model.calls == 1
    # 网关据此判断 selected 的 score 是精排分还是召回分。
    assert reranker.scorer == "cross_encoder"


def test_cross_encoder_empty_candidates_skips_model():
    model = FakeCrossEncoder([])
    assert CrossEncoderReranker(model).rank("q", [], 2) == []
    assert model.calls == 0


def test_noop_preserves_recall_order_and_truncates():
    reranker = NoopReranker()
    ranked = reranker.rank("q", ["a", "b", "c"], 2)

    assert [(r.index, r.text, r.score) for r in ranked] == [
        (0, "a", 0.0),
        (1, "b", 0.0),
    ]
    # noop 不产出精排分，score 恒为 0；网关按 scorer 回填召回分。
    assert reranker.scorer == "noop"


def test_noop_handles_empty_candidates():
    assert NoopReranker().rank("q", [], 2) == []


# ---------- PgVectorRetriever（§7）----------
# 用 fake DB-API 连接验证 SQL 参数与排序语义，不依赖真实 pgvector 实例；
# 真库的 parity 对比见 test_pgvector_parity.py。


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executions = []

    def execute(self, sql, params=None):
        self.executions.append((sql, params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, rows):
        self.cursor_obj = FakeCursor(rows)

    def cursor(self):
        return self.cursor_obj

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnect:
    def __init__(self, rows):
        self.rows = rows
        self.connections = []

    def __call__(self, conninfo):
        # retriever 每次 search 都开新连接（with 结束即关），fake 记录每一次。
        conn = FakeConnection(self.rows)
        self.connections.append(conn)
        return conn


def test_pgvector_maps_rows_and_oversamples_candidates():
    connect = FakeConnect([(5, "售后", "七天无理由退货", 0.9)])
    retriever = PgVectorRetriever(
        "postgresql://x", lambda q: [0.1, 0.2], connect=connect
    )

    hits = retriever.search("怎么退", 10)

    assert [(h.id, h.text, h.score, h.topic) for h in hits] == [
        (5, "七天无理由退货", 0.9, "售后")
    ]
    sql, params = connect.connections[0].cursor_obj.executions[0]
    # 候选池放大：先按纯向量取 top_k*OVERSAMPLE，再应用层重排。
    assert params[2] == 10 * OVERSAMPLE
    assert "embedding <=> %s::vector" in sql


def test_pgvector_topic_boost_reorders_within_candidate_pool():
    connect = FakeConnect(
        [
            (1, "物流", "运费说明", 0.60),
            (2, "售后", "七天无理由退货", 0.57),
        ]
    )
    retriever = PgVectorRetriever(
        "postgresql://x", lambda q: [0.0], connect=connect, intent_boost=0.05
    )

    hits = retriever.search("怎么退", 10, topic="售后")

    # 0.57 + 0.05 = 0.62 > 0.60，售后候选被推到第一，但分数仍保留原始余弦。
    assert [h.id for h in hits] == [2, 1]
    assert hits[0].score == pytest.approx(0.62)


def test_pgvector_without_topic_preserves_vector_order_and_truncates():
    connect = FakeConnect(
        [
            (1, "物流", "a", 0.9),
            (2, "售后", "b", 0.8),
            (3, "售后", "c", 0.7),
        ]
    )
    retriever = PgVectorRetriever("postgresql://x", lambda q: [0.0], connect=connect)

    hits = retriever.search("q", 2)

    assert [h.id for h in hits] == [1, 2]


def test_pgvector_encodes_query_exactly_once():
    encoded = []
    retriever = PgVectorRetriever(
        "postgresql://x",
        lambda q: encoded.append(q) or [0.0],
        connect=FakeConnect([]),
    )

    retriever.search("怎么退", 3)

    assert encoded == ["怎么退"]


def test_pgvector_rejects_unsafe_table_name():
    with pytest.raises(ValueError):
        PgVectorRetriever(
            "postgresql://x",
            lambda q: [0.0],
            table="knowledge; DROP TABLE knowledge",
            connect=FakeConnect([]),
        )


def test_pgvector_ping_reads_the_table():
    connect = FakeConnect([])
    retriever = PgVectorRetriever("postgresql://x", lambda q: [0.0], connect=connect)

    retriever.ping()

    sql, _ = connect.connections[0].cursor_obj.executions[0]
    assert "FROM knowledge" in sql

