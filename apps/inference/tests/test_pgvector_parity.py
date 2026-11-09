"""pgvector 与 InMemory 的召回 parity（architecture.md §7）。

只在 DATABASE_URL 指向已灌数据的 pgvector 时运行——它会真正加载嵌入模型并连库。
CI 默认跳过；本地回归或夜间 e2e 带上 DATABASE_URL 跑。

验收口径：同一 query，两种实现的前 top_k 文本集合应高度重合。允许少量边界
交换（pgvector 用 oversample 截断局部候选池，InMemory 全表打分），但不应整片不同。
"""

import json
import os
from pathlib import Path

import pytest

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="需要 DATABASE_URL 指向已灌数据的 pgvector"
)

QUERIES = 10
TOP_K = 10
MIN_OVERLAP = 8


def _sample_queries() -> list[str]:
    processed = Path(__file__).resolve().parents[3] / "ml" / "data" / "processed"
    rows = [
        json.loads(line)
        for line in (processed / "qa_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    # 去重后取前若干条，避免同一问题重复算。
    seen: list[str] = []
    for row in rows:
        if row["question"] not in seen:
            seen.append(row["question"])
        if len(seen) == QUERIES:
            break
    return seen


def test_pgvector_matches_in_memory_top_k():
    from pipeline import PROCESSED_DIR, QUERY_INSTRUCTION, CustomerServiceAgent

    from inference.retrieval.in_memory import InMemoryRetriever
    from inference.retrieval.pgvector import PgVectorRetriever

    agent = CustomerServiceAgent(PROCESSED_DIR / "qa_train.jsonl", load_reranker=False)
    in_memory = InMemoryRetriever(agent)
    pgvector = PgVectorRetriever(
        DATABASE_URL,
        lambda text: agent.embedder.encode(
            [QUERY_INSTRUCTION + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist(),
    )

    for question in _sample_queries():
        expected = {hit.text for hit in in_memory.search(question, TOP_K)}
        actual = {hit.text for hit in pgvector.search(question, TOP_K)}
        overlap = len(expected & actual)
        assert overlap >= MIN_OVERLAP, (
            f"{question}: InMemory/PgVector 前 {TOP_K} 只重合 {overlap} 条"
        )
