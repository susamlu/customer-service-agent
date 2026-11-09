"""把 qa_train 语料灌进 PostgreSQL + pgvector（architecture.md §7）。

InMemory 召回是进程内暴力点积，万级以内够用；语料变大后把同一批向量搬进
pgvector，运行时通过 DATABASE_URL 切换（apps/inference/model_service.py:
build_retriever），接口不变、上层无感。

文档侧编码不加指令前缀，与 pipeline.py 的 corpus_emb 口径一致；query 侧的指令
前缀由 model_service 在运行时补，两边对得上才能做 parity 对比。

用法：
  DATABASE_URL=postgresql://csagent:csagent@localhost:5432/csagent \\
    PYTHONPATH=apps:ml/runtime .venv/bin/python3 ml/experiments/build_vector_index.py --reset
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELSCOPE_DIR = PROJECT_DIR / "models" / "models"

sys.path.insert(0, str(PROJECT_DIR / "runtime"))
from topic_labels import MIN_COUNT, OTHER  # noqa: E402

EMBED_MODEL = "BAAI--bge-small-zh-v1.5"
DIM = 512
BATCH = 64


def find_snapshot(name: str) -> Path:
    matches = list(MODELSCOPE_DIR.glob(f"{name}/snapshots/*"))
    if not matches:
        raise FileNotFoundError(f"找不到模型 {name}，先运行 ModelScope 下载")
    return matches[0]


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def partition_labels(corpus: list[dict]) -> list[str]:
    """复刻 pipeline 的分区标签：与运行时召回的分区必须落在同一标签空间。"""
    counts = Counter(row["intent"] for row in corpus if row["intent"])
    keep = {name for name, count in counts.items() if count >= MIN_COUNT}
    return [row["intent"] if row["intent"] in keep else OTHER for row in corpus]


def vector_literal(embedding) -> str:
    return "[" + ",".join(f"{float(value):.7f}" for value in embedding) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="把 qa_train 语料灌进 pgvector")
    parser.add_argument("--corpus", default=str(PROCESSED_DIR / "qa_train.jsonl"))
    parser.add_argument("--table", default="knowledge")
    parser.add_argument("--reset", action="store_true", help="先清空表再灌")
    parser.add_argument("--device", default="cpu", help="编码设备，默认 cpu")
    args = parser.parse_args()

    conninfo = os.environ.get("DATABASE_URL", "").strip()
    if not conninfo:
        raise SystemExit(
            "需要 DATABASE_URL，例如 postgresql://csagent:csagent@localhost:5432/csagent"
        )
    table = args.table
    if not table.replace("_", "").isalnum():
        raise SystemExit(f"非法表名：{table}")

    corpus = load(Path(args.corpus))
    labels = partition_labels(corpus)
    embedder = SentenceTransformer(str(find_snapshot(EMBED_MODEL)), device=args.device)

    import psycopg

    # with 连接块在正常退出时提交、异常时回滚，整批插入是一个事务。
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "id BIGINT PRIMARY KEY, intent TEXT, answer TEXT NOT NULL, "
                f"embedding vector({DIM}) NOT NULL)"
            )
            if args.reset:
                cur.execute(f"TRUNCATE TABLE {table}")

        for start in range(0, len(corpus), BATCH):
            batch = corpus[start : start + BATCH]
            embeddings = embedder.encode(
                [row["answer"] for row in batch],
                batch_size=BATCH,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            rows = [
                (
                    start + offset,
                    labels[start + offset],
                    row["answer"],
                    vector_literal(embedding),
                )
                for offset, (row, embedding) in enumerate(zip(batch, embeddings))
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {table} (id, intent, answer, embedding) "
                    "VALUES (%s, %s, %s, %s::vector) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "intent = EXCLUDED.intent, answer = EXCLUDED.answer, "
                    "embedding = EXCLUDED.embedding",
                    rows,
                )
            print(f"已写入 {min(start + BATCH, len(corpus))}/{len(corpus)}", flush=True)

        # 索引放在灌完之后建：边插边维护 hnsw 比一次性建慢一个数量级。
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_embedding "
                f"ON {table} USING hnsw (embedding vector_cosine_ops)"
            )

    print(f"完成：{len(corpus)} 条写入 {table}")


if __name__ == "__main__":
    main()
