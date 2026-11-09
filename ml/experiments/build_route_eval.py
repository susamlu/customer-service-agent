"""构建路由评测集（architecture.md §4.5）。

300 条，两类来源，都不在训练集里：

  boundary  60 条取自 tool_eval.jsonl。这份是手写的边界问法（每条带 reason），
            capabilities 取标注的单一能力，用来压测 route 头的边界判断。
  heldout   252 条从 qa_test 分层抽样。topic 是语料人工标注，capabilities 由
            intent_capability_map.yaml 派生（与训练标签同源，但 topic 本身是
            标注真值，不是模型预测）。

字段：
  query         用户问法
  capabilities  多标签能力集
  entities      问法里显式出现的业务实体（SO 开头的订单号）
  source        期望的识别来源（rule / bert），供分层命中率统计

用法：
  .venv/bin/python3 ml/experiments/build_route_eval.py
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPO_DIR = PROJECT_DIR.parent
MAP_FILE = PROJECT_DIR / "experiments" / "intent_capability_map.yaml"

# 转人工正则与线上 L0 规则同源，避免评测与运行时口径漂移。
sys.path.insert(0, str(REPO_DIR / "apps"))
sys.path.insert(0, str(PROJECT_DIR / "runtime"))
from inference.intent.rule import HUMAN_PATTERN  # noqa: E402
from topic_labels import MIN_COUNT  # noqa: E402

OUT_PATH = PROCESSED_DIR / "route_eval.jsonl"
SEED = 42
# 目标 300 条：boundary 60 + heldout 240。heldout 与 boundary 会撞重复问法，
# 去重后不足 300，所以多抽样补齐；实际条数由 main() 打印。
HELDOUT_TOTAL = 252
BOUNDARY_TOTAL = 60
ORDER_ID = re.compile(r"SO\d{6,}", re.IGNORECASE)

TOOL_TO_CAPABILITY = {
    "query_order": "order",
    "search_knowledge": "knowledge",
    "transfer_human": "human",
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def label_space() -> set[str]:
    """与训练脚本同一口径：训练集中出现 >= 100 次的意图。"""
    counts = Counter(
        row["intent"]
        for row in load_jsonl(PROCESSED_DIR / "qa_train.jsonl")
        if row["intent"]
    )
    return {intent for intent, count in counts.items() if count >= MIN_COUNT}


def entities_of(query: str) -> dict[str, str]:
    match = ORDER_ID.search(query)
    return {"orderId": match.group()} if match else {}


def expected_source(query: str) -> str:
    return "rule" if HUMAN_PATTERN.search(query) else "bert"


def build_boundary() -> list[dict]:
    rows = load_jsonl(PROCESSED_DIR / "tool_eval.jsonl")
    items = []
    for row in rows[:BOUNDARY_TOTAL]:
        items.append(
            {
                "query": row["question"],
                "capabilities": [TOOL_TO_CAPABILITY[row["expected_tool"]]],
                "entities": entities_of(row["question"]),
                "source": expected_source(row["question"]),
                "provenance": "boundary",
            }
        )
    return items


def build_heldout(
    keep: set[str], capability_map: dict[str, tuple[str, ...]]
) -> list[dict]:
    rows = [
        row
        for row in load_jsonl(PROCESSED_DIR / "qa_test.jsonl")
        if row["intent"] in keep
    ]
    by_topic: dict[str, list[dict]] = {}
    for row in rows:
        by_topic.setdefault(row["intent"], []).append(row)

    rng = random.Random(SEED)
    topics = sorted(by_topic)
    base, extra = divmod(HELDOUT_TOTAL, len(topics))

    chosen: list[dict] = []
    for index, topic in enumerate(topics):
        candidates = by_topic[topic][:]
        rng.shuffle(candidates)
        take = base + (1 if index < extra else 0)
        chosen.extend(candidates[:take])

    items = []
    for row in chosen:
        items.append(
            {
                "query": row["question"],
                "capabilities": sorted(capability_map[row["intent"]]),
                "entities": entities_of(row["question"]),
                "source": expected_source(row["question"]),
                "provenance": "heldout",
            }
        )
    return items


def main() -> None:
    capability_map = {
        name: tuple(caps)
        for name, caps in (
            yaml.safe_load(MAP_FILE.read_text(encoding="utf-8")) or {}
        ).get("topics", {}).items()
    }
    keep = label_space()
    missing = keep - capability_map.keys()
    if missing:
        raise SystemExit(f"能力映射表缺少主题：{sorted(missing)}")

    boundary = build_boundary()
    heldout = build_heldout(keep, capability_map)
    items = boundary + heldout

    seen: set[str] = set()
    deduped = []
    for item in items:
        if item["query"] in seen:
            continue
        seen.add(item["query"])
        deduped.append(item)

    if len(deduped) != len(items):
        print(f"去重丢弃 {len(items) - len(deduped)} 条重复问法")

    OUT_PATH.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False) for item in deduped
        )
        + "\n",
        encoding="utf-8",
    )

    counts = Counter(cap for item in deduped for cap in item["capabilities"])
    print(f"wrote {OUT_PATH}  共 {len(deduped)} 条")
    print(f"能力分布：{dict(sorted(counts.items()))}")
    print(f"来源分布：{dict(Counter(item['provenance'] for item in deduped))}")


if __name__ == "__main__":
    main()
