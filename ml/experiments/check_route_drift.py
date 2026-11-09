"""检测 YAML 声明的能力集与 route 头实际输出的能力集是否漂移。

architecture.md §4.3 末段明确：`intent_capability_map.yaml` 只是**训练标签**的
来源，运行时能力集来自 route 头，不做 topic→能力查表。这换来「模型能修正
topic 误判」，代价是：

    YAML 上规则变了，线上未必真的变了。

改 YAML 没重训、重训了忘了同步 YAML、或者换了底座模型，`check_capability_map.py`
都不会报错——它只校验 YAML 对 41 个主题完整、能力值合法。本脚本补上行为层
的另一半：每个主题取 qa_train 样本跑一遍分类器，统计能力命中率，与 YAML 声明
逐格对比。**只做漂移告警，不做正确性断言。**

为什么用 qa_train 而不是 qa_test：这里测的是「模型学到的映射是否仍等于声明」，
不是泛化能力。训练集上的命中率不受样本难度干扰，信号最干净。

口径：
  每个 (主题, 能力) 一格，rate = 该主题样本中该能力被 route 头命中的比例。
  声明为真且 rate < --floor → 漏（YAML 说要有，模型没学会）
  声明为假且 rate > --ceil  → 多（YAML 说没有，模型自己在加）
  两个阈值之间留死区，避免二项噪声导致告警抖动（40 样本时标准差约 8%）。

规则层短路的样本（source=rule）不参与统计——它们根本不经过 route 头。
主题 top-1 准确率一并打印，用来区分「真漂移」和「模型本来就不擅长这个主题」：
top-1 高而能力率与声明相反，才是 YAML 与模型脱节；top-1 低则更可能是模型弱。

默认退出码 0（只告警）；--strict 时有漂移返回 1，供需要硬门禁的场景使用。

用法：
  .venv/bin/python3 ml/experiments/check_route_drift.py
  .venv/bin/python3 ml/experiments/check_route_drift.py --per-topic 80 --strict
  .venv/bin/python3 ml/experiments/check_route_drift.py --json /tmp/route-drift.json
"""

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "apps"))
sys.path.insert(0, str(PROJECT_DIR / "runtime"))

from inference.intent.multitask import build_classifier  # noqa: E402
from topic_labels import OTHER  # noqa: E402

QA_TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "qa_train.jsonl"
MAP_FILE = PROJECT_DIR / "experiments" / "intent_capability_map.yaml"
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "intent-multitask-bert"

CAPABILITIES = ("knowledge", "order", "human")
DEFAULT_PER_TOPIC = 40
# 支撑数低于这个值的主题不参与漂移判定：几 条样本算出来的比例没有意义。
MIN_SUPPORT = 5
# 声明为真 / 为假两侧的门槛，中间留死区。
DECLARED_FLOOR = 0.60
UNDECLARED_CEIL = 0.40


def _display_width(text: str) -> int:
    """中文按 2 列宽计算，用于表格对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def _load_yaml_topics() -> dict[str, set[str]]:
    data = yaml.safe_load(MAP_FILE.read_text(encoding="utf-8")) or {}
    return {name: set(caps) for name, caps in (data.get("topics") or {}).items()}


def _load_samples(per_topic: int, seed: int) -> dict[str, list[str]]:
    """按规范化主题分组；不在模型标签空间内的 intent 归入「其他」。"""
    config = json.loads((DEFAULT_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    known = set(config["topic_id2label"].values())

    grouped: dict[str, list[str]] = {}
    with QA_TRAIN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question = (row.get("question") or "").strip()
            if not question:
                continue
            intent = row.get("intent") or ""
            topic = intent if intent in known else OTHER
            grouped.setdefault(topic, []).append(question)

    rng = random.Random(seed)
    for topic, questions in grouped.items():
        if len(questions) > per_topic:
            grouped[topic] = rng.sample(questions, per_topic)
        else:
            grouped[topic] = list(questions)
    return grouped


def _measure(classifier, questions: list[str], topic: str) -> dict:
    """返回某主题的命中率、top-1 准确率与规则短路条数。"""
    hits = {cap: 0 for cap in CAPABILITIES}
    topic_hits = 0
    counted = 0
    rule_skipped = 0
    for question in questions:
        result = classifier.classify(question)
        if result.source == "rule":
            # 规则短路不经过 route 头，混进来会虚高 human 命中率。
            rule_skipped += 1
            continue
        counted += 1
        emitted = set(result.capabilities)
        for cap in CAPABILITIES:
            if cap in emitted:
                hits[cap] += 1
        if result.topic == topic:
            topic_hits += 1

    rates = {cap: (hits[cap] / counted if counted else 0.0) for cap in CAPABILITIES}
    return {
        "support": counted,
        "rule_skipped": rule_skipped,
        "topic_top1": (topic_hits / counted if counted else 0.0),
        "rates": rates,
    }


def _drifts(
    topic: str, declared: set[str], rates: dict[str, float], floor: float, ceil: float
) -> list[str]:
    issues: list[str] = []
    for cap in CAPABILITIES:
        rate = rates[cap]
        if cap in declared and rate < floor:
            issues.append(f"{topic}: 声明有 {cap}，实际命中率仅 {rate:.0%}")
        if cap not in declared and rate > ceil:
            issues.append(f"{topic}: 声明无 {cap}，实际命中率 {rate:.0%}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="route 头与能力映射表的漂移检测")
    parser.add_argument("--per-topic", type=int, default=DEFAULT_PER_TOPIC,
                        help=f"每个主题取多少条样本（默认 {DEFAULT_PER_TOPIC}）")
    parser.add_argument("--seed", type=int, default=42, help="抽样种子")
    parser.add_argument("--floor", type=float, default=DECLARED_FLOOR,
                        help=f"声明为真的最低命中率（默认 {DECLARED_FLOOR}）")
    parser.add_argument("--ceil", type=float, default=UNDECLARED_CEIL,
                        help=f"声明为假的最高命中率（默认 {UNDECLARED_CEIL}）")
    parser.add_argument("--device", default=None, help="cpu / mps，默认自动选择")
    parser.add_argument("--strict", action="store_true", help="有漂移时退出码为 1")
    parser.add_argument("--json", metavar="PATH", help="把结果写成 JSON")
    args = parser.parse_args()

    if args.per_topic < MIN_SUPPORT:
        print(f"--per-topic 不能小于 {MIN_SUPPORT}", file=sys.stderr)
        return 2

    declared = _load_yaml_topics()
    grouped = _load_samples(args.per_topic, args.seed)
    classifier = build_classifier(device=args.device)

    rows: list[dict] = []
    all_issues: list[str] = []
    uncovered: list[str] = []

    for topic in sorted(declared):
        questions = grouped.get(topic, [])
        measured = _measure(classifier, questions, topic)
        if measured["support"] < MIN_SUPPORT:
            uncovered.append(topic)
            rows.append({"topic": topic, "declared": sorted(declared[topic]), **measured,
                         "issues": []})
            continue
        issues = _drifts(topic, declared[topic], measured["rates"], args.floor, args.ceil)
        all_issues.extend(issues)
        rows.append({"topic": topic, "declared": sorted(declared[topic]), **measured,
                     "issues": issues})

    print(f"{_pad('主题', 22)}{'n_bert':>7}  {'top1':>5}  "
          f"{'knowledge':>10}{'order':>8}{'human':>8}  漂移")
    print("-" * 78)
    for row in rows:
        rates = row["rates"]
        if row["support"] < MIN_SUPPORT:
            print(f"{_pad(row['topic'], 22)}{row['support']:>4}  样本不足，跳过")
            continue
        flag = "!" if row["issues"] else ""
        print(
            f"{_pad(row['topic'], 22)}{row['support']:>4}  {row['topic_top1']:>5.0%}  "
            f"{rates['knowledge']:>10.0%}{rates['order']:>8.0%}{rates['human']:>8.0%}  {flag}"
        )

    total_cells = len(rows) * len(CAPABILITIES)
    print("-" * 78)
    print(f"主题 {len(rows)} 个，能力格 {total_cells} 个，漂移 {len(all_issues)} 处")
    if uncovered:
        print(f"样本不足（<{MIN_SUPPORT} 条，未判定）：{', '.join(uncovered)}")
    for issue in all_issues:
        print(f"  ! {issue}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"issues": all_issues, "topics": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if all_issues and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
