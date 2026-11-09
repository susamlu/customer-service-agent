"""一级意图评测（architecture.md §4.5 / §10.3）：多任务 BERT 的能力集、主题与线上口径。

评测集 `ml/data/processed/route_eval.jsonl` 共 300 条，但两个来源的标注口径
不同，必须分层计算。混在一起算会同时低估模型、放大误判：

  boundary  60 条，来自 tool_eval.jsonl。gold 是「该问法必须触发的单一工具」，
            不含额外的 knowledge。因此用**必中率**衡量：gold ⊆ predicted。
            多预测一个 knowledge 不算错——映射表本来就允许 order+knowledge，
            最终由 /api/plan 的门控裁剪（§4.4）。
  heldout   240 条，来自 qa_test。gold 是 intent_capability_map.yaml 派生的
            完整候选能力集，与训练标签同源。用**完全匹配**（subset accuracy）
            衡量：这就是路由器应该输出的集合，门控发生在模型服务之后。

指标：
  boundary 必中率     必须触发的工具是否触发（完全匹配仅作参考）
  heldout subset      多标签候选集完全匹配，路由主指标
  能力级 P/R/F1       把「某能力是否出现」当成独立二分类（只在 heldout 上算，
                      boundary 的 gold 不是完整集合，混入会污染 precision）
  order precision     同时报「heldout 实测」与「自然分布加权」两个口径。加权用
                      qa_test 上映射表派生的 order 先验，确认分层抽样没有把
                      precision 抬高。
  human recall        全部 human 正例的召回，再按实际命中层（rule / bert / llm）拆开；
                      规则接不住的灰区才是模型的真实能力。
  topic top-1         问法原文对回 qa_test 的人工主题标注；一问多标注的跳过。
  来源分布            rule / bert 各占多少条。
  延迟                avg / p95

口径分层（§10.3）：
  默认跑法（CI）      BERT-only、门控前。0.867 / 0.813 等基线就是这个口径。
  --gate on          追加门控后（线上单轮）口径，两套并列。评测没有会话历史，
                     等价于首轮对话：order 只在问法自带订单号时保留。

任何一项门槛不过退出码为 1，供 CI 拦截。退出码只由**门控前**口径决定，
与同一条 CI 门槛口径一致；`--gate on` 的门控后数值仅作参考。
回归线低于产品目标是有意的：在当前模型达标前，CI 的职责是拦住倒退；目标差值
记录在 TODO，由模型改进收敛。

为什么用 subset accuracy 而不是单标签准确率：设计本来就允许
`order + knowledge` 同时命中（如「物流全程跟踪」既要查物流也要讲时效政策），
单标签口径会把合法的多标签输出判错。

用法：
  .venv/bin/python3 ml/experiments/eval_router.py
  .venv/bin/python3 ml/experiments/eval_router.py --limit 20
  .venv/bin/python3 ml/experiments/eval_router.py --gate on
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "apps"))
sys.path.insert(0, str(PROJECT_DIR / "runtime"))

from inference.intent.gate import apply_gate  # noqa: E402
from inference.intent.multitask import build_classifier  # noqa: E402

EVAL_PATH = PROJECT_DIR / "data" / "processed" / "route_eval.jsonl"
QA_TEST_PATH = PROJECT_DIR / "data" / "processed" / "qa_test.jsonl"
MAP_FILE = PROJECT_DIR / "experiments" / "intent_capability_map.yaml"

CAPABILITIES = ("knowledge", "order", "human")
BOUNDARY_RECALL_FLOOR = 0.85
HELDOUT_SUBSET_FLOOR = 0.80
ORDER_PRECISION_FLOOR = 0.88
HUMAN_RECALL_FLOOR = 0.75

# (显示名, summarize() 字典键, 回归线)
FLOORS = (
    ("boundary 必中率", "coverage", BOUNDARY_RECALL_FLOOR),
    ("heldout subset", "subset", HELDOUT_SUBSET_FLOOR),
    ("order precision（自然分布加权）", "weighted", ORDER_PRECISION_FLOOR),
    ("human recall", "human_recall", HUMAN_RECALL_FLOOR),
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_eval(limit: int) -> list[dict]:
    rows = load_jsonl(EVAL_PATH)
    return rows[:limit] if limit else rows


def percentile(values: list[float], percent: float) -> float:
    """最近秩法取分位数，样本量小也用得上。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(percent / 100 * (len(ordered) - 1)))
    return ordered[index]


def prf(pairs: list[tuple[set[str], set[str]]], capability: str) -> tuple[float, float, float, int, int, int]:
    """单个能力的 precision / recall / F1 与 TP/FP/FN。"""
    tp = sum(capability in predicted and capability in gold for predicted, gold in pairs)
    fp = sum(capability in predicted and capability not in gold for predicted, gold in pairs)
    fn = sum(capability not in predicted and capability in gold for predicted, gold in pairs)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


def required_recall(pairs: list[tuple[set[str], set[str]]]) -> float:
    """gold ⊆ predicted：必须触发的工具是否触发，多预测不扣分。"""
    if not pairs:
        return 0.0
    return sum(gold <= predicted for predicted, gold in pairs) / len(pairs)


def natural_order_prevalence() -> tuple[float, int]:
    """qa_test 上按映射表派生 order 的自然先验，用来校正分层抽样的口径。"""
    capability_map = yaml.safe_load(MAP_FILE.read_text(encoding="utf-8"))["topics"]
    rows = [row for row in load_jsonl(QA_TEST_PATH) if row["intent"] in capability_map]
    if not rows:
        return 0.0, 0
    positives = sum("order" in set(capability_map[row["intent"]]) for row in rows)
    return positives / len(rows), len(rows)


def gold_topics() -> tuple[dict[str, str], int]:
    """qa_test 的问法 → 人工主题标注。

    heldout 的问法逐字来自 qa_test，所以按原文回查即可；同一问法有两个主题
    标注的跳过（不硬选一个，否则 topic 准确率会失真）。
    """
    grouped: dict[str, set[str]] = {}
    for row in load_jsonl(QA_TEST_PATH):
        if row.get("intent"):
            grouped.setdefault(row["question"], set()).add(row["intent"])
    topics = {q: next(iter(t)) for q, t in grouped.items() if len(t) == 1}
    ambiguous = sum(1 for t in grouped.values() if len(t) > 1)
    return topics, ambiguous


def prevalence_weighted_precision(
    pairs: list[tuple[set[str], set[str]]], capability: str, prevalence: float
) -> float:
    """把混淆矩阵重加权到自然先验后的 precision。

    TP/FP/FN 是在分层样本上数出来的。换算成 TPR/FPR 后，用 p 替换样本里的
    正例比例，得到真实流量分布下的 precision。
    """
    if not pairs:
        return 0.0
    _, _, _, tp, fp, fn = prf(pairs, capability)
    tpr = tp / (tp + fn) if tp + fn else 1.0
    negatives = len(pairs) - (tp + fn)
    fpr = fp / negatives if negatives else 0.0
    denominator = prevalence * tpr + (1 - prevalence) * fpr
    return prevalence * tpr / denominator if denominator else 1.0


class ClassifierPredictor:
    """一次 classify，同时给出门控前/后两套能力集。"""

    def __init__(self, gate_enabled: bool) -> None:
        self.classifier = build_classifier()
        self.gate_enabled = gate_enabled

    def __call__(self, query: str) -> tuple[set[str], set[str], str, str]:
        result = self.classifier.classify(query)
        ungated = set(result.capabilities)
        gated = ungated
        if self.gate_enabled:
            allowed, _ = apply_gate(
                result.capabilities,
                result.route_scores,
                result.entities,
                (),
                human_threshold=self.classifier.route_thresholds["human"],
            )
            gated = set(allowed)
        return ungated, gated, result.source, result.topic


def summarize(
    records: list[dict],
    key: str,
    title: str,
    natural_prevalence: float,
    natural_rows: int,
    *,
    detail: bool,
) -> dict:
    """计算并打印一组指标；key 决定看门控前（predicted）还是门控后（gated）。"""
    boundary = [r for r in records if r["row"].get("provenance") == "boundary"]
    heldout = [r for r in records if r["row"].get("provenance") == "heldout"]
    boundary_pairs = [(r[key], r["gold"]) for r in boundary]
    heldout_pairs = [(r[key], r["gold"]) for r in heldout]
    metrics: dict[str, float | None] = {
        "coverage": None,
        "subset": None,
        "weighted": None,
        "human_recall": None,
    }

    print(f"\n{title}")

    # ── boundary：必中率 ────────────────────────────────────────────────
    if boundary_pairs:
        coverage = required_recall(boundary_pairs)
        exact = sum(predicted == gold for predicted, gold in boundary_pairs) / len(boundary_pairs)
        metrics["coverage"] = coverage
        print(
            f"  boundary（{len(boundary)} 条）必中率 {coverage:.4f}  "
            f"完全匹配 {exact:.4f}（参考）"
        )
        if detail:
            missed = [r for r in boundary if not r["gold"] <= r[key]]
            extras = [r for r in boundary if r["gold"] <= r[key] and r[key] != r["gold"]]
            if missed:
                print(f"  漏触发（真错，{len(missed)} 条）：")
                for r in missed:
                    print(
                        f"    {r['row']['query']}\n"
                        f"      必须 {sorted(r['gold'])}  预测 {sorted(r[key])}（{r['source']}）"
                    )
            if extras:
                print(f"  多预测（可接受，门控裁剪，{len(extras)} 条）")
    else:
        print("  boundary 样本不足，跳过必中率门槛")

    # ── heldout：完全匹配 + 能力级 P/R/F1 ─────────────────────────────
    if heldout_pairs:
        subset = sum(predicted == gold for predicted, gold in heldout_pairs) / len(heldout_pairs)
        metrics["subset"] = subset
        print(f"  heldout（{len(heldout)} 条）subset accuracy {subset:.4f}")
        print(f"  {'能力':<10}{'P':>7}{'R':>7}{'F1':>7}{'TP':>6}{'FP':>6}{'FN':>6}")
        for capability in CAPABILITIES:
            precision, recall, f1, tp, fp, fn = prf(heldout_pairs, capability)
            print(f"  {capability:<10}{precision:>7.3f}{recall:>7.3f}{f1:>7.3f}{tp:>6}{fp:>6}{fn:>6}")

        order_precision = prf(heldout_pairs, "order")[0]
        weighted = prevalence_weighted_precision(heldout_pairs, "order", natural_prevalence)
        metrics["weighted"] = weighted
        print(
            f"  order precision 实测 {order_precision:.3f}  "
            f"自然分布加权 {weighted:.3f}（先验 {natural_prevalence:.3f}，qa_test {natural_rows} 条）"
        )

        if detail:
            gold_counts = Counter(cap for r in heldout for cap in r["gold"])
            print(f"  标注分布 {dict(sorted(gold_counts.items()))}")
            errors = [r for r in heldout if r[key] != r["gold"]]
            if errors:
                print(f"  heldout 误判明细（{len(errors)} 条）：")
                for r in errors:
                    print(
                        f"    {r['row']['query']}\n"
                        f"      期望 {sorted(r['gold'])}  预测 {sorted(r[key])}（{r['source']}）"
                    )
    else:
        print("  heldout 样本不足，跳过 subset / order precision 门槛")

    # ── human recall：全部正例，按实际命中层拆开 ──────────────────────
    human_records = [r for r in records if "human" in r["gold"]]
    if human_records:
        hits = sum("human" in r[key] for r in human_records)
        human_recall = hits / len(human_records)
        metrics["human_recall"] = human_recall
        by_layer = {}
        for layer in ("rule", "bert", "llm"):
            layer_records = [r for r in human_records if r["source"] == layer]
            if layer_records:
                hits_at_layer = sum("human" in r[key] for r in layer_records)
                by_layer[layer] = f"{hits_at_layer}/{len(layer_records)}"
        print(f"  human recall {human_recall:.3f}（{hits}/{len(human_records)}）按层 {by_layer}")
        if detail:
            for r in (r for r in human_records if "human" not in r[key]):
                print(f"    漏 {r['row']['query']}（{r['source']}）")
    else:
        print("  human 正例不足，跳过 recall 门槛")

    return metrics


def report_floors(metrics: dict, *, affect_exit_code: bool) -> bool:
    """打印门槛；返回是否失败。affect_exit_code=False 时只展示不改变退出码。"""
    print("  门槛：")
    failed = False
    for name, key, floor in FLOORS:
        value = metrics.get(key)
        if value is None:
            print(f"    {name} 样本不足，跳过")
            continue
        ok = value >= floor
        failed = failed or not ok
        print(f"    {name} {value:.4f} {'≥' if ok else '<'} {floor:.2f}  {'PASS' if ok else 'FAIL'}")
    return failed if affect_exit_code else False


def evaluate(rows: list[dict], predict: ClassifierPredictor, gate_enabled: bool) -> int:
    natural_prevalence, natural_rows = natural_order_prevalence()
    topic_gold, ambiguous = gold_topics()

    records: list[dict] = []
    latencies: list[float] = []
    started_total = 0.0

    for index, row in enumerate(rows):
        started = time.perf_counter()
        ungated, gated, source, topic = predict(row["query"])
        elapsed = time.perf_counter() - started
        started_total += elapsed
        latencies.append(elapsed * 1000)

        gold = set(row["capabilities"])
        hit = ungated == gold
        gate_changed = gated != ungated
        records.append(
            {
                "row": row,
                "predicted": ungated,
                "gated": gated,
                "gold": gold,
                "source": source,
                "topic": topic,
                "gold_topic": topic_gold.get(row["query"]),
                "ms": elapsed * 1000,
            }
        )

        print(
            f"[{index + 1}/{len(rows)}] {source:<9} {elapsed * 1000:6.1f}ms  "
            f"{'✓' if hit else '✗'} {row['query']}"
            + ("  [门控改动]" if gate_changed else ""),
            flush=True,
        )

    total = len(records)
    print("\n" + "=" * 68)
    print(
        f"样本 {total}  平均 {started_total / total * 1000:.1f}ms  "
        f"P95 {percentile(latencies, 95):.1f}ms"
    )
    print(f"来源分布 {dict(sorted(Counter(r['source'] for r in records).items()))}")

    topic_records = [r for r in records if r["gold_topic"]]
    if topic_records:
        topic_hits = sum(r["topic"] == r["gold_topic"] for r in topic_records)
        print(
            f"topic top-1 {topic_hits / len(topic_records):.4f}"
            f"（{topic_hits}/{len(topic_records)}；"
            f"{total - len(topic_records)} 条无主题真值，qa_test 有 {ambiguous} 条一问多标注）"
        )
    else:
        print("topic top-1 跳过：没有可回查的主题标注")

    baseline = summarize(
        records, "predicted", "门控前（BERT-only，CI 口径）",
        natural_prevalence, natural_rows, detail=True,
    )
    failed = report_floors(baseline, affect_exit_code=True)

    if gate_enabled:
        gated = summarize(
            records, "gated", "门控后（线上单轮口径，仅参考）",
            natural_prevalence, natural_rows, detail=False,
        )
        report_floors(gated, affect_exit_code=False)

    return 0 if not failed else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gate", choices=("off", "on"), default="off")
    args = parser.parse_args()

    rows = load_eval(args.limit)
    predictor = ClassifierPredictor(gate_enabled=args.gate == "on")
    raise SystemExit(evaluate(rows, predictor, gate_enabled=args.gate == "on"))


if __name__ == "__main__":
    main()
