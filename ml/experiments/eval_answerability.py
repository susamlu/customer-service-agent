"""可答性闸门评测（architecture.md §5.4）。

闸门要回答的不是「排得对不对」，而是「知识库到底有没有能答的内容」。
信号取线上同源的召回首位分数：

  多任务 BERT topic → retrieve_indexed top10（topic +0.05）→ hits[0] 分数

检索为空时没有分数，按 no_context 计（线上同口径）。判定三档与网关
`AnswerabilityGate` 的双阈值同值，改阈值必须两边一起改。

评测集两类标签（见 build_answerability_eval.py）：
  in_kb    可答，闸门必须放行（量误拦）
  out_kb   不可答，闸门应拦（量拦截）
boundary 是 in_kb 的子集，单独报告。

预登记门槛（先写死，再测量；产品取向「宁可漏拦，不可误拦」）：
  qa_test in_kb → no_context   ≤ 5%（把可答问题变固定话术的代价）
  boundary  in_kb → no_context ≤ 20%
  qa_test in_kb → answer       ≥ 75%（weak 只是提示，answer 才是直接生成）
  out_kb    → no_context       ≥ 50%（漏拦靠 weak 兜底，不追求全拦）

退出码：任一门槛不通过返回 1。**不进 per-commit CI**——依赖模型权重与 MPS，
放手动回归。

用法：
  PYTHONPATH=apps:ml/runtime .venv/bin/python3 ml/experiments/eval_answerability.py
  .venv/bin/python3 ml/experiments/eval_answerability.py --sweep --head 10
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "apps"))
sys.path.insert(0, str(PROJECT_DIR / "runtime"))

from inference.intent.multitask import build_classifier  # noqa: E402
from pipeline import PROCESSED_DIR, CustomerServiceAgent  # noqa: E402

EVAL_PATH = PROCESSED_DIR / "answerability_eval.jsonl"
T_LOW = 0.58
T_HIGH = 0.64
TOP_K = 10
# 召回分低于这个值就别再往下试了，域外问题也常能召回一两条弱相关。
MIN_SWEEP = 0.50
MAX_SWEEP = 0.72


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def tier(score: float | None, low: float, high: float) -> str:
    if score is None:
        return "no_context"
    if score >= high:
        return "answer"
    if score >= low:
        return "weak"
    return "no_context"


def separation(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """Mann-Whitney AUC：P(域内分数 > 域外分数)，0.5 = 无判别力。"""
    if len(in_scores) == 0 or len(out_scores) == 0:
        return float("nan")
    allv = np.concatenate([in_scores, out_scores])
    ranks = allv.argsort().argsort() + 1
    u = ranks[: len(in_scores)].sum() - len(in_scores) * (len(in_scores) + 1) / 2
    return float(u / (len(in_scores) * len(out_scores)))


def rates(rows: list[dict], low: float, high: float) -> dict[str, float]:
    tiers = Counter(tier(row["score"], low, high) for row in rows)
    n = len(rows) or 1
    return {
        "n": len(rows),
        "no_context": tiers["no_context"] / n,
        "weak": tiers["weak"] / n,
        "answer": tiers["answer"] / n,
    }


def fmt(row: dict) -> str:
    return (
        f"{row['n']:>4} | {row['no_context']:>10.3f} {row['weak']:>7.3f} "
        f"{row['answer']:>8.3f}"
    )


def sweep(rows: list[dict], in_rows: list[dict], out_rows: list[dict]) -> None:
    print("\n阈值网格（看 in 误拦与 out 拦截的取舍）：")
    print(f"{'T_low':>6} {'T_high':>7} | {'in no_ctx':>10} {'in answer':>10} "
          f"{'out no_ctx':>11} {'out answer':>11}")
    for low in np.arange(MIN_SWEEP, MAX_SWEEP, 0.04):
        for high in np.arange(low + 0.04, MAX_SWEEP + 0.02, 0.04):
            ri = rates(in_rows, float(low), float(high))
            ro = rates(out_rows, float(low), float(high))
            print(f"{low:>6.2f} {high:>7.2f} | {ri['no_context']:>10.3f} "
                  f"{ri['answer']:>10.3f} {ro['no_context']:>11.3f} "
                  f"{ro['answer']:>11.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(EVAL_PATH))
    parser.add_argument("--device", default="mps", help="mps / cuda / cpu")
    parser.add_argument("--low", type=float, default=T_LOW)
    parser.add_argument("--high", type=float, default=T_HIGH)
    parser.add_argument("--sweep", action="store_true", help="打印阈值网格")
    parser.add_argument("--head", type=int, default=0, help="打印最危险的误拦/漏拦各 N 条")
    parser.add_argument("--out", default="/tmp/eval_answerability.jsonl")
    args = parser.parse_args()

    if args.high <= args.low:
        raise SystemExit("T_high 必须大于 T_low")

    agent = CustomerServiceAgent(PROCESSED_DIR / "qa_train.jsonl")
    classifier = build_classifier()
    rows = load(Path(args.data))

    for row in rows:
        topic = classifier.classify(row["query"]).topic
        hits = agent.retrieve_indexed(row["query"], top_k=TOP_K, intent=topic)
        row["topic"] = topic
        row["score"] = round(float(hits[0][2]), 4) if hits else None
        row["top_intent"] = hits[0][1]["intent"] if hits else ""
        row["tier"] = tier(row["score"], args.low, args.high)

    slices: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        slices[row["provenance"]].append(row)
    in_rows = [r for r in rows if r["label"] == "in_kb"]
    out_rows = [r for r in rows if r["label"] == "out_kb"]

    print(f"评测集 {len(rows)} 条  T_low={args.low:g} T_high={args.high:g}")
    print(f"\n{'切片':<12} {'n':>4} | {'no_context':>10} {'weak':>7} {'answer':>8}")
    for name in ("qa_test", "boundary", "handwritten"):
        if slices[name]:
            print(f"{name:<12} {fmt(rates(slices[name], args.low, args.high))}")
    print(f"{'in_kb 合计':<12} {fmt(rates(in_rows, args.low, args.high))}")
    print(f"{'out_kb':<12} {fmt(rates(out_rows, args.low, args.high))}")

    in_scores = np.array([r["score"] for r in in_rows if r["score"] is not None])
    out_scores = np.array([r["score"] for r in out_rows if r["score"] is not None])
    print(f"\n召回首位分数 AUC(域内 > 域外) = {separation(in_scores, out_scores):.4f}")
    for name, vals in (("in_kb", in_scores), ("out_kb", out_scores)):
        print(
            f"  {name:<6} "
            + " ".join(f"p{q}={np.percentile(vals, q):.3f}" for q in (5, 25, 50, 75, 95))
        )

    if args.sweep:
        sweep(rows, in_rows, out_rows)

    qa = slices["qa_test"]
    boundary = slices["boundary"]
    gate_qa = rates(qa, args.low, args.high)
    gate_boundary = rates(boundary, args.low, args.high)
    gate_out = rates(out_rows, args.low, args.high)
    checks = [
        ("qa_test in_kb → no_context ≤ 5%", gate_qa["no_context"], 0.05, "<="),
        ("boundary in_kb → no_context ≤ 20%", gate_boundary["no_context"], 0.20, "<="),
        ("qa_test in_kb → answer ≥ 75%", gate_qa["answer"], 0.75, ">="),
        ("out_kb → no_context ≥ 50%", gate_out["no_context"], 0.50, ">="),
    ]
    print("\n预登记门槛：")
    failed = 0
    for label, actual, floor, op in checks:
        ok = actual <= floor if op == "<=" else actual >= floor
        failed += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<38} 实际 {actual:.3f}")

    if args.head:
        print(f"\n最危险的误拦（in_kb 判 no_context，前 {args.head} 条）：")
        for row in sorted(
            (r for r in in_rows if r["tier"] == "no_context"),
            key=lambda r: r["score"] if r["score"] is not None else -1,
        )[: args.head]:
            print(f"  {str(row['score']):>7}  {row['query']}")
        print(f"\n漏拦样本（out_kb 判 answer，前 {args.head} 条）：")
        for row in sorted(
            (r for r in out_rows if r["tier"] == "answer"), key=lambda r: -r["score"]
        )[: args.head]:
            print(f"  {row['score']:>7}  {row['query']}")

    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    print(f"\n逐条结果 → {args.out}")
    if failed:
        print(f"{failed} 项门槛未通过")
        return 1
    print("全部门槛通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
