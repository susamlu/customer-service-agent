"""在测试集上评估重排模型。

每个样本给出 (query, positive) 和 (query, negative) 两条，模型打分后判断
正例是否排在负例之前。指标：

  pairwise accuracy  正例得分高于负例的比例
  MRR@2             正例在两条候选中的排名倒数均值

`--model` 不传时用零样本 bge-reranker-base，传微调目录则评估微调模型。

门槛：pairwise accuracy ≥ 0.774（零样本基线，architecture.md §10.3），
不达标退出码 1，供 CI 直接拦截。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sentence_transformers import CrossEncoder

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models" / "models"
MODEL_NAME = "BAAI--bge-reranker-base"
PAIRWISE_FLOOR = 0.774


def find_model() -> Path:
    matches = list(MODEL_DIR.glob(f"{MODEL_NAME}/snapshots/*"))
    if not matches:
        raise FileNotFoundError(f"找不到模型 {MODEL_NAME}，先运行 ModelScope 下载")
    return matches[0]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="模型目录，默认零样本 bge-reranker-base")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="mps", help="mps / cuda / cpu")
    parser.add_argument("--min-pairwise", type=float, default=PAIRWISE_FLOOR)
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_model()
    rows = load(PROCESSED_DIR / "rerank_test.jsonl")
    model = CrossEncoder(str(model_path), max_length=args.max_length, device=args.device)

    pairs = []
    for row in rows:
        pairs.append((row["query"], row["positive"]))
        pairs.append((row["query"], row["negative"]))
    scores = model.predict(pairs, batch_size=32, show_progress_bar=True)

    correct = 0
    mrr_sum = 0.0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for i, row in enumerate(rows):
        positive_wins = bool(scores[2 * i] > scores[2 * i + 1])
        correct += int(positive_wins)
        mrr_sum += 1.0 if positive_wins else 0.5

        bucket = by_type[row["negative_type"]]
        bucket[0] += int(positive_wins)
        bucket[1] += 1

    total = len(rows)
    pairwise = correct / total
    print(f"样本 {total}")
    print(f"pairwise accuracy {pairwise:.4f}")
    print(f"MRR@2            {mrr_sum / total:.4f}")
    for negative_type, (wins, count) in sorted(by_type.items()):
        print(f"  {negative_type:<12} {wins}/{count} = {wins / count:.4f}")

    verdict = "PASS" if pairwise >= args.min_pairwise else "FAIL"
    print(
        f"\n门槛 pairwise {pairwise:.4f} "
        f"{'≥' if verdict == 'PASS' else '<'} {args.min_pairwise:.3f}  {verdict}"
    )
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
