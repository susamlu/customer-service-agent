"""在 CSDS 重排三元组上微调 bge-reranker-base。

每条三元组展开成正例对（label=1）和负例对（label=0）。
"""

import argparse
import json
import random
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "models"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "rerank-finetuned"
BASE_MODEL = "BAAI--bge-reranker-base"
SEED = 42


def find_model() -> Path:
    matches = list(MODEL_DIR.glob(f"{BASE_MODEL}/snapshots/*"))
    if not matches:
        raise FileNotFoundError(f"找不到模型 {BASE_MODEL}，先运行 ModelScope 下载")
    return matches[0]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条三元组，0 表示全部")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    random.seed(SEED)
    rows = load(PROCESSED_DIR / "rerank_train.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    examples = []
    for row in rows:
        examples.append(InputExample(texts=[row["query"], row["positive"]], label=1.0))
        examples.append(InputExample(texts=[row["query"], row["negative"]], label=0.0))
    random.shuffle(examples)

    model = CrossEncoder(
        str(find_model()), num_labels=1, max_length=args.max_length, device="mps"
    )
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=max(100, int(len(train_dataloader) * 0.1)),
        optimizer_params={"lr": 2e-5},
        show_progress_bar=True,
        use_amp=False,
    )
    # CrossEncoder.fit 在没有 evaluator 时不会自动保存，必须显式落盘
    model.save(str(OUTPUT_DIR))
    print(f"模型已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
