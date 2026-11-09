"""从原始问答对构造 rerank 三元组。

负例分三级，按难度递减：
  same_session  同一次会话中其它问题的答案，话题最接近，最难
  same_intent   同一意图下其它问题的答案，主题相同但问题不同
  random        其它任意问题的答案，容易区分

每个正例配一个 hard negative（按上面的优先级）和一个 random negative。
"""

import json
import random
from collections import defaultdict
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SEED = 42


def load(split: str) -> list[dict]:
    path = PROCESSED_DIR / f"qa_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build(split: str) -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    rows = load(split)

    by_session: dict[str, list[int]] = defaultdict(list)
    by_intent: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_session[row["session_id"]].append(i)
        if row["intent"]:
            by_intent[row["intent"]].append(i)

    out: list[dict] = []
    stats = {"same_session": 0, "same_intent": 0, "only_random": 0}

    for i, row in enumerate(rows):
        positive = row["answer"]
        used = {i}

        candidates = [
            j for j in by_session[row["session_id"]]
            if j != i and rows[j]["answer"] != positive
        ]
        if candidates:
            j = rng.choice(candidates)
            hard_type, hard_text = "same_session", rows[j]["answer"]
            used.add(j)
        else:
            candidates = [
                j for j in by_intent.get(row["intent"], [])
                if j != i and rows[j]["answer"] != positive
            ]
            if candidates:
                j = rng.choice(candidates)
                hard_type, hard_text = "same_intent", rows[j]["answer"]
                used.add(j)
            else:
                hard_type, hard_text = None, None

        random_index = None
        for _ in range(50):
            j = rng.randrange(len(rows))
            if j not in used and rows[j]["answer"] != positive:
                random_index = j
                break
        if random_index is None:
            continue

        if hard_text is not None:
            out.append({
                "query": row["question"],
                "positive": positive,
                "negative": hard_text,
                "negative_type": hard_type,
                "intent": row["intent"],
            })
            stats[hard_type] += 1
        else:
            stats["only_random"] += 1

        out.append({
            "query": row["question"],
            "positive": positive,
            "negative": rows[random_index]["answer"],
            "negative_type": "random",
            "intent": row["intent"],
        })

    return out, stats


def main() -> None:
    for split in ("train", "val", "test"):
        triples, stats = build(split)
        out_path = PROCESSED_DIR / f"rerank_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for triple in triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")
        print(
            f"{split}: 三元组 {len(triples)}  "
            f"(同会话 {stats['same_session']} / 同意图 {stats['same_intent']} "
            f"/ 仅随机 {stats['only_random']})  -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
