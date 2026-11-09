"""把 CSDS 的摘要式答案替换为 Dialogue 中的原始客服回复，输出问答对。

CSDS 的 AnsSummLong 是人工写的摘要（第三人称、书面语），不适合作为生成目标。
原始回复可通过 AnsSummLongUttIDs 回溯到 Dialogue 中的 speaker == "A" 句子。
"""

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# 中文汉字与中文标点
CJK = "\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"


def clean(text: str) -> str:
    """还原分词产生的空格，保留中英文之间的必要间隔。"""
    text = re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collect(dialogue_by_turn: dict, turn_ids: list, speaker: str) -> str:
    """按 turn 取回原始句子，逐句清洗后用空格分隔，避免多条回复粘连。"""
    parts = [
        clean(dialogue_by_turn[t]["utterance"])
        for t in turn_ids
        if t in dialogue_by_turn and dialogue_by_turn[t]["speaker"] == speaker
    ]
    return " ".join(p for p in parts if p)


def build(split: str) -> tuple[list[dict], dict]:
    data = json.loads((RAW_DIR / f"{split}.json").read_text(encoding="utf-8"))
    rows = []
    stats = {"total": 0, "no_answer": 0, "no_question": 0, "fallback_summ": 0}

    for session in data:
        dialogue_by_turn = {u["turn"]: u for u in session["Dialogue"]}
        for qa in session.get("QA", []):
            stats["total"] += 1

            answer = collect(dialogue_by_turn, qa.get("AnsSummLongUttIDs") or [], "A")
            if not answer:
                stats["no_answer"] += 1
                continue

            question = collect(dialogue_by_turn, qa.get("QueSummUttIDs") or [], "Q")
            if not question:
                question = qa.get("QueSumm", "")
                stats["no_question"] += 1
                if not question:
                    stats["fallback_summ"] += 1
                    continue

            rows.append({
                "session_id": session.get("Session_id"),
                "dialogue_id": session.get("DialogueID"),
                "intent": (qa.get("intent") or "").strip(),
                "question": question,
                "answer": answer,
                "question_summ": qa.get("QueSumm", ""),
                "answer_summ": qa.get("AnsSummLong", ""),
            })

    return rows, stats


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows, stats = build(split)
        out_path = OUT_DIR / f"qa_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        lengths = sorted(len(r["answer"]) for r in rows)
        median = lengths[len(lengths) // 2] if lengths else 0
        print(
            f"{split}: 保留 {len(rows)}/{stats['total']}  "
            f"无原始答案 {stats['no_answer']}  "
            f"无原始问题 {stats['no_question']}  "
            f"答案长度 中位 {median} / 最大 {lengths[-1] if lengths else 0}  "
            f"-> {out_path.name}"
        )


if __name__ == "__main__":
    main()
