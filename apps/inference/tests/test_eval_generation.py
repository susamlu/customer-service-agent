"""端到端生成评测的纯逻辑测试（不联网、不加载模型）。

覆盖评测脚本里最容易和线上口径悄悄分叉的四段：知识块拼装、消息格式、
评审 JSON 解析、汇总与门槛判定。
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR / "ml" / "experiments"))

import eval_generation as eg  # noqa: E402


def test_limit_text_keeps_short_and_truncates_long():
    assert eg.limit_text("短", 90) == "短"
    cut = eg.limit_text("a" * 100, 90)
    assert len(cut) == 90
    assert cut.endswith("…")


def test_assemble_knowledge_dedups_limits_items_and_chars():
    long_answer = "第二条" + "x" * 200
    context = eg.assemble_knowledge(["第一条", "第一条", "", long_answer, "第三条"])
    blocks = context.split("\n")
    assert len(blocks) == 2
    assert blocks[0] == "[知识] 第一条"
    assert blocks[1].startswith("[知识] 第二条")
    assert len(blocks[1]) == len("[知识] ") + 90


def test_build_messages_uses_placeholder_when_context_blank():
    messages = eg.build_messages("今天 {today}", "   ", "什么时候发货？", "2026-11-09")
    assert messages[0] == {"role": "system", "content": "今天 2026-11-09"}
    assert messages[1]["content"] == (
        eg.EMPTY_CONTEXT + "\n\n用户问题：什么时候发货？"
    )


def test_build_messages_keeps_context_verbatim():
    messages = eg.build_messages("sys {today}", "[知识] 48 小时内发货", "多久发货", "2026-11-09")
    assert messages[1]["content"] == "[知识] 48 小时内发货\n\n用户问题：多久发货"


def test_parse_judge_json_accepts_plain_and_fenced():
    plain = '{"faithfulness": 5, "relevance": 4, "style": 3, "reason": "ok"}'
    fenced = "```json\n" + plain + "\n```"
    for text in (plain, fenced):
        parsed = eg.parse_judge_json(text)
        assert parsed is not None
        assert (parsed["faithfulness"], parsed["relevance"], parsed["style"]) == (5, 4, 3)


def test_parse_judge_json_rejects_out_of_range_and_missing_fields():
    assert eg.parse_judge_json('{"faithfulness": 6, "relevance": 4, "style": 3}') is None
    assert eg.parse_judge_json('{"faithfulness": 4, "relevance": 4}') is None
    assert eg.parse_judge_json("完全不是 JSON") is None


def test_percentile_nearest_rank():
    values = [float(i) for i in range(1, 101)]
    assert eg.percentile(values, 95) == 95.0
    assert eg.percentile([], 95) == 0.0


def test_sample_rows_is_reproducible_and_filters_short_answers():
    rows = [
        {"question": "q1", "answer": "这是一个足够长的答案"},
        {"question": "q2", "answer": "短"},
        {"question": "", "answer": "空问题"},
    ]
    first = eg.sample_rows(rows, limit=5, seed=42, min_answer_chars=10)
    second = eg.sample_rows(rows, limit=5, seed=42, min_answer_chars=10)
    assert first == second
    assert [row["question"] for row in first] == ["q1"]


def test_summarize_and_verdict_pass_and_fail():
    def record(faithfulness, total_ms, error=None, cosine=0.6):
        return {
            "error": error,
            "judge": None
            if error
            else {"faithfulness": faithfulness, "relevance": 4, "style": 4},
            "cosine": None if error else cosine,
            "retrieval_ms": 20.0,
            "first_token_ms": 800.0,
            "total_ms": total_ms,
        }

    passing = eg.summarize([record(5, 1000), record(4, 1200)])
    assert passing["error_rate"] == 0.0
    assert passing["scores"]["faithfulness"] == 4.5
    assert eg.verdict(passing) is True

    failing = eg.summarize([record(4, 1000, error="boom")])
    assert failing["error_rate"] == 1.0
    assert eg.verdict(failing) is False

    low_quality = eg.summarize([record(3, 1000), record(3, 1000)])
    assert eg.verdict(low_quality) is False
