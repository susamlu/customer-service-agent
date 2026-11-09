"""Java ↔ Python 契约夹具（architecture.md §3）。

apps/contract/fixtures 是两侧共用的唯一真源：Python 侧用 pydantic 校验，
Java 侧用 ModelServiceClient 解析。任何一边改字段而不改夹具，这个测试会先红。
"""

import json
from pathlib import Path

from inference.api_models import ErrorResponse, PlanResponse

FIXTURES = Path(__file__).resolve().parents[2] / "contract" / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_text(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def frame(payload):
    """与 model_service.sse 完全一致的线上格式。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def parse_stream(text):
    events = []
    for line in text.splitlines():
        if not line:
            continue
        assert line.startswith("data: "), f"非 SSE 帧：{line!r}"
        events.append(json.loads(line[len("data: "):]))
    return events


def test_plan_fixture_matches_plan_response_model():
    model = PlanResponse.model_validate(load("plan_response.json"))

    assert model.capabilities == ["knowledge", "order"]
    assert model.topic.label == "物流"
    assert model.gate["order"].allowed is True
    assert model.gate["order"].reason == "order_id_present"
    assert set(model.gate) == {"knowledge", "order"}


def test_plan_fixture_keys_stay_in_sync_with_model():
    payload = load("plan_response.json")

    assert set(payload) == set(PlanResponse.model_fields)


def test_retrieve_fixture_shape():
    payload = load("retrieve_response.json")

    assert set(payload) == {"candidates", "intent", "elapsed"}
    assert payload["candidates"]
    for item in payload["candidates"]:
        assert set(item) == {"text", "intent", "score"}
        assert isinstance(item["score"], float)


def test_rerank_fixture_shape():
    payload = load("rerank_response.json")

    assert set(payload) == {"selected", "elapsed"}
    assert payload["selected"]
    for item in payload["selected"]:
        assert set(item) == {"index", "text", "score"}
        assert isinstance(item["index"], int)


def test_error_envelope_matches_error_response_model():
    model = ErrorResponse.model_validate(load("error_envelope.json"))

    assert model.code == "UPSTREAM_RATE_LIMITED"
    assert model.retriable is True


def test_generate_stream_fixture_is_valid_sse_and_round_trips():
    text = load_text("generate_stream.txt")
    events = parse_stream(text)

    assert [event["type"] for event in events] == ["token", "token", "token", "done"]
    assert events[-1]["reply"] == "".join(event["text"] for event in events[:-1])
    # 逐帧重编码必须与夹具逐字节一致：这就是 Java 侧收到的线上格式
    assert "".join(frame(event) for event in events) == text


def test_generate_error_stream_preserves_partial():
    events = parse_stream(load_text("generate_error_stream.txt"))

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "STREAM_INTERRUPTED"
    assert events[-1]["partial"] == "".join(
        event["text"] for event in events if event["type"] == "token"
    )
