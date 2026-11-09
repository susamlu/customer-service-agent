"""可观测中间件：trace_id 透传 + 请求指标（architecture.md §11）。"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from prometheus_client import REGISTRY
from starlette.testclient import TestClient

from inference.observability import ObservabilityMiddleware, current_trace_id

VALID = "abc123_-."
GENERATED_LEN = 16


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ObservabilityMiddleware)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"trace": current_trace_id()}

    @app.post("/api/plan")
    def plan() -> dict:
        return {"trace": current_trace_id()}

    @app.post("/api/generate")
    def generate() -> StreamingResponse:
        def body():
            yield "data: one\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    return app


def _counter(path: str, method: str, status: str) -> float:
    return REGISTRY.get_sample_value(
        "model_service_requests_total",
        {"path": path, "method": method, "status": status},
    ) or 0.0


def _duration_count(path: str, method: str, status: str) -> float:
    return REGISTRY.get_sample_value(
        "model_service_request_seconds_count",
        {"path": path, "method": method, "status": status},
    ) or 0.0


def test_valid_trace_id_is_echoed_and_visible_in_handler():
    client = TestClient(_build_app())
    resp = client.get("/healthz", headers={"X-Trace-Id": VALID})
    assert resp.status_code == 200
    assert resp.headers["x-trace-id"] == VALID
    assert resp.json()["trace"] == VALID


def test_missing_trace_id_is_generated():
    client = TestClient(_build_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200
    generated = resp.headers["x-trace-id"]
    assert len(generated) == GENERATED_LEN
    assert generated == resp.json()["trace"]


def test_invalid_trace_id_is_replaced_not_rejected():
    client = TestClient(_build_app())
    resp = client.get("/healthz", headers={"X-Trace-Id": "not valid value"})
    assert resp.status_code == 200
    assert resp.headers["x-trace-id"] != "not valid value"
    assert len(resp.headers["x-trace-id"]) == GENERATED_LEN


def test_known_path_is_measured():
    client = TestClient(_build_app())
    before_total = _counter("/api/plan", "POST", "200")
    before_count = _duration_count("/api/plan", "POST", "200")

    client.post("/api/plan")

    assert _counter("/api/plan", "POST", "200") == before_total + 1
    assert _duration_count("/api/plan", "POST", "200") == before_count + 1


def test_unknown_path_folds_into_other():
    client = TestClient(_build_app())
    before = _counter("other", "GET", "404")

    client.get("/scanner/not-a-route")

    assert _counter("other", "GET", "404") == before + 1


def test_generate_stream_is_excluded_from_metrics():
    client = TestClient(_build_app())
    before_total = _counter("/api/generate", "POST", "200")
    before_other = _counter("other", "POST", "200")

    with client.stream("POST", "/api/generate") as resp:
        assert resp.status_code == 200
        assert "".join(resp.iter_text()) == "data: one\n\n"

    assert _counter("/api/generate", "POST", "200") == before_total
    assert _counter("other", "POST", "200") == before_other


def test_streaming_response_is_not_buffered():
    """中间件只包 header，不准把 SSE 攒完再发。"""
    client = TestClient(_build_app())
    with client.stream("POST", "/api/generate") as resp:
        first = next(resp.iter_text())
        assert first == "data: one\n\n"


def test_context_var_is_cleared_between_requests():
    client = TestClient(_build_app())
    client.get("/healthz", headers={"X-Trace-Id": "first"})
    resp = client.get("/healthz", headers={"X-Trace-Id": "second"})
    assert resp.json()["trace"] == "second"
    assert current_trace_id() is None
