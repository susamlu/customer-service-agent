"""可观测底座（architecture.md §11）：trace_id 透传 + Prometheus 指标。

网关为每个请求生成 ``X-Trace-Id``，模型服务继承合法值、自己补造非法的，
并在响应头回写；不在请求上下文中就返回 None，绝不因为脏头让请求失败。

必须是纯 ASGI 中间件：``BaseHTTPMiddleware`` 会把 ``StreamingResponse`` 拆开
再包一层，首字延迟被推迟到整个生成结束——SSE 的观测价值恰好就在首字上。
"""

from __future__ import annotations

import re
import time
import uuid
from contextvars import ContextVar

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

TRACE_HEADER = "x-trace-id"
_TRACE_VALUE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# 指标只认识这些路径：其余（含 404、扫描器乱打）统一归到 other，防止标签爆炸。
_KNOWN_PATHS = frozenset({"/healthz", "/api/plan", "/api/retrieve", "/api/rerank"})
# SSE 的生命周期等于整段生成，混进请求直方图会污染 P95；/metrics 自己也别算。
_UNMEASURED_PATHS = frozenset({"/api/generate", "/metrics"})

REQUEST_SECONDS = Histogram(
    "model_service_request_seconds",
    "模型服务单次请求耗时",
    ["path", "method", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
REQUEST_TOTAL = Counter(
    "model_service_requests_total",
    "模型服务请求数",
    ["path", "method", "status"],
)

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def current_trace_id() -> str | None:
    """当前请求的 trace_id；不在请求里（如启动预热）返回 None。"""
    return _trace_id.get()


def window_metrics() -> Response:
    """Prometheus 抓取入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _resolve_trace_id(headers) -> str:
    for name, value in headers:
        if name.decode("latin-1").lower() != TRACE_HEADER:
            continue
        candidate = value.decode("latin-1").strip()
        if _TRACE_VALUE.match(candidate):
            return candidate
        break
    return uuid.uuid4().hex[:16]


class ObservabilityMiddleware:
    """trace_id + 请求耗时/状态码；不缓冲响应体。"""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")
        trace_id = _resolve_trace_id(scope.get("headers", []))
        token = _trace_id.set(trace_id)
        started = time.perf_counter()
        status = 500  # 未走到 response.start 就异常时按 500 记账

        async def send_with_trace(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                headers.append((TRACE_HEADER.encode("latin-1"), trace_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        finally:
            if path not in _UNMEASURED_PATHS:
                labels = (path if path in _KNOWN_PATHS else "other", method, str(status))
                REQUEST_SECONDS.labels(*labels).observe(time.perf_counter() - started)
                REQUEST_TOTAL.labels(*labels).inc()
            _trace_id.reset(token)
