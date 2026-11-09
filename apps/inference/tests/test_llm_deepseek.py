"""DeepSeek Provider（§2.1 / §9）：SSE 解析与错误码映射。

网络层用 fake client 替换：`_open` 只依赖 build_request/send，
给一串预先编排的结果（Response 或异常）就能覆盖全部失败分支。
"""

import json

import httpx
import pytest

from inference.llm import deepseek
from inference.llm.deepseek import DeepSeekProvider
from inference.llm.base import LlmError


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.sends = 0

    def build_request(self, method, url, headers=None, json=None):
        return {"method": method, "url": url, "headers": headers, "json": json}

    def send(self, request, stream=True):
        self.sends += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ExplodingResponse:
    status_code = 200

    def __init__(self):
        self.closed = False

    def iter_lines(self):
        yield 'data: {"choices": [{"delta": {"content": "半句"}}]}'
        raise httpx.ReadError("连接中断")

    def close(self):
        self.closed = True


def delta(content):
    return json.dumps({"choices": [{"delta": {"content": content}}]})


def sse_body(*payloads):
    return "".join(f"data: {p}\n\n" for p in payloads)


def sse_response(*payloads):
    return httpx.Response(200, content=sse_body(*payloads).encode("utf-8"))


def provider_with(outcomes):
    client = FakeClient(outcomes)
    return DeepSeekProvider("test-key", client=client), client


def collect(provider):
    return list(provider.stream([{"role": "user", "content": "hi"}], 8))


def test_stream_yields_tokens_and_stops_at_done():
    provider, _ = provider_with([
        sse_response(delta("你好"), delta("，在的"), "[DONE]", delta("不该出现")),
    ])
    assert collect(provider) == ["你好", "，在的"]


def test_stream_skips_non_data_lines_and_malformed_json():
    body = (
        "event: message\n"
        "data: {not json}\n\n"
        'data: {"choices": []}\n\n'
        'data: {"choices": [{"delta": {}}]}\n\n'
        f"data: {delta('ok')}\n\n"
    )
    provider, _ = provider_with([httpx.Response(200, content=body.encode("utf-8"))])
    assert collect(provider) == ["ok"]


def test_429_then_success_retries_once(monkeypatch):
    sleeps = []
    monkeypatch.setattr(deepseek.time, "sleep", lambda seconds: sleeps.append(seconds))
    provider, client = provider_with([
        httpx.Response(429, content=b"rate limited"),
        sse_response(delta("重试成功")),
    ])

    assert collect(provider) == ["重试成功"]
    assert client.sends == 2
    assert sleeps == [deepseek.RATE_LIMIT_BACKOFF_SECONDS]


def test_429_twice_maps_to_upstream_rate_limited(monkeypatch):
    monkeypatch.setattr(deepseek.time, "sleep", lambda seconds: None)
    provider, client = provider_with([
        httpx.Response(429, content=b"rate limited"),
        httpx.Response(429, content=b"rate limited"),
    ])

    with pytest.raises(LlmError) as err:
        collect(provider)

    assert err.value.code == "UPSTREAM_RATE_LIMITED"
    assert client.sends == 2


def test_timeout_maps_to_upstream_timeout():
    provider, _ = provider_with([httpx.ReadTimeout("too slow")])

    with pytest.raises(LlmError) as err:
        collect(provider)

    assert err.value.code == "UPSTREAM_TIMEOUT"


def test_connection_error_maps_to_internal():
    provider, _ = provider_with([httpx.ConnectError("dns")])

    with pytest.raises(LlmError) as err:
        collect(provider)

    assert err.value.code == "INTERNAL"


def test_500_includes_status_and_request_body():
    provider, _ = provider_with([httpx.Response(500, content=b"server exploded")])

    with pytest.raises(LlmError) as err:
        collect(provider)

    assert err.value.code == "INTERNAL"
    assert "500" in str(err.value)


def test_mid_stream_error_keeps_partial_and_maps_to_stream_interrupted():
    response = ExplodingResponse()
    provider, _ = provider_with([response])
    stream = provider.stream([], 8)

    assert next(stream) == "半句"
    with pytest.raises(LlmError) as err:
        next(stream)

    assert err.value.code == "STREAM_INTERRUPTED"
    assert response.closed


def test_empty_api_key_rejected():
    with pytest.raises(ValueError):
        DeepSeekProvider("")
