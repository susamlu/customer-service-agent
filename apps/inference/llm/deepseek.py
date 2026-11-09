"""DeepSeek 生成（默认实现，architecture.md §2.1）。

DeepSeek 走 OpenAI 兼容协议，用 httpx 直接发 SSE 请求，不引入 openai SDK：
这里只需要一个流式 chat completions，多一个依赖就多一份版本约束。

错误映射（§8 / §9）：
  429            退避 1 次后仍失败 → UPSTREAM_RATE_LIMITED
  连接/读超时     → UPSTREAM_TIMEOUT
  流中途断开      → STREAM_INTERRUPTED
"""

import json
import time
from typing import Iterator

import httpx

from inference.llm.base import LlmError

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
RATE_LIMIT_BACKOFF_SECONDS = 1.0
TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)


class DeepSeekProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek api_key 不能为空")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client or httpx.Client(timeout=TIMEOUT)

    def stream(
        self, messages: list[dict], max_new_tokens: int
    ) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": True,
        }
        # 429 只在「还没吐字」时退避重试：一旦开始流式输出就无法重放，
        # 重试会重复或跳过内容。
        response = self._open(stream_body=body, retry_on_429=True)
        try:
            yield from self._iter_delta(response)
        except (httpx.HTTPError, OSError) as exc:
            raise LlmError("STREAM_INTERRUPTED", f"生成流中断：{exc}") from exc
        finally:
            response.close()

    def _open(self, stream_body: dict, retry_on_429: bool) -> httpx.Response:
        for attempt in range(2 if retry_on_429 else 1):
            try:
                response = self.client.send(
                    self.client.build_request(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=stream_body,
                    ),
                    stream=True,
                )
            except httpx.TimeoutException as exc:
                raise LlmError("UPSTREAM_TIMEOUT", f"DeepSeek 超时：{exc}") from exc
            except httpx.HTTPError as exc:
                raise LlmError("INTERNAL", f"DeepSeek 请求失败：{exc}") from exc

            if response.status_code == 429 and attempt == 0:
                response.close()
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            if response.status_code == 429:
                response.close()
                raise LlmError("UPSTREAM_RATE_LIMITED", "DeepSeek 限流，请稍后重试")
            if response.status_code >= 400:
                detail = response.read().decode("utf-8", errors="replace")[:200]
                code = response.status_code
                response.close()
                raise LlmError("INTERNAL", f"DeepSeek 返回 {code}：{detail}")
            return response
        raise AssertionError("unreachable")

    @staticmethod
    def _iter_delta(response: httpx.Response) -> Iterator[str]:
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                yield content
