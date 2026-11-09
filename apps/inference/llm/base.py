"""生成抽象。线上实现是 DeepSeek（architecture.md §2.1），网关只依赖接口。

stream 必须是惰性迭代：SSE 靠逐 token 推送压首字延迟，把整段生成完再返回
等于把流式退化成阻塞调用。

LlmError 携带 §9 的错误码，服务层把它折成 SSE error 事件；不允许把底层
异常文本直接当协议字段透传。
"""

from typing import Iterator, Protocol


class LlmError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LlmProvider(Protocol):
    def stream(
        self, messages: list[dict], max_new_tokens: int
    ) -> Iterator[str]: ...
