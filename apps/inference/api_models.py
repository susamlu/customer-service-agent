"""模型服务对外契约，与 docs/architecture.md §3 一一对应。

只放已接入的协议对象，不放推理逻辑。每接入一个端点，就在这里补一套模型；
改字段必须同步三处：
  1. docs/architecture.md §3
  2. Java 侧 ModelServiceClient
  3. docs/openapi.json（由 dump_openapi.py 生成）
"""

from typing import Literal

from pydantic import BaseModel, Field

Capability = Literal["knowledge", "order", "human"]
Source = Literal["rule", "bert"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class PlanRequest(BaseModel):
    query: str
    session_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class Topic(BaseModel):
    label: str
    confidence: float


class GateDecision(BaseModel):
    allowed: bool
    reason: str


class PlanResponse(BaseModel):
    capabilities: list[Capability]
    route_scores: dict[str, float]
    topic: Topic
    entities: dict[str, str]
    gate: dict[str, GateDecision]
    source: Source
    elapsed_ms: float


class ErrorResponse(BaseModel):
    """§3.5 统一错误封装。code 取值见 §9。"""

    code: str
    message: str
    retriable: bool
