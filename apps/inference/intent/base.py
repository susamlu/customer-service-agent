"""意图识别抽象：一次调用给出能力集、主题、实体与来源（architecture.md §4.2）。

IntentResult 只承载「这一句问的是什么」，不含门控结果。门控依赖会话历史，
放在 /api/plan 端点内统一裁剪（单一事实来源，§4.4）；分类器本身不感知历史，
网关的 CapabilityPlanner 只消费已裁剪的能力集。
"""

from dataclasses import dataclass
from typing import Protocol

from inference.api_models import Capability, Source


@dataclass(frozen=True)
class IntentResult:
    """一次意图识别的完整结果，字段与 /api/plan 响应几乎一一对应。"""

    capabilities: list[Capability]
    route_scores: dict[str, float]
    topic: str
    topic_confidence: float
    entities: dict[str, str]
    source: Source


class IntentClassifier(Protocol):
    """实现方只负责 query → IntentResult，不访问网络、不写状态。

    线上实现是多任务 BERT（apps/inference/intent/multitask.py）；协议留着是为了
    门控与评测能对分类器做替身，不改 /api/plan 的签名。

    `route_thresholds` 是门控判定 human 的阈值来源（§4.4）：门控不再自带
    0.5，必须与能力集判定共用同一份阈值。
    """

    route_thresholds: dict[str, float]

    def classify(self, query: str) -> IntentResult: ...
