"""前置门控（architecture.md §4.4）：把候选能力集裁成可执行集合。

纯函数，不依赖 FastAPI 与运行时状态；`/api/plan` 与离线评测共用同一实现，
避免「评测口径」与「线上口径」各走一条代码（§4.4）。

human 阈值必须由分类器传入（`route_thresholds["human"]`），这里不设独立默认：
两个真相源会让改阈值后的判定在模型头与门控之间悄悄分叉。
"""

import re
from typing import Iterable, Protocol

from inference.api_models import Capability, GateDecision

ORDER_ID = re.compile(r"SO\d{6,}", re.IGNORECASE)
DEFAULT_HUMAN_THRESHOLD = 0.5


class HistoryItem(Protocol):
    """会话历史条目：只读 content，测试可传任意带该属性的对象。"""

    content: str


def apply_gate(
    caps: Iterable[Capability],
    scores: dict[str, float],
    entities: dict[str, str],
    history: Iterable[HistoryItem] = (),
    *,
    human_threshold: float = DEFAULT_HUMAN_THRESHOLD,
) -> tuple[list[Capability], dict[str, GateDecision]]:
    """返回（可执行能力集，每个候选能力的判定与理由）。

    * knowledge 恒放行。
    * order 需要显式订单号或会话上下文；被裁掉时补回 knowledge（§4.4），
      不让用户只剩一句澄清。
    * human 需要分数 ≥ 分类器阈值。阈值与 `_capabilities` 同源，比较口径也
      一致（含等号），否则恰好等于阈值时两处判定会相反。
    """
    allowed = list(caps)
    gate: dict[str, GateDecision] = {}

    if "knowledge" in allowed:
        gate["knowledge"] = GateDecision(allowed=True, reason="always")

    if "order" in allowed:
        has_id = "orderId" in entities
        has_context = has_id or any(ORDER_ID.search(item.content) for item in history)
        reason = (
            "order_id_present"
            if has_id
            else "order_context_present"
            if has_context
            else "no_order_context"
        )
        gate["order"] = GateDecision(allowed=has_context, reason=reason)
        if not has_context:
            allowed.remove("order")
            if "knowledge" not in allowed:
                allowed.insert(0, "knowledge")
                gate["knowledge"] = GateDecision(allowed=True, reason="always")

    if "human" in allowed:
        hit = scores.get("human", 0.0) >= human_threshold
        comparator = "ge" if hit else "lt"
        gate["human"] = GateDecision(
            allowed=hit, reason=f"p_human_{comparator}_{human_threshold:g}"
        )
        if not hit:
            allowed.remove("human")

    return allowed, gate
