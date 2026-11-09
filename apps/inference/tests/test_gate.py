"""前置门控（§4.4）：候选能力集裁成可执行集合。

门控与 /api/plan、离线评测共用 `inference.intent.gate.apply_gate`（§4.4），
human 阈值必须由调用方传入分类器的 `route_thresholds["human"]`，不再自带 0.5。
"""

from inference.api_models import ChatMessage
from inference.intent.gate import apply_gate


def gate(caps, scores=None, entities=None, history=None, threshold=0.5):
    return apply_gate(
        caps,
        scores or {},
        entities or {},
        history or [],
        human_threshold=threshold,
    )


def test_knowledge_is_always_allowed():
    allowed, decisions = gate(["knowledge"])

    assert allowed == ["knowledge"]
    assert decisions["knowledge"].allowed is True
    assert decisions["knowledge"].reason == "always"


def test_order_with_entity_is_allowed():
    allowed, decisions = gate(["order"], entities={"orderId": "SO123456"})

    assert allowed == ["order"]
    assert decisions["order"].allowed is True
    assert decisions["order"].reason == "order_id_present"


def test_order_without_context_is_removed_and_knowledge_is_added():
    allowed, decisions = gate(["order"])

    assert allowed == ["knowledge"]
    assert decisions["order"].allowed is False
    assert decisions["order"].reason == "no_order_context"
    assert decisions["knowledge"].allowed is True


def test_order_context_from_history_is_allowed():
    history = [ChatMessage(role="user", content="我的订单 SO123456 呢")]

    allowed, decisions = gate(["order"], history=history)

    assert allowed == ["order"]
    assert decisions["order"].reason == "order_context_present"


def test_history_without_order_id_does_not_help():
    history = [ChatMessage(role="user", content="你好")]

    allowed, decisions = gate(["order"], history=history)

    assert allowed == ["knowledge"]
    assert decisions["order"].reason == "no_order_context"


def test_order_removal_does_not_duplicate_existing_knowledge():
    allowed, decisions = gate(["knowledge", "order"])

    assert allowed == ["knowledge"]
    assert list(allowed).count("knowledge") == 1
    assert decisions["knowledge"].reason == "always"


def test_human_uses_inclusive_threshold():
    allowed, decisions = gate(["human"], scores={"human": 0.5})
    assert allowed == ["human"]
    assert decisions["human"].reason == "p_human_ge_0.5"

    allowed, decisions = gate(["human"], scores={"human": 0.49})
    assert allowed == []
    assert decisions["human"].allowed is False
    assert decisions["human"].reason == "p_human_lt_0.5"


def test_human_threshold_comes_from_caller():
    allowed, decisions = gate(["human"], scores={"human": 0.6}, threshold=0.7)

    assert allowed == []
    assert decisions["human"].reason == "p_human_lt_0.7"


def test_empty_candidate_set_stays_empty():
    allowed, decisions = gate([])

    assert allowed == []
    assert decisions == {}


def test_combined_caps_all_removed_falls_back_to_knowledge():
    allowed, decisions = gate(["order", "human"], scores={"human": 0.2})

    assert allowed == ["knowledge"]
    assert set(decisions) == {"order", "human", "knowledge"}


def test_gate_only_records_capabilities_that_were_candidates():
    allowed, decisions = gate(["knowledge"])

    assert allowed == ["knowledge"]
    assert set(decisions) == {"knowledge"}


def test_input_list_is_not_mutated():
    caps = ["order", "human"]

    gate(caps, scores={"human": 0.0})

    assert caps == ["order", "human"]
