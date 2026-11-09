"""多任务 BERT（§4.2）：每标签阈值、L0 短路。

不加载真模型：用 stub 替换 encoder，`_forward` 直接给分数。这样测的是
分类器自己的决策逻辑，而不是 BERT 权重。

意图识别没有第三层兜底（§4.1），低置信度直接返回 route 头结果。
"""

from prometheus_client import REGISTRY

from inference.intent.multitask import BertIntentClassifier, _capabilities

THRESHOLDS = {"knowledge": 0.35, "order": 0.7, "human": 0.9}


class StubModel:
    def to(self, device):
        return self

    def eval(self):
        return self


CONFIG = {
    "route_labels": ["knowledge", "order", "human"],
    "route_threshold": 0.5,
    "route_thresholds": THRESHOLDS,
    "topic_id2label": {"0": "其他", "1": "物流"},
    "max_length": 64,
}


def make_classifier(config=None):
    classifier = BertIntentClassifier(
        StubModel(), tokenizer=None, config=config or CONFIG, device="cpu",
    )
    classifier._forward = lambda query: (
        {"knowledge": 0.2, "order": 0.8, "human": 0.1}, "物流", 0.9,
    )
    return classifier


def test_capabilities_apply_per_label_thresholds():
    probs = {"knowledge": 0.3, "order": 0.75, "human": 0.55}
    assert _capabilities(probs, THRESHOLDS) == ["order"]


def test_capabilities_returns_route_order():
    probs = {"knowledge": 0.4, "order": 0.8, "human": 0.95}
    assert _capabilities(probs, THRESHOLDS) == ["knowledge", "order", "human"]


def test_capabilities_empty_when_all_below_thresholds():
    probs = {"knowledge": 0.34, "order": 0.69, "human": 0.89}
    assert _capabilities(probs, THRESHOLDS) == []


def test_human_threshold_is_inclusive():
    probs = {"knowledge": 0.0, "order": 0.0, "human": 0.9}
    assert _capabilities(probs, THRESHOLDS) == ["human"]


def test_classify_human_short_circuits_before_bert():
    classifier = make_classifier()
    classifier._forward = lambda query: (_ for _ in ()).throw(AssertionError("不该跑 BERT"))

    result = classifier.classify("帮我转人工")

    assert result.capabilities == ["human"]
    assert result.source == "rule"
    assert result.route_scores["human"] == 1.0


def test_classify_bert_path_returns_threshold_capabilities():
    result = make_classifier().classify("订单 SO123456 到哪了")

    assert result.capabilities == ["order"]
    assert result.source == "bert"
    assert result.entities == {"orderId": "SO123456"}
    assert result.topic == "物流"


def test_low_confidence_stays_on_bert():
    classifier = make_classifier()
    classifier._forward = lambda query: (
        {"knowledge": 0.4, "order": 0.4, "human": 0.4}, "其他", 0.4,
    )

    result = classifier.classify("说不清的问题")

    assert result.source == "bert"
    assert result.capabilities == ["knowledge"]


def _plan_count(source: str) -> float:
    return REGISTRY.get_sample_value("intent_plan_total", {"source": source}) or 0.0


def test_plan_metric_records_final_source():
    before = {source: _plan_count(source) for source in ("rule", "bert")}

    make_classifier().classify("帮我转人工")
    make_classifier().classify("订单 SO123456 到哪了")

    assert _plan_count("rule") == before["rule"] + 1
    assert _plan_count("bert") == before["bert"] + 1


def test_missing_per_label_thresholds_fall_back_to_global():
    config = {
        "route_labels": ["knowledge", "order", "human"],
        "route_threshold": 0.62,
        "topic_id2label": {"0": "其他"},
    }
    classifier = make_classifier(config=config)
    assert classifier.route_thresholds == {
        "knowledge": 0.62, "order": 0.62, "human": 0.62,
    }
