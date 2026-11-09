"""L0 规则层（§4.1）：只验高精度短路与订单号抽取的边界。"""

import pytest

from inference.intent import rule


@pytest.mark.parametrize(
    "query",
    [
        "我要转人工",
        "找人工客服",
        "不要机器人，转真人",
        "我要投诉",
        "帮我叫你们经理",
        "客服电话是多少",
    ],
)
def test_human_match_hits_explicit_phrases(query):
    assert rule.human_match(query) is not None


@pytest.mark.parametrize(
    "query",
    [
        "怎么退货",
        "我的订单到哪了",
        "运费多少",
        "这个颜色还有别的吗",
    ],
)
def test_human_match_returns_none_without_explicit_phrase(query):
    assert rule.human_match(query) is None


def test_human_match_returns_matched_fragment():
    assert rule.human_match("你好，请帮我转人工") == "转人工"


def test_entities_extracts_so_order_id():
    assert rule.entities("订单 SO123456 发货了吗") == {"orderId": "SO123456"}


def test_entities_is_case_insensitive_but_keeps_original_text():
    assert rule.entities("so998877 到哪了") == {"orderId": "so998877"}


def test_entities_requires_so_prefix():
    # 裸数字（手机号、快递单号）不认，避免误吞
    assert rule.entities("13812345678 打不通") == {}
    assert rule.entities("单号 12345678901") == {}


def test_entities_requires_at_least_six_digits():
    assert rule.entities("SO12345") == {}
