"""L0 规则前置（architecture.md §4.1）：只做高精度短路与实体抽取。

规则只认两种东西：
  1. 转人工的显式表达——命中即短路，不必再花 15ms 跑 BERT；
  2. 订单号——任何分支都要带上，门控靠它判断能不能查订单。

宁可漏判，不可误判：漏了还有 BERT 兜底，误判会直接走错分支。
"""

import re

# 只认 SO 开头的订单号。裸 11 位数字会误吞手机号，客服场景里代价太高。
ORDER_ID = re.compile(r"SO\d{6,}", re.IGNORECASE)

# 转人工表达的唯一事实来源：线上 L0 与离线评测（build_route_eval.py）共用。
HUMAN_PATTERN = re.compile(
    r"转人工|转真人|人工客服|找人工|要人工|不要机器人|不跟机器人|"
    r"客服电话|客服热线|投诉|举报|负责人|经理|店长"
)


def human_match(query: str) -> str | None:
    """命中转人工表达时返回命中的片段（便于日志与前端解释），否则 None。"""
    match = HUMAN_PATTERN.search(query)
    return match.group() if match else None


def entities(query: str) -> dict[str, str]:
    match = ORDER_ID.search(query)
    return {"orderId": match.group()} if match else {}
