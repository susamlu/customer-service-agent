"""意图路由指标（architecture.md §11.2）。

只做记账，不参与判定。意图识别只有规则短路与 BERT 两层，这里记录最终来源分布，
用于观察两者各占多少。
"""

from prometheus_client import Counter

PLAN_TOTAL = Counter(
    "intent_plan_total",
    "一次意图识别的最终来源分布（rule / bert）",
    ["source"],
)
