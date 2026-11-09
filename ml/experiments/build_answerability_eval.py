"""构建可答性评测集（architecture.md §5.4）。

可答性闸门要回答的问题不是「检索排得对不对」，而是「知识库里到底有没有
能回答这条问题的内容」。现有评测集回答不了：qa_test 每条问题在 qa_train 里
都有对应答案，天然可答，没有「知识库不覆盖」的负样本。

两类来源，都由人工定义标签，不依赖模型预测：

  in_kb    120 条，从 qa_test 按固定 seed 抽样。KB（qa_train）里必有同意图
           答案，是「必须放行」的正样本；用来量误拦率。
  out_kb    60 条，手写。天气、编程、医疗、生活常识等和电商客服无关的问题。
           用来量拦截率。
  boundary  10 条，手写。字面含域外实体（iPhone、猫粮、口红），但问的是
           退换货 / 物流等域内政策。属于 in_kb，单独切片观察，不计入硬门槛。

字段：
  query       用户问法
  label       期望判定：in_kb（可答）| out_kb（不可答）
  provenance  qa_test | handwritten | boundary
  note        抽样出处或人工备注

用法：
  .venv/bin/python3 ml/experiments/build_answerability_eval.py
"""

import json
import random
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "answerability_eval.jsonl"
SEED = 42
IN_KB_TOTAL = 120

# 手写域外问题：确保和电商客服知识库无关，不被召回「碰巧」命中。
OUT_KB = [
    "今天天气怎么样",
    "明天会下雨吗",
    "帮我写一首关于春天的诗",
    "推荐一部好看的电影",
    "你能讲个笑话吗",
    "给我讲个睡前故事",
    "教我怎么做红烧肉",
    "麦当劳外卖怎么点",
    "附近有什么好吃的",
    "帮我订一张去上海的机票",
    "高铁票怎么改签",
    "北京的旅游景点有哪些",
    "怎么学好英语口语",
    "帮我翻译一句英文",
    "我的房贷利率是多少",
    "今天股市会涨吗",
    "比特币现在多少钱",
    "你们公司股票代码是多少",
    "帮我算一下 37 乘以 89",
    "3 的平方根是多少",
    "世界和平是什么意思",
    "人为什么会有梦想",
    "我失恋了怎么办",
    "怎么和老板提加薪",
    "明天是星期几",
    "现在几点了",
    "你是谁开发的",
    "你会下围棋吗",
    "帮我写一段 Python 代码",
    "什么是对称加密",
    "推荐几本好看的小说",
    "周杰伦的新歌叫什么",
    "今晚的球赛几点开始",
    "世界杯冠军是谁",
    "减肥应该吃什么",
    "怎么练出腹肌",
    "感冒了要吃什么药",
    "我的车该保养了",
    "汽车胎压多少合适",
    "怎么养一只猫",
    "猫粮哪个牌子好",
    "我的手机屏幕碎了去哪修",
    "iphone 15 多少钱",
    "华为和小米哪个好",
    "怎么申请签证",
    "护照过期了怎么办",
    "水电费怎么交",
    "怎么交个人所得税",
    "帮我写一份简历",
    "面试时怎么自我介绍",
    "世界上最高的山是哪座",
    "地球到月球有多远",
    "帮我起个男孩的名字",
    "结婚纪念日送什么礼物",
    "情人节是几月几号",
    "怎么学弹吉他",
    "钢琴考级有什么用",
    "怎么减肥最快",
    "为什么晚上睡不着",
    "帮我推荐一首好听的歌",
]

# 字面有域外实体、实际问的是域内政策。这些必须放行，否则闸门会把可答问题拦掉。
BOUNDARY = [
    "我买的 iPhone 15 能七天无理由退货吗",
    "我买的猫粮拆封了还能退吗",
    "我老公送的口红不合适能换色号吗",
    "买错了尺码怎么换货",
    "双十一买的东西什么时候发货",
    "快递到了但我出差不在家怎么办",
    "买的护肤品过敏了能退吗",
    "下单后多久发货",
    "商品降价了能补差价吗",
    "退款审核要多久",
]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def build_in_kb() -> list[dict]:
    rows = [
        row
        for row in load_jsonl(PROCESSED_DIR / "qa_test.jsonl")
        if row["intent"]
    ]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return [
        {
            "query": row["question"],
            "label": "in_kb",
            "provenance": "qa_test",
            "note": row["intent"],
        }
        for row in rows[:IN_KB_TOTAL]
    ]


def handwritten(items: list[str], label: str, provenance: str) -> list[dict]:
    return [
        {"query": query, "label": label, "provenance": provenance, "note": ""}
        for query in items
    ]


def main() -> None:
    items = (
        build_in_kb()
        + handwritten(OUT_KB, "out_kb", "handwritten")
        + handwritten(BOUNDARY, "in_kb", "boundary")
    )

    seen: set[str] = set()
    deduped = []
    for item in items:
        if item["query"] in seen:
            continue
        seen.add(item["query"])
        deduped.append(item)
    if len(deduped) != len(items):
        print(f"去重丢弃 {len(items) - len(deduped)} 条重复问法")

    OUT_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in deduped) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_PATH}  共 {len(deduped)} 条")
    print(f"标签分布：{dict(Counter(item['label'] for item in deduped))}")
    print(f"来源分布：{dict(Counter(item['provenance'] for item in deduped))}")


if __name__ == "__main__":
    main()
