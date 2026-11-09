"""L1 多任务 BERT（architecture.md §4.2）：一次前向同时给出 route 与 topic。

这是一个自包含推理实现，只依赖 `ml/models/intent-multitask-bert/` 目录：
  encoder/        BertModel 权重（AutoModel 加载）
  heads.pt        route_head / topic_head 的 state_dict
  config.json     标签空间、阈值、max_length

训练侧（ml/experiments/train_intent_multitask.py）保存的正是这套结构。
训练时对 [CLS] 向量做的 dropout 在 eval 模式下是恒等变换，所以推理侧直接取
`last_hidden_state[:, 0]`，结果与训练时完全一致。

本类是 `/api/plan` 的唯一实现；旧的「规则 → 向量 → 3B」分层路由已下线。
"""

import json
import os
import threading
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from inference.api_models import Capability
from inference.intent import metrics, rule
from inference.intent.base import IntentResult
from inference.paths import project_root

DEFAULT_MODEL_DIR = project_root() / "ml" / "models" / "intent-multitask-bert"
OTHER = "其他"
ROUTE_ORDER: tuple[Capability, ...] = ("knowledge", "order", "human")


class MultiTaskIntentModel(torch.nn.Module):
    """编码器 + 两个头。结构与训练脚本保持一致，只省掉 eval 下无作用的 dropout。"""

    def __init__(self, encoder, topic_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.topic_head = torch.nn.Linear(hidden, topic_size)
        self.route_head = torch.nn.Linear(hidden, len(ROUTE_ORDER))

    def forward(self, **batch):
        output = self.encoder(**batch)
        pooled = output.last_hidden_state[:, 0]
        return self.topic_head(pooled), self.route_head(pooled)


class BertIntentClassifier:
    """L0 规则短路 → L1 BERT，两层；低置信度直接返回 route 头结果。"""

    def __init__(
        self,
        model: MultiTaskIntentModel,
        tokenizer,
        config: dict,
        device: str,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = int(config.get("max_length", 64))
        self.route_labels: tuple[Capability, ...] = tuple(config["route_labels"])
        # 阈值取自 config.json 的 route_thresholds（训练脚本在 val 上逐标签扫描写入，
        # §4.2），缺键回落到全局 route_threshold。三个能力正例率差 25 倍，共用一个
        # 阈值会同时压不住 human 误触发和 order 召回。
        # 三个值都为 0.5 时有两种可能：扫描最优恰好是全 0.5，或没有组合能让
        # order precision 达标而走了回退分支——训练日志会打印是哪一种。
        default_threshold = float(config.get("route_threshold", 0.5))
        configured = config.get("route_thresholds") or {}
        self.route_thresholds = {
            label: float(configured.get(label, default_threshold))
            for label in ROUTE_ORDER
        }
        self.topic_labels = [
            config["topic_id2label"][str(i)] for i in range(len(config["topic_id2label"]))
        ]
        # FastAPI 的同步端点跑在线程池里，同一个 encoder 并发前向会互相抢占 MPS。
        # 单次前向只有 ~15ms，加锁的代价远小于排队的抖动。
        self._lock = threading.Lock()

    def classify(self, query: str) -> IntentResult:
        entities = rule.entities(query)

        # L0：转人工是最明确的显式表达，短路后连 BERT 都不跑。
        if rule.human_match(query):
            metrics.PLAN_TOTAL.labels("rule").inc()
            return IntentResult(
                capabilities=["human"],
                route_scores={"knowledge": 0.0, "order": 0.0, "human": 1.0},
                topic=OTHER,
                topic_confidence=1.0,
                entities=entities,
                source="rule",
            )

        probs, topic, topic_confidence = self._forward(query)
        metrics.PLAN_TOTAL.labels("bert").inc()
        return IntentResult(
            capabilities=_capabilities(probs, self.route_thresholds),
            route_scores={label: round(probs[label], 4) for label in ROUTE_ORDER},
            topic=topic,
            topic_confidence=topic_confidence,
            entities=entities,
            source="bert",
        )

    def warmup(self) -> None:
        """把 Metal kernel 编译成本挪到启动阶段，首个请求不吃这笔钱。"""
        self.classify("预热")

    def _forward(self, query: str) -> tuple[dict[str, float], str, float]:
        batch = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with self._lock, torch.no_grad():
            topic_logits, route_logits = self.model(**batch)
            route_probs = torch.sigmoid(route_logits)[0].cpu().tolist()
            topic_probs = torch.softmax(topic_logits[0], dim=-1)
            topic_confidence, topic_index = topic_probs.max(dim=-1)
        probs = {label: float(p) for label, p in zip(self.route_labels, route_probs)}
        missing = [label for label in ROUTE_ORDER if label not in probs]
        if missing:
            raise ValueError(f"模型 config 的 route_labels 缺少 {missing}，换模型需同步标签顺序")
        return probs, self.topic_labels[int(topic_index)], round(float(topic_confidence), 4)


def _capabilities(
    probs: dict[str, float], thresholds: dict[str, float]
) -> list[Capability]:
    """route 多标签按各自阈值取命中。

    这里不补 knowledge：纯订单主题（如「订单状态解释」）在映射表里就是 [order]，
    无知识可讲；是否要知识兜底由门控在 `order` 被裁掉时决定（§4.4）。
    """
    caps = [label for label in ROUTE_ORDER if probs[label] >= thresholds[label]]
    return sorted(caps, key=ROUTE_ORDER.index)


def build_classifier(
    model_dir: Path | None = None,
    device: str | None = None,
) -> BertIntentClassifier:
    """从自包含目录重建分类器；线上与离线评测共用这一个入口，避免装配分叉。

    默认路径可用 INTENT_MODEL_DIR 覆盖。
    """
    directory = Path(
        model_dir or os.environ.get("INTENT_MODEL_DIR") or DEFAULT_MODEL_DIR
    )
    if not directory.exists():
        raise FileNotFoundError(f"找不到意图模型目录：{directory}（先跑 T6 训练脚本）")

    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    topic_size = len(config["topic_id2label"])
    encoder = AutoModel.from_pretrained(str(directory / "encoder"))
    model = MultiTaskIntentModel(encoder, topic_size)
    heads = torch.load(directory / "heads.pt", map_location="cpu")
    model.topic_head.load_state_dict(heads["topic_head"])
    model.route_head.load_state_dict(heads["route_head"])

    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    return BertIntentClassifier(model, tokenizer, config, resolved_device)
