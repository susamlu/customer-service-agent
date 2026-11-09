"""多任务意图微调：共享编码器 + 两个分类头（architecture.md §4.2）。

  route 头  3 类多标签 sigmoid → {knowledge, order, human}
  topic 头  41 类 softmax       → 检索分区 / 可解释主题

route 标签不是人工标的，而是由 topic 经 intent_capability_map.yaml 派生
（§4.3）：零额外标注成本，改表即可改标签，缺键在训练前直接报错。

产物是一个自包含目录：
  encoder/      BertModel 权重
  heads.pt      route_head / topic_head 的 state_dict
  config.json   标签空间、阈值、max_length、基座名

serving 侧按 config.json 重建编码器并挂两个头，不依赖 ModelScope 快照。

阈值调参在训练结束后进行：在 val 上按标签搜阈值（subset accuracy 最优，且 order
precision 不低于 0.90），结果写进 config.json 的 `route_thresholds`。

用法：
  .venv/bin/python3 ml/experiments/train_intent_multitask.py
  .venv/bin/python3 ml/experiments/train_intent_multitask.py --limit 500 --epochs 1
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELSCOPE_DIR = PROJECT_DIR / "models" / "models"
MAP_FILE = PROJECT_DIR / "experiments" / "intent_capability_map.yaml"

sys.path.insert(0, str(PROJECT_DIR / "runtime"))
from topic_labels import MIN_COUNT, OTHER  # noqa: E402

DEFAULT_MODEL = "google-bert--bert-base-chinese"
DEFAULT_OUTPUT = PROJECT_DIR / "models" / "intent-multitask-bert"
SEED = 42
MAX_LENGTH = 64
ROUTE_LABELS = ("knowledge", "order", "human")

# 阈值在 val 上按标签各调各的，目标函数是 subset accuracy，order 的 precision
# 不得低于 §10.3 的门槛。单标签共用一个 0.5 会同时踩两个坑：human 正例只占
# 3.7%，0.5 时大面积误触发；knowledge 占 92%，0.5 时又偏保守。
THRESHOLD_GRID = tuple(round(0.20 + 0.05 * i, 2) for i in range(16))
ORDER_PRECISION_FLOOR = 0.90


def find_snapshot(name: str) -> Path:
    matches = list(MODELSCOPE_DIR.glob(f"{name}/snapshots/*"))
    if not matches:
        raise FileNotFoundError(f"找不到模型 {name}，先运行 ModelScope 下载")
    return matches[0]


def load(split: str) -> list[dict]:
    path = PROCESSED_DIR / f"qa_{split}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [row for row in rows if row["intent"]]


def load_capability_map() -> dict[str, tuple[str, ...]]:
    data = yaml.safe_load(MAP_FILE.read_text(encoding="utf-8")) or {}
    return {
        name: tuple(caps) for name, caps in (data.get("topics") or {}).items()
    }


class MultiTaskDataset(Dataset):
    def __init__(self, questions, topics, routes, tokenizer) -> None:
        self.encodings = tokenizer(
            questions,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        self.topics = torch.tensor(topics, dtype=torch.long)
        self.routes = torch.tensor(routes, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.topics)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["topic"] = self.topics[index]
        item["route"] = self.routes[index]
        return item


class MultiTaskIntentModel(torch.nn.Module):
    def __init__(self, encoder, topic_labels: list[str], dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(dropout)
        hidden = encoder.config.hidden_size
        self.topic_head = torch.nn.Linear(hidden, len(topic_labels))
        self.route_head = torch.nn.Linear(hidden, len(ROUTE_LABELS))

    def forward(self, **batch):
        output = self.encoder(**batch)
        pooled = self.dropout(output.last_hidden_state[:, 0])
        return self.topic_head(pooled), self.route_head(pooled)


@torch.no_grad()
def predict(model, loader, device: str):
    model.eval()
    topic_preds, topic_true, route_probs, route_true = [], [], [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        topic_labels = batch.pop("topic")
        route_labels = batch.pop("route")
        topic_logits, route_logits = model(**batch)
        topic_preds.extend(topic_logits.argmax(dim=-1).cpu().tolist())
        topic_true.extend(topic_labels.cpu().tolist())
        route_probs.append(torch.sigmoid(route_logits).cpu().numpy())
        route_true.append(route_labels.cpu().numpy())
    return (
        np.array(topic_preds),
        np.array(topic_true),
        np.concatenate(route_probs),
        np.concatenate(route_true),
    )


def route_metrics(
    probs: np.ndarray, truth: np.ndarray, thresholds: dict[str, float]
) -> dict:
    preds = np.stack(
        [probs[:, i] >= thresholds[label] for i, label in enumerate(ROUTE_LABELS)],
        axis=1,
    ).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, preds, average=None, zero_division=0
    )
    per_label = {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(truth[:, i].sum()),
        }
        for i, label in enumerate(ROUTE_LABELS)
    }
    return {
        "thresholds": thresholds,
        "subset_accuracy": round(float((preds == truth).all(axis=1).mean()), 4),
        "per_label": per_label,
    }


def tune_thresholds(probs: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """在 val 上逐个标签扫阈值：最大化 subset accuracy，order precision 不破门槛。

    返回的字典直接写进 config.json 的 `route_thresholds`，serving 侧按标签取阈值，
    与训练时看到的分布保持一致。
    """
    truth = truth.astype(bool)
    masks = {
        label: {value: probs[:, i] >= value for value in THRESHOLD_GRID}
        for i, label in enumerate(ROUTE_LABELS)
    }
    order_truth = truth[:, ROUTE_LABELS.index("order")]

    best_subset = -1.0
    best: dict[str, float] | None = None
    for knowledge in THRESHOLD_GRID:
        for order in THRESHOLD_GRID:
            order_pred = masks["order"][order]
            true_positives = int((order_pred & order_truth).sum())
            false_positives = int((order_pred & ~order_truth).sum())
            order_precision = true_positives / (true_positives + false_positives) if (
                true_positives + false_positives
            ) else 1.0
            if order_precision < ORDER_PRECISION_FLOOR:
                continue
            for human in THRESHOLD_GRID:
                preds = np.stack(
                    [masks["knowledge"][knowledge], order_pred, masks["human"][human]],
                    axis=1,
                )
                subset = float((preds == truth).all(axis=1).mean())
                if subset > best_subset:
                    best_subset = subset
                    best = {"knowledge": knowledge, "order": order, "human": human}

    # 极端情况下（模型完全学不出 order）没有任何组合能过门槛，退回 0.5 并告警，
    # 由训练脚本打印的指标暴露问题，而不是悄悄挑一个破门槛的阈值。
    # 两种 0.5 必须能区分：显式打印走了哪一支，读 config.json 的人才知道阈值
    # 是扫描结果还是回退值。
    if best is None:
        print("警告：val 上没有任何阈值组合能让 order precision 达标，回退全部 0.5", flush=True)
        return {label: 0.5 for label in ROUTE_LABELS}

    print(f"阈值扫描命中：val subset {best_subset:.4f} @ {best}", flush=True)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    # 3 epoch / 2e-5 训出来明显欠拟合（topic acc 0.54）：17.8k 样本 × 558 step/epoch
    # 只走了 1674 步。默认改为 6 epoch / 3e-5。
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本，0 表示全部")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ModelScope 目录名")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    capability_map = load_capability_map()
    train_all = load("train")
    val = load("val")
    test = load("test")

    # 标签空间永远按全量训练集确定，--limit 只裁剪参与训练的样本，
    # 否则小样本冒烟时会算出一个残缺的 label2id。
    counts = Counter(row["intent"] for row in train_all)
    keep = {intent for intent, count in counts.items() if count >= MIN_COUNT}
    train = train_all[: args.limit] if args.limit else train_all
    topic_labels = sorted(keep | {OTHER})
    label2id = {label: index for index, label in enumerate(topic_labels)}
    other_id = label2id[OTHER]

    # 映射表是 route 标签的唯一来源（§4.3）：缺一个主题就训练出错误的 route 头，
    # 所以这里直接失败，不做隐式兜底。
    missing = sorted(set(topic_labels) - capability_map.keys())
    extra = sorted(capability_map.keys() - set(topic_labels))
    if missing or extra:
        raise SystemExit(
            f"intent_capability_map.yaml 与训练标签空间不一致："
            f"缺少 {missing}，多余 {extra}"
        )

    def encode(rows: list[dict]):
        questions, topics, routes = [], [], []
        for row in rows:
            topic = row["intent"] if row["intent"] in keep else OTHER
            questions.append(row["question"])
            topics.append(label2id[topic])
            caps = capability_map[topic]
            routes.append([1.0 if label in caps else 0.0 for label in ROUTE_LABELS])
        return questions, topics, routes

    train_q, train_y, train_r = encode(train)
    val_q, val_y, val_r = encode(val)
    test_q, test_y, test_r = encode(test)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(find_snapshot(args.model)))
    encoder = AutoModel.from_pretrained(str(find_snapshot(args.model)))
    model = MultiTaskIntentModel(encoder, topic_labels).to(device)

    train_loader = DataLoader(
        MultiTaskDataset(train_q, train_y, train_r, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MultiTaskDataset(val_q, val_y, val_r, tokenizer),
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(
        MultiTaskDataset(test_q, test_y, test_r, tokenizer),
        batch_size=64,
        shuffle=False,
    )

    # 与单任务基线相同的 class_weight="balanced" 口径，保证两条基线可对比。
    # 冒烟用的子集可能缺类，只对出现过的类算权重，其余保持 1.0。
    present = np.unique(train_y)
    weights = np.ones(len(topic_labels), dtype=np.float32)
    weights[present] = compute_class_weight(
        "balanced", classes=present, y=np.array(train_y)
    )
    topic_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    # route 头的正负例失衡用平方根阻尼，而不是完整的 (N-pos)/pos。human 只占 3.7%，
    # 完整权重是 25.7，会把 human 头推成「宁可错杀」——实测 P 只有 0.64。失衡的
    # 剩余部分交给每标签阈值去纠正（tune_thresholds）。
    route_counts = np.array(train_r).sum(axis=0)
    route_pos_weight = torch.tensor(
        np.sqrt((len(train_r) - route_counts) / np.maximum(route_counts, 1)),
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(10, int(total_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    topic_loss_fn = torch.nn.CrossEntropyLoss(weight=topic_weights)
    route_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=route_pos_weight)

    print(
        f"train={len(train)} val={len(val)} test={len(test)} "
        f"topics={len(topic_labels)} route_pos_weight={route_pos_weight.tolist()}",
        flush=True,
    )

    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        running_topic = 0.0
        running_route = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            topic_labels_batch = batch.pop("topic")
            route_labels_batch = batch.pop("route")
            topic_logits, route_logits = model(**batch)
            topic_loss = topic_loss_fn(topic_logits, topic_labels_batch)
            route_loss = route_loss_fn(route_logits, route_labels_batch)
            loss = topic_loss + args.route_weight * route_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_topic += topic_loss.item()
            running_route += route_loss.item()
            if step % 100 == 0:
                print(
                    f"epoch {epoch + 1} step {step}/{len(train_loader)} "
                    f"topic_loss {running_topic / step:.4f} "
                    f"route_loss {running_route / step:.4f}",
                    flush=True,
                )

        val_topic, val_topic_true, val_route, val_route_true = predict(
            model, val_loader, device
        )
        val_topic_acc = accuracy_score(val_topic_true, val_topic)
        val_topic_f1 = f1_score(val_topic_true, val_topic, average="macro")
        val_route_metrics = route_metrics(
            val_route, val_route_true, {label: 0.5 for label in ROUTE_LABELS}
        )
        score = 0.5 * val_route_metrics["subset_accuracy"] + 0.5 * val_topic_f1
        print(
            f"epoch {epoch + 1} 完成  topic_acc {val_topic_acc:.4f}  "
            f"topic_macro-F1 {val_topic_f1:.4f}  "
            f"route_subset {val_route_metrics['subset_accuracy']:.4f}  "
            f"综合 {score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            # 阈值此时还没调，先按 0.5 写；训练结束后 tune_thresholds 会覆写 config.json。
            save(
                model,
                tokenizer,
                topic_labels,
                args.model,
                output_dir,
                {label: 0.5 for label in ROUTE_LABELS},
            )
            print(f"  综合指标提升，已保存到 {output_dir}", flush=True)

    print(f"\n最佳 val 综合指标 {best_score:.4f}")
    model = load_model(output_dir).to(device)

    # 阈值必须在 val 上定，test 与 route_eval 只用来验收。
    _, _, val_route, val_route_true = predict(model, val_loader, device)
    thresholds = tune_thresholds(val_route, val_route_true)
    print(f"val 调参阈值 {thresholds}", flush=True)
    write_config(args.model, topic_labels, thresholds, output_dir)
    print(
        "val@调参阈值 "
        + json.dumps(route_metrics(val_route, val_route_true, thresholds), ensure_ascii=False)
    )

    test_topic, test_topic_true, test_route, test_route_true = predict(
        model, test_loader, device
    )
    test_f1 = f1_score(test_topic_true, test_topic, average="macro")
    print(
        f"test: topic accuracy {accuracy_score(test_topic_true, test_topic):.4f}  "
        f"topic macro-F1 {test_f1:.4f}"
    )
    print(json.dumps(route_metrics(test_route, test_route_true, thresholds), ensure_ascii=False, indent=2))

    other_mask = test_topic_true == other_id
    other_recall = (
        float((test_topic[other_mask] == other_id).mean())
        if other_mask.any()
        else 0.0
    )
    print(f"{OTHER} 类: n={int(other_mask.sum())}  recall {other_recall:.4f}")


def save(
    model,
    tokenizer,
    topic_labels,
    base_model,
    output_dir: Path,
    thresholds: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(output_dir / "encoder"))
    tokenizer.save_pretrained(str(output_dir))
    torch.save(
        {
            "topic_head": model.topic_head.state_dict(),
            "route_head": model.route_head.state_dict(),
        },
        output_dir / "heads.pt",
    )
    write_config(base_model, topic_labels, thresholds, output_dir)


def write_config(
    base_model: str,
    topic_labels: list[str],
    thresholds: dict[str, float],
    output_dir: Path,
) -> None:
    """只写 config.json：checkpoint 保存与阈值调参后各写一次，权重不重复落盘。"""
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "max_length": MAX_LENGTH,
                "route_labels": list(ROUTE_LABELS),
                "route_threshold": 0.5,
                "route_thresholds": thresholds,
                "topic_id2label": {
                    str(index): label for index, label in enumerate(topic_labels)
                },
                "topic_label2id": {
                    label: index for index, label in enumerate(topic_labels)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_model(model_dir: Path) -> MultiTaskIntentModel:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    topic_labels = [
        config["topic_id2label"][str(i)] for i in range(len(config["topic_id2label"]))
    ]
    encoder = AutoModel.from_pretrained(str(model_dir / "encoder"))
    model = MultiTaskIntentModel(encoder, topic_labels)
    heads = torch.load(model_dir / "heads.pt", map_location="cpu")
    model.topic_head.load_state_dict(heads["topic_head"])
    model.route_head.load_state_dict(heads["route_head"])
    return model


if __name__ == "__main__":
    main()
