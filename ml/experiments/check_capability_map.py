"""校验 intent_capability_map.yaml 覆盖模型全部主题标签。

architecture.md §4.3 规则 1：每个主题必须显式出现，不允许隐式默认，缺失键在 CI
里报错。CI 在训练与部署前调用本脚本；本地改表后先跑一遍再提交。

用法：
  python3 ml/experiments/check_capability_map.py
"""

import json
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
MULTITASK_CONFIG = PROJECT_DIR / "models" / "intent-multitask-bert" / "config.json"
MAP_FILE = PROJECT_DIR / "experiments" / "intent_capability_map.yaml"

ALLOWED = {"knowledge", "order", "human"}


def load_labels() -> set[str]:
    """标签唯一来源是多任务 BERT；权重缺失直接失败，不做隐式回退。"""
    data = json.loads(MULTITASK_CONFIG.read_text(encoding="utf-8"))
    return set(data["topic_id2label"].values())


def main() -> int:
    labels = load_labels()
    data = yaml.safe_load(MAP_FILE.read_text(encoding="utf-8")) or {}
    topics = data.get("topics") or {}

    errors: list[str] = []
    missing = sorted(labels - topics.keys())
    extra = sorted(topics.keys() - labels)
    if missing:
        errors.append(f"缺少主题 {len(missing)} 个：{missing}")
    if extra:
        errors.append(f"未知主题（模型标签里没有）：{extra}")

    for name, caps in topics.items():
        if not isinstance(caps, list) or not caps:
            errors.append(f"{name}: 能力集必须是非空列表")
            continue
        bad = set(caps) - ALLOWED
        if bad:
            errors.append(f"{name}: 非法能力 {sorted(bad)}")
        if len(caps) != len(set(caps)):
            errors.append(f"{name}: 能力重复")

    if errors:
        print("\n".join(errors))
        return 1
    print(f"OK: {len(topics)} 个主题全部覆盖")
    return 0


if __name__ == "__main__":
    sys.exit(main())
