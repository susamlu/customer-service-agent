"""读取主题→能力映射表（ml/experiments/intent_capability_map.yaml）。

这张表只服务训练与校验：能力集直接来自 route 头的多标签输出（§4.3），
运行时不做 topic→能力查表。只读 + 缓存；改表走 PR，不在运行时写。
"""

from functools import lru_cache

import yaml

from inference.paths import project_root

MAP_FILE = project_root() / "ml" / "experiments" / "intent_capability_map.yaml"


@lru_cache(maxsize=1)
def _topics() -> dict[str, tuple[str, ...]]:
    data = yaml.safe_load(MAP_FILE.read_text(encoding="utf-8")) or {}
    return {
        name: tuple(caps) for name, caps in (data.get("topics") or {}).items()
    }


def all_topics() -> tuple[str, ...]:
    return tuple(_topics())
