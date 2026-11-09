"""项目根定位与运行时资源可达性。

本地是 `apps/inference/...`，容器里拍平成 `/app/inference/...`；固定 parents[N]
在两种布局下只能对一个。这里锁住 project_root() 的约定，避免容器里静默指到 `/`。
"""

from inference import paths
from inference.intent import capability_map


def test_project_root_contains_runtime_markers():
    root = paths.project_root()
    assert (root / "ml" / "models").is_dir()
    assert (root / "ml" / "experiments").is_dir()
    assert (root / "ml" / "runtime").is_dir()


def test_project_root_env_override_wins(monkeypatch, tmp_path):
    paths.project_root.cache_clear()
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    try:
        assert paths.project_root() == tmp_path.resolve()
    finally:
        monkeypatch.delenv("PROJECT_ROOT")
        paths.project_root.cache_clear()


def test_capability_map_is_loadable_from_resolved_root():
    """映射表是训练标签与运行时能力集的唯一来源，路径错了部署时才炸。"""
    assert capability_map.MAP_FILE.is_file()
    assert len(capability_map.all_topics()) == 41
