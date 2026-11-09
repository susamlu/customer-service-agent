"""定位项目根目录。

本地仓库是 `apps/inference/...`（四层），容器里被拍平成 `/app/inference/...`
（三层），所以 `parents[3]` 这类固定层级在两种布局下只能对一个。统一从这里取，
避免每个模块各算一次、容器里静默指到 `/`。
"""

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """优先环境变量，其次向上找同时含 ml/models 与 ml/experiments 的目录。

    容器里 ml/models 由 volume 挂载、ml/experiments 通过 COPY 提供；两者缺一个
    就再退一层，只认 ml/models。都找不到时回退到本地布局的推导值。
    """
    explicit = os.environ.get("PROJECT_ROOT", "").strip()
    if explicit:
        return Path(explicit).resolve()

    for directory in Path(__file__).resolve().parents:
        if (directory / "ml" / "models").is_dir() and (directory / "ml" / "experiments").is_dir():
            return directory
    for directory in Path(__file__).resolve().parents:
        if (directory / "ml" / "models").is_dir():
            return directory
    return Path(__file__).resolve().parents[2]
