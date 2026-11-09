"""导出模型服务 OpenAPI 快照，供 Java 侧契约测试与评审 diff。

用法：
  PYTHONPATH=apps .venv/bin/python apps/inference/dump_openapi.py

改动接口后必须重跑本脚本并提交 docs/openapi.json，否则契约测试失败。
"""

import json
from pathlib import Path

from inference.model_service import app

OUT = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"


def main() -> None:
    OUT.write_text(
        json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
