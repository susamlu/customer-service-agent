"""让测试不依赖调用时的 PYTHONPATH：统一把 apps/ 放进 sys.path。"""

import sys
from pathlib import Path

APPS_DIR = Path(__file__).resolve().parents[2]
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))
