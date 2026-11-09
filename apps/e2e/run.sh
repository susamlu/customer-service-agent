#!/usr/bin/env bash
# 端到端黑盒测试（architecture.md §10.2）：真实网关 + 模型服务。
# 用法（项目根执行）：
#   ./apps/e2e/run.sh           # 栈没起就先起，跑完保留
#   ./apps/e2e/run.sh --stop    # 跑完停掉由本脚本启动的栈
set -euo pipefail
cd "$(dirname "$0")/../.."

GATEWAY=${E2E_GATEWAY:-http://127.0.0.1:8090}
STOP_AFTER=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--stop" ]; then
    STOP_AFTER=1
  else
    ARGS+=("$arg")
  fi
done

if ! curl -s --max-time 2 "${GATEWAY}/api/model-status" > /dev/null; then
  ./apps/deploy/run.sh
fi

export E2E_REQUIRED=1
export E2E_GATEWAY="${GATEWAY}"
status=0
PYTHONPATH=apps:ml/runtime .venv/bin/python3 -m pytest apps/e2e -q "${ARGS[@]}" || status=$?

if [ "${STOP_AFTER}" = "1" ]; then
  ./apps/deploy/run.sh --stop
fi
exit "${status}"
