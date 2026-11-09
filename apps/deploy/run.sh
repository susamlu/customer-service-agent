#!/usr/bin/env bash
# 一键启动客服 Chat：模型服务(8000) + Spring Boot 网关(8090) + 浏览器
# 用法（在项目根执行）：
#   ./apps/deploy/run.sh            前端已构建则直接启动，否则先构建
#   ./apps/deploy/run.sh --build    强制重新构建前端
#   ./apps/deploy/run.sh --stop     停止全部服务
set -euo pipefail
cd "$(dirname "$0")/../.."   # 项目根

MODEL_PORT=8000
GATEWAY_PORT=8090
MODEL_LOG=/tmp/csagent-model.log
GATEWAY_LOG=/tmp/csagent-gateway.log

stop_all() {
  for pidfile in /tmp/csagent-model.pid /tmp/csagent-gateway.pid; do
    if [ -f "$pidfile" ]; then
      kill "$(cat "$pidfile")" 2>/dev/null || true
      rm -f "$pidfile"
    fi
  done
  echo "已停止"
}

if [ "${1:-}" = "--stop" ]; then
  stop_all
  exit 0
fi

echo "== 1/4 检查依赖 =="
if [ ! -d .venv ]; then
  echo "缺少 .venv，请先阅读 README「环境准备」" >&2
  exit 1
fi
.venv/bin/python3 -c "import fastapi, uvicorn" 2>/dev/null || \
  uv pip install --python .venv/bin/python3 fastapi 'uvicorn[standard]'
JAVA17=$(/usr/libexec/java_home -v 17 2>/dev/null || true)
if [ -z "$JAVA17" ]; then
  echo "缺少 JDK 17（brew install --cask temurin@17）" >&2
  exit 1
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "未设置 DEEPSEEK_API_KEY；生成固定走 DeepSeek，没有本地降级路径。" >&2
  echo "  本地运行：export DEEPSEEK_API_KEY=sk-... 后重试" >&2
  exit 1
fi

echo "== 2/4 构建前端（产物进 Spring Boot 静态目录）=="
if [ "${1:-}" = "--build" ] || [ ! -f apps/backend/src/main/resources/static/index.html ]; then
  (cd apps/frontend && npm install --silent && npm run build)
fi

echo "== 3/4 启动模型服务（MPS 加速，首次加载约 30-60 秒）=="
if curl -s --max-time 2 "http://127.0.0.1:${MODEL_PORT}/healthz" | grep -q ready; then
  echo "模型服务已在运行"
else
  PYTHONPATH=apps:ml/runtime nohup .venv/bin/python3 -m uvicorn inference.model_service:app \
    --host 127.0.0.1 --port "${MODEL_PORT}" > "${MODEL_LOG}" 2>&1 &
  echo $! > /tmp/csagent-model.pid
fi

echo "== 4/4 启动 Spring Boot 网关 =="
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/api/model-status"; then
  echo "网关已在运行"
else
  JAVA_HOME="$JAVA17" nohup mvn -q -f apps/backend/pom.xml spring-boot:run > "${GATEWAY_LOG}" 2>&1 &
  echo $! > /tmp/csagent-gateway.pid
fi

echo -n "等待模型加载"
for _ in $(seq 1 90); do
  status=$(curl -s --max-time 2 "http://127.0.0.1:${MODEL_PORT}/healthz" 2>/dev/null || true)
  echo "${status}" | grep -q '"ready"' && break
  echo -n "."
  sleep 2
done
echo

if echo "${status:-}" | grep -q '"ready"'; then
  echo "模型就绪"
else
  echo "模型加载超时，查看日志：tail -f ${MODEL_LOG}" >&2
  exit 1
fi

echo -n "等待网关"
gateway_ok=0
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/" 2>/dev/null || true)
  if [ "$code" = "200" ]; then gateway_ok=1; break; fi
  echo -n "."
  sleep 2
done
echo

if [ "$gateway_ok" = "1" ]; then
  echo "全部就绪：http://127.0.0.1:${GATEWAY_PORT}"
  open "http://127.0.0.1:${GATEWAY_PORT}" 2>/dev/null || true
else
  echo "网关启动超时，查看日志：tail -f ${GATEWAY_LOG}" >&2
  exit 1
fi
