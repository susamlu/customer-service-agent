# 模型服务容器：CPU 推理（容器内无 MPS）。
# 模型权重与语料通过 volume 挂载，不打进镜像。
# 构建上下文为项目根（见同目录 docker-compose.yml）。
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/ml/runtime

COPY apps/inference/requirements.txt .
RUN pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 容器内拍平：inference/ 在 /app/inference，runtime 代码在 /app/ml/runtime。
# 路径统一走 inference.paths.project_root()（apps/inference/paths.py），它按
# 「同含 ml/models 与 ml/experiments」向上定位，两种布局都成立。
# ml/models 由 volume 挂载；能力映射表必须打进镜像（README 的 CI 也校验同一张表）。
COPY apps/inference/ inference/
COPY ml/runtime/ ml/runtime/
COPY ml/experiments/intent_capability_map.yaml ml/experiments/intent_capability_map.yaml

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "inference.model_service:app", "--host", "0.0.0.0", "--port", "8000"]
