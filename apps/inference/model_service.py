"""模型服务：只暴露原子推理能力，不承载业务编排。

启动：
  PYTHONPATH=apps:ml/runtime .venv/bin/python3 -m uvicorn inference.model_service:app --host 127.0.0.1 --port 8000

接口：
  GET  /healthz          健康检查
  POST /api/plan         意图规划 → capabilities / route_scores / topic / gate
  POST /api/retrieve     意图分区加权的向量召回 → {candidates: [{text, intent, score}]}
  POST /api/rerank       重排                  → {selected: [{index, text, score}], elapsed}
  POST /api/generate     流式生成（SSE）        → token / done / error 事件
  GET  /metrics          Prometheus 指标        （不暴露在 OpenAPI 中）

生成固定走 DeepSeek（architecture.md §2.1）：DEEPSEEK_API_KEY 缺失时启动失败，
不做静默降级。服务常驻内存只有多任务 BERT + bge；精排默认 NoopReranker，不加载
1.1GB 精排权重（docs/experiments.md「rerank 取舍消融」）。
"""

import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from inference.api_models import (
    ErrorResponse,
    PlanRequest,
    PlanResponse,
    Topic,
)
from inference.intent.gate import apply_gate
from inference.intent.multitask import BertIntentClassifier, build_classifier
from inference.llm.base import LlmError, LlmProvider
from inference.llm.deepseek import (
    DEFAULT_BASE_URL as DEEPSEEK_BASE_URL,
    DEFAULT_MODEL as DEEPSEEK_MODEL,
    DeepSeekProvider,
)
from inference.observability import ObservabilityMiddleware, window_metrics
from inference.paths import project_root
from inference.retrieval.in_memory import InMemoryRetriever
from inference.retrieval.reranker import NoopReranker

RUNTIME_DIR = project_root() / "ml" / "runtime"
sys.path.insert(0, str(RUNTIME_DIR))

from pipeline import PROCESSED_DIR, QUERY_INSTRUCTION, CustomerServiceAgent  # noqa: E402

state: dict = {
    "status": "loading",
    "error": None,
    "agent": None,
    "intent": None,
    "retriever": None,
    "reranker": None,
    "llm": None,
}


def load_models() -> None:
    """后台线程加载全部组件；任何异常都落到 state["error"]，healthz 能看见。"""
    try:
        agent = CustomerServiceAgent(
            PROCESSED_DIR / "qa_train.jsonl", load_reranker=False
        )
        llm = build_llm()
        intent = build_classifier()

        warmup(agent, intent)
        state["agent"] = agent
        state["intent"] = intent
        state["retriever"] = build_retriever(agent)
        state["reranker"] = NoopReranker()
        state["llm"] = llm
        state["status"] = "ready"
    except Exception as exc:  # noqa: BLE001 - 加载失败必须暴露给 healthz
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"模型加载失败：{state['error']}", file=sys.stderr, flush=True)


def build_llm() -> LlmProvider:
    """构造 DeepSeek 生成提供方（architecture.md §2.1）。

    缺 key 直接失败：没有本地降级路径，静默退回只会把错误推迟到首个请求。
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "未设置 DEEPSEEK_API_KEY，生成服务无法启动（architecture.md §2.1：不提供本地降级）"
        )
    return DeepSeekProvider(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
        model=os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL),
    )


def build_retriever(agent: CustomerServiceAgent):
    """有 DATABASE_URL 就用 pgvector，否则留在进程内暴力点积（§7）。

    两种实现的接口完全一致，切换不影响上层。没有隐藏回退：连库或读表失败会把
    异常抛给 load_models，落进 healthz 的 error，而不是悄悄退回 InMemory 让人
    以为向量库在工作。query 侧加 bge 指令前缀，与 InMemory 的召回口径一致。
    """
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        return InMemoryRetriever(agent)

    from inference.retrieval.pgvector import PgVectorRetriever

    retriever = PgVectorRetriever(
        database_url,
        lambda text: agent.embedder.encode(
            [QUERY_INSTRUCTION + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist(),
    )
    retriever.ping()
    return retriever


def warmup(agent: CustomerServiceAgent, intent: BertIntentClassifier) -> None:
    """把首次推理的 Metal kernel 编译成本挪到启动阶段。

    多任务 BERT、嵌入首次调用都要现场编译 Metal kernel，放在 ready 之前跑一遍
    代表路径，首个用户请求就不用承担这笔开销。精排默认 noop，无 kernel 可预热。
    预热失败只记日志，不阻塞启动。
    """
    try:
        t0 = time.perf_counter()
        intent.warmup()
        agent.embedder.encode(["预热"], normalize_embeddings=True, show_progress_bar=False)
        print(f"预热完成，耗时 {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001 - 预热失败不应阻塞服务启动
        print(f"预热失败（不影响启动）：{e}", file=sys.stderr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(title="customer-service model service", lifespan=lifespan)

# 纯 ASGI 中间件：trace_id 透传 + 请求指标；不拆 StreamingResponse（§11）。
app.add_middleware(ObservabilityMiddleware)


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus 抓取入口。"""
    return window_metrics()


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    """统一错误封装（architecture.md §3.5）。

    detail 传了完整的 {code, message, retriable} 就原样返回；否则一律折成
    INTERNAL，绝不再把裸异常字符串当成协议字段透传。
    """
    detail = exc.detail
    if isinstance(detail, dict) and {"code", "message", "retriable"} <= detail.keys():
        body = ErrorResponse(**detail).model_dump()
    else:
        body = ErrorResponse(
            code="INTERNAL", message=str(detail), retriable=True
        ).model_dump()
    return JSONResponse(status_code=exc.status_code, content=body)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10
    intent: str | None = None


class RerankRequest(BaseModel):
    query: str
    candidates: list[str]
    top_k: int = 2


class GenerateRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 200


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": state["status"],
        "error": state["error"],
        "llm_provider": "deepseek" if state["llm"] is not None else None,
        "retriever": getattr(state["retriever"], "backend", None),
    }


def require(name: str):
    """取一个已就绪的组件；未就绪统一 503，交给错误封装折成契约格式。"""
    if state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"模型仍在加载：{state['status']}")
    return state[name]


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post(
    "/api/plan",
    response_model=PlanResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def plan(req: PlanRequest) -> PlanResponse:
    """一次原子调用给出能力集、主题、实体与门控结果（architecture.md §3.1）。

    分类器只回答「问的是什么」，门控依赖会话历史，留在这里算。
    """
    classifier: BertIntentClassifier = require("intent")
    t0 = time.perf_counter()

    result = classifier.classify(req.query)
    capabilities, gate = apply_gate(
        result.capabilities,
        result.route_scores,
        result.entities,
        req.history,
        human_threshold=classifier.route_thresholds["human"],
    )

    return PlanResponse(
        capabilities=capabilities,
        route_scores=result.route_scores,
        topic=Topic(
            label=result.topic, confidence=round(result.topic_confidence, 4)
        ),
        entities=result.entities,
        gate=gate,
        source=result.source,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    retriever = require("retriever")
    t0 = time.perf_counter()
    hits = retriever.search(req.query, req.top_k, topic=req.intent)
    return {
        "candidates": [
            {
                "text": hit.text,
                "intent": hit.topic,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ],
        "intent": req.intent,
        "elapsed": round(time.perf_counter() - t0, 2),
    }


@app.post("/api/rerank")
def rerank(req: RerankRequest) -> dict:
    reranker = require("reranker")
    if not req.candidates:
        return {"selected": [], "scorer": reranker.scorer, "elapsed": 0.0}
    t0 = time.perf_counter()
    selected = reranker.rank(req.query, req.candidates, req.top_k)
    return {
        "selected": [
            {
                "index": item.index,
                "text": item.text,
                "score": round(item.score, 4),
            }
            for item in selected
        ],
        # noop 精排的 score 统一为 0，网关按 scorer 回填召回分，不把 0 当分数展示。
        "scorer": reranker.scorer,
        "elapsed": round(time.perf_counter() - t0, 2),
    }


@app.post("/api/generate")
def generate(req: GenerateRequest) -> StreamingResponse:
    provider = require("llm")
    return StreamingResponse(
        generate_events(provider, req.messages, req.max_new_tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def generate_events(provider: LlmProvider, messages: list[dict], max_new_tokens: int):
    t0 = time.perf_counter()
    full = []
    try:
        for piece in provider.stream(messages, max_new_tokens):
            full.append(piece)
            yield sse({"type": "token", "text": piece})
    except LlmError as exc:
        # 已吐出的内容保留：流式中断时用户至少看到部分回答（§8）。
        yield sse(
            {
                "type": "error",
                "code": exc.code,
                "message": exc.message,
                "partial": "".join(full).strip(),
            }
        )
        return
    except Exception as exc:  # noqa: BLE001 - 未分类异常也必须是合法 SSE 事件
        yield sse({"type": "error", "code": "INTERNAL", "message": f"生成失败：{exc}"})
        return
    yield sse(
        {
            "type": "done",
            "reply": "".join(full).strip(),
            "elapsed": round(time.perf_counter() - t0, 2),
        }
    )
