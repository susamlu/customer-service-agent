"""端到端黑盒测试：真实 Spring Boot 网关 + 模型服务（architecture.md §10.2）。

不 mock 任何一层：HTTP POST /api/chat 拿 SSE，断言决策链事件是否符合线上主流程。
覆盖四条路径：订单 / 知识 / 转人工 / 订单+知识并行。

网关或模型服务不可达时默认 skip（本地没起栈时不误报）；CI 设 E2E_REQUIRED=1
把 skip 变成失败，否则门禁形同虚设。

运行：apps/e2e/run.sh，或
  PYTHONPATH=apps:ml/runtime .venv/bin/python3 -m pytest apps/e2e -q
"""

import json
import os
import time
import urllib.request

import pytest

GATEWAY = os.environ.get("E2E_GATEWAY", "http://127.0.0.1:8090").rstrip("/")
READY_TIMEOUT = float(os.environ.get("E2E_READY_TIMEOUT", "180"))
CHAT_TIMEOUT = float(os.environ.get("E2E_CHAT_TIMEOUT", "180"))
REQUIRED = os.environ.get("E2E_REQUIRED", "").strip().lower() in {"1", "true", "yes"}

USER = "u_chen"
ORDER_ID = "SO2026102901"


def _skip_or_fail(message: str) -> None:
    """本地缺栈时跳过；CI（E2E_REQUIRED=1）必须失败，不静默跳过。"""
    if REQUIRED:
        pytest.fail(message)
    pytest.skip(message)


def _stack_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{GATEWAY}/api/model-status", timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - 任何异常都视为未就绪
        return False
    return isinstance(body, dict) and body.get("status") == "ready"


@pytest.fixture(scope="session", autouse=True)
def live_stack():
    """等模型服务 ready（加载 BERT + bge + reranker 需要时间）。"""
    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        if _stack_ready():
            return
        time.sleep(2)
    _skip_or_fail(
        f"{GATEWAY} 未就绪（{READY_TIMEOUT:.0f}s 内 /api/model-status 未返回 ready）；"
        "先起栈：./apps/deploy/run.sh"
    )


def chat(query: str, session_id: str) -> list[dict]:
    """发一条消息，读完 SSE，返回全部 data 事件（按出现顺序）。"""
    request = urllib.request.Request(
        f"{GATEWAY}/api/chat?sessionId={session_id}",
        data=json.dumps({"query": query, "userId": USER}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=CHAT_TIMEOUT) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def by_type(events: list[dict], event_type: str) -> list[dict]:
    return [event for event in events if event.get("type") == event_type]


def reply_text(events: list[dict]) -> str:
    """生成链路的正文可能来自 token 流或 done.reply；固定话术走 reply.text。"""
    reply = by_type(events, "reply")
    if reply:
        return reply[0].get("text", "")
    tokens = "".join(event.get("text", "") for event in by_type(events, "token"))
    done = by_type(events, "done")
    return tokens or (done[0].get("reply", "") if done else "")


def assert_clean_end(events: list[dict]) -> None:
    """每条请求都必须以 done 收口，且没有 error 事件。"""
    assert not by_type(events, "error"), f"出现 error 事件：{by_type(events, 'error')}"
    assert by_type(events, "done"), "SSE 未以 done 结束"


def test_order_journey():
    """订单主流程：plan 判定 order，产出 order 能力结果并最终生成回答。"""
    events = chat(f"帮我查一下订单 {ORDER_ID} 的物流状态", "e2e-order")

    assert_clean_end(events)
    plan = by_type(events, "plan")[0]
    assert plan["route"] == "order", plan
    assert "order" in plan["capabilities"], plan
    assert by_type(events, "order"), "缺少 order 能力事件"
    assert reply_text(events).strip(), "回答为空"


def test_knowledge_journey():
    """知识主流程：plan 判定 knowledge，召回 → 重排 → 上下文 → 生成。"""
    events = chat("退款政策是怎么规定的？", "e2e-knowledge")

    assert_clean_end(events)
    plan = by_type(events, "plan")[0]
    assert plan["route"] == "knowledge", plan
    assert "knowledge" in plan["capabilities"], plan
    assert by_type(events, "retrieve"), "缺少 retrieve 事件"
    context = by_type(events, "context")[0]
    assert "knowledge" in context["sources"], context
    assert reply_text(events).strip(), "回答为空"


def test_human_journey():
    """转人工：规则短路，固定话术直接收口，不走检索与生成。"""
    events = chat("我要转人工", "e2e-human")

    assert_clean_end(events)
    plan = by_type(events, "plan")[0]
    assert plan["route"] == "human", plan
    assert "人工" in reply_text(events), "转人工话术缺失"


def test_parallel_journey():
    """并行场景：一句话同时要订单与政策，两个能力都要落地。"""
    events = chat(
        f"我的订单 {ORDER_ID} 现在到哪了，另外退款政策怎么说的？", "e2e-parallel"
    )

    assert_clean_end(events)
    plan = by_type(events, "plan")[0]
    assert {"order", "knowledge"} <= set(plan["capabilities"]), plan
    assert by_type(events, "order"), "并行场景缺少 order 能力事件"
    context = by_type(events, "context")[0]
    assert {"order", "knowledge"} <= set(context["sources"]), context
    assert reply_text(events).strip(), "回答为空"
