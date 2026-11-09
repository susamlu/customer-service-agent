"""生成提供方：固定 DeepSeek，不做本地降级（architecture.md §2.1）。

缺 DEEPSEEK_API_KEY 必须启动失败并把错误暴露到 /healthz，不能静默退回本地模型。
"""

import pytest

from inference import model_service
from inference.retrieval.reranker import NoopReranker


class FakeDeepSeek:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(model_service, "DeepSeekProvider", FakeDeepSeek)


def test_missing_key_fails_loudly():
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        model_service.build_llm()


def test_key_builds_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    provider = model_service.build_llm()
    assert isinstance(provider, FakeDeepSeek)
    assert provider.kwargs["api_key"] == "sk-x"


def test_base_url_and_model_are_configurable(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    provider = model_service.build_llm()
    assert provider.kwargs["base_url"] == "https://example.test"
    assert provider.kwargs["model"] == "deepseek-reasoner"


def test_load_models_assembles_noop_reranker(monkeypatch):
    """精排默认不加载：agent 以 load_reranker=False 构造，装配 NoopReranker。"""
    seen: dict = {}

    class FakeAgent:
        def __init__(self, _path, load_reranker=True):
            seen["load_reranker"] = load_reranker

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setattr(model_service, "CustomerServiceAgent", FakeAgent)
    monkeypatch.setattr(model_service, "build_classifier", lambda: object())
    monkeypatch.setattr(model_service, "build_retriever", lambda _agent: object())
    monkeypatch.setattr(model_service, "warmup", lambda _agent, _intent: None)
    monkeypatch.setitem(model_service.state, "status", "loading")

    model_service.load_models()

    assert model_service.state["status"] == "ready"
    assert seen["load_reranker"] is False
    assert isinstance(model_service.state["reranker"], NoopReranker)


def test_startup_without_key_marks_failed(monkeypatch):
    """缺 key 时 load_models 把错误暴露给 healthz，而不是退回本地模型。"""

    class FakeAgent:
        pass

    monkeypatch.setattr(
        model_service,
        "CustomerServiceAgent",
        lambda _path, **_kwargs: FakeAgent(),
    )
    monkeypatch.setitem(model_service.state, "status", "loading")
    monkeypatch.setitem(model_service.state, "error", None)
    model_service.load_models()
    assert model_service.state["status"] == "failed"
    assert "DEEPSEEK_API_KEY" in model_service.state["error"]
