"""端到端生成评测（architecture.md §10.4）。

同一条 query 走完整链路——意图 → 检索 → rerank → 上下文拼装 → DeepSeek 生成——
再判回答质量。前三段各有独立评测（eval_router / eval_retrieval / eval_rerank），
这里只测「拼起来之后用户看到的那句话」。

三个口径锚点必须在运行时对齐线上，否则数字不可信：

  1. 系统 prompt 直接读网关的资源文件，不复制。
  2. 用户消息格式复刻 ChatOrchestrator.chatMessages()。
  3. 知识块形状复刻 ContextAssembler（前 2 条、各限 90 字、`[知识] ` 前缀）。

判分器走独立 httpx 调用（temperature=0）：DeepSeekProvider 不暴露 temperature，
拿它做评审模型无法保证确定性。

门槛（预登记于 §10.4）：faithfulness 均值 ≥ 4.0、三项均值 ≥ 3.5、error_rate ≤ 2%。
cosine 与延迟只报告，首次基线落地后再设回归线。
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ML_DIR = HERE.parent
REPO_DIR = ML_DIR.parent
sys.path.insert(0, str(REPO_DIR / "apps"))
sys.path.insert(0, str(ML_DIR / "runtime"))

from inference.intent.multitask import build_classifier  # noqa: E402
from inference.llm.deepseek import DeepSeekProvider  # noqa: E402
from pipeline import TOP_K, CustomerServiceAgent  # noqa: E402

PROCESSED_DIR = ML_DIR / "data" / "processed"
CORPUS_PATH = PROCESSED_DIR / "qa_train.jsonl"
SYSTEM_PROMPT_PATH = (
    REPO_DIR / "apps" / "backend" / "src" / "main" / "resources" / "prompts" / "chat-system.txt"
)

# 与 ContextAssembler.java 同名同值；Java 侧无法 import 这里，改一处必须改另一处。
KNOWLEDGE_MAX_ITEMS = 2
KNOWLEDGE_MAX_CHARS = 90
EMPTY_CONTEXT = "（本轮没有可用的订单或知识库上下文）"
MAX_NEW_TOKENS = 200

FAITHFULNESS_FLOOR = 4.0
OVERALL_FLOOR = 3.5
ERROR_RATE_CEILING = 0.02

JUDGE_PROMPT = """你是电商客服回答质量评审。根据下面四项内容打分，输出严格 JSON。

【用户问题】
{question}

【参考回答】（真实客服的历史答复，只作参考，不是唯一正确答案）
{gold}

【上下文】（生成时模型能看到的资料）
{context}

【待评回答】
{reply}

从三个维度各打 1-5 分（整数）：
- faithfulness：待评回答是否只使用了上下文中出现的事实，有没有编造数字、赔偿金额、时效、承诺。
  上下文不足以回答时，明确说明「暂无相关说明」并给出谨慎建议，视为忠实（4-5 分）。
- relevance：是否直接回答了用户问题，没有跑题、没有答非所问。
- style：是否客服口吻，先共情再给信息，简洁（120 字左右），不罗列资料原文。

只输出 JSON，不要其他文字：
{{"faithfulness": 4, "relevance": 4, "style": 4, "reason": "一句话理由"}}"""


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sample_rows(rows: list[dict], limit: int, seed: int, min_answer_chars: int) -> list[dict]:
    """过滤空问题/过短答案后按 seed 采样，保证可复现。limit<=0 取全部。"""
    pool = [
        row
        for row in rows
        if (row.get("question") or "").strip()
        and len((row.get("answer") or "").strip()) >= min_answer_chars
    ]
    if limit <= 0 or limit >= len(pool):
        return pool
    return random.Random(seed).sample(pool, limit)


def limit_text(text: str, max_chars: int) -> str:
    """对齐 Java `ContextAssembler.limit`：超长截断并补 `…`。"""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def assemble_knowledge(answers: list[str]) -> str:
    """把重排后的答案拼成 `[知识]` 块：strip 后去重、最多 2 条、各限 90 字。"""
    blocks: list[str] = []
    seen: set[str] = set()
    for raw in answers:
        snippet = (raw or "").strip()
        if not snippet or snippet in seen:
            continue
        seen.add(snippet)
        blocks.append("[知识] " + limit_text(snippet, KNOWLEDGE_MAX_CHARS))
        if len(blocks) >= KNOWLEDGE_MAX_ITEMS:
            break
    return "\n".join(blocks)


def build_messages(system_prompt: str, context: str, question: str, today: str) -> list[dict]:
    """复刻 ChatOrchestrator.chatMessages()。"""
    evidence = context if context.strip() else EMPTY_CONTEXT
    system = system_prompt.replace("{today}", today)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": evidence + "\n\n用户问题：" + question},
    ]


def parse_judge_json(text: str) -> dict | None:
    """从评审输出里抠 JSON；分数必须是 1-5 的整数，否则视为判分失败。"""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    scores = {}
    for key in ("faithfulness", "relevance", "style"):
        value = data.get(key)
        if not isinstance(value, int) or not 1 <= value <= 5:
            return None
        scores[key] = value
    scores["reason"] = str(data.get("reason", ""))[:200]
    return scores


def percentile(values: list[float], pct: float) -> float:
    """nearest-rank P95，避免依赖 numpy 版本细节。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(pct / 100 * len(ordered))) - 1))
    return ordered[index]


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if not r.get("error")]
    total = len(records) or 1
    judged = [r for r in ok if r.get("judge")]
    cosines = [r["cosine"] for r in ok if r.get("cosine") is not None]
    latencies = [r["total_ms"] for r in ok]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    scores = {
        key: mean([r["judge"][key] for r in judged])
        for key in ("faithfulness", "relevance", "style")
    }
    scores["overall"] = mean(list(scores.values())) if judged else 0.0
    return {
        "n": len(records),
        "ok": len(ok),
        "judged": len(judged),
        "judge_errors": sum(1 for r in ok if not r.get("judge")),
        "error_rate": (len(records) - len(ok)) / total,
        "scores": scores,
        "cosine_mean": mean(cosines),
        "retrieval_ms_avg": mean([r["retrieval_ms"] for r in ok]),
        "first_token_ms_avg": mean([r["first_token_ms"] for r in ok]),
        "total_ms_avg": mean(latencies),
        "total_ms_p95": percentile(latencies, 95),
    }


def verdict(summary: dict) -> bool:
    """预登记硬门槛；cosine 与延迟不参与。

    `--judge off` 时没有分数，只剩链路冒烟：错了不算质量不过关，也不能当质量过关。
    """
    if summary["error_rate"] > ERROR_RATE_CEILING:
        return False
    if summary["judged"] == 0:
        return True
    return (
        summary["scores"]["faithfulness"] >= FAITHFULNESS_FLOOR
        and summary["scores"]["overall"] >= OVERALL_FLOOR
    )


def classify_context(agent, classifier, row: dict) -> dict:
    """pipeline 口径：意图 → 检索 → rerank → 知识块。检索串行执行。"""
    start = time.perf_counter()
    topic = classifier.classify(row["question"]).topic
    candidates = agent.retrieve_indexed(row["question"], top_k=TOP_K, intent=topic)
    reranked = agent.rerank(row["question"], [item for _, item, _ in candidates])
    context = assemble_knowledge([item.get("answer", "") for item in reranked])
    return {
        "topic": topic,
        "context": context,
        "retrieval_ms": (time.perf_counter() - start) * 1000,
    }


def oracle_context(row: dict) -> dict:
    """oracle 口径：gold answer 直接当上下文，不截断，测生成上限。"""
    return {"topic": None, "context": row["answer"].strip(), "retrieval_ms": 0.0}


def generate(provider, messages: list[dict]) -> tuple[str, float, float]:
    """返回 (reply, first_token_ms, total_ms)；空输出按失败抛错由调用方记。"""
    start = time.perf_counter()
    first_token_ms = None
    pieces: list[str] = []
    for piece in provider.stream(messages, MAX_NEW_TOKENS):
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - start) * 1000
        pieces.append(piece)
    total_ms = (time.perf_counter() - start) * 1000
    reply = "".join(pieces).strip()
    if not reply:
        raise RuntimeError("生成结果为空")
    return reply, first_token_ms or total_ms, total_ms


def judge(client: httpx.Client, base_url: str, model: str, api_key: str, payload: dict) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(**payload)}],
        "temperature": 0,
        "max_tokens": 300,
        "stream": False,
    }
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    parsed = parse_judge_json(text)
    if parsed is None:
        raise ValueError(f"评审输出无法解析：{text[:120]}")
    return parsed


def cosine(embedder, left: str, right: str) -> float | None:
    if not left.strip() or not right.strip():
        return None
    vectors = embedder.encode([left, right], normalize_embeddings=True, show_progress_bar=False)
    return float(vectors[0] @ vectors[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROCESSED_DIR / "qa_test.jsonl"))
    parser.add_argument("--limit", type=int, default=100, help="<=0 表示全量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-answer-chars", type=int, default=10)
    parser.add_argument("--context", choices=("pipeline", "oracle"), default="pipeline")
    parser.add_argument("--judge", choices=("auto", "off"), default="auto")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--today", default=time.strftime("%Y-%m-%d"))
    parser.add_argument("--out", default="/tmp/eval_generation.jsonl")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("未设置 DEEPSEEK_API_KEY，无法评测生成")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    rows = sample_rows(load_rows(Path(args.data)), args.limit, args.seed, args.min_answer_chars)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    print(f"样本 {len(rows)} 条，口径 {args.context}，判分 {args.judge}", flush=True)

    agent = CustomerServiceAgent(CORPUS_PATH)
    classifier = build_classifier(device=args.device) if args.context == "pipeline" else None

    print("构建上下文…", flush=True)
    inputs = []
    for row in rows:
        built = (
            classify_context(agent, classifier, row)
            if args.context == "pipeline"
            else oracle_context(row)
        )
        payload = {
            "question": row["question"].strip(),
            "gold": (row.get("answer") or "").strip(),
            "context": built["context"],
        }
        inputs.append(
            {
                "row": row,
                "topic": built["topic"],
                "context": built["context"],
                "retrieval_ms": built["retrieval_ms"],
                "messages": build_messages(system_prompt, built["context"], payload["question"], args.today),
                "payload": payload,
            }
        )

    provider = DeepSeekProvider(api_key=api_key, base_url=base_url, model=model)
    judge_client = httpx.Client(
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
    )
    want_judge = args.judge == "auto"

    def run(item: dict) -> dict:
        record = {
            "question": item["payload"]["question"],
            "gold": item["payload"]["gold"],
            "topic": item["topic"],
            "context": item["context"],
            "retrieval_ms": item["retrieval_ms"],
            "reply": "",
            "judge": None,
            "cosine": None,
            "first_token_ms": 0.0,
            "total_ms": 0.0,
            "error": None,
        }
        try:
            reply, first_token_ms, total_ms = generate(provider, item["messages"])
            record.update(reply=reply, first_token_ms=first_token_ms, total_ms=total_ms)
            record["cosine"] = cosine(agent.embedder, reply, record["gold"])
        except Exception as exc:  # 单条失败不中断整轮
            record["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return record
        if want_judge:
            try:
                payload = dict(item["payload"], reply=record["reply"])
                record["judge"] = judge(judge_client, base_url, model, api_key, payload)
            except Exception as exc:
                record["judge"] = None
                record["judge_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return record

    print("生成与判分…", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        records = list(pool.map(run, inputs))

    summary = summarize(records)
    output = Path(args.out)
    output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    passed = verdict(summary)
    if summary["judged"] == 0:
        print(f"\n冒烟口径（未判分）：error_rate≤{ERROR_RATE_CEILING:.0%}  -> {'PASS' if passed else 'FAIL'}")
    else:
        print(
            f"\n门槛 faithfulness≥{FAITHFULNESS_FLOOR} / 均分≥{OVERALL_FLOOR} / "
            f"error_rate≤{ERROR_RATE_CEILING:.0%}  -> {'PASS' if passed else 'FAIL'}"
        )
    print(f"明细：{output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
