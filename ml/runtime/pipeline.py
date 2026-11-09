"""检索运行时：语料加载、向量召回、重排。

被 `apps/inference/model_service.py` 复用：模型服务只做原子推理，业务编排在
Spring Boot 网关。生成固定走 DeepSeek（`apps/inference/llm/deepseek.py`），
这里不再加载任何本地大模型。
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from topic_labels import MIN_COUNT, OTHER

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
MODELSCOPE_DIR = MODEL_DIR / "models"

EMBED_MODEL = "BAAI--bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

TOP_K = 10
TOP_N = 2

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# MIN_COUNT / OTHER 从 topic_labels 导入，与训练、评测共用同一标签空间。
# 同意图候选的相似度加成。取值由离线扫描确定（脚本已随旧链路清理）：
# 意图分类器准确率只有 0.55，boost 超过 0.1 就会把错误分区的候选抬过头，
# recall@10 反而下降；0.05 是收益/风险平衡点（recall@10 +3.2，MRR +0.18）。
INTENT_BOOST = 0.05


def find_snapshot(name: str) -> Path:
    matches = list(MODELSCOPE_DIR.glob(f"{name}/snapshots/*"))
    if not matches:
        raise FileNotFoundError(f"找不到模型 {name}，先运行 ModelScope 下载")
    return matches[0]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CustomerServiceAgent:
    """常驻检索组件：语料 + bge 嵌入；精排模型按需加载。

    `load_reranker=False` 用于线上默认不加载 1.1GB 精排权重（docs/experiments.md
    「rerank 取舍消融」），此时整条链路取召回序；评测脚本需要精排再传 True。
    """

    def __init__(self, corpus_path: Path, load_reranker: bool = True) -> None:
        self.corpus = load(corpus_path)
        # 语料的原始 intent 是 208 类长尾，分类器只输出 41 类。检索分区必须落在
        # 同一标签空间，否则「价保申请流程」永远匹配不到细分标签的语料行。
        counts = Counter(row["intent"] for row in self.corpus if row["intent"])
        keep = {name for name, count in counts.items() if count >= MIN_COUNT}
        self.corpus_labels = np.array(
            [row["intent"] if row["intent"] in keep else OTHER for row in self.corpus],
            dtype=object,
        )
        for row, label in zip(self.corpus, self.corpus_labels):
            row["_label"] = label

        self.embedder = SentenceTransformer(str(find_snapshot(EMBED_MODEL)), device=DEVICE)
        self.corpus_emb = self.embedder.encode(
            [row["answer"] for row in self.corpus],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.reranker = (
            CrossEncoder(
                str(MODEL_DIR / "rerank-finetuned"), max_length=128, device=DEVICE
            )
            if load_reranker
            else None
        )

    def retrieve_indexed(
        self,
        query: str,
        top_k: int = TOP_K,
        intent: str | None = None,
        intent_boost: float = INTENT_BOOST,
    ) -> list[tuple[int, dict, float]]:
        """向量召回，返回 (语料下标, 行, 分数)。

        给了 intent 就对同意图分区的候选加权。用加权而不是硬过滤：意图分类器
        test 准确率只有 0.55，分错区时硬过滤会把正确候选整片排除，加权最多损失
        一点排序。

        返回下标是为了让上层（Reranker、追踪）能回指语料行；只需要行的调用方
        用 retrieve_scored。
        """
        query_emb = self.embedder.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.corpus_emb @ query_emb
        if intent:
            scores = scores + intent_boost * (self.corpus_labels == intent)
        top_indices = np.argsort(-scores)[:top_k]
        return [
            (int(i), self.corpus[i], float(scores[i])) for i in top_indices
        ]

    def retrieve_scored(
        self,
        query: str,
        top_k: int = TOP_K,
        intent: str | None = None,
        intent_boost: float = INTENT_BOOST,
    ) -> list[tuple[dict, float]]:
        return [
            (row, score)
            for _, row, score in self.retrieve_indexed(
                query, top_k, intent, intent_boost
            )
        ]

    def retrieve(
        self, query: str, top_k: int = TOP_K, intent: str | None = None
    ) -> list[dict]:
        return [row for row, _ in self.retrieve_scored(query, top_k, intent)]

    def rerank(self, query: str, candidates: list[dict], top_n: int = TOP_N) -> list[dict]:
        if self.reranker is None:
            raise RuntimeError("以 load_reranker=False 构造，未加载精排模型")
        scores = self.reranker.predict([(query, row["answer"]) for row in candidates])
        order = np.argsort(-scores)[:top_n]
        return [candidates[i] for i in order]
