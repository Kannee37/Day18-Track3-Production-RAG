"""Module 3: Reranking and latency benchmarking."""

import math
import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _lexical_score(query: str, document: str) -> float:
    query_tokens = _tokens(query)
    doc_tokens = _tokens(document)
    if not query_tokens or not doc_tokens:
        return 0.0

    qset = set(query_tokens)
    dset = set(doc_tokens)
    overlap = len(qset & dset)
    coverage = overlap / len(qset)
    density = sum(1 for token in doc_tokens if token in qset) / len(doc_tokens)
    phrase_bonus = 0.2 if any(token in document.lower() for token in qset) else 0.0
    return coverage * 0.75 + density * 0.25 + phrase_bonus


class _LexicalReranker:
    def compute_score(self, pairs):
        return [_lexical_score(query, doc) for query, doc in pairs]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load a neural reranker only when explicitly enabled; otherwise use lexical scoring."""
        if self._model is not None:
            return self._model

        if os.getenv("USE_NEURAL_RERANKER", "").lower() in {"1", "true", "yes"}:
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_name, use_fp16=True)
                return self._model
            except Exception:
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
                    return self._model
                except Exception:
                    pass

        self._model = _LexicalReranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents and return the top_k highest scoring items."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]
        if hasattr(model, "compute_score"):
            scores = model.compute_score(pairs)
        else:
            scores = model.predict(pairs)

        if isinstance(scores, float):
            scores = [scores]
        ranked = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)[:top_k]
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(ranked)
        ]


class FlashrankReranker:
    """Lightweight optional reranker with lexical fallback."""

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [{"text": doc.get("text", ""), "metadata": doc.get("metadata", {})} for doc in documents]
            response = self._model.rerank(RerankRequest(query=query, passages=passages))
            ranked = response[:top_k]
            return [
                RerankResult(
                    text=item.get("text", ""),
                    original_score=0.0,
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=item.get("metadata", {}),
                    rank=i + 1,
                )
                for i, item in enumerate(ranked)
            ]
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark reranker latency in milliseconds."""
    times = []
    runs = max(1, n_runs)
    for _ in range(runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    return {"avg_ms": avg, "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for result in reranker.rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
