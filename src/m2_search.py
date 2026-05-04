"""Module 2: Hybrid search with BM25, dense search, and RRF."""

import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words, with a regex fallback."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented if isinstance(segmented, str) and segmented.strip() else text
    except Exception:
        return text


def _tokenize(text: str) -> list[str]:
    segmented = segment_vietnamese(text).lower()
    return re.findall(r"\w+", segmented, flags=re.UNICODE)


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokenize(chunk.get("text", "")) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def _fallback_scores(self, query_tokens: list[str]) -> list[float]:
        query = set(query_tokens)
        scores = []
        for tokens in self.corpus_tokens:
            if not tokens or not query:
                scores.append(0.0)
                continue
            tf = sum(1 for token in tokens if token in query)
            coverage = len(query & set(tokens)) / len(query)
            scores.append(tf + coverage)
        return scores

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25 or a lexical fallback."""
        if not self.documents:
            return []

        query_tokens = _tokenize(query)
        if self.bm25 is not None:
            scores = list(self.bm25.get_scores(query_tokens))
        else:
            scores = self._fallback_scores(query_tokens)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i].get("text", ""),
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in ranked
            if scores[i] > 0 or top_k > 0
        ]


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._memory_chunks: list[dict] = []
        self._memory_vectors: list[list[float]] = []
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
        except Exception:
            self.client = None

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("USE_NEURAL_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
                try:
                    from sentence_transformers import SentenceTransformer

                    self._encoder = SentenceTransformer(EMBEDDING_MODEL)
                except Exception:
                    self._encoder = False
            else:
                self._encoder = False
        return self._encoder

    def _hash_vector(self, text: str, dim: int = 256) -> list[float]:
        vector = [0.0] * dim
        for token in _tokenize(text):
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _encode_many(self, texts: list[str]):
        encoder = self._get_encoder()
        if encoder:
            return encoder.encode(texts, show_progress_bar=False)
        return [self._hash_vector(text) for text in texts]

    def _encode_one(self, text: str):
        encoder = self._get_encoder()
        if encoder:
            return encoder.encode(text)
        return self._hash_vector(text)

    def _index_memory(self, chunks: list[dict]) -> None:
        self._memory_chunks = chunks
        self._memory_vectors = [list(v) for v in self._encode_many([c.get("text", "") for c in chunks])]

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant, falling back to in-memory vectors."""
        if not chunks:
            self._index_memory([])
            return

        vectors = self._encode_many([c.get("text", "") for c in chunks])
        vectors = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

        if self.client is None:
            self._memory_chunks = chunks
            self._memory_vectors = vectors
            return

        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            size = len(vectors[0]) if vectors else EMBEDDING_DIM
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i].get("text", "")},
                )
                for i, vector in enumerate(vectors)
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception:
            self._memory_chunks = chunks
            self._memory_vectors = vectors

    def _search_memory(self, query: str, top_k: int) -> list[SearchResult]:
        if not self._memory_chunks:
            return []
        q = self._encode_one(query)
        q = q.tolist() if hasattr(q, "tolist") else list(q)
        q_norm = math.sqrt(sum(v * v for v in q)) or 1.0

        scored = []
        for i, vector in enumerate(self._memory_vectors):
            v_norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            score = sum(a * b for a, b in zip(q, vector)) / (q_norm * v_norm)
            scored.append((score, i))

        scored.sort(reverse=True)
        return [
            SearchResult(
                text=self._memory_chunks[i].get("text", ""),
                score=float(score),
                metadata=self._memory_chunks[i].get("metadata", {}),
                method="dense",
            )
            for score, i in scored[:top_k]
        ]

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Search using Qdrant or the in-memory fallback."""
        query_vector = self._encode_one(query)
        query_vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        if self.client is not None:
            try:
                if hasattr(self.client, "query_points"):
                    response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
                    hits = response.points
                else:
                    hits = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=hit.payload.get("text", ""),
                        score=float(hit.score),
                        metadata=hit.payload or {},
                        method="dense",
                    )
                    for hit in hits
                ]
            except Exception:
                pass

        return self._search_memory(query, top_k)


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked lists using reciprocal rank fusion."""
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            item = fused.setdefault(result.text, {"score": 0.0, "result": result})
            item["score"] += 1.0 / (k + rank)
            if result.score > item["result"].score:
                item["result"] = result

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 and dense search with RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    sample = "Nhan vien duoc nghi phep nam"
    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
