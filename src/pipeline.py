"""Production RAG pipeline: M1 + M2 + M3 + M4 + optional M5."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    print("\n[1/4] Chunking documents...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        _, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")

    print("\n[2/4] Enriching chunks (M5)...")
    enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    if enriched:
        all_chunks = [{"text": item.enriched_text, "metadata": item.auto_metadata} for item in enriched]
        print(f"  Enriched {len(enriched)} chunks")
    else:
        print("  No chunks to enrich; using raw chunks.")

    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)

    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run one query through retrieval and reranking."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    answer = contexts[0] if contexts else "Khong tim thay thong tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on the test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i + 1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running evaluation...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        score = results.get(metric, 0)
        status = "PASS" if score >= 0.75 else "LOW"
        print(f"  [{status}] {metric}: {score:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
