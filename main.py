"""Main entry point for Lab 18."""

import json
import os
import time


def main():
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)

    print("\nSTEP 1: Running Basic RAG Baseline")
    print("-" * 40)
    from naive_baseline import main as run_baseline

    run_baseline()

    print("\nSTEP 2: Running Production Pipeline")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline

    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)

    for filename in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(filename):
            os.replace(filename, os.path.join("reports", filename))

    print("\nSTEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Delta':>8}")
        print("-" * 58)
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            basic = naive.get("aggregate", {}).get(metric, 0)
            production = prod.get("aggregate", {}).get(metric, 0)
            delta = production - basic
            status = "PASS" if production >= 0.75 else "LOW"
            print(f"{status:<5} {metric:<19} {basic:>8.4f} {production:>12.4f} {delta:>+8.4f}")
    else:
        print("Reports were not generated.")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("\nNext steps:")
    print("  1. Fill analysis/failure_analysis.md")
    print("  2. Fill analysis/group_report.md")
    print("  3. Write analysis/reflections/reflection_[Name].md")
    print("  4. Run: python check_lab.py")


if __name__ == "__main__":
    main()
