"""Module 4: RAGAS-style evaluation and failure analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap_score(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _heuristic_evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question: list[EvalResult] = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = " ".join(ctxs)
        faithfulness = _overlap_score(answer, context_text)
        answer_relevancy = _overlap_score(question, answer)
        context_precision = sum(_overlap_score(question, ctx) for ctx in ctxs) / max(len(ctxs), 1)
        context_recall = _overlap_score(ground_truth, context_text)
        per_question.append(EvalResult(
            question=question,
            answer=answer,
            contexts=ctxs,
            ground_truth=ground_truth,
            faithfulness=float(faithfulness),
            answer_relevancy=float(answer_relevancy),
            context_precision=float(context_precision),
            context_recall=float(context_recall),
        ))

    def avg(metric: str) -> float:
        if not per_question:
            return 0.0
        return sum(getattr(item, metric) for item in per_question) / len(per_question)

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS when configured; otherwise use deterministic local metrics."""
    use_real_ragas = bool(OPENAI_API_KEY) and os.getenv("USE_REAL_RAGAS", "").lower() in {"1", "true", "yes"}
    if use_real_ragas:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            df = result.to_pandas()
            per_question = [
                EvalResult(
                    question=row["question"],
                    answer=row["answer"],
                    contexts=row["contexts"],
                    ground_truth=row["ground_truth"],
                    faithfulness=float(row.get("faithfulness", 0.0)),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                    context_precision=float(row.get("context_precision", 0.0)),
                    context_recall=float(row.get("context_recall", 0.0)),
                )
                for _, row in df.iterrows()
            ]
            return {
                "faithfulness": float(df.get("faithfulness", [0]).mean()),
                "answer_relevancy": float(df.get("answer_relevancy", [0]).mean()),
                "context_precision": float(df.get("context_precision", [0]).mean()),
                "context_recall": float(df.get("context_recall", [0]).mean()),
                "per_question": per_question,
            }
        except Exception:
            pass

    return _heuristic_evaluate(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using a diagnostic mapping."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def average(result: EvalResult) -> float:
        return sum(getattr(result, metric) for metric in metric_names) / len(metric_names)

    failures = []
    for result in sorted(eval_results, key=average)[:bottom_n]:
        scores = {metric: float(getattr(result, metric)) for metric in metric_names}
        worst_metric = min(scores, key=scores.get)
        score = scores[worst_metric]

        if worst_metric == "faithfulness":
            diagnosis = "LLM hallucinating"
            suggested_fix = "Tighten prompt and force answers to cite retrieved context."
        elif worst_metric == "context_recall":
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking, add BM25 recall, or index enriched queries."
        elif worst_metric == "context_precision":
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking, metadata filters, or reduce top_k."
        else:
            diagnosis = "Answer does not match question"
            suggested_fix = "Improve prompt template and answer extraction."

        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": score,
            "avg_score": average(result),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    per_question = []
    for item in results.get("per_question", []):
        per_question.append(asdict(item) if hasattr(item, "__dataclass_fields__") else item)

    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(per_question),
        "per_question": per_question,
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
