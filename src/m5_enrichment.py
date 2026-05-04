"""Module 5: Chunk enrichment pipeline."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """A chunk after enrichment."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if s.strip()]


def _use_openai() -> bool:
    return bool(OPENAI_API_KEY) and os.getenv("USE_OPENAI_ENRICHMENT", "").lower() in {"1", "true", "yes"}


def _chat(system: str, user: str, max_tokens: int = 150) -> str:
    if not _use_openai():
        return ""
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_ENRICHMENT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


def summarize_chunk(text: str) -> str:
    """Create a short summary for a chunk."""
    response = _chat(
        "Summarize the passage in 2 concise Vietnamese sentences.",
        text,
        max_tokens=150,
    )
    if response:
        return response

    sentences = _sentences(text)
    return " ".join(sentences[:2]).strip()


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate likely questions that this chunk can answer."""
    response = _chat(
        f"Generate {n_questions} Vietnamese questions answered by this passage. One question per line.",
        text,
        max_tokens=200,
    )
    if response:
        questions = [line.strip().lstrip("0123456789.-) ") for line in response.splitlines()]
        return [q for q in questions if q][:n_questions]

    summary = summarize_chunk(text)
    keywords = re.findall(r"\w+", summary, flags=re.UNICODE)[:8]
    topic = " ".join(keywords[:5]) if keywords else "noi dung nay"
    templates = [
        f"{topic} noi ve dieu gi?",
        f"Thong tin chinh cua {topic} la gi?",
        f"Can luu y gi ve {topic}?",
    ]
    return templates[:n_questions]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a compact context sentence to the original chunk."""
    response = _chat(
        "Write one short Vietnamese sentence describing where this chunk sits in the document.",
        f"Document: {document_title}\n\nChunk:\n{text}",
        max_tokens=80,
    )
    if response:
        return f"{response}\n\n{text}"

    title = document_title or "tai lieu"
    first_line = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), "")
    context = f"Doan nay trich tu {title}"
    if first_line:
        context += f", noi ve {first_line[:80]}"
    return f"{context}.\n\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract lightweight metadata fields from a chunk."""
    response = _chat(
        'Extract JSON metadata: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|general", "language": "vi|en"}.',
        text,
        max_tokens=150,
    )
    if response:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    if any(word in lowered for word in ["vpn", "mat khau", "password", "wireguard"]):
        category = "it"
    elif any(word in lowered for word in ["nghi", "phep", "nhan vien", "thu viec"]):
        category = "hr"
    elif any(word in lowered for word in ["hoa don", "chi phi", "finance", "ngan sach"]):
        category = "finance"
    elif any(word in lowered for word in ["policy", "chinh sach", "quy dinh"]):
        category = "policy"
    else:
        category = "general"

    words = re.findall(r"\w+", text, flags=re.UNICODE)
    topic = " ".join(words[:6]) if words else ""
    entities = sorted({w.strip(".,:;()") for w in re.findall(r"\b[A-Z][\w-]+\b", text)})[:5]
    language = "vi" if re.search(r"[ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", lowered) else "en"
    return {"topic": topic, "entities": entities, "category": category, "language": language}


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Run selected enrichment methods over chunks."""
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    method_set = set(methods)
    use_all = "full" in method_set
    enriched: list[EnrichedChunk] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        base_metadata = chunk.get("metadata", {})
        summary = summarize_chunk(text) if use_all or "summary" in method_set else ""
        questions = generate_hypothesis_questions(text) if use_all or "hyqa" in method_set else []
        enriched_text = (
            contextual_prepend(text, base_metadata.get("source", ""))
            if use_all or "contextual" in method_set
            else text
        )
        auto_meta = extract_metadata(text) if use_all or "metadata" in method_set else {}

        if questions:
            enriched_text = f"{enriched_text}\n\nLikely questions:\n" + "\n".join(f"- {q}" for q in questions)
        if summary:
            enriched_text = f"Summary: {summary}\n\n{enriched_text}"

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**base_metadata, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam."
    print(summarize_chunk(sample))
    print(generate_hypothesis_questions(sample))
    print(contextual_prepend(sample, "So tay nhan vien"))
    print(extract_metadata(sample))
