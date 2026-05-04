"""Module 1: Advanced chunking strategies."""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown/text files and PDF files from data/."""
    docs = []
    patterns = ["*.md", "*.txt"]
    for pattern in patterns:
        for fp in sorted(glob.glob(os.path.join(data_dir, pattern))):
            with open(fp, encoding="utf-8") as f:
                docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        try:
            from pypdf import PdfReader

            reader = PdfReader(fp)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if text:
                docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        except Exception:
            continue
    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Baseline paragraph chunking."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def _sentence_split(text: str) -> list[str]:
    pattern = r"(?<=[.!?。！？])\s+|\n\s*\n"
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _lexical_similarity(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group adjacent sentences that appear to discuss the same topic."""
    metadata = metadata or {}
    sentences = _sentence_split(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current = [sentences[0]]
    split_threshold = max(0.05, threshold * 0.35)

    for sentence in sentences[1:]:
        sim = _lexical_similarity(current[-1], sentence)
        starts_new_section = sentence.lstrip().startswith("#")
        current_is_large = sum(len(s) for s in current) >= 900
        if (len(current) >= 2 and sim < split_threshold) or starts_new_section or current_is_large:
            chunks.append(Chunk(
                text=" ".join(current).strip(),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
            current = []
        current.append(sentence)

    if current:
        chunks.append(Chunk(
            text=" ".join(current).strip(),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))
    return chunks


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks and smaller child chunks linked by parent_id."""
    metadata = metadata or {}
    units = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not units and text.strip():
        units = [text.strip()]

    parents: list[Chunk] = []
    current: list[str] = []
    for unit in units:
        proposed = "\n\n".join(current + [unit])
        if current and len(proposed) > parent_size:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text="\n\n".join(current).strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
            ))
            current = [unit]
        else:
            current.append(unit)

    if current:
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text="\n\n".join(current).strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
        ))

    children: list[Chunk] = []
    overlap = max(20, child_size // 5)
    step = max(1, child_size - overlap)
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parent_text = parent.text.strip()
        if len(parent_text) <= child_size:
            child_texts = [parent_text]
        else:
            child_texts = []
            start = 0
            while start < len(parent_text):
                end = min(len(parent_text), start + child_size)
                cut = parent_text.rfind(" ", start, end)
                if end < len(parent_text) and cut > start + child_size * 0.6:
                    end = cut
                child_text = parent_text[start:end].strip()
                if child_text:
                    child_texts.append(child_text)
                if end >= len(parent_text):
                    break
                start += step

        for child_text in child_texts:
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "chunk_index": len(children), "parent_id": pid},
                parent_id=pid,
            ))

    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Split markdown by headers while preserving the header in each chunk."""
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    current_header = ""
    current_content: list[str] = []

    def flush() -> None:
        content = "".join(current_content).strip()
        if not current_header and not content:
            return
        section = current_header.lstrip("#").strip() if current_header else "Untitled"
        chunk_text = f"{current_header}\n\n{content}".strip()
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                **metadata,
                "section": section,
                "header": current_header,
                "strategy": "structure",
                "chunk_index": len(chunks),
            },
        ))

    for part in parts:
        if not part:
            continue
        if re.match(r"^#{1,3}\s+", part):
            flush()
            current_header = part.strip()
            current_content = []
        else:
            current_content.append(part)
    flush()
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Run all chunkers and return basic length statistics."""
    collected = {"basic": [], "semantic": [], "hierarchical": [], "structure": []}
    parent_count = 0
    child_count = 0

    for doc in documents:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        collected["basic"].extend(chunk_basic(text, metadata=meta))
        collected["semantic"].extend(chunk_semantic(text, metadata=meta))
        parents, children = chunk_hierarchical(text, metadata=meta)
        parent_count += len(parents)
        child_count += len(children)
        collected["hierarchical"].extend(children)
        collected["structure"].extend(chunk_structure_aware(text, metadata=meta))

    def stats(chunks: list[Chunk]) -> dict:
        lengths = [len(c.text) for c in chunks]
        if not lengths:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        return {
            "num_chunks": len(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    results = {name: stats(chunks) for name, chunks in collected.items()}
    results["hierarchical"]["parents"] = parent_count
    results["hierarchical"]["children"] = child_count

    print(f"{'Strategy':<14} | {'Chunks':>8} | {'Avg Len':>8} | {'Min':>5} | {'Max':>5}")
    print("-" * 52)
    for name, item in results.items():
        count = f"{item['parents']}p/{item['children']}c" if name == "hierarchical" else str(item["num_chunks"])
        print(f"{name:<14} | {count:>8} | {item['avg_length']:>8.1f} | {item['min_length']:>5} | {item['max_length']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    compare_strategies(docs)
