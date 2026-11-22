"""
corpus_statistics.py
====================
Compute and export statistics for the domain corpus.

Metrics collected per document and in aggregate:
- Page count (from PAGE markers in extracted text)
- Character count
- Word count
- Estimated token count (using a lightweight tokenizer)
- Sentence count (heuristic)

Output: JSON (full stats) and CSV (per-document summary).
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_PAGE_MARKER_RE = re.compile(r"=== PAGE \d+ ===")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _count_pages(text: str) -> int:
    """Count PAGE markers in extracted text."""
    return len(_PAGE_MARKER_RE.findall(text))


def _count_sentences(text: str) -> int:
    """Heuristic sentence count using punctuation boundaries."""
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    return max(1, len([s for s in sentences if s.strip()]))


def _estimate_tokens_whitespace(text: str) -> int:
    """Rough token estimate: split on whitespace (≈ 0.75 words/token for English)."""
    words = len(text.split())
    return max(1, int(words / 0.75))


def compute_document_stats(name: str, text: str) -> dict:
    """Compute statistics for a single document.

    Parameters
    ----------
    name:
        Document filename or identifier.
    text:
        Full document text.

    Returns
    -------
    Dict with keys: ``filename``, ``pages``, ``chars``, ``words``,
    ``sentences``, ``estimated_tokens``.
    """
    return {
        "filename": name,
        "pages": _count_pages(text),
        "chars": len(text),
        "words": len(text.split()),
        "sentences": _count_sentences(text),
        "estimated_tokens": _estimate_tokens_whitespace(text),
    }


def compute_corpus_statistics(
    corpus_dir: Path,
    output_json: Path,
    output_csv: Path,
) -> dict:
    """Compute statistics for all .txt files in *corpus_dir*.

    Parameters
    ----------
    corpus_dir:
        Directory containing cleaned corpus text files.
    output_json:
        Path where full JSON stats will be written.
    output_csv:
        Path where per-document CSV summary will be written.

    Returns
    -------
    Aggregate statistics dict.
    """
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        logger.warning("No .txt files found in %s", corpus_dir)

    doc_stats: list[dict] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        stats = compute_document_stats(f.name, text)
        doc_stats.append(stats)
        logger.debug("Stats for %s: %s", f.name, stats)

    aggregate: dict = {
        "num_documents": len(doc_stats),
        "total_pages": sum(d["pages"] for d in doc_stats),
        "total_chars": sum(d["chars"] for d in doc_stats),
        "total_words": sum(d["words"] for d in doc_stats),
        "total_sentences": sum(d["sentences"] for d in doc_stats),
        "total_estimated_tokens": sum(d["estimated_tokens"] for d in doc_stats),
        "avg_chars_per_doc": (
            sum(d["chars"] for d in doc_stats) / len(doc_stats) if doc_stats else 0
        ),
        "avg_tokens_per_doc": (
            sum(d["estimated_tokens"] for d in doc_stats) / len(doc_stats) if doc_stats else 0
        ),
        "documents": doc_stats,
    }

    # Write JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2)
    logger.info("Corpus stats JSON written to %s", output_json)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if doc_stats:
        fieldnames = list(doc_stats[0].keys())
        with output_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(doc_stats)
        logger.info("Corpus stats CSV written to %s", output_csv)

    logger.info(
        "Corpus summary: %d docs | %d pages | %d chars | ~%d tokens",
        aggregate["num_documents"],
        aggregate["total_pages"],
        aggregate["total_chars"],
        aggregate["total_estimated_tokens"],
    )
    return aggregate


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parents[2]
    stats = compute_corpus_statistics(
        corpus_dir=base / "data" / "domain_corpus",
        output_json=base / "data" / "processed" / "corpus_stats.json",
        output_csv=base / "data" / "processed" / "corpus_stats.csv",
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "documents"}, indent=2))
