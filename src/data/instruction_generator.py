"""
instruction_generator.py
=========================
Heuristic instruction-response pair generator.

Converts cleaned domain corpus text into supervised fine-tuning (SFT) data
using sliding-window sentence segmentation and a small set of question
templates.  No external LLM is required.

Output format (each record)::

    {
        "id": "doc0_win3",
        "instruction": "What does the following text explain? ...",
        "input": "",
        "response": "<the window text>",
        "source": "filename.txt",
        "text": "<full formatted prompt+response for SFTTrainer>"
    }
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator

import jsonlines

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence segmentation (heuristic, no NLTK dependency)
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    # Remove page markers inserted by pdf_extractor
    text = re.sub(r"=== PAGE \d+ ===", "", text)
    sentences = _SENT_END_RE.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TEMPLATES: list[str] = [
    "What does the following text explain?\n\n{context}",
    "Summarize the key technical steps described here:\n\n{context}",
    "Based on the following content, what problem is being solved?\n\n{context}",
    "Explain the concept described in this passage:\n\n{context}",
    "What are the main takeaways from this technical text?\n\n{context}",
]


def _format_alpaca(instruction: str, response: str) -> str:
    """Format as Alpaca-style prompt for SFTTrainer."""
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}"
    )


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_pairs_from_text(
    text: str,
    source_name: str,
    window_size: int = 3,
    stride: int = 2,
    max_pairs: int = 50,
    min_response_length: int = 80,
) -> list[dict]:
    """Generate instruction-response pairs from a single document.

    Parameters
    ----------
    text:
        Cleaned document text.
    source_name:
        Identifier for the source document (e.g. filename).
    window_size:
        Number of sentences per sliding window.
    stride:
        Sentence stride between consecutive windows.
    max_pairs:
        Maximum number of pairs to generate from this document.
    min_response_length:
        Minimum character length for the response (window text).

    Returns
    -------
    List of instruction-response dicts.
    """
    sentences = _split_sentences(text)
    pairs: list[dict] = []
    pair_idx = 0
    template_cycle = 0

    i = 0
    while i <= len(sentences) - window_size and len(pairs) < max_pairs:
        window = sentences[i : i + window_size]
        context = " ".join(window).strip()

        if len(context) < min_response_length:
            i += stride
            continue

        template = TEMPLATES[template_cycle % len(TEMPLATES)]
        instruction = template.format(context=context[:600])  # cap context in instruction
        response = context

        record = {
            "id": f"{source_name}_win{pair_idx}",
            "instruction": instruction,
            "input": "",
            "response": response,
            "source": source_name,
            "text": _format_alpaca(instruction, response),
        }
        pairs.append(record)
        pair_idx += 1
        template_cycle += 1
        i += stride

    return pairs


def generate_dataset_from_directory(
    corpus_dir: Path,
    output_path: Path,
    window_size: int = 3,
    stride: int = 2,
    max_pairs_per_doc: int = 50,
    min_response_length: int = 80,
) -> dict[str, int]:
    """Generate a full instruction dataset from all corpus documents.

    Parameters
    ----------
    corpus_dir:
        Directory of cleaned .txt files.
    output_path:
        Path to write the JSONL output file.
    window_size, stride, max_pairs_per_doc, min_response_length:
        Passed to :func:`generate_pairs_from_text`.

    Returns
    -------
    Stats dict: ``total_pairs``, ``num_documents``.
    """
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        logger.warning("No .txt files found in %s", corpus_dir)
        return {"total_pairs": 0, "num_documents": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0

    with jsonlines.open(str(output_path), mode="w") as writer:
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            pairs = generate_pairs_from_text(
                text=text,
                source_name=f.stem,
                window_size=window_size,
                stride=stride,
                max_pairs=max_pairs_per_doc,
                min_response_length=min_response_length,
            )
            for pair in pairs:
                writer.write(pair)
            total_pairs += len(pairs)
            logger.info("Generated %d pairs from %s", len(pairs), f.name)

    stats = {"total_pairs": total_pairs, "num_documents": len(files)}
    logger.info(
        "Instruction dataset: %d pairs from %d documents → %s",
        total_pairs,
        len(files),
        output_path,
    )
    return stats


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parents[2]
    stats = generate_dataset_from_directory(
        corpus_dir=base / "data" / "domain_corpus",
        output_path=base / "data" / "processed" / "instruction_dataset.jsonl",
    )
    print(stats)
