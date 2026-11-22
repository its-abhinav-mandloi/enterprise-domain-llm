"""
deduplicator.py
===============
Exact and near-duplicate detection for text documents.

Exact duplicates  → SHA-256 hash of full text.
Near-duplicates   → Jaccard similarity on character-level word shingles.

Usage::

    from src.data.deduplicator import deduplicate_texts
    unique_texts = deduplicate_texts(texts, shingle_size=5, jaccard_threshold=0.8)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sha256_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    """Compute the set of k-word shingles for *text*.

    Parameters
    ----------
    text:
        Input text.
    k:
        Number of words per shingle.

    Returns
    -------
    Set of k-tuples of words.
    """
    words = text.lower().split()
    if len(words) < k:
        return {tuple(words)}
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    Returns a float in [0, 1].
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def deduplicate_texts(
    texts: list[str],
    shingle_size: int = 5,
    jaccard_threshold: float = 0.8,
) -> tuple[list[str], dict[str, int]]:
    """Remove exact and near-duplicate texts.

    Parameters
    ----------
    texts:
        List of input text strings.
    shingle_size:
        Word-shingle size for near-duplicate comparison.
    jaccard_threshold:
        Jaccard similarity at or above which a text is considered a near-duplicate.

    Returns
    -------
    Tuple of (unique_texts, stats_dict) where stats_dict has keys
    ``total``, ``exact_dupes``, ``near_dupes``, ``kept``.
    """
    seen_hashes: set[str] = set()
    seen_shingles: list[set[tuple[str, ...]]] = []
    unique_texts: list[str] = []
    exact_dupes = 0
    near_dupes = 0

    for text in texts:
        digest = sha256_hash(text)

        # --- Exact duplicate check ---
        if digest in seen_hashes:
            exact_dupes += 1
            logger.debug("Exact duplicate removed (hash %s…)", digest[:12])
            continue
        seen_hashes.add(digest)

        # --- Near-duplicate check ---
        shingles = word_shingles(text, shingle_size)
        is_near_dup = False
        for existing_shingles in seen_shingles:
            if jaccard_similarity(shingles, existing_shingles) >= jaccard_threshold:
                near_dupes += 1
                logger.debug("Near-duplicate removed (Jaccard >= %.2f)", jaccard_threshold)
                is_near_dup = True
                break

        if not is_near_dup:
            seen_shingles.append(shingles)
            unique_texts.append(text)

    stats = {
        "total": len(texts),
        "exact_dupes": exact_dupes,
        "near_dupes": near_dupes,
        "kept": len(unique_texts),
    }
    logger.info(
        "Deduplication: %d → %d texts (%d exact, %d near-dup removed)",
        stats["total"],
        stats["kept"],
        exact_dupes,
        near_dupes,
    )
    return unique_texts, stats


def deduplicate_directory(
    input_dir: Path,
    output_dir: Path,
    shingle_size: int = 5,
    jaccard_threshold: float = 0.8,
) -> dict[str, int]:
    """Deduplicate all .txt files in *input_dir* and write unique ones to *output_dir*.

    Returns deduplication stats.
    """
    files = sorted(input_dir.glob("*.txt"))
    texts = [f.read_text(encoding="utf-8", errors="replace") for f in files]
    names = [f.name for f in files]

    unique_texts, stats = deduplicate_texts(texts, shingle_size, jaccard_threshold)

    # We need to track which names survive; rebuild with index mapping
    # Re-run with index to map surviving indices to filenames
    seen_hashes: set[str] = set()
    seen_shingles: list[set[tuple[str, ...]]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, text in zip(names, texts):
        digest = sha256_hash(text)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        shingles = word_shingles(text, shingle_size)
        is_near_dup = any(
            jaccard_similarity(shingles, ex) >= jaccard_threshold
            for ex in seen_shingles
        )
        if is_near_dup:
            continue

        seen_shingles.append(shingles)
        (output_dir / name).write_text(text, encoding="utf-8")
        written += 1

    stats["written"] = written
    return stats


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parents[2]
    stats = deduplicate_directory(
        input_dir=base / "data" / "domain_corpus",
        output_dir=base / "data" / "domain_corpus",
    )
    print(stats)
