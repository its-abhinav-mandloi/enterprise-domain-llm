"""
text_cleaner.py
===============
Clean and normalise raw extracted text before corpus construction.

Steps applied (in order):
1. Remove control characters (except newline/tab).
2. Normalise whitespace (collapse runs, strip trailing spaces).
3. Filter by minimum / maximum length.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# Control-char regex: matches everything in Unicode category "C" except \n \t
_CTRL_RE = re.compile(r"[^\S\n\t]+")          # collapse horizontal whitespace runs
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")       # collapse 3+ newlines → 2


def remove_control_chars(text: str) -> str:
    """Remove Unicode control characters (Cc category) except newline and tab."""
    return "".join(
        ch for ch in text if unicodedata.category(ch) != "Cc" or ch in ("\n", "\t")
    )


def normalise_whitespace(text: str) -> str:
    """Collapse horizontal whitespace runs and trim trailing spaces on each line."""
    # Collapse multiple spaces/tabs on a single line
    text = _CTRL_RE.sub(" ", text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse excessive blank lines
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(
    text: str,
    min_length: int = 100,
    max_length: int = 100_000,
    normalize_whitespace: bool = True,
    remove_ctrl: bool = True,
) -> str | None:
    """Clean a single text string.

    Parameters
    ----------
    text:
        Raw input text.
    min_length:
        Discard texts shorter than this (after cleaning).
    max_length:
        Discard texts longer than this (after cleaning).
    normalize_whitespace:
        Apply whitespace normalisation.
    remove_ctrl:
        Strip Unicode control characters.

    Returns
    -------
    Cleaned text, or ``None`` if the result fails length filters.
    """
    if remove_ctrl:
        text = remove_control_chars(text)
    if normalize_whitespace:
        text = normalise_whitespace(text)

    if len(text) < min_length:
        logger.debug("Discarding short text (%d chars < %d)", len(text), min_length)
        return None
    if len(text) > max_length:
        logger.debug("Discarding long text (%d chars > %d)", len(text), max_length)
        return None
    return text


def clean_file(
    input_path: Path,
    output_path: Path,
    min_length: int = 100,
    max_length: int = 100_000,
) -> bool:
    """Read, clean, and write a single text file.

    Returns ``True`` if the file passed filters and was written, ``False`` otherwise.
    """
    raw = input_path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw, min_length=min_length, max_length=max_length)
    if cleaned is None:
        logger.info("Filtered out %s (length filter)", input_path.name)
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    logger.info("Cleaned %s → %s (%d chars)", input_path.name, output_path.name, len(cleaned))
    return True


def clean_all_files(
    input_dir: Path,
    output_dir: Path,
    min_length: int = 100,
    max_length: int = 100_000,
) -> dict[str, int]:
    """Clean all .txt files in *input_dir*, writing results to *output_dir*.

    Returns a summary dict with keys ``total``, ``kept``, ``filtered``.
    """
    txt_files = sorted(input_dir.glob("*.txt"))
    total = len(txt_files)
    kept = 0

    for src in txt_files:
        dst = output_dir / src.name
        if clean_file(src, dst, min_length, max_length):
            kept += 1

    summary = {"total": total, "kept": kept, "filtered": total - kept}
    logger.info("Cleaning complete: %d/%d files kept", kept, total)
    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parents[2]
    summary = clean_all_files(
        input_dir=base / "data" / "extracted_text",
        output_dir=base / "data" / "domain_corpus",
    )
    print(summary)
