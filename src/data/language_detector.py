"""
language_detector.py
====================
Wrapper around *langdetect* for filtering non-English texts.

langdetect is non-deterministic by default.  We seed it for reproducibility
and retry up to ``max_retries`` times if detection raises an error.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langdetect import DetectorFactory, detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

# Seed for reproducible language detection
DetectorFactory.seed = 42


def detect_language(text: str, max_retries: int = 3) -> tuple[str | None, float]:
    """Detect the dominant language of *text*.

    Parameters
    ----------
    text:
        Input text to analyse.
    max_retries:
        How many times to retry on ``LangDetectException``.

    Returns
    -------
    Tuple ``(language_code, confidence)`` where *language_code* is a BCP-47
    string (e.g. ``"en"``) or ``None`` on failure, and *confidence* is a
    float in [0, 1].
    """
    for attempt in range(max_retries):
        try:
            probabilities = detect_langs(text)
            if probabilities:
                top = probabilities[0]
                return top.lang, top.prob
        except LangDetectException as exc:
            logger.debug("LangDetect attempt %d failed: %s", attempt + 1, exc)
    return None, 0.0


def is_english(
    text: str,
    min_confidence: float = 0.8,
) -> bool:
    """Return ``True`` if *text* is detected as English with sufficient confidence.

    Parameters
    ----------
    text:
        Input text.
    min_confidence:
        Required minimum probability score.
    """
    lang, conf = detect_language(text)
    return lang == "en" and conf >= min_confidence


def filter_english_texts(
    texts: list[str],
    min_confidence: float = 0.8,
) -> tuple[list[str], dict[str, int]]:
    """Filter *texts* to only English ones.

    Returns a tuple of (filtered_texts, stats) where stats has keys
    ``total``, ``english``, ``non_english``, ``failed``.
    """
    english: list[str] = []
    non_english = 0
    failed = 0

    for text in texts:
        lang, conf = detect_language(text)
        if lang is None:
            failed += 1
            logger.debug("Language detection failed for text snippet")
        elif lang == "en" and conf >= min_confidence:
            english.append(text)
        else:
            non_english += 1
            logger.debug("Non-English text removed (lang=%s, conf=%.2f)", lang, conf)

    stats = {
        "total": len(texts),
        "english": len(english),
        "non_english": non_english,
        "failed": failed,
    }
    logger.info(
        "Language filtering: %d/%d texts kept as English", len(english), len(texts)
    )
    return english, stats


def filter_english_directory(
    input_dir: Path,
    output_dir: Path,
    min_confidence: float = 0.8,
) -> dict[str, int]:
    """Filter all .txt files in *input_dir* to English only.

    Non-English files are not copied.  Returns stats dict.
    """
    files = sorted(input_dir.glob("*.txt"))
    output_dir.mkdir(parents=True, exist_ok=True)

    english_count = 0
    non_english_count = 0
    failed_count = 0

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        lang, conf = detect_language(text)
        if lang is None:
            failed_count += 1
            logger.warning("Language detection failed for %s", f.name)
        elif lang == "en" and conf >= min_confidence:
            (output_dir / f.name).write_text(text, encoding="utf-8")
            english_count += 1
        else:
            non_english_count += 1
            logger.info("Skipping non-English file: %s (lang=%s, conf=%.2f)", f.name, lang, conf)

    return {
        "total": len(files),
        "english": english_count,
        "non_english": non_english_count,
        "failed": failed_count,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parents[2]
    stats = filter_english_directory(
        input_dir=base / "data" / "domain_corpus",
        output_dir=base / "data" / "domain_corpus",
    )
    print(stats)
