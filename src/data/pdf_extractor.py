"""
pdf_extractor.py
================
Extract text from PDF files page-by-page using PyMuPDF (fitz).

Each PDF produces one plain-text file in the output directory.
Page boundaries are preserved with markers so downstream code
can count pages precisely.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    min_page_chars: int = 100,
) -> dict[str, int | str]:
    """Extract text from a single PDF and save to a .txt file.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF file.
    output_dir:
        Directory where the extracted .txt file will be written.
    min_page_chars:
        Pages with fewer characters than this threshold are skipped.

    Returns
    -------
    dict with keys: ``filename``, ``pages_total``, ``pages_kept``,
    ``chars_extracted``, ``output_path``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (pdf_path.stem + ".txt")

    doc = fitz.open(str(pdf_path))
    pages_total = len(doc)
    pages_kept = 0
    chars_extracted = 0
    lines: list[str] = []

    for page_num in range(pages_total):
        page = doc[page_num]
        text: str = page.get_text("text")  # plain text extraction
        text = text.strip()
        if len(text) < min_page_chars:
            logger.debug("Skipping page %d of %s (too short: %d chars)", page_num + 1, pdf_path.name, len(text))
            continue
        lines.append(f"=== PAGE {page_num + 1} ===")
        lines.append(text)
        pages_kept += 1
        chars_extracted += len(text)

    doc.close()

    combined = "\n\n".join(lines)
    output_path.write_text(combined, encoding="utf-8")

    logger.info(
        "Extracted %s: %d/%d pages kept, %d chars → %s",
        pdf_path.name,
        pages_kept,
        pages_total,
        chars_extracted,
        output_path,
    )
    return {
        "filename": pdf_path.name,
        "pages_total": pages_total,
        "pages_kept": pages_kept,
        "chars_extracted": chars_extracted,
        "output_path": str(output_path),
    }


def extract_all_pdfs(
    raw_pdf_dir: Path,
    output_dir: Path,
    min_page_chars: int = 100,
) -> list[dict[str, int | str]]:
    """Extract text from every PDF in *raw_pdf_dir*.

    Parameters
    ----------
    raw_pdf_dir:
        Directory containing input PDF files.
    output_dir:
        Directory where extracted .txt files will be written.
    min_page_chars:
        Passed through to :func:`extract_text_from_pdf`.

    Returns
    -------
    List of result dicts, one per PDF processed.
    """
    pdf_files = sorted(raw_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", raw_pdf_dir)
        return []

    results: list[dict[str, int | str]] = []
    for pdf_path in pdf_files:
        try:
            result = extract_text_from_pdf(pdf_path, output_dir, min_page_chars)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to extract %s: %s", pdf_path.name, exc)
            results.append({"filename": pdf_path.name, "error": str(exc)})

    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parents[2]
    stats = extract_all_pdfs(
        raw_pdf_dir=base / "data" / "raw_pdfs",
        output_dir=base / "data" / "extracted_text",
    )
    for s in stats:
        print(s)
