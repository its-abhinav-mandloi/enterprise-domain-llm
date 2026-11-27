#!/usr/bin/env python3
"""
Full data pipeline: extract, clean, deduplicate, generate instruction dataset.
Optimized for large PDFs (9000+ pages total).
"""

import os
import re
import json
import csv
import hashlib
import random
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")

import fitz  # pymupdf
from langdetect import detect, LangDetectException

# ── Paths (portable relative to repository root) ─────────────────────────────
BASE = Path(__file__).resolve().parent
RAW_PDF_DIR = BASE / "data" / "raw_pdfs"
EXTRACTED_DIR = BASE / "data" / "extracted_text"
CORPUS_DIR = BASE / "data" / "domain_corpus"
RESULTS_DIR = BASE / "results"

for d in [RAW_PDF_DIR, EXTRACTED_DIR, CORPUS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── PDF METADATA ─────────────────────────────────────────────────────────────
PDF_SOURCES = {
    "bash_manual.pdf":         "https://www.gnu.org/software/bash/manual/bash.pdf",
    "aws_ec2_userguide.pdf":   "https://docs.aws.amazon.com/pdfs/AWSEC2/latest/UserGuide/ec2-ug.pdf",
    "redhat_sysadmin.pdf":     "https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/pdf/configuring_basic_system_settings/red_hat_enterprise_linux-9-configuring_basic_system_settings-en-us.pdf",
    "postgresql_manual.pdf":   "https://www.postgresql.org/files/documentation/pdf/17/postgresql-17-A4.pdf",
    "ubuntu_server_guide.pdf": "https://assets.ubuntu.com/v1/f954307f-ubuntu-server-guide.pdf",
    "gcc_manual.pdf":          "https://gcc.gnu.org/onlinedocs/gcc.pdf",
}
FAILED_DOWNLOADS = [
    {"url": "https://nginx.org/en/docs/nginx.pdf", "reason": "HTML response returned, not a valid PDF"},
]

# ── 1. TEXT EXTRACTION ───────────────────────────────────────────────────────
def extract_pdf(pdf_path: Path, out_dir: Path) -> Dict:
    """Extract text from PDF, page by page, return stats."""
    doc = fitz.open(str(pdf_path))
    pages_text = []
    total_chars = 0
    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
        total_chars += len(text)
    out_path = out_dir / (pdf_path.stem + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n--- PAGE BREAK ---\n\n".join(pages_text))
    doc.close()
    return {
        "filename": pdf_path.name,
        "txt_file": out_path.name,
        "pages": len(pages_text),
        "chars_raw": total_chars,
        "pages_text": pages_text,
    }

print("=" * 60)
print("STEP 1: DOWNLOADING (IF MISSING) & EXTRACTING TEXT FROM PDFs")
print("=" * 60)

def ensure_pdfs_downloaded(pdf_sources: Dict[str, str], dest_dir: Path) -> None:
    """Download any missing source PDFs from their official URLs for fresh-clone reproducibility."""
    import urllib.request
    for pdf_name, url in pdf_sources.items():
        pdf_path = dest_dir / pdf_name
        if not pdf_path.exists():
            print(f"  Downloading {pdf_name} from {url} ...")
            try:
                urllib.request.urlretrieve(url, str(pdf_path))
            except Exception as exc:
                print(f"  WARNING: Failed to download {pdf_name}: {exc}")

ensure_pdfs_downloaded(PDF_SOURCES, RAW_PDF_DIR)

extraction_results = []
for pdf_name in PDF_SOURCES:
    pdf_path = RAW_PDF_DIR / pdf_name
    if not pdf_path.exists():
        print(f"  SKIP (not found): {pdf_name}")
        continue
    result = extract_pdf(pdf_path, EXTRACTED_DIR)
    extraction_results.append(result)
    print(f"  {result['filename']}: {result['pages']} pages, {result['chars_raw']:,} chars")

total_pdfs_valid = len(extraction_results)
total_pages = sum(r["pages"] for r in extraction_results)
total_chars_before = sum(r["chars_raw"] for r in extraction_results)
print(f"\nTotal: {total_pdfs_valid} PDFs, {total_pages:,} pages, {total_chars_before:,} chars\n")


# ── 2. CLEANING ───────────────────────────────────────────────────────────────
def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_shingles(text: str, k: int = 5) -> set:
    text = text.lower()
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def is_english(text: str) -> bool:
    """Detect language using first 500 chars for speed."""
    try:
        return detect(text[:500]) == "en"
    except LangDetectException:
        return True  # assume English for technical docs if detection fails

print("=" * 60)
print("STEP 2: CLEANING CORPUS")
print("=" * 60)

total_chars_after_length = 0
total_chars_after_dedup = 0
total_chars_after_lang = 0
docs_removed_dups = 0
docs_removed_non_english = 0

seen_hashes: set = set()
seen_shingles: List[Tuple[set, str]] = []

cleaned_results = []

for result in extraction_results:
    fname = result["filename"]
    pages_text = result["pages_text"]
    n_pages = result["pages"]

    # -- Length filter: keep pages >= 100 chars
    filtered_pages = [normalize_whitespace(p) for p in pages_text if len(p.strip()) >= 100]
    chars_after_len = sum(len(p) for p in filtered_pages)
    total_chars_after_length += chars_after_len

    # -- Exact dedup (SHA256) globally
    unique_pages = []
    for page in filtered_pages:
        h = hashlib.sha256(page.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_pages.append(page)
        else:
            docs_removed_dups += 1

    # -- Near-dup detection (Jaccard on 5-grams, threshold 0.85, first 5000 chars)
    full_sample = " ".join(unique_pages)[:5000]
    doc_shingles = get_shingles(full_sample)
    is_near_dup = False
    for seen_shin, seen_name in seen_shingles:
        if jaccard(doc_shingles, seen_shin) >= 0.85:
            print(f"  NEAR-DUP detected: {fname} ≈ {seen_name} — excluding")
            is_near_dup = True
            docs_removed_dups += len(unique_pages)
            break
    if not is_near_dup:
        seen_shingles.append((doc_shingles, fname))

    chars_after_dedup = sum(len(p) for p in unique_pages) if not is_near_dup else 0
    total_chars_after_dedup += chars_after_dedup

    # -- Language filter
    # For large docs (>200 pages), sample first + last 20 pages for speed
    if is_near_dup:
        lang_pages: List[str] = []
        lang_ok = False
    else:
        if n_pages > 200:
            # Sample to check language, then assume rest is same
            sample = unique_pages[:5]
            sample_text = " ".join(sample)
            doc_is_english = is_english(sample_text)
            lang_pages = unique_pages if doc_is_english else []
            if not doc_is_english:
                docs_removed_non_english += len(unique_pages)
        else:
            lang_pages = []
            for page in unique_pages:
                if is_english(page):
                    lang_pages.append(page)
                else:
                    docs_removed_non_english += 1
        lang_ok = len(lang_pages) > 0

    chars_after_lang = sum(len(p) for p in lang_pages)
    total_chars_after_lang += chars_after_lang

    # -- Save to domain_corpus
    if lang_ok and lang_pages:
        out_path = CORPUS_DIR / (Path(fname).stem + "_clean.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(lang_pages))
        status = "included"
    else:
        status = "excluded"

    cleaned_results.append({
        "filename": fname,
        "url": PDF_SOURCES.get(fname, ""),
        "pages": n_pages,
        "chars": chars_after_lang,
        "chars_raw": result["chars_raw"],
        "status": status,
        "lang_pages": lang_pages,
        "txt_stem": Path(fname).stem,
    })
    print(f"  {fname}: {len(lang_pages)} clean pages, {chars_after_lang:,} chars [{status}]")

final_corpus_chars = total_chars_after_lang

reductions = {
    "length_filter": total_chars_before - total_chars_after_length,
    "deduplication": total_chars_after_length - total_chars_after_dedup,
    "language_filter": total_chars_after_dedup - total_chars_after_lang,
}
greatest_stage = max(reductions, key=reductions.get)
print(f"\nGreatest reduction at: {greatest_stage}")
print(f"  Before cleaning:        {total_chars_before:,}")
print(f"  After length filter:    {total_chars_after_length:,} (removed {reductions['length_filter']:,})")
print(f"  After dedup:            {total_chars_after_dedup:,} (removed {reductions['deduplication']:,})")
print(f"  After lang filter:      {total_chars_after_lang:,} (removed {reductions['language_filter']:,})")


# ── 3. CORPUS STATISTICS ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: SAVING CORPUS STATISTICS")
print("=" * 60)

stats = {
    "total_pdfs_attempted": len(PDF_SOURCES) + len(FAILED_DOWNLOADS),
    "total_pdfs_valid": total_pdfs_valid,
    "total_pages_extracted": total_pages,
    "total_chars_before_cleaning": total_chars_before,
    "total_chars_after_length_filter": total_chars_after_length,
    "total_chars_after_dedup": total_chars_after_dedup,
    "total_chars_after_language_filter": total_chars_after_lang,
    "docs_removed_duplicates": docs_removed_dups,
    "docs_removed_non_english": docs_removed_non_english,
    "final_corpus_size_chars": final_corpus_chars,
    "greatest_reduction_stage": greatest_stage,
    "sources": [
        {
            "filename": r["filename"],
            "url": r["url"],
            "pages": r["pages"],
            "chars": r["chars"],
            "status": r["status"],
        }
        for r in cleaned_results
    ],
}

json_path = RESULTS_DIR / "corpus_statistics.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(f"  Saved: {json_path}")

csv_path = RESULTS_DIR / "corpus_statistics.csv"
flat = {k: v for k, v in stats.items() if k != "sources"}
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=flat.keys())
    writer.writeheader()
    writer.writerow(flat)
print(f"  Saved: {csv_path}")


# ── 4. INSTRUCTION DATASET GENERATION ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: GENERATING INSTRUCTION DATASET")
print("=" * 60)

def chunk_text(text: str, chars_per_chunk: int = 1800, overlap: int = 300) -> List[str]:
    """Split text into overlapping chunks."""
    step = chars_per_chunk - overlap
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chars_per_chunk]
        if len(chunk) >= 150:
            chunks.append(chunk)
        i += step
    return chunks

def extract_commands(text: str) -> List[str]:
    cmds = []
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^\$\s+\S", s):
            cmds.append(s.lstrip("$ ").strip())
        elif re.match(r"^(sudo|apt|yum|dnf|systemctl|kubectl|docker|git|ls|cd|mkdir|chmod|chown|grep|awk|sed|curl|wget|ssh|scp|rsync|psql|gcc|make|pg_|createdb|createuser)\s", s):
            cmds.append(s)
    return cmds[:4]

def extract_technical_terms(text: str) -> List[str]:
    terms = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)
    camel = re.findall(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]+\b", text)
    combined = list(dict.fromkeys(terms + camel))
    return [t for t in combined if len(t) > 4][:4]

def extract_procedure(text: str) -> Optional[str]:
    matches = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s+.{20,}", text)
    if len(matches) >= 2:
        return "\n".join(m.strip() for m in matches[:6])
    return None

def clean_instruction(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\x09\x0a\x20-\x7e]", "", s)
    return s

def generate_pairs(lang_pages: List[str], source_name: str, max_pairs: int = 120) -> List[Dict]:
    full_text = "\n\n".join(lang_pages)
    # For very large docs, sample evenly
    if len(full_text) > 500_000:
        step = len(full_text) // 500_000
        full_text = full_text[::step][:500_000]

    chunks = chunk_text(full_text)
    pairs: List[Dict] = []
    seen: set = set()
    doc_label = source_name.replace("_clean.txt", "").replace("_", " ")

    for chunk in chunks:
        if len(pairs) >= max_pairs:
            break
        cmds = extract_commands(chunk)
        terms = extract_technical_terms(chunk)
        proc = extract_procedure(chunk)
        snippet = chunk[:60].strip()
        snippet_clean = re.sub(r"[^a-zA-Z0-9 ]", " ", snippet).strip().lower()
        if len(snippet_clean) < 5:
            continue

        candidates = [
            # 0 — general how-to
            {
                "instruction": clean_instruction(f"How do I {snippet_clean} in {doc_label}?"),
                "response": chunk[:800].strip(),
                "source": source_name,
            },
            # 1 — command
            ({
                "instruction": clean_instruction(f"What is the command to {snippet_clean}?"),
                "response": f"The following command(s) can be used:\n\n" + "\n".join(f"  {c}" for c in cmds) + f"\n\nExplanation:\n{chunk[:500].strip()}",
                "source": source_name,
            } if cmds else None),
            # 2 — technical term
            ({
                "instruction": clean_instruction(f"What does {terms[0]} mean in the context of {doc_label}?"),
                "response": chunk[:800].strip(),
                "source": source_name,
            } if terms else None),
            # 3 — step-by-step
            {
                "instruction": clean_instruction(f"What steps should I follow to {snippet_clean}?"),
                "response": proc if proc else f"To accomplish this:\n\n{chunk[:700].strip()}",
                "source": source_name,
            },
            # 4 — troubleshooting
            {
                "instruction": clean_instruction(f"How can I troubleshoot issues with {snippet_clean} in {doc_label}?"),
                "response": f"To troubleshoot this issue, consider the following:\n\n{chunk[:700].strip()}",
                "source": source_name,
            },
            # 5 — best practices
            {
                "instruction": clean_instruction(f"What are the best practices for {snippet_clean} when using {doc_label}?"),
                "response": chunk[:700].strip(),
                "source": source_name,
            },
            # 6 — explain concept
            ({
                "instruction": clean_instruction(f"Can you explain {terms[1] if len(terms) > 1 else terms[0]} and its role in {doc_label}?"),
                "response": chunk[:800].strip(),
                "source": source_name,
            } if terms else None),
        ]

        for pair in candidates:
            if len(pairs) >= max_pairs:
                break
            if pair is None:
                continue
            inst = pair["instruction"]
            resp = pair["response"].strip()
            if len(inst) < 15 or len(resp) < 50:
                continue
            key = hashlib.sha256((inst + resp[:100]).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"instruction": inst, "response": resp, "source": source_name})

    return pairs

all_pairs: List[Dict] = []
for result in cleaned_results:
    if result["status"] != "included" or not result["lang_pages"]:
        continue
    src_name = result["txt_stem"] + "_clean.txt"
    pairs = generate_pairs(result["lang_pages"], src_name, max_pairs=120)
    print(f"  {result['filename']}: {len(pairs)} pairs generated")
    all_pairs.extend(pairs)

print(f"\nTotal pairs before global dedup: {len(all_pairs)}")

# Global dedup
global_seen: set = set()
deduped_pairs: List[Dict] = []
for p in all_pairs:
    key = hashlib.sha256((p["instruction"] + p["response"][:100]).encode()).hexdigest()
    if key not in global_seen:
        global_seen.add(key)
        deduped_pairs.append(p)
print(f"Total pairs after global dedup: {len(deduped_pairs)}")

# Shuffle + split 80/20 seed=42
random.seed(42)
random.shuffle(deduped_pairs)
n_train = int(len(deduped_pairs) * 0.8)
train_pairs = deduped_pairs[:n_train]
eval_pairs = deduped_pairs[n_train:]

def write_jsonl(path: Path, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

write_jsonl(BASE / "data" / "instruction_dataset.jsonl", deduped_pairs)
write_jsonl(BASE / "data" / "train.jsonl", train_pairs)
write_jsonl(BASE / "data" / "eval.jsonl", eval_pairs)

print(f"\nDataset saved:")
print(f"  instruction_dataset.jsonl : {len(deduped_pairs)} records")
print(f"  train.jsonl               : {len(train_pairs)} records (80%)")
print(f"  eval.jsonl                : {len(eval_pairs)} records (20%)")

# ── 5. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE COMPLETE — SUMMARY")
print("=" * 60)
print(f"PDFs attempted:          {stats['total_pdfs_attempted']}")
print(f"PDFs valid:              {stats['total_pdfs_valid']}")
print(f"Pages extracted:         {stats['total_pages_extracted']:,}")
print(f"Chars before clean:      {stats['total_chars_before_cleaning']:,}")
print(f"Chars after clean:       {final_corpus_chars:,}")
print(f"Dedup page removals:     {stats['docs_removed_duplicates']}")
print(f"Non-English removals:    {stats['docs_removed_non_english']}")
print(f"Greatest reduction:      {greatest_stage}")
print(f"Train pairs:             {len(train_pairs)}")
print(f"Eval pairs:              {len(eval_pairs)}")
print(f"Total instruction pairs: {len(deduped_pairs)}")

# Update stats with dataset counts
stats["instruction_pairs_total"] = len(deduped_pairs)
stats["instruction_pairs_train"] = len(train_pairs)
stats["instruction_pairs_eval"] = len(eval_pairs)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(f"\nFinal stats: {json_path}")
