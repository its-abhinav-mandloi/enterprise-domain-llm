"""
test_data_pipeline.py
=====================
Unit tests for data extraction, text cleaning, deduplication, and JSONL quality validation.
"""

import hashlib
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def test_domain_corpus_exists():
    """Verify that domain_corpus contains at least 5 cleaned .txt documents."""
    corpus_dir = DATA_DIR / "domain_corpus"
    assert corpus_dir.exists(), "domain_corpus directory does not exist"
    txt_files = list(corpus_dir.glob("*_clean.txt"))
    assert len(txt_files) >= 5, f"Expected >= 5 clean txt files, found {len(txt_files)}"


def test_cleaned_documents_non_empty():
    """Verify each cleaned document has substantial content (> 10,000 characters)."""
    corpus_dir = DATA_DIR / "domain_corpus"
    for f in corpus_dir.glob("*_clean.txt"):
        text = f.read_text(encoding="utf-8")
        assert len(text) > 10000, f"File {f.name} is too short ({len(text)} chars)"


def test_instruction_dataset_schema_and_quality():
    """Verify instruction_dataset.jsonl schema and strict text hygiene (no TOC dots or truncation)."""
    dataset_file = DATA_DIR / "instruction_dataset.jsonl"
    assert dataset_file.exists(), "instruction_dataset.jsonl not found"

    count = 0
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            inst = record["instruction"].strip()
            resp = record["response"].strip()
            assert len(inst) > 15, f"Instruction too short: {inst}"
            assert len(resp) >= 150, f"Response too short ({len(resp)} chars)"
            assert not re.search(r"\.{5,}", resp), f"TOC dot-leader artifact in response: {resp[:80]}"
            assert resp[0].isupper(), f"Response does not start cleanly on a sentence: {resp[:60]}"
            assert resp[-1] in ".!?", f"Response does not end cleanly on punctuation: {resp[-40:]}"
            count += 1
    assert count >= 300, f"Expected >= 300 clean instruction pairs, got {count}"


def test_train_eval_split_ratio():
    """Verify exact 80/20 train/eval split summing to instruction_dataset.jsonl."""
    dataset_file = DATA_DIR / "instruction_dataset.jsonl"
    train_file = DATA_DIR / "train.jsonl"
    eval_file = DATA_DIR / "eval.jsonl"
    assert train_file.exists() and eval_file.exists(), "Split files not found"

    n_full = sum(1 for line in open(dataset_file, "r", encoding="utf-8") if line.strip())
    n_train = sum(1 for line in open(train_file, "r", encoding="utf-8") if line.strip())
    n_eval = sum(1 for line in open(eval_file, "r", encoding="utf-8") if line.strip())
    assert n_train + n_eval == n_full, f"Split sum {n_train + n_eval} != dataset size {n_full}"

    ratio = n_train / n_full
    assert 0.79 <= ratio <= 0.81, f"Expected 80% train ratio, got {ratio:.2%}"


def test_deduplication_exact():
    """Verify zero duplicate instructions exist in instruction_dataset."""
    dataset_file = DATA_DIR / "instruction_dataset.jsonl"
    seen = set()
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            h = hashlib.sha256(record["instruction"].strip().lower().encode("utf-8")).hexdigest()
            assert h not in seen, f"Duplicate instruction detected: {record['instruction'][:50]}"
            seen.add(h)
