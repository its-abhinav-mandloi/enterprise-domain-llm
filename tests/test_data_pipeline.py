"""
test_data_pipeline.py
=====================
Unit tests for data extraction, text cleaning, deduplication, and JSONL validation.
"""

import json
import hashlib
from pathlib import Path
import pytest

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


def test_instruction_dataset_schema():
    """Verify instruction_dataset.jsonl schema: 'instruction' and 'response' non-empty."""
    dataset_file = DATA_DIR / "instruction_dataset.jsonl"
    assert dataset_file.exists(), "instruction_dataset.jsonl not found"

    count = 0
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            assert "instruction" in record, "Missing 'instruction' field"
            assert "response" in record, "Missing 'response' field"
            assert len(record["instruction"].strip()) > 10, "Instruction too short"
            assert len(record["response"].strip()) > 20, "Response too short"
            count += 1
    assert count >= 100, f"Expected >= 100 instruction pairs, got {count}"


def test_train_eval_split_ratio():
    """Verify 80/20 train/eval dataset split."""
    train_file = DATA_DIR / "train.jsonl"
    eval_file = DATA_DIR / "eval.jsonl"
    assert train_file.exists() and eval_file.exists(), "Split files not found"

    n_train = sum(1 for line in open(train_file, "r", encoding="utf-8") if line.strip())
    n_eval = sum(1 for line in open(eval_file, "r", encoding="utf-8") if line.strip())
    total = n_train + n_eval

    ratio = n_train / total
    assert 0.78 <= ratio <= 0.82, f"Expected ~80% train ratio, got {ratio:.2%}"


def test_deduplication_exact():
    """Verify no duplicate instructions exist in instruction_dataset."""
    dataset_file = DATA_DIR / "instruction_dataset.jsonl"
    seen = set()
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            h = hashlib.sha256(record["instruction"].strip().encode("utf-8")).hexdigest()
            assert h not in seen, f"Duplicate instruction detected: {record['instruction'][:40]}"
            seen.add(h)
