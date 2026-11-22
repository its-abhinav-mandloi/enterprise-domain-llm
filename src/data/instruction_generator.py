"""
instruction_generator.py
========================
High-precision, corpus-grounded instruction–response pair generator for
Assignment 1B (Part B1 — Domain V5: IT Helpdesk & Incident Resolution Assistant).

Reads ONLY the cleaned domain corpus `.txt` files from `data/domain_corpus/`,
normalizes PDF typography artifacts (ligatures, hyphenated line breaks, TOC/page
headers), extracts clean section headings and complete multi-sentence technical
passages, and formats natural, strictly corpus-grounded Q&A pairs.

Outputs:
- data/instruction_dataset.jsonl  (all unique deduplicated pairs)
- data/train.jsonl                (80% deterministic split, seed=42)
- data/eval.jsonl                 (20% deterministic split, seed=42)
- results/instruction_dataset_stats.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Friendly display names for each cleaned manual in data/domain_corpus/
SOURCE_LABELS: Dict[str, str] = {
    "aws_ec2_userguide_clean.txt": "AWS EC2",
    "bash_manual_clean.txt": "GNU Bash",
    "gcc_manual_clean.txt": "GCC",
    "postgresql_manual_clean.txt": "PostgreSQL 17",
    "redhat_sysadmin_clean.txt": "Red Hat Enterprise Linux 9",
    "ubuntu_server_guide_clean.txt": "Ubuntu Server",
}

# Running headers/footers to strip before parsing
RUNNING_HEADERS_RE = re.compile(
    r"^(?:"
    r"Amazon Elastic Compute Cloud|"
    r"User Guide|"
    r"Using the GNU Compiler Collection \(GCC\)|"
    r"Chapter \d+:.*|"
    r"Ubuntu Server Guide|"
    r"Red Hat Enterprise Linux 9.*|"
    r"Configuring basic system settings|"
    r"PostgreSQL 17.*Documentation|"
    r"Bash Reference Manual|"
    r"\d{1,5}|"
    r"[ivxlcdm]{1,6}"
    r")$",
    re.IGNORECASE,
)

# TOC / dot-leader lines
TOC_LINE_RE = re.compile(r"(\.{4,}|(\.\s){4,}|…{2,})")

# Spaced-out PDF monospace glyph artifacts like "f i l e" or "1 2 7 . 0 . 0 . 1"
SPACED_GLYPH_RE = re.compile(r"\b(?:[A-Za-z0-9]\s){4,}[A-Za-z0-9]\b")

# Ligature normalization map
LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2212": "-",
    "\u2013": "-",
    "\u2014": " - ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
}


def normalize_pdf_text(raw: str) -> str:
    """Normalize ligatures, hyphenated line wraps, and whitespace."""
    text = raw
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    # Rejoin hyphenated line breaks like "descrip-\ntion" -> "description"
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def clean_lines_and_skip_toc(text: str, fname: str) -> List[str]:
    """Strip TOC preamble and running page headers/footers."""
    lines = text.splitlines()
    # Skip initial TOC block: find end of heavy TOC dot-leader region if present
    toc_indices = [
        idx
        for idx, ln in enumerate(lines[: int(len(lines) * 0.20)])
        if TOC_LINE_RE.search(ln)
    ]
    start_idx = (max(toc_indices) + 5) if toc_indices else int(len(lines) * 0.03)
    if fname == "redhat_sysadmin_clean.txt":
        # Red Hat TOC uses uppercase chapter lines without dots before CHAPTER 1 body
        chap1_hits = [
            i for i, ln in enumerate(lines)
            if ln.strip().upper().startswith("CHAPTER 1.")
        ]
        if len(chap1_hits) >= 2:
            start_idx = chap1_hits[1]

    cleaned: List[str] = []
    for ln in lines[start_idx:]:
        s = ln.strip()
        if not s:
            cleaned.append("")
            continue
        if TOC_LINE_RE.search(s):
            continue
        if RUNNING_HEADERS_RE.match(s):
            continue
        if SPACED_GLYPH_RE.search(s):
            continue
        cleaned.append(s)
    return cleaned


def _reflow_paragraph(lines: List[str]) -> str:
    """Join wrapped PDF lines into clean prose while preserving numbered lists."""
    out_parts: List[str] = []
    for ln in lines:
        if not ln:
            if out_parts and not out_parts[-1].endswith("\n\n"):
                out_parts.append("\n\n")
            continue
        if re.match(r"^(\d+\.\s|[•*-]\s)", ln):
            if out_parts and not out_parts[-1].endswith("\n"):
                out_parts.append("\n")
            out_parts.append(ln + " ")
        else:
            out_parts.append(ln + " ")
    text = "".join(out_parts)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_clean_sentences(body: str, min_chars: int = 200, max_chars: int = 850) -> Optional[str]:
    """Select clean, complete sentences starting at a valid sentence boundary and ending at punctuation."""
    if not body:
        return None
    # Reject tables / mostly numeric / garbled blocks
    alpha_chars = sum(c.isalpha() for c in body)
    if alpha_chars < 120 or (alpha_chars / max(len(body), 1)) < 0.62:
        return None
    if SPACED_GLYPH_RE.search(body) or TOC_LINE_RE.search(body):
        return None

    # Split into sentences while keeping paragraph integrity
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])", body)
    selected: List[str] = []
    cur_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # First sentence MUST start cleanly with an uppercase letter (no mid-word truncation)
        if not selected:
            if not re.match(r"^[A-Z][a-zA-Z0-9'\"`(/\-]", s):
                continue
            # Avoid starting on orphan pronouns or bare section fragments
            if re.match(r"^(Note|Tip|Warning|Important|Table|Figure|Example|See)\b[:.]?$", s):
                continue
        if cur_len + len(s) + 1 > max_chars and cur_len >= min_chars:
            break
        selected.append(s)
        cur_len += len(s) + 1

    if not selected:
        return None

    result = " ".join(selected).strip()
    # Ensure clean sentence ending
    if not re.search(r"[.!?]$", result):
        # Trim back to last complete sentence ending
        m = re.search(r"^(.*[.!?])[^.!?]*$", result, re.DOTALL)
        if m:
            result = m.group(1).strip()
        else:
            return None

    if len(result) < min_chars or len(result) > max_chars + 100:
        return None
    # Final sanity checks
    if re.search(r"\b(page \d+|see section \d+\.\d+\.\d+)\b", result.lower()) and len(result) < 260:
        return None
    if result.count("\n") > 12:
        return None
    return result


def _clean_heading_text(heading: str) -> str:
    """Strip section numbers and normalize heading casing cleanly."""
    h = re.sub(r"^(?:CHAPTER\s+\d+\.?\s*|\d+(?:\.\d+){0,4}\.?\s+)", "", heading.strip(), flags=re.IGNORECASE)
    h = re.sub(r"\s+", " ", h).strip(" :-.")
    # Convert ALL-CAPS headings (like RHEL) to readable sentence/title case while preserving acronyms
    if h.isupper() and len(h) > 4:
        acronyms = {"SSH", "DNS", "DHCP", "IP", "IPV4", "IPV6", "TCP", "UDP", "TLS", "SSL", "VPN",
                    "RHEL", "NMCLI", "SELINUX", "NFS", "SMB", "NTP", "UUID", "LVM", "RAID", "BIOS",
                    "UEFI", "HTTP", "HTTPS", "API", "CLI", "CPU", "RAM", "GPU", "AWS", "EC2", "EBS",
                    "VPC", "IAM", "AMI", "SQL", "WAL", "GCC", "GNU", "URL", "PAM", "LDAP", "UID", "GID"}
        words = []
        for idx, w in enumerate(h.split()):
            clean_w = re.sub(r"[^A-Z0-9]", "", w)
            if clean_w in acronyms:
                words.append(w)
            elif idx == 0:
                words.append(w.capitalize())
            else:
                words.append(w.lower())
        h = " ".join(words)
    return h


def _is_valid_heading(h: str) -> bool:
    """Validate that a heading is a genuine technical topic, not noise or a full sentence."""
    if not h or len(h) < 5 or len(h) > 80:
        return False
    if h.endswith(".") or h.endswith(",") or h.endswith(";"):
        return False
    words = h.split()
    if len(words) < 2 or len(words) > 12:
        return False
    # Reject generic structural labels
    low = h.lower()
    banned_exact = {
        "additional resources", "prerequisites", "procedure", "verification",
        "overview", "introduction", "conclusion", "notes", "note", "warning",
        "important", "tip", "summary", "examples", "example", "contents",
        "references", "syntax", "options", "parameters", "description",
        "troubleshooting", "next steps", "related information", "limitations",
        "considerations", "requirements", "table of contents", "index",
    }
    if low in banned_exact:
        return False
    if any(tok in low for tok in ["http://", "https://", "example.com", "copyright", "license", "appendix"]):
        return False
    # Must contain mostly alphabetic characters
    if sum(c.isalpha() for c in h) / len(h) < 0.70:
        return False
    return True


def _build_question(topic: str, source_label: str, idx: int, is_flag: bool = False) -> str:
    """Generate a natural, grammatically sound IT Helpdesk question for a verified heading/flag."""
    if is_flag:
        templates = [
            f"What does the {source_label} compiler option `{topic}` do?",
            f"How is the `{topic}` flag used in {source_label}, and what behavior does it control?",
            f"In {source_label}, explain the purpose and effect of the `{topic}` option.",
        ]
        return templates[idx % len(templates)]

    t_low = topic.lower()
    # Verb-led headings (gerunds or imperatives)
    if re.match(r"^(configuring|managing|monitoring|troubleshooting|viewing|creating|connecting|installing|enabling|disabling|updating|securing|filtering|authenticating|backing|restoring|setting|starting|stopping|checking|using|working|deploying|optimizing|tuning|mounting|attaching|detaching|launching|terminating|migrating|reviewing|assigning|adding|removing|deleting|changing|verifying|testing|diagnosing)\b", t_low):
        phrase = topic[0].lower() + topic[1:] if not topic[:2].isupper() else topic
        templates = [
            f"In {source_label}, what is the recommended procedure for {phrase}?",
            f"How should an administrator handle {phrase} in {source_label}?",
            f"Explain the key steps and considerations for {phrase} in {source_label}.",
        ]
        return templates[idx % len(templates)]

    if (
        re.match(r"^(configure|manage|monitor|troubleshoot|view|create|connect|install|enable|disable|update|secure|filter|authenticate|back up|restore|set up|start|stop|check|use|deploy|optimize|mount|attach|detach|launch|terminate|migrate|verify|test|diagnose|expand|resize|modify|change|add|remove|delete|grant|revoke)\b", t_low)
        and not re.match(r"^(use|check|view|start|stop|update|change|mount|launch|filter|test)\s+(of|for|in|to|with|on|at)\b", t_low)
    ):
        phrase = topic[0].lower() + topic[1:] if not topic[:2].isupper() else topic
        templates = [
            f"How do you {phrase} in {source_label}?",
            f"What steps are documented in {source_label} to {phrase}?",
            f"In {source_label}, how can an IT engineer {phrase}?",
        ]
        return templates[idx % len(templates)]

    # Noun-phrase technical topics
    templates = [
        f"In {source_label}, how does {topic} work?",
        f"Explain {topic} in {source_label} based on the official documentation.",
        f"What should a system administrator know about {topic} in {source_label}?",
        f"Provide an overview of {topic} in {source_label}.",
    ]
    return templates[idx % len(templates)]


def extract_sections_from_manual(lines: List[str], fname: str) -> List[Tuple[str, str, bool]]:
    """Extract (topic, body_text, is_flag) tuples tailored to each manual's layout."""
    sections: List[Tuple[str, str, bool]] = []
    n = len(lines)

    if fname == "gcc_manual_clean.txt":
        # Extract both numbered sections and documented compiler flags (-W..., -f..., -O..., -g...)
        i = 0
        while i < n:
            ln = lines[i]
            is_flag = bool(re.match(r"^-[WfOgmd][A-Za-z0-9_=-]{2,35}$", ln))
            is_num_sec = bool(re.match(r"^\d+\.\d+(?:\.\d+)?\s+[A-Z][A-Za-z0-9 ,:/()-]{4,65}$", ln))
            if is_flag or is_num_sec:
                topic = ln if is_flag else _clean_heading_text(ln)
                body_lines: List[str] = []
                j = i + 1
                while j < n and len(body_lines) < 25:
                    nxt = lines[j]
                    if re.match(r"^-[WfOgmd][A-Za-z0-9_=-]{2,35}$", nxt) or re.match(r"^\d+\.\d+(?:\.\d+)?\s+[A-Z]", nxt):
                        break
                    body_lines.append(nxt)
                    j += 1
                body = _reflow_paragraph(body_lines)
                if is_flag or _is_valid_heading(topic):
                    sections.append((topic, body, is_flag))
                i = j
            else:
                i += 1
        return sections

    if fname in ("bash_manual_clean.txt", "postgresql_manual_clean.txt", "redhat_sysadmin_clean.txt"):
        # Numbered headings: "3.6.1 Redirecting Input", "9.8. Data Type Formatting", "1.2. CONFIGURING..."
        heading_re = re.compile(r"^(?:\d+(?:\.\d+){1,4}\.?\s+[A-Za-z][A-Za-z0-9 ,:/()'\"-]{4,75})$")
        i = 0
        while i < n:
            ln = lines[i]
            if heading_re.match(ln):
                topic = _clean_heading_text(ln)
                body_lines = []
                j = i + 1
                while j < n and len(body_lines) < 35:
                    nxt = lines[j]
                    if heading_re.match(nxt) or nxt.upper().startswith("CHAPTER "):
                        break
                    body_lines.append(nxt)
                    j += 1
                if _is_valid_heading(topic):
                    sections.append((topic, _reflow_paragraph(body_lines), False))
                i = j
            else:
                i += 1
        return sections

    # Standalone Title/Sentence-case headings (Ubuntu Server Guide & AWS EC2 User Guide)
    # Heuristic: short line (2–10 words), starts with capital, no trailing period/colon,
    # followed immediately by a full prose line starting with an uppercase letter.
    i = 0
    while i < n - 3:
        ln = lines[i]
        if (
            6 <= len(ln) <= 72
            and re.match(r"^[A-Z][A-Za-z0-9 /()-]+$", ln)
            and not ln.endswith((".", ",", ";", ":", "?"))
            and _is_valid_heading(ln)
        ):
            # Look ahead past at most 1 blank line for prose
            k = i + 1
            while k < n and not lines[k]:
                k += 1
            if k < n and len(lines[k]) >= 35 and re.match(r"^[A-Z][a-z]", lines[k]):
                body_lines = []
                j = k
                while j < n and len(body_lines) < 28:
                    nxt = lines[j]
                    # Stop at next heading candidate
                    if (
                        body_lines
                        and 6 <= len(nxt) <= 72
                        and re.match(r"^[A-Z][A-Za-z0-9 /()-]+$", nxt)
                        and not nxt.endswith((".", ",", ";", ":", "?"))
                        and _is_valid_heading(nxt)
                        and j + 1 < n
                        and len(lines[j + 1]) >= 35
                    ):
                        break
                    body_lines.append(nxt)
                    j += 1
                sections.append((ln.strip(), _reflow_paragraph(body_lines), False))
                i = j
                continue
        i += 1
    return sections


def build_instruction_dataset(
    corpus_dir: Path | str = "data/domain_corpus",
    output_dir: Path | str = "data",
    results_dir: Path | str = "results",
    target_per_doc: int = 100,
    seed: int = 42,
) -> Dict:
    """Build clean, balanced, deduplicated instruction dataset + 80/20 train/eval split."""
    corpus_path = Path(corpus_dir)
    out_path = Path(output_dir)
    res_path = Path(results_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    all_records: List[Dict[str, str]] = []
    seen_inst_hashes: set = set()
    seen_resp_hashes: set = set()
    per_source_counts: Dict[str, int] = {}

    for txt_file in sorted(corpus_path.glob("*_clean.txt")):
        fname = txt_file.name
        label = SOURCE_LABELS.get(fname, txt_file.stem.replace("_clean", "").replace("_", " ").title())
        raw = txt_file.read_text(encoding="utf-8", errors="replace")
        norm = normalize_pdf_text(raw)
        lines = clean_lines_and_skip_toc(norm, fname)
        raw_sections = extract_sections_from_manual(lines, fname)

        doc_candidates: List[Dict[str, str]] = []
        for idx, (topic, body, is_flag) in enumerate(raw_sections):
            clean_resp = _extract_clean_sentences(body, min_chars=180, max_chars=820)
            if not clean_resp:
                continue
            question = _build_question(topic, label, idx, is_flag=is_flag)
            inst_h = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()
            resp_h = hashlib.sha256(clean_resp[:250].strip().lower().encode("utf-8")).hexdigest()
            if inst_h in seen_inst_hashes or resp_h in seen_resp_hashes:
                continue
            seen_inst_hashes.add(inst_h)
            seen_resp_hashes.add(resp_h)
            doc_candidates.append({
                "instruction": question,
                "response": clean_resp,
                "source": fname,
                "topic": topic,
            })

        # Uniformly subsample across the document for broad topic coverage
        if len(doc_candidates) > target_per_doc:
            step = len(doc_candidates) / target_per_doc
            selected = [doc_candidates[int(i * step)] for i in range(target_per_doc)]
        else:
            selected = doc_candidates

        per_source_counts[fname] = len(selected)
        all_records.extend(selected)
        logger.info("Extracted %d clean pairs from %s (from %d valid candidates)", len(selected), fname, len(doc_candidates))

    # Deterministic shuffle and exact 80/20 split
    rng.shuffle(all_records)
    split_idx = int(round(len(all_records) * 0.80))
    train_records = all_records[:split_idx]
    eval_records = all_records[split_idx:]

    # Save JSONL deliverables
    full_file = out_path / "instruction_dataset.jsonl"
    train_file = out_path / "train.jsonl"
    eval_file = out_path / "eval.jsonl"

    for path, rows in [(full_file, all_records), (train_file, train_records), (eval_file, eval_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Compute summary statistics
    avg_inst_chars = round(sum(len(r["instruction"]) for r in all_records) / max(len(all_records), 1), 1)
    avg_resp_chars = round(sum(len(r["response"]) for r in all_records) / max(len(all_records), 1), 1)
    avg_words = round(sum(len((r["instruction"] + " " + r["response"]).split()) for r in all_records) / max(len(all_records), 1), 1)

    stats = {
        "total_pairs": len(all_records),
        "train_pairs": len(train_records),
        "eval_pairs": len(eval_records),
        "train_ratio": round(len(train_records) / max(len(all_records), 1), 4),
        "eval_ratio": round(len(eval_records) / max(len(all_records), 1), 4),
        "random_seed": seed,
        "generation_method": (
            "Section-anchored, sentence-boundary grounded extraction directly from the 6 cleaned "
            "domain corpus manuals (ligature normalization, hyphen rejoining, TOC/header stripping, "
            "strict sentence-start/end validation, and SHA-256 deduplication; zero external facts)."
        ),
        "avg_instruction_chars": avg_inst_chars,
        "avg_response_chars": avg_resp_chars,
        "avg_words_per_example": avg_words,
        "estimated_avg_tokens_per_example": round(avg_words * 1.33, 1),
        "per_source_counts": per_source_counts,
        "sample_pairs": all_records[:5],
    }

    stats_path = res_path / "instruction_dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(
        "Built clean dataset: %d total (%d train / %d eval) -> %s",
        len(all_records), len(train_records), len(eval_records), full_file,
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    base = Path(__file__).resolve().parents[2]
    stats = build_instruction_dataset(
        corpus_dir=base / "data" / "domain_corpus",
        output_dir=base / "data",
        results_dir=base / "results",
        target_per_doc=100,
        seed=42,
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "sample_pairs"}, indent=2))
    print("\n--- 5 SAMPLE PAIRS ---")
    for i, s in enumerate(stats["sample_pairs"], 1):
        print(f"\n[{i}] ({s['source']})\nQ: {s['instruction']}\nA: {s['response'][:260]}...")
