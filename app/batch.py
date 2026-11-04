"""Batch mode: many denial letters in, one summary out, soonest deadline first, plus a draft letter per file.

This is what a practice's biller would use. The AI drafts; a person still reviews every row, and rows that need
attention (missing facts, document conflicts, suspicious text, errors) say so.
"""
from __future__ import annotations

import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any

from .appeal import AppealResult, TextLLM, UnsupportedDocument, build_appeal
from .retrieval import Retriever

MAX_FILES = 20
URGENCY_ORDER = {"urgent": 0, "ok": 1, "unknown": 2, "expired": 3}
COLUMNS = ["file", "status", "insurer", "claim_number", "denial_code", "denial_reason", "deadline", "days_left",
           "urgency", "needs_review"]


@dataclass(frozen=True)
class BatchItem:
    name: str
    result: AppealResult | None = None
    error: str | None = None


def _one(llm: TextLLM, retr: Retriever, name: str, data: bytes, today: date | None) -> BatchItem:
    try:
        return BatchItem(name, result=build_appeal(llm, retr, data, today=today))
    except UnsupportedDocument as exc:
        return BatchItem(name, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad file must not sink the batch; don't leak details
        return BatchItem(name, error=f"could not process ({type(exc).__name__})")


def run_batch(llm: TextLLM, retr: Retriever, files: list[tuple[str, bytes]], workers: int = 3,
              today: date | None = None) -> list[BatchItem]:
    if len(files) > MAX_FILES:
        raise ValueError(f"At most {MAX_FILES} files per batch.")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        items = list(ex.map(lambda f: _one(llm, retr, f[0], f[1], today), files))
    return sort_items(items)


def sort_items(items: list[BatchItem]) -> list[BatchItem]:
    """Urgent first (soonest), then ok (soonest), then unknown deadline, then expired, then errors."""
    def key(it: BatchItem):
        if it.result is None:
            return (9, 0, it.name)
        r = it.result
        return (URGENCY_ORDER.get(r.urgency, 2), r.days_left if r.days_left is not None else 0, it.name)
    return sorted(items, key=key)


def review_flags(r: AppealResult) -> list[str]:
    flags: list[str] = []
    if r.not_found:
        flags.append("missing: " + ", ".join(r.not_found))
    if r.mismatches:
        flags.append("documents disagree")
    if r.suspicious:
        flags.append("instruction-like text in document")
    if r.urgency == "expired":
        flags.append("deadline appears passed")
    if r.urgency == "unknown":
        flags.append("no readable deadline")
    return flags


def summary_row(it: BatchItem) -> dict[str, Any]:
    if it.result is None:
        return {"file": it.name, "status": "error", "insurer": "", "claim_number": "", "denial_code": "",
                "denial_reason": "", "deadline": "", "days_left": "", "urgency": "", "needs_review": it.error or "error"}
    r, v = it.result, (lambda k: it.result.facts[k].value if k in it.result.facts else "")
    return {"file": it.name, "status": "drafted" if r.letter else "no letter", "insurer": v("payer"),
            "claim_number": v("claim_number"), "denial_code": v("denial_codes"), "denial_reason": v("denial_reason"),
            "deadline": r.deadline_iso or v("appeal_deadline"), "days_left": "" if r.days_left is None else r.days_left,
            "urgency": r.urgency, "needs_review": "; ".join(review_flags(r)) or "standard review"}


_FORMULA = re.compile(r"^[=+\-@\t\r]")


def _safe_cell(v: Any) -> str:
    """Cells come from untrusted documents. Neutralise spreadsheet formulas (CSV injection)."""
    s = str(v)
    return "'" + s if _FORMULA.match(s) else s


def to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: _safe_cell(r.get(c, "")) for c in COLUMNS})
    return buf.getvalue()


def safe_filename(name: str, taken: set[str], suffix: str = ".txt") -> str:
    leaf = name.replace("\\", "/").split("/")[-1]
    leaf = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", leaf)          # drop the extension first
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._-")[:80].strip("._-") or "file"
    out, n = base + suffix, 2
    while out in taken:
        out, n = f"{base}_{n}{suffix}", n + 1
    taken.add(out)
    return out
