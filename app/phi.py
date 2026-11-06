"""Deterministic check for personal details that should have been redacted before a letter is sent anywhere.

It never calls a model, and it never returns the sensitive values themselves: only the KIND of detail found and a
masked preview, so a log line or error message can't leak what it was meant to protect. It is a safety net, not a
guarantee: it catches the common labelled patterns (member ID, date of birth, SSN, patient name, phone).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_REDACTED = re.compile(r"(x{3,}|\*{3,}|_{3,}|#{3,}|redact|\[.*\]|<.*>|n/?a\b)", re.I)

_DATE = r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}|\d{4}-\d{2}-\d{2})"

# (kind, human label, pattern with the sensitive value in group 1)
_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("ssn", "Social Security number", re.compile(r"\b(\d{3}-\d{2}-\d{4})\b")),
    ("dob", "date of birth",
     re.compile(r"(?i)\b(?:date of birth|d\.?o\.?b\.?|born(?: on)?)\b[\s:.\-]{0,6}(" + _DATE + r")")),
    ("member_id", "member or policy ID",
     re.compile(r"(?i)\b(?:member|subscriber|policy|insured|group)\s*(?:id|number|no\.?|#)\s*[:#\-]?\s*"
                r"([A-Z0-9][A-Z0-9\-]{5,})")),
    ("patient_name", "patient name",
     re.compile(r"(?im)^[ \t]*(?:patient|member|subscriber|insured|beneficiary)(?:[ \t]*name)?[ \t]*:[ \t]*"
                r"([A-Z][A-Za-z'\-]+(?:[ \t]+[A-Z][A-Za-z'\-\.]*){1,3})")),
    ("salutation_name", "patient name in the greeting",
     re.compile(r"(?m)^[ \t]*Dear[ \t]+(?:Mr\.|Ms\.|Mrs\.|Mx\.)?[ \t]*([A-Z][a-z]+[ \t]+[A-Z][a-z]+)[ \t]*[,:]")),
    ("phone", "phone number",
     re.compile(r"(?i)\b(?:phone|tel|cell|mobile)\b[^\n\d]{0,12}(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})")),
)

# Words that follow "Dear" in a form letter and are not a person's name.
_NOT_NAMES = {"member services", "claims department", "appeals department", "valued member", "health plan"}


@dataclass(frozen=True)
class PHIFinding:
    kind: str
    label: str
    preview: str      # masked, e.g. "AB****"


def _mask(value: str) -> str:
    v = value.strip()
    return (v[:2] + "*" * max(len(v) - 2, 3)) if len(v) > 2 else "***"


def scan_phi(text: str) -> tuple[PHIFinding, ...]:
    """Every labelled personal detail in `text` that has not been blanked out. Empty tuple means nothing found."""
    found: list[PHIFinding] = []
    seen: set[tuple[str, str]] = set()
    for kind, label, pat in _PATTERNS:
        for m in pat.finditer(text or ""):
            value = m.group(1)
            if _REDACTED.search(value) or value.lower() in _NOT_NAMES:
                continue
            key = (kind, value.lower())
            if key in seen:
                continue
            seen.add(key)
            found.append(PHIFinding(kind, label, _mask(value)))
    return tuple(found)


def phi_kinds(text: str) -> tuple[str, ...]:
    """Just the kinds, in a stable order. Safe to store, log or show."""
    return tuple(sorted({f.kind for f in scan_phi(text)}))


def phi_message(kinds: tuple[str, ...]) -> str:
    labels = {k: lab for k, lab, _ in _PATTERNS}
    names = ", ".join(labels.get(k, k) for k in kinds)
    return (f"This letter still seems to contain personal details ({names}). "
            "Please black them out, then send it again.")
