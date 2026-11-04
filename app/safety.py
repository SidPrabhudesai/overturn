"""Deterministic safety rules. These never depend on the model."""
from __future__ import annotations

import re

_CRISIS = re.compile(
    r"\b(kill myself|end my life|want to die|thinking (about|of) suicide|hurt myself|hurting myself)\b",
    re.I,
)
_EMERGENCY = re.compile(
    r"\b(can'?t|cannot) breathe\b"
    r"|\b(i|he|she|they|my \w+|someone|friend)\b[^.?!]{0,40}\boverdos(e|ed|ing)\b"
    r"|\b(i|he|she|they|my \w+|someone|friend)\b[^.?!]{0,30}\b(swallowed|drank|ate)\b[^.?!]{0,30}\b(bleach|poison|antifreeze)\b"
    r"|\b(having|got|feel|feeling)\b[^.?!]{0,20}\bchest pain\b"
    r"|\b(unconscious|not breathing|seizure right now)\b",
    re.I,
)

CRISIS_NOTE = (
    "If you're thinking about hurting yourself, you don't have to handle it alone. In the US you can "
    "call or text 988 (Suicide & Crisis Lifeline); elsewhere, contact your local emergency number."
)
EMERGENCY_NOTE = (
    "If this is happening now, call your local emergency number (911 in the US) instead of "
    "relying on a chat bot."
)


def safety_notes(text: str) -> list[str]:
    notes = []
    if _CRISIS.search(text):
        notes.append(CRISIS_NOTE)
    if _EMERGENCY.search(text):
        notes.append(EMERGENCY_NOTE)
    return notes
