"""Turn a Report into a WhatsApp-sized reply (Twilio caps WhatsApp bodies at 1600 chars)."""
from __future__ import annotations

from .pipeline import Report

MAX_REPLY_CHARS = 1500

LABELS = {
    "supported": "✅ Supported by evidence",
    "contradicted": "❌ Contradicted by evidence",
    "misleading": "⚠️ Misleading or missing context",
    "insufficient_evidence": "❓ Not enough evidence in my sources",
}

HELP = (
    "Send me a health claim you've seen (for example, something forwarded to you) and I'll check it "
    "against public-health sources like WHO, CDC, NIH and NHS."
)
FOOTER = "General information only, not medical advice. Talk to a clinician about your own situation."


def format_report(report: Report) -> str:
    parts: list[str] = list(report.safety_notes)
    if not report.results:
        parts.append(HELP)
        return "\n\n".join(parts)[:MAX_REPLY_CHARS]

    for r in report.results:
        block = [f"*Claim:* {r.claim}", f"*Verdict:* {LABELS[r.verdict]}"]
        if r.explanation:
            block.append(r.explanation)
        urls = list(dict.fromkeys(p.url for p in r.sources))[:2]
        if urls:
            block.append("Sources: " + " ".join(urls))
        parts.append("\n".join(block))
    parts.append(FOOTER)

    reply = "\n\n".join(parts)
    if len(reply) <= MAX_REPLY_CHARS:
        return reply
    # Keep the footer; trim the middle rather than cut off the disclaimer.
    keep = MAX_REPLY_CHARS - len(FOOTER) - 6
    return reply[:keep].rstrip() + "…\n\n" + FOOTER
