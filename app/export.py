"""Turn an AppealResult into a file a person can open, edit and send: Word (.docx), PDF, or Markdown.

One list of blocks is built once and rendered three ways, so the files always say the same thing. python-docx and
reportlab are imported only when that format is asked for, so the app still runs without them.

Block kinds: ("title", text) ("h", text) ("p", text) ("note", text) ("li", text) ("table", rows) ("quote", text)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .appeal import AppealResult
from .phi import phi_message
from .playbooks import playbook_for

Block = tuple[str, Any]

FIELD_LABELS = {
    "payer": "Insurer", "service": "Service denied", "denial_reason": "Reason for denial",
    "denial_codes": "Denial codes", "appeal_deadline": "Appeal deadline", "claim_number": "Claim number",
    "plan_type": "Plan type",
}
URGENCY_TEXT = {
    "expired": "The deadline in the letter appears to have PASSED. Check the date and ask the plan about late appeals.",
    "urgent": "Deadline is close. Treat this as the first appeal to file.",
    "ok": "Deadline is not immediate, but calendar it now.",
    "unknown": "No usable deadline found in the letter. Check the letter and the plan's appeal instructions.",
}


def build_blocks(r: AppealResult, title: str = "Overturn appeal draft") -> list[Block]:
    b: list[Block] = [("title", title)]
    b.append(("note", "Draft prepared with AI assistance. Review every fact against the original letter before filing."))

    # 1. Deadline first: it is the one thing that cannot slip.
    b.append(("h", "Deadline"))
    if r.deadline_text:
        when = f" ({r.deadline_iso}, {r.days_left} days from today)" if r.deadline_iso and r.days_left is not None else ""
        b.append(("p", f"{r.deadline_text}{when}"))
    else:
        b.append(("p", "The letter does not state a deadline."))
    b.append(("note", URGENCY_TEXT.get(r.urgency, URGENCY_TEXT["unknown"])))

    # 2. Things to check before anything is sent.
    flags: list[str] = []
    if r.phi_flags:
        flags.append(phi_message(r.phi_flags))
    if r.suspicious:
        flags.append("The document contains text that looks like instructions to an AI. It was ignored. Read the original.")
    for m in r.mismatches:
        flags.append(f"Letter and EOB disagree on {m.field}: letter says '{m.letter_value}', EOB says '{m.eob_value}'.")
    if r.not_found:
        flags.append("Not found or not verifiable in the letter: " + ", ".join(FIELD_LABELS.get(f, f) for f in r.not_found) + ".")
    if flags:
        b.append(("h", "Check before sending"))
        b.extend(("li", f) for f in flags)

    # 3. Facts, each with the words it came from.
    b.append(("h", "Facts from the letter"))
    rows = [["Item", "Found", "Quoted from the letter"]]
    for k, f in r.facts.items():
        rows.append([FIELD_LABELS.get(k, k), f.value, f'"{f.quote}"'])
    b.append(("table", rows))

    # 4. The draft itself.
    b.append(("h", "Draft appeal letter"))
    for para in (r.letter or "(no letter was produced)").split("\n\n"):
        b.append(("p", para.strip()))

    # 5. Reviewer notes by denial type.
    pb = playbook_for(r.checklist_kind)
    b.append(("h", f"Reviewer notes: {pb.title}"))
    b.append(("note", "Suggestions for the reviewer. These are not facts about the patient and are not in the letter."))
    b.append(("p", "Points the appeal should address:"))
    b.extend(("li", x) for x in pb.address)
    b.append(("p", "Ask the clinician for:"))
    b.extend(("li", x) for x in pb.ask_clinician)
    b.append(("p", "Ask the plan for:"))
    b.extend(("li", x) for x in pb.ask_plan)

    b.append(("h", "Attachment checklist"))
    b.extend(("li", "[ ] " + x) for x in (r.checklist or ()))

    if r.rights:
        b.append(("h", "Appeal rights (from official sources)"))
        for x in r.rights:
            urls = ", ".join(dict.fromkeys(p.url for p in x.sources))
            b.append(("li", f"{x.statement}  Source: {urls}"))

    b.append(("note", r.disclaimer))
    return b


def to_markdown(r: AppealResult, title: str = "Overturn appeal draft") -> str:
    out: list[str] = []
    for kind, v in build_blocks(r, title):
        if kind == "title":
            out.append(f"# {v}")
        elif kind == "h":
            out.append(f"\n## {v}")
        elif kind in ("p", "quote"):
            out.append(v)
        elif kind == "note":
            out.append(f"*{v}*")
        elif kind == "li":
            out.append(f"- {v}")
        elif kind == "table":
            head, *body = v
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + "---|" * len(head))
            out.extend("| " + " | ".join(c.replace("|", "/") for c in row) + " |" for row in body)
    return "\n".join(out) + "\n"


def to_docx(r: AppealResult, path: str | Path, title: str = "Overturn appeal draft") -> Path:
    from docx import Document
    from docx.shared import Pt, RGBColor

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    for kind, v in build_blocks(r, title):
        if kind == "title":
            doc.add_heading(v, level=0)
        elif kind == "h":
            doc.add_heading(v, level=2)
        elif kind == "p":
            doc.add_paragraph(v)
        elif kind == "note":
            run = doc.add_paragraph().add_run(v)
            run.italic = True
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x56, 0x63, 0x73)
        elif kind == "li":
            doc.add_paragraph(v, style="List Bullet")
        elif kind == "table":
            t = doc.add_table(rows=len(v), cols=len(v[0]))
            t.style = "Light Grid Accent 1"
            for i, row in enumerate(v):
                for j, cell in enumerate(row):
                    t.cell(i, j).text = cell
    p = Path(path)
    doc.save(str(p))
    return p


def to_pdf(r: AppealResult, path: str | Path, title: str = "Overturn appeal draft") -> Path:
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    base = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=14)
    note = ParagraphStyle("n", parent=body, fontSize=8.5, textColor=colors.HexColor("#566373"))
    h1 = ParagraphStyle("t", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("h", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=10,
                        spaceAfter=4, textColor=colors.HexColor("#1f5fbf"))
    flow: list = []
    for kind, v in build_blocks(r, title):
        if kind == "title":
            flow.append(Paragraph(escape(v), h1))
        elif kind == "h":
            flow.append(Paragraph(escape(v), h2))
        elif kind == "p":
            flow.append(Paragraph(escape(v).replace("\n", "<br/>"), body))
            flow.append(Spacer(1, 4))
        elif kind == "note":
            flow.append(Paragraph(escape(v), note))
        elif kind == "li":
            flow.append(Paragraph("&bull; " + escape(v), body))
        elif kind == "table":
            t = Table([[Paragraph(escape(c), body) for c in row] for row in v], colWidths=[100, 130, 270])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef9")),
                                   ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#d9e0e7")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            flow.append(t)
    p = Path(path)
    SimpleDocTemplate(str(p), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=48, bottomMargin=48,
                      title=title).build(flow)
    return p


def write_all(r: AppealResult, out_dir: str | Path, stem: str = "appeal-draft") -> list[Path]:
    """Write .md always, and .docx / .pdf when their libraries are installed. Returns the files written."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    md = d / f"{stem}.md"
    md.write_text(to_markdown(r), encoding="utf-8")
    written.append(md)
    for fn, ext in ((to_docx, "docx"), (to_pdf, "pdf")):
        try:
            written.append(fn(r, d / f"{stem}.{ext}"))
        except ImportError:
            pass
    return written
