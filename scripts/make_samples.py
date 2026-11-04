"""Make fictional denial-letter PDFs (invented payers, no real people) for testing and the demo.

  python scripts/make_samples.py        # writes samples/*.pdf

Open one and screenshot it to test photo upload too. Everything in them is made up.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

LETTERS = {
    "mri_not_necessary": ("Northwind Health Plan", "MRI of the left knee", "the service was not medically necessary",
                          "CO-50", "March 14, 2027", "NW-2291847", "PPO"),
    "prior_auth_missing": ("Bluestem Insurance", "outpatient physical therapy (12 sessions)", "prior authorization was not obtained",
                           "CO-197", "April 2, 2027", "BS-5540021", "HMO"),
    "out_of_network": ("Harborview Mutual", "emergency room visit on 01/09/2027", "the facility was out of network",
                       "CO-242", "February 28, 2027", "HM-7782310", "PPO"),
    "experimental": ("Cedar Ridge Health", "genetic panel testing", "the service is considered experimental or investigational",
                     "CO-55", "May 20, 2027", "CR-3309915", "EPO"),
    "not_covered": ("Summit Valley Plan", "durable medical equipment (CPAP machine)", "the item is not a covered benefit under your plan",
                    "CO-96", "June 9, 2027", "SV-6614402", "HDHP"),
    "frequency_limit": ("Lakeshore Health Alliance", "chiropractic visits 13 through 20", "the plan limit of 12 visits per year was reached",
                        "CO-119", "January 31, 2027", "LH-1175530", "PPO"),
}


def body(payer, service, reason, code, deadline, claim, plan) -> str:
    return (
        f"{payer}\nMember Services - Appeals Department\n\nNOTICE OF ADVERSE BENEFIT DETERMINATION\n\n"
        f"Claim number: {claim}\nPlan type: {plan}\n\n"
        f"Dear Member,\n\nWe have reviewed the claim for {service}. We are unable to approve this claim because {reason}. "
        f"Reason code: {code}.\n\n"
        f"If you disagree with this decision you have the right to appeal. Your appeal must be received by {deadline}. "
        "You may request, free of charge, copies of the documents and criteria used to make this decision.\n\n"
        "Sincerely,\nClaims Review Team"
    )


def make_pdf(text: str) -> bytes:
    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, 90) or [""])
    esc = lambda t: t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = "BT /F1 11 Tf 60 740 Td 15 TL " + " ".join(f"({esc(t)}) Tj T*" for t in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1", "replace")
    x = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n".encode()


def main() -> None:
    d = Path(__file__).resolve().parent.parent / "samples"
    d.mkdir(exist_ok=True)
    for name, args in LETTERS.items():
        (d / f"{name}.pdf").write_bytes(make_pdf(body(*args)))
    print(f"wrote {len(LETTERS)} fictional letters to {d}")


if __name__ == "__main__":
    main()
