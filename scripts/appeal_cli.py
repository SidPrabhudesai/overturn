"""Run the appeal pipeline on one file from the terminal (no server).

  python scripts/appeal_cli.py samples/mri_not_necessary.pdf
  python scripts/appeal_cli.py letter.pdf --out out/        # also writes .docx, .pdf and .md
  python scripts/appeal_cli.py letter.pdf --strict          # refuse a PDF/Word file that still shows a member ID, DOB, SSN or name
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.appeal import PHIDetected, UnsupportedDocument, build_appeal  # noqa: E402
from app.export import write_all  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.llm import LLMError, MetaLLM  # noqa: E402
from app.retrieval import LocalCorpusRetriever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Draft an appeal from one denial letter.")
    ap.add_argument("file")
    ap.add_argument("--out", help="folder to write the .docx, .pdf and .md into")
    ap.add_argument("--strict", action="store_true", help="refuse files that still show personal details")
    args = ap.parse_args()
    s = load_settings()
    retr = LocalCorpusRetriever.from_dir(s.corpus_dir)
    if not len(retr):
        print("NOTE: corpus is empty, so no appeal-rights section will appear. See README (appeal corpus).\n")
    try:
        r = build_appeal(MetaLLM(s), retr, Path(args.file).read_bytes(), strict_phi=args.strict)
    except PHIDetected as exc:
        print(f"NOT SENT TO THE AI: {exc}")
        return 3
    except (UnsupportedDocument, LLMError) as exc:
        print(f"FAILED: {exc}")
        return 1
    print("FACTS (each backed by a quote from the letter):")
    for k, f in r.facts.items():
        print(f"  {k}: {f.value}   <- \"{f.quote}\"")
    print("NOT FOUND / UNVERIFIED:", ", ".join(r.not_found) or "none")
    print("\nRIGHTS:")
    for x in r.rights:
        print(f"  - {x.statement}  [{', '.join(p.url for p in x.sources)}]")
    print("\nLETTER:\n" + r.letter + "\n\n" + r.disclaimer)
    if r.phi_flags:
        print("\nWARNING: personal details still visible in the letter: " + ", ".join(r.phi_flags))
    if args.out:
        for f in write_all(r, args.out, Path(args.file).stem + "-appeal"):
            print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
