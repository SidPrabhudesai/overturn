"""Run the appeal pipeline on one file from the terminal (no server).

  python scripts/appeal_cli.py samples/mri_not_necessary.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.appeal import UnsupportedDocument, build_appeal  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.llm import LLMError, MetaLLM  # noqa: E402
from app.retrieval import LocalCorpusRetriever  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    s = load_settings()
    retr = LocalCorpusRetriever.from_dir(s.corpus_dir)
    if not len(retr):
        print("NOTE: corpus is empty, so no appeal-rights section will appear. See README (appeal corpus).\n")
    try:
        r = build_appeal(MetaLLM(s), retr, Path(sys.argv[1]).read_bytes())
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
