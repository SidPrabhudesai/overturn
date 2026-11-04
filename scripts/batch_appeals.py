"""Process a folder of denial letters (PDF/PNG/JPG/WebP/GIF) into a summary CSV and one draft letter each.

  python scripts/batch_appeals.py path/to/letters out_folder

out_folder/summary.csv is sorted soonest-deadline first. out_folder/letters/*.txt are drafts for a person to review.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.batch import MAX_FILES, run_batch, safe_filename, summary_row, to_csv  # noqa: E402

EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}


def process(folder: Path, out: Path, llm, retriever, workers: int = 3) -> int:
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        print(f"No supported files in {folder}")
        return 1
    if len(files) > MAX_FILES:
        print(f"{len(files)} files found; processing the first {MAX_FILES}. Run again for the rest.")
        files = files[:MAX_FILES]
    items = run_batch(llm, retriever, [(p.name, p.read_bytes()) for p in files], workers)
    (out / "letters").mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    for it in items:
        if it.result and it.result.letter:
            (out / "letters" / safe_filename(it.name, taken)).write_text(it.result.letter + "\n\n" + it.result.disclaimer + "\n")
    (out / "summary.csv").write_text(to_csv([summary_row(i) for i in items]))
    print(f"Processed {len(items)} files -> {out}/summary.csv (soonest deadline first) and {out}/letters/")
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    from app.config import load_settings
    from app.llm import MetaLLM
    from app.retrieval import LocalCorpusRetriever

    s = load_settings()
    return process(Path(sys.argv[1]), Path(sys.argv[2]), MetaLLM(s), LocalCorpusRetriever.from_dir(s.corpus_dir))


if __name__ == "__main__":
    raise SystemExit(main())
