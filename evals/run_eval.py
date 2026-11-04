"""Evaluate the verifier on evals/claims.jsonl in three modes.

  ungrounded  model's own knowledge, no retrieval
  grounded    retrieval over trusted sources, guard OFF (verdicts accepted even without citations)
  guarded     retrieval over trusted sources, guard ON (the shipped pipeline)

Metrics (verdict only; claim extraction is bypassed to isolate verification):
  accuracy            correct / all claims (abstaining counts as not correct)
  accuracy_answered   correct / claims where the bot gave a verdict
  abstain_rate        share answered "insufficient_evidence"
  flip_rate           share where a TRUE claim was called contradicted or a FALSE one supported

Usage:
  python evals/run_eval.py --check-data
  python evals/run_eval.py --mode all --out evals/results
The labels are one person's reading of general public-health knowledge: review them before quoting results.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.llm import LLM, MetaLLM  # noqa: E402
from app.pipeline import INSUFFICIENT, verify_claim  # noqa: E402
from app.retrieval import LocalCorpusRetriever, Retriever  # noqa: E402

LABELS = ("supported", "contradicted", "misleading")
MODES = ("ungrounded", "grounded", "guarded")


def load_claims(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate claim ids")
    for r in rows:
        if r["label"] not in LABELS:
            raise ValueError(f"{r['id']}: bad label {r['label']!r}")
    return rows


def score(rows: list[dict]) -> dict:
    """rows: dicts with 'label' and 'pred'."""
    n = len(rows)
    if n == 0:
        raise ValueError("no rows to score")
    answered = [r for r in rows if r["pred"] != INSUFFICIENT]
    correct = sum(r["pred"] == r["label"] for r in rows)
    flips = sum(
        (r["label"] == "supported" and r["pred"] == "contradicted")
        or (r["label"] == "contradicted" and r["pred"] == "supported")
        for r in rows
    )
    confusion: dict[str, Counter] = {lab: Counter() for lab in LABELS}
    for r in rows:
        confusion[r["label"]][r["pred"]] += 1
    return {
        "n": n,
        "accuracy": correct / n,
        "accuracy_answered": (sum(r["pred"] == r["label"] for r in answered) / len(answered)) if answered else None,
        "abstain_rate": 1 - len(answered) / n,
        "flip_rate": flips / n,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def run_mode(mode: str, claims: list[dict], llm: LLM, retriever: Retriever | None, workers: int = 8) -> list[dict]:
    def one(c: dict) -> dict:
        try:
            if mode == "ungrounded":
                res = verify_claim(llm, c["claim"], None)
            else:
                assert retriever is not None
                res = verify_claim(llm, c["claim"], retriever.search(c["claim"], 4), guard=(mode == "guarded"))
            pred, note = res.verdict, res.note
        except Exception as exc:  # record failures instead of aborting a long run
            pred, note = INSUFFICIENT, f"error: {type(exc).__name__}"
        return {**c, "pred": pred, "note": note}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, claims))


def to_markdown(results: dict[str, dict], model: str, n_passages: int | None) -> str:
    lines = [
        f"Model: `{model}`. Corpus passages: {n_passages if n_passages is not None else 'n/a'}. Claims: "
        f"{next(iter(results.values()))['n']}.",
        "",
        "| Mode | Accuracy | Accuracy when answered | Abstain rate | Flip rate |",
        "|---|---|---|---|---|",
    ]
    for mode, s in results.items():
        aa = "n/a" if s["accuracy_answered"] is None else f"{s['accuracy_answered']:.0%}"
        lines.append(f"| {mode} | {s['accuracy']:.0%} | {aa} | {s['abstain_rate']:.0%} | {s['flip_rate']:.0%} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default=str(ROOT / "evals" / "claims.jsonl"))
    ap.add_argument("--mode", choices=(*MODES, "all"), default="all")
    ap.add_argument("--limit", type=int, default=0, help="only the first N claims (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "evals" / "results"))
    ap.add_argument("--check-data", action="store_true", help="validate the claims file and exit")
    args = ap.parse_args()

    claims = load_claims(Path(args.claims))
    if args.check_data:
        print(f"{len(claims)} claims OK: {dict(Counter(c['label'] for c in claims))}")
        return 0
    if args.limit:
        claims = claims[: args.limit]

    settings = load_settings()
    if not settings.api_key:
        print("MODEL_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 2
    llm = MetaLLM(settings)
    modes = MODES if args.mode == "all" else (args.mode,)
    retriever: LocalCorpusRetriever | None = None
    if any(m != "ungrounded" for m in modes):
        retriever = LocalCorpusRetriever.from_dir(settings.corpus_dir)
        if not len(retriever):
            print("Corpus is empty. Run scripts/build_corpus.py first.", file=sys.stderr)
            return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for mode in modes:
        print(f"running {mode} on {len(claims)} claims...", file=sys.stderr)
        rows = run_mode(mode, claims, llm, retriever, args.workers)
        (out / f"{mode}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        results[mode] = score(rows)
    (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = to_markdown(results, settings.model, len(retriever) if retriever else None)
    (out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
