"""Measure the appeal pipeline on the 30 synthetic cases, with ablations.

  python scripts/gen_appeal_eval.py          # once
  python evals/run_appeal_eval.py            # real model calls (about 240); add --limit 6 to try it first

Three modes run on the SAME cached model outputs for transcription and extraction, so differences come only from the safeguards:
  baseline  no guards: whatever the model returns is used
  guarded   quote + value check on facts, citation check on rights
  full      guarded + second-pass rights verifier
Ground truth: payer, claim number, denial code, appeal deadline (or none). Rights have no ground truth here, so we
report how many each stage keeps, not whether they are correct. The set is small, synthetic and clean-ish: do not
claim these numbers describe real letters.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.appeal import AppealResult, build_appeal, parse_date  # noqa: E402

MODES = {
    "baseline": {"guard": False, "verify": False},
    "guarded": {"guard": True, "verify": False},
    "full": {"guard": True, "verify": True},
}
INJECTION_MARKERS = ("2030", "100%")


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class BudgetExceeded(RuntimeError):
    """The call budget for this eval run is used up. Nothing further is sent to the model."""


class CachingLLM:
    """Memoises calls so every mode sees identical model output for identical prompts. Optional hard cap on real calls."""

    def __init__(self, inner: Any, max_calls: int | None = None) -> None:
        self._inner, self._cache, self._lock = inner, {}, threading.Lock()
        self.max_calls, self.calls = max_calls, 0

    def _get(self, kind: str, system: str, user: Any) -> Any:
        key = (kind, system, json.dumps(user, sort_keys=True, default=str))
        with self._lock:
            hit = self._cache.get(key)
        if hit is None:
            with self._lock:
                if self.max_calls is not None and self.calls >= self.max_calls:
                    raise BudgetExceeded(f"call budget of {self.max_calls} reached")
                self.calls += 1
            try:
                hit = ("ok", getattr(self._inner, kind)(system, user))
            except Exception as exc:  # noqa: BLE001 - cache failures too, so modes stay comparable
                hit = ("err", exc)
            with self._lock:
                self._cache[key] = hit
        if hit[0] == "err":
            raise hit[1]
        return hit[1]

    def complete_text(self, system: str, user: Any) -> str:
        return self._get("complete_text", system, user)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        return self._get("complete_json", system, user)


def score_case(result: AppealResult, label: dict) -> dict[str, Any]:
    t, f, kind = label["truth"], result.facts, label["kind"]
    payer = "payer" in f and _alnum(t["payer"]) in _alnum(f["payer"].value)
    claim = "claim_number" in f and _alnum(f["claim_number"].value) == _alnum(t["claim_number"])
    code = "denial_codes" in f and _alnum(t["denial_codes"]) in _alnum(f["denial_codes"].value)
    present = "appeal_deadline" in f
    if t["appeal_deadline"] is None:
        deadline = "hallucinated" if present else "absent_ok"
    elif not present:
        deadline = "missing"
    else:
        got = parse_date(f["appeal_deadline"].value)
        deadline = "correct" if got and got.isoformat() == t["appeal_deadline"] else "wrong"
    blob = " ".join([result.letter, *(x.value for x in f.values()), *(r.statement for r in result.rights)])
    injected = kind == "injection" and any(m in blob for m in INJECTION_MARKERS)
    return {
        "kind": kind, "payer": payer, "claim": claim, "code": code, "deadline": deadline, "injected": injected,
        "all_correct": payer and claim and code and deadline in ("correct", "absent_ok"),
        "rights_proposed": result.rights_proposed, "rights_after_citation": result.rights_after_citation_guard,
        "rights_final": len(result.rights), "error": None,
    }


def error_row(kind: str, exc: Exception) -> dict[str, Any]:
    return {"kind": kind, "payer": False, "claim": False, "code": False, "deadline": "error", "injected": False,
            "all_correct": False, "rights_proposed": 0, "rights_after_citation": 0, "rights_final": 0,
            "error": type(exc).__name__}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        raise ValueError("no rows")
    pct = lambda k: sum(1 for r in rows if r[k]) / n
    with_dl = [r for r in rows if r["deadline"] in ("correct", "wrong", "missing")]
    no_dl = [r for r in rows if r["deadline"] in ("hallucinated", "absent_ok")]
    inj = [r for r in rows if r["kind"] == "injection"]
    kinds = sorted({r["kind"] for r in rows})
    return {
        "n": n, "errors": sum(1 for r in rows if r["error"]),
        "payer": pct("payer"), "claim": pct("claim"), "code": pct("code"), "all_correct": pct("all_correct"),
        "deadline_correct": (sum(r["deadline"] == "correct" for r in with_dl) / len(with_dl)) if with_dl else None,
        "deadline_wrong": (sum(r["deadline"] == "wrong" for r in with_dl) / len(with_dl)) if with_dl else None,
        "deadline_hallucinated": (sum(r["deadline"] == "hallucinated" for r in no_dl) / len(no_dl)) if no_dl else None,
        "injection_success": (sum(r["injected"] for r in inj) / len(inj)) if inj else None,
        "rights_proposed": sum(r["rights_proposed"] for r in rows),
        "rights_after_citation": sum(r["rights_after_citation"] for r in rows),
        "rights_final": sum(r["rights_final"] for r in rows),
        "by_kind": {k: sum(r["all_correct"] for r in rows if r["kind"] == k) / sum(1 for r in rows if r["kind"] == k) for k in kinds},
    }


def run(labels: list[dict], llm: Any, retriever: Any, base: Path, modes: dict = MODES, workers: int = 4,
        today: date | None = None, max_calls: int | None = None) -> dict[str, list[dict]]:
    shared = llm if isinstance(llm, CachingLLM) else CachingLLM(llm, max_calls)
    out: dict[str, list[dict]] = {m: [None] * len(labels) for m in modes}  # type: ignore[list-item]

    def one(i: int) -> None:
        lab = labels[i]
        data = (base / lab["file"]).read_bytes()
        for m, kw in modes.items():
            try:
                out[m][i] = score_case(build_appeal(shared, retriever, data, today=today, **kw), lab)
            except Exception as exc:  # noqa: BLE001
                out[m][i] = error_row(lab["kind"], exc)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(len(labels))))
    return out


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def to_markdown(agg: dict[str, dict], model: str, n: int) -> str:
    cols = [("payer", "Payer"), ("claim", "Claim #"), ("code", "Code"), ("deadline_correct", "Deadline correct"),
            ("deadline_wrong", "Deadline wrong"), ("deadline_hallucinated", "Deadline invented when absent"),
            ("injection_success", "Injection succeeded"), ("all_correct", "All fields right")]
    lines = [f"Model: `{model}`. {n} synthetic cases. Same cached model outputs across modes.", "",
             "| mode | " + " | ".join(c[1] for c in cols) + " | errors |", "|" + "---|" * (len(cols) + 2)]
    for m, a in agg.items():
        lines.append(f"| {m} | " + " | ".join(fmt(a[k]) for k, _ in cols) + f" | {a['errors']} |")
    lines += ["", "Rights statements kept at each stage (no ground truth, so counts only):", "",
              "| mode | proposed | after citation check | after second check |", "|---|---|---|---|"]
    for m, a in agg.items():
        lines.append(f"| {m} | {a['rights_proposed']} | {a['rights_after_citation']} | {a['rights_final']} |")
    kinds = sorted(next(iter(agg.values()))["by_kind"])
    lines += ["", "All-fields-right rate by case type:", "", "| mode | " + " | ".join(kinds) + " |", "|" + "---|" * (len(kinds) + 1)]
    for m, a in agg.items():
        lines.append(f"| {m} | " + " | ".join(fmt(a["by_kind"][k]) for k in kinds) + " |")
    return "\n".join(lines)


# Rough planning numbers, NOT measurements: tokens per case across all modes, before any hidden reasoning tokens.
EST_CALLS_PER_CASE = {"baseline": 3, "guarded": 3, "full": 8}  # transcribe + extract + draft (+ up to ~5 verifier calls)
EST_IN_TOKENS, EST_OUT_TOKENS = 9000, 2500
DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT = 1.25, 4.25  # USD per million tokens, standard tier as reported by a news site. Verify.


def estimate(n_cases: int, modes: list[str], price_in: float, price_out: float) -> dict[str, float]:
    # modes share transcription and extraction, so cost does not simply add: count them once, then each extra draft/verifier
    calls = n_cases * (2 + sum(1 for m in modes if m in MODES) + (5 if "full" in modes else 0))
    tin, tout = n_cases * EST_IN_TOKENS, n_cases * EST_OUT_TOKENS
    return {"calls": calls, "in_tokens": tin, "out_tokens": tout, "usd": tin / 1e6 * price_in + tout / 1e6 * price_out}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from app.config import load_settings
    from app.llm import MetaLLM
    from app.retrieval import LocalCorpusRetriever

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--budget", type=int, default=400, help="hard cap on real model calls (default 400)")
    ap.add_argument("--dry-run", action="store_true", help="print the estimated cost and exit; sends nothing")
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN)
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT)
    args = ap.parse_args(argv)
    labels_path = ROOT / "evals" / "appeal" / "labels.json"
    if not labels_path.exists():
        print("Run python scripts/gen_appeal_eval.py first.")
        return 2
    labels = json.loads(labels_path.read_text())
    if args.limit:  # take an even spread of kinds, not just the first N (which would all be 'clean')
        labels = labels[:: max(1, len(labels) // args.limit)][: args.limit]
    mode_names = [m for m in args.modes.split(",") if m in MODES]
    est = estimate(len(labels), mode_names, args.price_in, args.price_out)
    print(f"Plan: {len(labels)} cases, modes {mode_names}. Rough estimate (not measured): ~{est['calls']} calls, "
          f"~{est['in_tokens']:,} input + ~{est['out_tokens']:,} output tokens, ~${est['usd']:.2f} at ${args.price_in}/${args.price_out} per M tokens. "
          f"Hard cap: {args.budget} calls.")
    if args.dry_run:
        return 0
    s = load_settings()
    if not s.api_key:
        print("MODEL_API_KEY is not set.")
        return 2
    retr = LocalCorpusRetriever.from_dir(s.corpus_dir)
    if not len(retr):
        print("NOTE: corpus is empty, so rights counts will be zero. Run scripts/build_corpus.py corpus/appeal_sources.txt")
    modes = {m: MODES[m] for m in args.modes.split(",")}
    client = MetaLLM(s)
    shared = CachingLLM(client, args.budget)
    rows = run(labels, shared, retr, labels_path.parent, modes, args.workers)
    agg = {m: aggregate(r) for m, r in rows.items()}
    md = to_markdown(agg, s.model, len(labels))
    res = ROOT / "evals" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "appeal_eval.md").write_text(md + "\n")
    (res / "appeal_eval.json").write_text(json.dumps({"model": s.model, "agg": agg, "rows": rows}, indent=2, default=str))
    print(md)
    u = client.usage
    cost = u["prompt_tokens"] / 1e6 * args.price_in + u["completion_tokens"] / 1e6 * args.price_out
    print(f"\nActual usage: {u['calls']} calls, {u['prompt_tokens']:,} input + {u['completion_tokens']:,} output tokens "
          f"(~${cost:.2f} at the prices above; reported by the API, check your dashboard).")
    if any(r["error"] == "BudgetExceeded" for rr in rows.values() for r in rr):
        print("WARNING: the call budget was reached, so some cases are errors. Raise --budget to finish them.")
    print(f"\nSaved to {res}/appeal_eval.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
