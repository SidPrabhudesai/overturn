"""Score a local model (base or base+LoRA adapter) on the extraction task, with the same Guard 1 the app uses.

  python scripts/eval_local_model.py --model meta-llama/Llama-3.2-1B-Instruct                      # base, no training
  python scripts/eval_local_model.py --model meta-llama/Llama-3.2-1B-Instruct --adapter train/adapter

Test sets:
  ood   train/sft_val_ood.jsonl  (held-out payers + phrasings, same synthetic generator family)
  eval  evals/appeal/cases/*.pdf with a text layer (clean, missing_deadline, injection, distractor; photos need OCR so are skipped)
Metrics per set: payer / claim / code / deadline correct, deadline hallucinated (stated when the letter has none), all_correct,
unparseable output. 'unguarded' = raw model output; 'guarded' = after Guard 1.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.appeal import EXTRACT_SYSTEM, FIELDS, parse_date, pdf_text_layer, verify_facts  # noqa: E402


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_json(text: str):
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def score(raw, letter: str, truth: dict, guard: bool) -> dict:
    if guard:
        facts, _ = verify_facts(raw, letter, FIELDS, True)
        get = lambda k: facts[k].value if k in facts else None  # noqa: E731
    else:
        f = (raw or {}).get("facts") if isinstance(raw, dict) else None
        f = f if isinstance(f, dict) else {}
        get = lambda k: f[k].get("value") if isinstance(f.get(k), dict) and isinstance(f[k].get("value"), str) else None  # noqa: E731
    payer, claim, code, dl = get("payer"), get("claim_number"), get("denial_codes"), get("appeal_deadline")
    p = bool(payer) and _alnum(truth["payer"]) in _alnum(payer)
    c = bool(claim) and _alnum(claim) == _alnum(truth["claim_number"])
    k = bool(code) and _alnum(truth["denial_codes"]) in _alnum(code)
    if truth["appeal_deadline"] is None:
        d = "hallucinated" if dl else "absent_ok"
    elif not dl:
        d = "missing"
    else:
        got = parse_date(dl)
        d = "correct" if got and got.isoformat() == truth["appeal_deadline"] else "wrong"
    return {"payer": p, "claim": c, "code": k, "deadline": d, "all": p and c and k and d in ("correct", "absent_ok")}


def summarize(rows: list[dict], bad_json: int) -> dict:
    n = len(rows) or 1
    pct = lambda x: round(100 * x / n, 1)  # noqa: E731
    return {"n": len(rows), "payer": pct(sum(r["payer"] for r in rows)), "claim": pct(sum(r["claim"] for r in rows)),
            "code": pct(sum(r["code"] for r in rows)), "deadline_ok": pct(sum(r["deadline"] in ("correct", "absent_ok") for r in rows)),
            "deadline_hallucinated": pct(sum(r["deadline"] == "hallucinated" for r in rows)),
            "all_correct": pct(sum(r["all"] for r in rows)), "unparseable_output": bad_json}


def run_set(cases: list[tuple[str, dict]], generate: Callable[[str], str]) -> dict:
    out = {}
    raws = []
    bad = 0
    for letter, truth in cases:
        raw = parse_json(generate(letter))
        bad += raw is None
        raws.append((letter, truth, raw))
    for name, guard in (("unguarded", False), ("guarded", True)):
        out[name] = summarize([score(r, l, t, guard) for l, t, r in raws], bad)
    return out


def load_ood(path: Path) -> list[tuple[str, dict]]:
    cases = []
    for line in path.read_text().splitlines():
        ex = json.loads(line)
        user = ex["messages"][1]["content"]
        cases.append((user.split("<letter>\n", 1)[1].rsplit("\n</letter>", 1)[0], ex["truth"]))
    return cases


def load_eval(base: Path) -> list[tuple[str, dict]]:
    cases = []
    for item in json.loads((base / "labels.json").read_text()):
        if item["kind"] == "photo":
            continue
        text = pdf_text_layer((base / item["file"]).read_bytes())
        if text:
            cases.append((text, item["truth"]))
    return cases


def make_generate(model_id: str, adapter: str | None) -> Callable[[str], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def generate(letter: str) -> str:
        msgs = [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": f"<letter>\n{letter[:12000]}\n</letter>"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=500, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="evals/results/local_model_eval.json")
    a = ap.parse_args()
    gen = make_generate(a.model, a.adapter)
    sets = {"ood": load_ood(ROOT / "train" / "sft_val_ood.jsonl"), "eval": load_eval(ROOT / "evals" / "appeal")}
    result = {"model": a.model, "adapter": a.adapter}
    for name, cases in sets.items():
        cases = cases[: a.limit] if a.limit else cases
        result[name] = run_set(cases, gen)
        print(name, json.dumps(result[name], indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
