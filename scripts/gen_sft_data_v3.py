"""v3 generator: v2 data plus targeted letters for (a) relative deadlines -> null, (b) many distractor dates, (c) several codes.
python scripts/gen_sft_data_v3.py --n-v2 1400 --n-each 200   # writes train/sft_train_v3.jsonl
"""
from __future__ import annotations
import argparse, json, random, sys
from datetime import date, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import gen_sft_data_v2 as v2  # noqa: E402

DESC = {"CO-197": "Precertification was not obtained.", "CO-50": "Not deemed medically necessary.", "CO-16": "Claim lacks required information.",
        "CO-97": "Included in the allowance for another service.", "CO-45": "Charge exceeds the fee schedule.", "CO-96": "Non-covered charge.",
        "CO-29": "The time limit for filing has expired.", "PR-204": "Service not covered under the benefit plan.", "N54": "Claim information is inconsistent with pre-certified services.",
        "N130": "Consult plan benefit documents for coverage limitations.", "CO-109": "Claim not covered by this payer.", "CO-18": "Duplicate claim."}
REL = ["You must ask for an appeal within {n} calendar days of the date of this notice.", "Your appeal must be filed within {n} days of receiving this letter.",
       "You have {n} days from the date of this notice to request a review.", "Requests for review are accepted for {n} days after the notice date.",
       "To appeal, contact us within {n} days. A State Fair Hearing may also be requested."]
ABS = ["A written appeal must be filed on or before {d}.", "Appeals must be received no later than {d}.", "Your appeal must arrive by {d}.",
       "Deadline to file an appeal: {d}", "The plan will not accept appeals after {d}."]
LAB = ["Claim Number:", "Claim #", "Claim ID:", "ICN (claim number):", "Reference Number:"]

def F(v, q, text):
    assert q in text and v in q, (v, q)
    return {"value": v, "quote": q}

def make3(rng, kind):
    payer = f"{rng.choice(v2.PLACES)} {rng.choice(v2.KINDS)}"
    claim = v2.claim_id(rng)
    svc = rng.choice(v2.SERVICES)
    reason, code = rng.choice(v2.REASONS)
    fmt = rng.choice(v2.DATEFMT)
    D = lambda dt: dt.strftime(fmt)  # noqa: E731
    notice = date(2026, rng.randint(9, 11), rng.randint(1, 28))
    dl = date(2027, 1, 1) + timedelta(days=rng.randint(0, 300))
    plan = rng.choice(v2.PLANS) if rng.random() < 0.4 else None
    claim_l = f"{rng.choice(LAB)} {claim}"
    plan_l = f"Plan type: {plan}" if plan else None
    lines = [payer, rng.choice(["Claims Review", "Appeals & Grievances", "Member Services", "Notice of Action"]), "", claim_l]
    lines.append(f"Date of this notice: {D(notice)}")
    codes = [code]
    if kind == "rel":
        n = rng.choice([30, 45, 60, 90, 120, 180])
        lines.append(f"Date of service: {D(notice - timedelta(days=rng.randint(20, 90)))}")
        code_l = f"Adjustment code {code}"
        dead_l, dl_val, dl_iso = rng.choice(REL).format(n=n), None, None
    elif kind == "dist":
        lines += [f"Date request received: {D(notice - timedelta(days=rng.randint(5, 25)))}",
                  f"Coverage effective date: {D(date(2026, 1, 1))}", f"Service date: {D(notice - timedelta(days=rng.randint(30, 80)))}"]
        c = code.split("-")[-1] if rng.random() < 0.3 else code
        codes = [c]
        code_l = f"Denial code: {c}"
        dl_val = D(dl)
        dead_l, dl_iso = rng.choice(ABS).format(d=dl_val), dl.isoformat()
    else:
        pool = [k for k in DESC if k != code]
        codes = [code] + rng.sample(pool, rng.choice([1, 2]))
        code_l = "Reason codes: " + ", ".join(codes)
        dl_val = D(dl)
        dead_l, dl_iso = rng.choice(ABS).format(d=dl_val), dl.isoformat()
    body = lines + [""]
    if plan_l:
        body.append(plan_l)
    reason_l = f"Your claim for {svc} was denied. Reason: {reason}."
    body += [reason_l, code_l]
    if kind == "multi":
        body += [f" {k} {DESC.get(k, 'See remark.')}" for k in codes]
    body += ["", dead_l, "", "Sincerely,", "Claims Review Team"]
    text = "\n".join(body)
    facts = {"payer": F(payer, payer, text), "service": F(svc, svc, text), "denial_reason": F(reason, reason, text),
             "denial_codes": F(codes[0], code_l, text), "appeal_deadline": None if dl_val is None else F(dl_val, dead_l, text),
             "claim_number": F(claim, claim_l, text), "plan_type": F(plan, plan_l, text) if plan else None}
    truth = {"payer": payer, "claim_number": claim, "denial_codes": codes[0], "appeal_deadline": dl_iso}
    return {"text": text, "facts": facts, "truth": truth}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-v2", type=int, default=1400)
    ap.add_argument("--n-each", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3031)
    ap.add_argument("--out", default=str(ROOT / "train" / "sft_train_v3.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    exs = [v2.make2(rng) for _ in range(a.n_v2)]
    for kind in ("rel", "dist", "multi"):
        exs += [make3(rng, kind) for _ in range(a.n_each)]
    rng.shuffle(exs)
    Path(a.out).write_text("\n".join(json.dumps(v2.to_example(e)) for e in exs) + "\n")
    print("wrote", len(exs), "examples to", a.out)
    for kind in ("rel", "dist", "multi"):
        e = make3(random.Random(1), kind)
        print("-----", kind, e["truth"]); print(e["text"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
