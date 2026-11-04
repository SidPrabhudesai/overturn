import json, subprocess, sys
from pathlib import Path
import random
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gen_sft_data as g
import eval_local_model as ev

def test_examples_are_self_consistent():
    rng = random.Random(1)
    for ood in (False, True):
        for _ in range(300):
            ex = g.make(rng, ood)
            for f in ex["facts"].values():
                if f:
                    assert f["quote"] in ex["text"] and f["value"] in f["quote"]

def test_train_never_uses_held_out_wording_or_eval_wording():
    rng = random.Random(2)
    texts = [g.make(rng, False)["text"] for _ in range(500)]
    bad = ["Last day to appeal", "Any request for reconsideration", "This letter explains why", "Our file no.", "Adjustment reason",
           "must be received by", "unable to approve this claim", *g.PAYERS_HELD]
    assert not [b for b in bad if any(b in t for t in texts)]

def test_ood_uses_held_out_payers_only():
    rng = random.Random(3)
    assert all(g.make(rng, True)["truth"]["payer"] in g.PAYERS_HELD for _ in range(50))

def test_oracle_scores_100_and_garbage_scores_0():
    rng = random.Random(4)
    exs = [g.make(rng, True) for _ in range(40)]
    cases = [(e["text"], e["truth"]) for e in exs]
    by_text = {e["text"]: json.dumps({"facts": e["facts"]}) for e in exs}
    ok = ev.run_set(cases, lambda t: by_text[t])
    assert ok["guarded"]["all_correct"] == 100 and ok["unguarded"]["unparseable_output"] == 0
    bad = ev.run_set(cases, lambda t: "no idea")
    assert bad["guarded"]["all_correct"] < 20 and bad["guarded"]["unparseable_output"] == 40

def test_guard_catches_fabricated_values():
    from app.appeal import FIELDS, verify_facts
    e = g.make(random.Random(5), True)
    fake = {"facts": dict(e["facts"], claim_number={"value": "ZZ99999999", "quote": "Claim number: ZZ99999999"})}
    kept, missing = verify_facts(fake, e["text"], FIELDS, True)
    assert "claim_number" not in kept and "claim_number" in missing
