from collections import Counter
from pathlib import Path

import pytest

from evals.run_eval import ROOT, LABELS, load_claims, run_mode, score, to_markdown
from tests.helpers import FakeLLM, FakeRetriever, cdc


def test_dataset_is_well_formed():
    claims = load_claims(ROOT / "evals" / "claims.jsonl")
    assert len(claims) == 50
    assert Counter(c["label"] for c in claims) == {"supported": 20, "contradicted": 20, "misleading": 10}
    assert all(c["claim"].strip().endswith(".") for c in claims)
    assert len({c["claim"] for c in claims}) == 50


def test_load_claims_rejects_bad_label_and_duplicates(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id":"a","claim":"x","label":"maybe"}\n')
    with pytest.raises(ValueError, match="bad label"):
        load_claims(p)
    p.write_text('{"id":"a","claim":"x","label":"supported"}\n{"id":"a","claim":"y","label":"supported"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_claims(p)


def test_score_known_values():
    rows = [
        {"label": "supported", "pred": "supported"},
        {"label": "supported", "pred": "contradicted"},       # flip
        {"label": "contradicted", "pred": "supported"},       # flip
        {"label": "contradicted", "pred": "insufficient_evidence"},  # abstain
        {"label": "misleading", "pred": "supported"},         # wrong but not a flip
        {"label": "misleading", "pred": "misleading"},
    ]
    s = score(rows)
    assert s["n"] == 6
    assert s["accuracy"] == pytest.approx(2 / 6)
    assert s["accuracy_answered"] == pytest.approx(2 / 5)
    assert s["abstain_rate"] == pytest.approx(1 / 6)
    assert s["flip_rate"] == pytest.approx(2 / 6)
    assert s["confusion"]["supported"] == {"supported": 1, "contradicted": 1}
    assert set(s["confusion"]) == set(LABELS)


def test_score_edge_cases():
    all_abstain = score([{"label": "supported", "pred": "insufficient_evidence"}])
    assert all_abstain["accuracy_answered"] is None and all_abstain["abstain_rate"] == 1
    with pytest.raises(ValueError):
        score([])


def test_run_mode_records_predictions_and_survives_errors():
    claims = [
        {"id": "a", "claim": "ok claim", "label": "supported"},
        {"id": "b", "claim": "boom", "label": "supported"},
    ]

    def verify(user):
        if "boom" in user:
            raise RuntimeError("model blew up")
        return {"verdict": "supported", "explanation": "x", "citations": ["S1"]}

    rows = run_mode("guarded", claims, FakeLLM(verify=verify), FakeRetriever([cdc()]), workers=2)
    assert [r["pred"] for r in rows] == ["supported", "insufficient_evidence"]
    assert rows[1]["note"].startswith("error:") and rows[0]["label"] == "supported"


def test_modes_differ_only_by_guard_and_retrieval():
    claim = [{"id": "a", "claim": "claim", "label": "supported"}]
    uncited = FakeLLM(verify={"verdict": "supported", "explanation": "x", "citations": []})
    assert run_mode("grounded", claim, uncited, FakeRetriever([cdc()]))[0]["pred"] == "supported"
    assert run_mode("guarded", claim, uncited, FakeRetriever([cdc()]))[0]["pred"] == "insufficient_evidence"
    assert run_mode("ungrounded", claim, uncited, None)[0]["pred"] == "supported"


def test_markdown_table_shape():
    s = score([{"label": "supported", "pred": "supported"}])
    md = to_markdown({"guarded": s}, "m", 12)
    assert "| guarded | 100% | 100% | 0% | 0% |" in md
