from datetime import date

import pytest

from app import appeal
from app.appeal import (EOB_FIELDS, Fact, Mismatch, Right, build_appeal, checklist_for, clean_placeholders,
                        cross_check, deadline_status, parse_date, verify_facts, verify_rights)
from tests.helpers import FakeRetriever
from tests.test_appeal import LETTER, PDF, AppealLLM, gov


# ---------- guard 1: value must be inside its quote ----------
def test_fabricated_value_with_unrelated_real_quote_is_dropped():
    raw = {"facts": {"appeal_deadline": {"value": "January 1, 2027", "quote": "Acme Health Plan"}}}
    assert verify_facts(raw, LETTER)[0] == {}


def test_value_inside_its_quote_is_kept():
    raw = {"facts": {"denial_codes": {"value": "CO-50", "quote": "code CO-50"}}}
    assert "denial_codes" in verify_facts(raw, LETTER)[0]


def test_guard_off_keeps_everything_well_formed():
    raw = {"facts": {"appeal_deadline": {"value": "January 1, 2027", "quote": "nonsense"}}}
    assert "appeal_deadline" in verify_facts(raw, LETTER, guard=False)[0]


# ---------- placeholders ----------
@pytest.mark.parametrize("bad", ["[MEMBER ID: [MEMBER ID]]", "[[MEMBER ID]]", "[MEMBER ID (see [MEMBER ID])]"])
def test_nested_placeholders_collapse(bad):
    out = clean_placeholders(f"Hi [YOUR NAME], {bad}, bye")
    assert out == "Hi [YOUR NAME], [MEMBER ID], bye"


def test_clean_text_unchanged():
    t = "Dear X,\n[YOUR NAME] and [DATE] (code CO-50) [A]"
    assert clean_placeholders(t) == t and clean_placeholders("no brackets") == "no brackets"


# ---------- dates ----------
@pytest.mark.parametrize("text,iso", [
    ("March 14, 2027", "2027-03-14"), ("Mar. 14, 2027", "2027-03-14"), ("03/14/2027", "2027-03-14"),
    ("2027-03-14", "2027-03-14"), ("14 March 2027", "2027-03-14"), ("March 14th, 2027", "2027-03-14"),
    ("received by March 14, 2027 at 5pm", "2027-03-14"),
])
def test_parse_date(text, iso):
    assert parse_date(text).isoformat() == iso


@pytest.mark.parametrize("text", ["", "soon", "within 60 days", "13/45/2027"])
def test_unparseable_dates(text):
    assert parse_date(text) is None


def test_deadline_status_buckets():
    today = date(2026, 11, 1)
    assert deadline_status("November 10, 2026", today) == ("2026-11-10", 9, "urgent")
    assert deadline_status("November 15, 2026", today)[2] == "urgent"      # exactly 14 days
    assert deadline_status("November 16, 2026", today)[2] == "ok"
    assert deadline_status("October 31, 2026", today) == ("2026-10-31", -1, "expired")
    assert deadline_status("soon", today) == (None, None, "unknown")
    assert deadline_status(None, today) == (None, None, "unknown")


# ---------- checklist ----------
@pytest.mark.parametrize("reason,kind", [
    ("the service was not medically necessary", "medical_necessity"),
    ("prior authorization was not obtained", "prior_authorization"),
    ("the facility was out of network", "out_of_network"),
    ("considered experimental or investigational", "experimental"),
    ("the plan limit of 12 visits per year was reached", "limit_reached"),
    ("the item is not a covered benefit", "not_covered"),
    ("something else entirely", "general"), ("", "general"), (None, "general"),
])
def test_checklist_kinds(reason, kind):
    k, tips = checklist_for(reason)
    assert k == kind and len(tips) >= 3


# ---------- rights verifier ----------
def _rights():
    p = gov()
    return (Right("a", (p,)), Right("b", (p,)))


class V:
    def __init__(self, fn): self.fn = fn
    def complete_json(self, system, user): return self.fn(user)


def test_verifier_keeps_only_explicit_true():
    out = {"a": {"supported": True}, "b": {"supported": False}}
    kept, dropped = verify_rights(V(lambda u: out["a" if "Statement: a" in u else "b"]), _rights())
    assert [r.statement for r in kept] == ["a"] and dropped == 1


@pytest.mark.parametrize("resp", [{}, {"supported": "true"}, {"supported": 1}, {"supported": None}, "yes", None, []])
def test_verifier_fails_closed_on_malformed(resp):
    kept, dropped = verify_rights(V(lambda u: resp), _rights())
    assert kept == () and dropped == 2


def test_verifier_fails_closed_on_exception():
    def boom(u): raise RuntimeError("down")
    assert verify_rights(V(boom), _rights()) == ((), 2)


def test_verifier_prompt_contains_statement_and_passage_text():
    seen = []
    verify_rights(V(lambda u: seen.append(u) or {"supported": True}), _rights()[:1])
    assert "Statement: a" in seen[0] and "external review" in seen[0]


# ---------- EOB cross-check ----------
def F(v): return Fact(v, v)


def test_cross_check_flags_real_disagreements_only():
    letter = {"claim_number": F("NW-2291847"), "payer": F("Northwind Health Plan"), "service": F("MRI of the left knee")}
    same = {"claim_number": F("nw 2291847"), "payer": F("Northwind Health Plan, Inc."), "service": F("MRI left knee")}
    assert cross_check(letter, same) == ()
    diff = {"claim_number": F("NW-9999999"), "payer": F("Other Mutual"), "service": F("dental cleaning")}
    assert {m.field for m in cross_check(letter, diff)} == {"claim_number", "payer", "service"}
    assert cross_check(letter, {}) == () and cross_check({}, diff) == ()


# ---------- pipeline wiring ----------
def test_verifier_drops_rights_in_pipeline_and_counts_are_reported():
    llm = AppealLLM()
    llm.verifier = lambda u: {"supported": False}
    r = build_appeal(llm, FakeRetriever([gov()]), PDF, today=date(2026, 11, 1))
    assert r.rights == () and r.rights_proposed == 1 and r.rights_after_citation_guard == 1


def test_verify_false_skips_second_pass():
    llm = AppealLLM()
    llm.verifier = lambda u: {"supported": False}
    r = build_appeal(llm, FakeRetriever([gov()]), PDF, verify=False)
    assert len(r.rights) == 1


def test_deadline_and_checklist_come_from_verified_facts():
    facts = {"appeal_deadline": {"value": "December 15, 2026", "quote": "appeal by December 15, 2026"},
             "denial_reason": {"value": "not medically necessary", "quote": "not medically necessary"}}
    r = build_appeal(AppealLLM(facts=facts), FakeRetriever([gov()]), PDF, today=date(2026, 12, 10))
    assert (r.deadline_iso, r.days_left, r.urgency) == ("2026-12-15", 5, "urgent")
    assert r.checklist_kind == "medical_necessity"


def test_unverified_deadline_is_never_computed():
    facts = {"appeal_deadline": {"value": "January 1, 2027", "quote": "Acme Health Plan"}}
    r = build_appeal(AppealLLM(facts=facts), FakeRetriever([gov()]), PDF)
    assert r.deadline_iso is None and r.urgency == "unknown"


def test_eob_mismatch_reaches_result():
    class Two(AppealLLM):
        def complete_json(self, system, user):
            if system.startswith("You extract facts from an explanation"):
                return {"facts": {"claim_number": {"value": "ZZ-1", "quote": "Claim ZZ-1"}}}
            return super().complete_json(system, user)

        def complete_text(self, system, user):
            self.text_calls.append((system, user))
            return LETTER if len(self.text_calls) == 1 else "EOB Claim ZZ-1 for knee MRI paid nothing at all"

    llm = Two(facts={"claim_number": {"value": "55-1234", "quote": "Claim number 55-1234"}})
    r = build_appeal(llm, FakeRetriever([gov()]), PDF, eob=PDF)
    assert [m.field for m in r.mismatches] == ["claim_number"] and r.eob_facts["claim_number"].value == "ZZ-1"


def test_letter_is_cleaned_in_pipeline():
    llm = AppealLLM(draft={"letter": "Dear X [MEMBER ID: [MEMBER ID]]", "rights": []})
    assert build_appeal(llm, FakeRetriever([gov()]), PDF).letter == "Dear X [MEMBER ID]"
