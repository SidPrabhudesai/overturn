import pytest

from app.appeal import PHIDetected, build_appeal
from app.export import build_blocks, to_docx, to_markdown, to_pdf, write_all
from app.phi import phi_kinds, scan_phi
from app.playbooks import PLAYBOOKS, playbook_for
from tests.helpers import FakeRetriever
from tests.test_appeal import AppealLLM, LETTER, PDF, gov


def test_scan_finds_labelled_details_and_masks_values():
    text = ("Patient: Jane Q Public\nMember ID: ABC123456\nDate of birth: 03/14/1985\nSSN 123-45-6789\n"
            "Phone: (410) 555-0199\nDear Jane Public,\n")
    kinds = phi_kinds(text)
    assert set(kinds) == {"patient_name", "member_id", "dob", "ssn", "phone", "salutation_name"}
    for f in scan_phi(text):
        assert "Jane" not in f.preview and "123456" not in f.preview and "1985" not in f.preview


def test_scan_ignores_blanked_out_values_and_form_greetings():
    text = "Patient: [REDACTED]\nMember ID: XXXXXXXX\nDate of birth: ********\nDear Member Services,\nClaim number NB-2049-118"
    assert scan_phi(text) == ()


def test_scan_does_not_flag_claim_numbers_or_dates_of_service():
    assert scan_phi("Claim no. NB-2049-118. Date of service 01/02/2026. Appeal by December 15, 2026.") == ()


def test_phi_flags_reported_on_result_and_not_values():
    llm = AppealLLM(transcript=LETTER + "\nMember ID: ZX9876543\nDOB: 01/02/1990")
    r = build_appeal(llm, FakeRetriever([gov()]), PDF)
    assert set(r.phi_flags) == {"member_id", "dob"}
    assert "ZX9876543" not in repr(r.phi_flags)


def test_strict_blocks_before_any_model_call(monkeypatch):
    from app import appeal
    monkeypatch.setattr(appeal, "pdf_text_layer", lambda data: "Patient: Jane Q Public\nMember ID: ABC123456 denied")
    llm = AppealLLM()
    with pytest.raises(PHIDetected) as e:
        build_appeal(llm, FakeRetriever([gov()]), PDF, strict_phi=True)
    assert llm.text_calls == [] and llm.json_calls == []
    assert "Jane" not in str(e.value) and "ABC123456" not in str(e.value)


def test_strict_passes_clean_files(monkeypatch):
    from app import appeal
    monkeypatch.setattr(appeal, "pdf_text_layer", lambda data: LETTER)
    r = build_appeal(AppealLLM(), FakeRetriever([gov()]), PDF, strict_phi=True)
    assert r.phi_flags == ()


def test_every_denial_kind_has_a_playbook():
    from app.appeal import CHECKLISTS
    assert set(CHECKLISTS) == set(PLAYBOOKS)
    assert playbook_for("nonsense") is PLAYBOOKS["general"]
    for pb in PLAYBOOKS.values():
        assert pb.address and pb.ask_clinician and pb.ask_plan


def _result():
    return build_appeal(AppealLLM(), FakeRetriever([gov()]), PDF)


def test_blocks_put_deadline_first_and_name_the_denial_type():
    blocks = build_blocks(_result())
    heads = [v for k, v in blocks if k == "h"]
    assert heads[0] == "Deadline"
    assert any(h.startswith("Reviewer notes:") for h in heads)


def test_markdown_has_facts_with_quotes_and_letter():
    md = to_markdown(_result())
    assert "December 15, 2026" in md and "Acme Health Plan" in md and "Dear Acme Health Plan" in md


def test_files_are_written(tmp_path):
    r = _result()
    files = write_all(r, tmp_path, "case1")
    names = {f.name for f in files}
    assert {"case1.md", "case1.docx", "case1.pdf"} <= names
    assert (tmp_path / "case1.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "case1.docx").read_bytes().startswith(b"PK")


def test_unreviewed_flags_appear_in_the_export():
    llm = AppealLLM(transcript=LETTER + "\nMember ID: ZX9876543")
    r = build_appeal(llm, FakeRetriever([gov()]), PDF)
    md = to_markdown(r)
    assert "Check before sending" in md and "member or policy ID" in md and "ZX9876543" not in md
