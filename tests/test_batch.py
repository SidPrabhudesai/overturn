import csv
import io
import json
from datetime import date
from pathlib import Path

import pytest

from app.appeal import Fact
from app.batch import (MAX_FILES, BatchItem, run_batch, safe_filename, sort_items, summary_row, to_csv)
from app.server import Services, create_app
from app.config import Settings
from scripts.batch_appeals import process
from scripts.make_samples import make_pdf
from tests.helpers import FakeRetriever, asgi
from tests.test_appeal import AppealLLM, gov

TODAY = date(2026, 11, 1)


class Res:  # minimal AppealResult stand-in
    def __init__(self, urgency, days, letter="L", not_found=(), facts=None, mism=(), susp=False):
        self.urgency, self.days_left, self.letter = urgency, days, letter
        self.not_found, self.mismatches, self.suspicious = not_found, mism, susp
        self.facts = facts or {}
        self.deadline_iso = None


def items():
    return [BatchItem("err.pdf", error="bad"), BatchItem("exp.pdf", Res("expired", -3)), BatchItem("unk.pdf", Res("unknown", None)),
            BatchItem("ok30.pdf", Res("ok", 30)), BatchItem("urg9.pdf", Res("urgent", 9)), BatchItem("urg2.pdf", Res("urgent", 2)),
            BatchItem("ok20.pdf", Res("ok", 20))]


def test_sort_order_urgent_first_then_ok_unknown_expired_error():
    assert [i.name for i in sort_items(items())] == ["urg2.pdf", "urg9.pdf", "ok20.pdf", "ok30.pdf", "unk.pdf", "exp.pdf", "err.pdf"]


def test_csv_neutralises_formula_injection_and_keeps_columns():
    row = summary_row(BatchItem("=cmd.pdf", Res("ok", 5, facts={"payer": Fact("=HYPERLINK(\"http://x\")", "q"), "claim_number": Fact("+1", "q"),
                                                          "denial_codes": Fact("@SUM(A1)", "q"), "denial_reason": Fact("-2+3", "q")})))
    out = list(csv.DictReader(io.StringIO(to_csv([row]))))[0]
    assert out["file"].startswith("'=") and out["insurer"].startswith("'=") and out["claim_number"] == "'+1"
    assert out["denial_code"] == "'@SUM(A1)" and out["denial_reason"] == "'-2+3"
    assert out["days_left"] == "5"


def test_csv_is_parseable_with_commas_quotes_and_newlines():
    row = summary_row(BatchItem("a.pdf", Res("ok", 1, facts={"payer": Fact('Smith, "Jones"\nHealth', "q")})))
    assert list(csv.DictReader(io.StringIO(to_csv([row]))))[0]["insurer"] == 'Smith, "Jones"\nHealth'


def test_needs_review_flags_explain_why():
    r = Res("expired", -1, not_found=("payer",), mism=(1,), susp=True)
    row = summary_row(BatchItem("x.pdf", r))
    for phrase in ("missing: payer", "documents disagree", "instruction-like", "deadline appears passed"):
        assert phrase in row["needs_review"]
    assert summary_row(BatchItem("y.pdf", Res("ok", 20)))["needs_review"] == "standard review"
    assert summary_row(BatchItem("z.pdf", error="could not read"))["status"] == "error"


@pytest.mark.parametrize("name,expect", [("../../etc/passwd", "passwd.txt"), ("C:\\x\\y.pdf", "y.txt"), ("my letter (1).pdf", "my_letter_1.txt"),
                                          ("", "file.txt"), ("...", "file.txt"), ("a" * 200 + ".pdf", "a" * 80 + ".txt")])
def test_safe_filename_blocks_traversal(name, expect):
    assert safe_filename(name, set()) == expect


def test_safe_filename_dedupes():
    taken = set()
    assert [safe_filename("a.pdf", taken) for _ in range(3)] == ["a.txt", "a_2.txt", "a_3.txt"]


import base64
BAD_B64 = base64.b64encode(b"%PDF-1.4 BAD").decode()


def test_run_batch_isolates_failures_and_caps_size():
    class L(AppealLLM):
        def complete_text(self, system, user):
            if BAD_B64 in json.dumps(user):
                raise RuntimeError("secret internal detail")
            return super().complete_text(system, user)

    good, bad = b"%PDF-1.4 ok", b"%PDF-1.4 BAD"
    out = run_batch(L(), FakeRetriever([gov()]), [("bad.pdf", bad), ("good.pdf", good), ("junk.txt", b"hello")], today=TODAY)
    by = {i.name: i for i in out}
    assert by["good.pdf"].result is not None
    assert by["bad.pdf"].error == "could not process (RuntimeError)" and "secret" not in by["bad.pdf"].error
    assert by["junk.txt"].error and by["junk.txt"].result is None
    with pytest.raises(ValueError):
        run_batch(AppealLLM(), FakeRetriever([]), [("a.pdf", good)] * (MAX_FILES + 1))


def test_cli_process_writes_sorted_csv_and_safe_letters(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    (src / "a.pdf").write_bytes(make_pdf("x"))
    (src / "notes.docx").write_bytes(b"ignored")
    assert process(src, out, AppealLLM(), FakeRetriever([gov()])) == 0
    assert (out / "summary.csv").read_text().splitlines()[0].startswith("file,status,insurer")
    assert [p.name for p in (out / "letters").iterdir()] == ["a.txt"]
    assert "not legal or medical advice" in (out / "letters" / "a.txt").read_text()
    empty = tmp_path / "empty"; empty.mkdir()
    assert process(empty, tmp_path / "o2", AppealLLM(), FakeRetriever([])) == 1


def _multi(parts):
    b = b"XBX"
    body = b"".join(b"--" + b + b'\r\nContent-Disposition: form-data; name="files"; filename="' + n.encode() + b'"\r\nContent-Type: application/pdf\r\n\r\n' + d + b"\r\n" for n, d in parts) + b"--" + b + b"--\r\n"
    return body, {"content-type": "multipart/form-data; boundary=XBX"}


def _svc():
    return Services(settings=Settings(), llm=AppealLLM(), retriever=FakeRetriever([gov()]))


def test_batch_endpoint_returns_rows_csv_and_letters():
    body, h = _multi([("a.pdf", b"%PDF-1.4 a"), ("b.pdf", b"%PDF-1.4 b")])
    status, data = asgi(create_app(_svc()), "POST", "/api/batch", body=body, headers=h)
    out = json.loads(data)
    assert status == 200 and len(out["items"]) == 2 and out["items"][0]["letter"] and out["csv"].startswith("file,status")


def test_batch_endpoint_rejects_empty_toomany_and_nonmultipart_and_page_serves():
    app = create_app(_svc())
    assert asgi(app, "GET", "/batch")[0] == 200
    assert asgi(app, "POST", "/api/batch", body=b"{}", headers={"content-type": "application/json"})[0] == 400
    body, h = _multi([(f"{i}.pdf", b"%PDF-1.4 a") for i in range(MAX_FILES + 1)])
    assert asgi(app, "POST", "/api/batch", body=body, headers=h)[0] == 400
    from app import server
    assert "innerHTML" not in server.BATCH_PAGE


def test_samples_endpoints_list_serve_and_refuse_traversal(tmp_path, monkeypatch):
    from app import server
    (tmp_path / "a_sample.pdf").write_bytes(b"%PDF-1.4 sample")
    (tmp_path / "secret.txt").write_text("nope")
    monkeypatch.setattr(server, "SAMPLES_DIR", tmp_path)
    app = create_app(_svc())
    assert json.loads(asgi(app, "GET", "/api/samples")[1]) == {"samples": ["a_sample.pdf"]}
    status, body = asgi(app, "GET", "/api/samples/a_sample.pdf")
    assert status == 200 and body == b"%PDF-1.4 sample"
    for bad in ("secret.txt", "..%2Fsecret.txt", "nope.pdf", "a_sample.pdf%00"):
        assert asgi(app, "GET", "/api/samples/" + bad)[0] == 404


def test_samples_missing_directory_is_empty_list(tmp_path, monkeypatch):
    from app import server
    monkeypatch.setattr(server, "SAMPLES_DIR", tmp_path / "does-not-exist")
    assert json.loads(asgi(create_app(_svc()), "GET", "/api/samples")[1]) == {"samples": []}
