import json
import urllib.parse

import pytest

from app import server
from app.config import Settings
from app.format import FOOTER, HELP, LABELS, MAX_REPLY_CHARS, format_report
from app.llm import LLMError
from app.pipeline import ClaimResult, Report
from app.server import RateLimiter, Services, create_app
from app.twilio_utils import twiml_message, valid_signature
from tests.helpers import FakeLLM, FakeRetriever, asgi, cdc


# ---------- format ----------
def result(verdict="supported", explanation="Short.", sources=(cdc(),)):
    return ClaimResult("Some claim", verdict, explanation, tuple(sources))


def test_format_has_label_sources_and_footer():
    text = format_report(Report((result(),), "en"))
    assert LABELS["supported"] in text and "https://www.cdc.gov/measles/vaccine.html" in text and text.endswith(FOOTER)


def test_format_never_exceeds_whatsapp_limit_and_keeps_footer():
    big = result(explanation="word " * 400)
    text = format_report(Report((big, big, big), "en"))
    assert len(text) <= MAX_REPLY_CHARS and text.endswith(FOOTER)


def test_format_help_and_safety_note_ordering():
    assert format_report(Report((), "en")) == HELP
    text = format_report(Report((result(),), "en", ("CALL 911",)))
    assert text.startswith("CALL 911")


# ---------- twilio ----------
def test_signature_matches_twilios_published_example():
    params = {
        "CallSid": "CA1234567890ABCDE", "Caller": "+14158675310", "Digits": "1234",
        "From": "+14158675310", "To": "+18005551212",
    }
    url = "https://mycompany.com/myapp.php?foo=1&bar=2"
    assert valid_signature("12345", url, params, "GvWf1cFY/Q7PnoempGyD5oXAezc=")
    assert not valid_signature("12345", url, params, "AAAA")
    assert not valid_signature("12345", url, {**params, "Digits": "9999"}, "GvWf1cFY/Q7PnoempGyD5oXAezc=")
    assert not valid_signature("12345", url, params, "")


def test_twiml_escapes_markup():
    assert "&lt;script&gt;&amp;" in twiml_message("<script>&")


# ---------- server ----------
OK_VERIFY = {"verdict": "supported", "explanation": "Yes.", "citations": ["S1"]}


def svc(settings=None, llm=None, retr=None):
    return Services(
        settings=settings or Settings(),
        llm=llm or FakeLLM(extract={"language": "en", "claims": ["MMR prevents measles"]}, verify=OK_VERIFY),
        retriever=retr or FakeRetriever([cdc()]),
    )


def post_json(app, payload, **kw):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return asgi(app, "POST", "/api/check", body=body, headers={"content-type": "application/json"}, **kw)


def test_index_and_health():
    app = create_app(svc())
    status, html = asgi(app, "GET", "/")
    assert status == 200 and b"Health claim checker" in html
    assert asgi(app, "GET", "/healthz")[0] == 200


def test_api_check_happy_path():
    status, data = post_json(create_app(svc()), {"text": "Does MMR prevent measles?"})
    body = json.loads(data)
    assert status == 200
    assert body["results"][0]["verdict"] == "supported"
    assert body["results"][0]["sources"][0]["url"].startswith("https://www.cdc.gov")


@pytest.mark.parametrize("payload", [b"{not json", {"text": ""}, {"text": "   "}, {"text": 5}, {"nope": 1}, [1, 2]])
def test_api_check_rejects_bad_input(payload):
    assert post_json(create_app(svc()), payload)[0] == 400


def test_api_check_rate_limits_per_client():
    s = svc()
    s.limiter = RateLimiter(limit=2)
    app = create_app(s)
    codes = [post_json(app, {"text": "x"})[0] for _ in range(3)]
    other = post_json(app, {"text": "x"}, client=("9.9.9.9", 1))[0]
    assert codes == [200, 200, 429] and other == 200


def test_api_check_model_failure_is_503_without_details():
    class Down:
        def complete_json(self, system, user):
            raise LLMError("HTTP 500: internal secret details")

    status, data = post_json(create_app(svc(llm=Down())), {"text": "x"})
    assert status == 503 and b"secret" not in data


def form(**fields):
    return urllib.parse.urlencode(fields).encode(), {"content-type": "application/x-www-form-urlencoded"}


def test_twilio_sync_reply_is_twiml():
    body, headers = form(From="whatsapp:+15550001111", Body="Does MMR prevent measles?")
    status, data = asgi(create_app(svc()), "POST", "/webhook/twilio", body=body, headers=headers)
    assert status == 200 and data.startswith(b"<?xml") and LABELS["supported"].encode() in data


def test_twilio_empty_body_gets_help():
    body, headers = form(From="whatsapp:+15550001111", Body="")
    _, data = asgi(create_app(svc()), "POST", "/webhook/twilio", body=body, headers=headers)
    assert HELP.encode()[:30] in data


def test_twilio_rejects_bad_signature_and_accepts_good_one():
    import base64, hashlib, hmac

    settings = Settings(twilio_auth_token="tok", public_url="https://x.ngrok.app/")
    app = create_app(svc(settings))
    fields = {"From": "whatsapp:+15550001111", "Body": "hi"}
    body, headers = form(**fields)
    assert asgi(app, "POST", "/webhook/twilio", body=body, headers={**headers, "x-twilio-signature": "bad"})[0] == 403
    assert asgi(app, "POST", "/webhook/twilio", body=body, headers=headers)[0] == 403

    data = "https://x.ngrok.app/webhook/twilio" + "".join(k + fields[k] for k in sorted(fields))
    sig = base64.b64encode(hmac.new(b"tok", data.encode(), hashlib.sha1).digest()).decode()
    assert asgi(app, "POST", "/webhook/twilio", body=body, headers={**headers, "x-twilio-signature": sig})[0] == 200


def test_twilio_async_mode_acks_then_replies_via_rest(monkeypatch):
    sent = []
    monkeypatch.setattr(server, "send_whatsapp", lambda *a: sent.append(a))
    settings = Settings(twilio_account_sid="AC1", twilio_auth_token="tok")  # no public_url: signature check off
    body, headers = form(From="whatsapp:+15550001111", Body="Does MMR prevent measles?")
    status, data = asgi(create_app(svc(settings)), "POST", "/webhook/twilio", body=body, headers=headers)
    assert status == 200 and b"<Message>" not in data
    assert len(sent) == 1 and sent[0][4].startswith("*Claim:*") and sent[0][3] == "whatsapp:+15550001111"


def test_twilio_model_down_gets_friendly_message():
    class Down:
        def complete_json(self, system, user):
            raise LLMError("x")

    body, headers = form(From="whatsapp:+15550001111", Body="claim")
    _, data = asgi(create_app(svc(llm=Down())), "POST", "/webhook/twilio", body=body, headers=headers)
    assert b"couldn't reach" in data


def test_rate_limiter_window_expires():
    now = [0.0]
    rl = RateLimiter(limit=1, window=10, clock=lambda: now[0])
    assert rl.allow("a") and not rl.allow("a") and rl.allow("b")
    now[0] = 11
    assert rl.allow("a")


# ---------- appeal endpoint ----------
def _multipart(data: bytes):
    b = b"XBOUNDARYX"
    body = b"--" + b + b'\r\nContent-Disposition: form-data; name="file"; filename="l.pdf"\r\nContent-Type: application/pdf\r\n\r\n' + data + b"\r\n--" + b + b"--\r\n"
    return body, {"content-type": "multipart/form-data; boundary=XBOUNDARYX"}


def _appeal_svc():
    from tests.test_appeal import AppealLLM, gov
    return Services(settings=Settings(), llm=AppealLLM(), retriever=FakeRetriever([gov()]))


def test_appeal_page_and_endpoint_happy_path():
    app = create_app(_appeal_svc())
    assert asgi(app, "GET", "/appeal")[0] == 200
    body, headers = _multipart(b"%PDF-1.4 x")
    status, data = asgi(app, "POST", "/api/appeal", body=body, headers=headers)
    out = json.loads(data)
    assert status == 200 and out["facts"]["payer"]["quote"] == "Acme Health Plan" and out["letter"]


def test_appeal_rejects_wrong_filetype_and_missing_file_and_rate_limits():
    app = create_app(_appeal_svc())
    body, headers = _multipart(b"just text")
    assert asgi(app, "POST", "/api/appeal", body=body, headers=headers)[0] == 400
    assert asgi(app, "POST", "/api/appeal", body=b"{}", headers={"content-type": "application/json"})[0] == 400
    s = _appeal_svc(); s.limiter = RateLimiter(limit=1)
    app = create_app(s)
    body, headers = _multipart(b"%PDF-1.4 x")
    codes = [asgi(app, "POST", "/api/appeal", body=body, headers=headers)[0] for _ in range(2)]
    assert codes == [200, 429]


def test_appeal_response_has_deadline_checklist_and_trust_fields():
    body, headers = _multipart(b"%PDF-1.4 x")
    out = json.loads(asgi(create_app(_appeal_svc()), "POST", "/api/appeal", body=body, headers=headers)[1])
    assert out["trust"]["facts_kept"] == 3 and out["trust"]["rights_final"] == 1
    assert out["deadline"]["iso"] == "2026-12-15" and out["checklist"]["tips"] and out["has_eob"] is False


def test_appeal_with_eob_upload_sets_has_eob():
    b = b"XBOUNDARYX"
    part = lambda name, data: (b"--" + b + b'\r\nContent-Disposition: form-data; name="' + name + b'"; filename="x.pdf"\r\nContent-Type: application/pdf\r\n\r\n' + data + b"\r\n")
    body = part(b"file", b"%PDF-1.4 a") + part(b"eob", b"%PDF-1.4 b") + b"--" + b + b"--\r\n"
    out = json.loads(asgi(create_app(_appeal_svc()), "POST", "/api/appeal", body=body,
                          headers={"content-type": "multipart/form-data; boundary=XBOUNDARYX"})[1])
    assert out["has_eob"] is True


def test_appeal_page_never_uses_innerhtml():
    assert "innerHTML" not in server.APPEAL_PAGE and "insertAdjacentHTML" not in server.APPEAL_PAGE


def test_appeal_response_includes_letter_text_and_studio_page_is_served():
    app = create_app(_appeal_svc())
    body, headers = _multipart(b"%PDF-1.4 x")
    out = json.loads(asgi(app, "POST", "/api/appeal", body=body, headers=headers)[1])
    assert isinstance(out["letter_text"], str) and "Acme Health Plan" in out["letter_text"]
    status, page = asgi(app, "GET", "/studio")
    assert status == 200 and b"Denial Letter Helper" in page
