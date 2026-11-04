import pytest

from app.config import Settings, is_trusted_url, load_settings
from app.llm import LLMError, MetaLLM, parse_json_object


@pytest.mark.parametrize(
    "url,ok",
    [
        ("https://www.cdc.gov/measles/index.html", True),
        ("https://cdc.gov/x", True),
        ("http://www.who.int/news", True),
        ("https://medlineplus.gov/a.html", True),
        ("https://cdc.gov.evil.com/x", False),
        ("https://evilcdc.gov/x", False),
        ("https://example.com/cdc.gov", False),
        ("ftp://who.int/file", False),
        ("javascript:alert(1)", False),
        ("", False),
    ],
)
def test_is_trusted_url(url, ok):
    assert is_trusted_url(url) is ok


def test_contributor_tier_refused_by_default():
    with pytest.raises(ValueError, match="Contributor"):
        load_settings({"MODEL": "muse-spark-1.2-contributor", "MODEL_API_KEY": "k"})


def test_contributor_tier_needs_explicit_opt_in():
    s = load_settings({"MODEL": "muse-spark-1.2-contributor", "ALLOW_CONTRIBUTOR_TIER": "1"})
    assert s.model.endswith("-contributor")


def test_load_settings_reads_key_and_defaults():
    s = load_settings({"MODEL_API_KEY": " abc "})
    assert s.api_key == "abc" and s.base_url == "https://api.meta.ai/v1" and s.retriever == "local"


def test_parse_json_object_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here you go: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    for bad in ("", "no json", "[1,2]", "{broken"):
        with pytest.raises(ValueError):
            parse_json_object(bad)


class Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text

    def json(self):
        return self._p


def completion(content):
    return {"choices": [{"message": {"content": content}}]}


class Session:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make(session, key="secret-key"):
    sleeps = []
    return MetaLLM(Settings(api_key=key), session=session, sleep=sleeps.append), sleeps


def test_request_shape():
    sess = Session(Resp(200, completion('{"ok": true}')))
    llm, _ = make(sess)
    assert llm.complete_json("sys", "usr") == {"ok": True}
    req = sess.requests[0]
    assert req["url"] == "https://api.meta.ai/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer secret-key"
    assert req["json"]["model"] == "muse-spark-1.2"
    assert req["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert req["json"]["reasoning_effort"] == "low"


def test_retries_then_succeeds_with_backoff():
    sess = Session(Resp(429, text="slow down"), Resp(503), Resp(200, completion('{"x": 1}')))
    llm, sleeps = make(sess)
    assert llm.complete_json("s", "u") == {"x": 1}
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_retries_without_leaking_key():
    sess = Session(*[Resp(500)] * 4)
    llm, _ = make(sess)
    with pytest.raises(LLMError) as e:
        llm.complete_json("s", "u")
    assert "secret-key" not in str(e.value)


def test_client_error_is_not_retried():
    sess = Session(Resp(401, text="bad key"), Resp(200, completion("{}")))
    llm, sleeps = make(sess)
    with pytest.raises(LLMError, match="401"):
        llm.complete_json("s", "u")
    assert len(sess.requests) == 1 and sleeps == []


def test_repairs_non_json_once_then_fails():
    ok = Session(Resp(200, completion("not json")), Resp(200, completion('{"fixed": true}')))
    llm, _ = make(ok)
    assert llm.complete_json("s", "u") == {"fixed": True}
    assert ok.requests[1]["json"]["messages"][-1]["content"].startswith("Reply with ONLY")

    bad = Session(Resp(200, completion("nope")), Resp(200, completion("still nope")))
    llm, _ = make(bad)
    with pytest.raises(LLMError, match="valid JSON"):
        llm.complete_json("s", "u")


def test_content_parts_and_bad_shape():
    parts = Session(Resp(200, completion([{"type": "text", "text": '{"a":'}, {"type": "text", "text": "1}"}])))
    llm, _ = make(parts)
    assert llm.complete_json("s", "u") == {"a": 1}
    with pytest.raises(LLMError, match="shape"):
        make(Session(Resp(200, {"oops": 1})))[0].complete_json("s", "u")


def test_missing_key_rejected():
    with pytest.raises(LLMError, match="MODEL_API_KEY"):
        MetaLLM(Settings(api_key=""))


def test_usage_is_accumulated_from_api_responses_and_tolerates_missing_or_bad_fields():
    good = {**completion('{"a": 1}'), "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    bad = {**completion('{"a": 2}'), "usage": {"prompt_tokens": "lots", "completion_tokens": -5}}
    none = completion('{"a": 3}')
    llm, _ = make(Session(Resp(200, good), Resp(200, bad), Resp(200, none)))
    for _ in range(3):
        llm.complete_json("s", "u")
    assert llm.usage == {"calls": 3, "prompt_tokens": 100, "completion_tokens": 20}


def test_failed_calls_are_not_counted_as_billed():
    llm, _ = make(Session(Resp(500), Resp(200, completion('{"a": 1}'))))
    llm.complete_json("s", "u")
    assert llm.usage["calls"] == 1
