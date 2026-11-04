"""Full path over real HTTP: MetaLLM -> (local stand-in for Meta's API) with real retrieval and formatting.

The stand-in only mimics the OpenAI-compatible request/response shape from Meta's docs. It proves our
client, parsing and pipeline work end to end. It does NOT prove Meta's live API accepts our request.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.config import Settings
from app.server import Services

CORPUS = [
    {"url": "https://www.cdc.gov/measles/vaccine.html", "title": "Measles vaccine",
     "text": "The MMR vaccine is highly effective at preventing measles, mumps and rubella."},
    {"url": "https://www.who.int/salt", "title": "Salt", "text": "Excess sodium raises blood pressure."},
]


@pytest.fixture()
def fake_meta():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            system, user = body["messages"][0]["content"], body["messages"][-1]["content"]
            if system.startswith("You extract"):
                out = {"language": "en", "claims": ["The MMR vaccine prevents measles."]}
            elif "[S1]" in user:
                out = {"verdict": "supported", "explanation": "The CDC says it works.", "citations": ["S1"]}
            else:
                out = {"verdict": "supported", "explanation": "Trust me.", "citations": []}
            payload = json.dumps({"choices": [{"message": {"content": "```json\n" + json.dumps(out) + "\n```"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()


def services(tmp_path, base_url, corpus):
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(c) for c in corpus) + "\n")
    return Services(Settings(api_key="test-key", base_url=base_url, corpus_dir=str(tmp_path)))


def test_whatsapp_reply_end_to_end(tmp_path, fake_meta):
    base, seen = fake_meta
    reply = services(tmp_path, base, CORPUS).reply_text("My aunt says the MMR shot doesn't stop measles??")
    assert "✅ Supported by evidence" in reply
    assert "https://www.cdc.gov/measles/vaccine.html" in reply
    assert reply.rstrip().endswith("not medical advice. Talk to a clinician about your own situation.")
    assert [s["path"] for s in seen] == ["/v1/chat/completions"] * 2
    assert all(s["auth"] == "Bearer test-key" for s in seen)
    assert seen[0]["body"]["model"] == "muse-spark-1.2" and "reasoning_effort" in seen[0]["body"]
    assert "[S1]" in seen[1]["body"]["messages"][-1]["content"]  # retrieved passage reached the model


def test_empty_corpus_means_no_verdict_and_no_verification_call(tmp_path, fake_meta):
    base, seen = fake_meta
    reply = services(tmp_path, base, []).reply_text("The MMR vaccine prevents measles.")
    assert "❓ Not enough evidence" in reply and "✅" not in reply
    assert len(seen) == 1  # extraction only; nothing retrieved, so the model was never asked to judge


def test_unrelated_corpus_cannot_produce_a_confident_verdict(tmp_path, fake_meta):
    base, _ = fake_meta
    only_salt = [CORPUS[1]]
    reply = services(tmp_path, base, only_salt).reply_text("The MMR vaccine prevents measles.")
    assert "✅" not in reply and "❓" in reply
