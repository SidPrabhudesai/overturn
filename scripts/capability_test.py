"""Which ambitious features can your Meta key actually do? Run this BEFORE we commit to scope.

  python scripts/capability_test.py                      # tool calling + model access
  python scripts/capability_test.py --audio speech.wav   # also tries speech-to-text

Make a speech file on a Mac:  say -o /tmp/t.wav --data-format=LEI16@16000 "My denial code is C O fifty"
Use synthetic text only. Some models may train on prompts (check Meta's terms before sending anything real).
Prints PASS/FAIL per capability. Paste the whole output back.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402


def post(s, path, **kw):
    return requests.post(s.base_url.rstrip("/") + path, headers={"Authorization": f"Bearer {s.api_key}"}, timeout=120, **kw)


def chat(s, model, messages, **extra):
    return post(s, "/chat/completions", json={"model": model, "messages": messages, **extra})


def short(r):
    return f"HTTP {r.status_code} {r.text[:250]!r}"


def test_tool_calling(s) -> bool:
    tools = [{"type": "function", "function": {
        "name": "add_days", "description": "Add days to an ISO date and return the new ISO date.",
        "parameters": {"type": "object", "properties": {"date": {"type": "string"}, "days": {"type": "integer"}},
                       "required": ["date", "days"]}}}]
    msgs = [{"role": "user", "content": "What date is 15 days after 2026-11-01? You must use the add_days tool."}]
    r = chat(s, s.model, msgs, tools=tools, tool_choice="auto")
    if not r.ok:
        print(f"[FAIL] tool calling, turn 1: {short(r)}")
        return False
    msg = r.json()["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        print(f"[FAIL] tool calling: model answered without calling the tool: {str(msg.get('content'))[:200]!r}")
        return False
    call = calls[0]
    print(f"       model called {call['function']['name']}({call['function']['arguments']})")
    msgs += [msg, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"date": "2026-11-16"})}]
    r2 = chat(s, s.model, msgs, tools=tools)
    if not r2.ok:
        print(f"[FAIL] tool calling, turn 2: {short(r2)}")
        return False
    text = r2.json()["choices"][0]["message"].get("content") or ""
    ok = "2026-11-16" in text or "November 16" in text
    print(f"[{'PASS' if ok else 'WEAK'}] tool calling full loop: reply={text.strip()[:150]!r}")
    return ok


def test_model(s, model) -> bool:
    for extra in ({"reasoning_effort": "low"}, {}):
        r = chat(s, model, [{"role": "user", "content": "Reply with exactly: ok"}], **extra)
        if r.ok:
            text = r.json()["choices"][0]["message"].get("content") or ""
            print(f"[PASS] model {model} {'with' if extra else 'without'} reasoning_effort: reply={text.strip()[:80]!r}")
            return True
        last = short(r)
    print(f"[FAIL] model {model}: {last}")
    return False


def test_asr(s, path) -> bool:
    data = Path(path).read_bytes()
    for model in ("muse-voice-transcribe-1.0",):
        r = post(s, "/asr/transcribe", files={"file": (Path(path).name, data)}, data={"model": model})
        if r.ok:
            print(f"[PASS] speech-to-text ({model}): {r.text[:250]!r}")
            return True
        print(f"[FAIL] speech-to-text ({model}): {short(r)}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    args = ap.parse_args()
    s = load_settings()
    if not s.api_key:
        print("MODEL_API_KEY is not set.")
        return 2
    results = {"tool calling": test_tool_calling(s)}
    for m in ("muse-spark-1.3", "muse-wonder-1.0"):
        results[f"model {m}"] = test_model(s, m)
    if args.audio:
        results["speech-to-text"] = test_asr(s, args.audio)
    print("\nSummary:", ", ".join(f"{k}={'ok' if v else 'no'}" for k, v in results.items()))
    print("Paste this whole output back to me.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
