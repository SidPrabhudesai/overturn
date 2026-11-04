"""30-second check that your key, base URL, model id and request shape work against the REAL API.

  python scripts/smoke_test.py

This is the one thing the offline tests cannot prove. Run it before anything else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.llm import LLMError, MetaLLM  # noqa: E402


def main() -> int:
    s = load_settings()
    if not s.api_key:
        print("MODEL_API_KEY is not set (copy .env.example to .env).")
        return 2
    print(f"base_url={s.base_url} model={s.model} reasoning_effort={s.reasoning_effort}")

    try:
        r = requests.get(f"{s.base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {s.api_key}"}, timeout=20)
        ids = [m.get("id") for m in r.json().get("data", [])] if r.ok else []
        print(f"GET /models -> {r.status_code}; ids: {ids or r.text[:200]}")
        if ids and s.model not in ids:
            print(f"WARNING: {s.model!r} is not in that list. Set MODEL to one of the ids above.")
    except (requests.RequestException, ValueError) as exc:
        print(f"GET /models failed ({type(exc).__name__}); not fatal, trying a completion.")

    try:
        out = MetaLLM(s).complete_json('Reply with exactly this JSON: {"ok": true}', "ping")
    except LLMError as exc:
        print(f"FAILED: {exc}")
        print("If the error mentions reasoning_effort or a role, tell me the message and I'll adjust llm.py.")
        return 1
    print(f"completion OK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
