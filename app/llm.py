"""Minimal client for Meta Model API's OpenAI-compatible Chat Completions endpoint.

Plain `requests` on purpose: no SDK to install, and the whole request/response shape is visible.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable, Protocol

import requests

from .config import Settings


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of model output, tolerating code fences and chatter around it."""
    s = _FENCE.sub("", text.strip())
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("JSON value is not an object")
    return obj


def _message_text(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("unexpected response shape") from exc
    if isinstance(content, list):  # some servers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


class MetaLLM:
    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.api_key:
            raise LLMError("MODEL_API_KEY is not set")
        self._s = settings
        self._http = session or requests.Session()
        self._sleep = sleep
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}  # what this client was actually billed for, as reported

    def _record(self, data: Any) -> None:
        u = data.get("usage") if isinstance(data, dict) else None
        u = u if isinstance(u, dict) else {}
        num = lambda k: u.get(k) if isinstance(u.get(k), int) and u.get(k) >= 0 else 0  # noqa: E731
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += num("prompt_tokens")
            self.usage["completion_tokens"] += num("completion_tokens")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._s.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self._s.api_key}", "Content-Type": "application/json"}
        delay = 1.0
        last = "unknown error"
        for attempt in range(4):
            try:
                resp = self._http.post(url, headers=headers, json=payload, timeout=self._s.timeout)
            except requests.RequestException as exc:
                last = type(exc).__name__
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    self._record(data)
                    return data
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                last = f"HTTP {resp.status_code}"
            if attempt < 3:
                self._sleep(delay)
                delay *= 2
        raise LLMError(f"model call failed after retries ({last})")

    def complete_text(self, system: str, user: Any) -> str:
        """Plain-text completion. `user` is a string or a list of OpenAI-style content parts (text/image/file)."""
        payload: dict[str, Any] = {
            "model": self._s.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "reasoning_effort": self._s.reasoning_effort,
        }
        return _message_text(self._post(payload))

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._s.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning_effort": self._s.reasoning_effort,
        }
        for _ in range(2):  # one repair attempt if the model returns non-JSON
            text = _message_text(self._post(payload))
            try:
                return parse_json_object(text)
            except ValueError:
                payload["messages"] = payload["messages"] + [
                    {"role": "assistant", "content": text[:500]},
                    {"role": "user", "content": "Reply with ONLY the JSON object, nothing else."},
                ]
        raise LLMError("model did not return valid JSON")
