"""Test doubles and a tiny ASGI client (httpx, which Starlette's TestClient needs, isn't installed)."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from app.retrieval import Passage


class FakeLLM:
    """Answers extraction calls and verification calls from canned data. Records every call."""

    def __init__(self, extract: dict | None = None, verify: dict | Callable[[str], dict] | None = None):
        self.extract = extract if extract is not None else {"language": "en", "claims": []}
        self.verify = verify
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if system.startswith("You extract"):
            return self.extract
        return self.verify(user) if callable(self.verify) else (self.verify or {})


class FakeRetriever:
    def __init__(self, passages: list[Passage] | None = None, error: Exception | None = None):
        self.passages = passages if passages is not None else []
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, k: int = 4) -> list[Passage]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.passages[:k]


def cdc(pid: str = "S1", text: str = "The MMR vaccine is highly effective against measles.") -> Passage:
    return Passage(id=pid, url="https://www.cdc.gov/measles/vaccine.html", title="Measles vaccine", text=text)


def asgi(app, method: str, path: str, *, body: bytes = b"", headers: dict[str, str] | None = None,
         client: tuple[str, int] = ("1.2.3.4", 1000)) -> tuple[int, bytes]:
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": client, "server": ("test", 80),
    }
    sent: list[dict] = []
    state = {"first": True}

    async def receive():
        if state["first"]:
            state["first"] = False
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()  # stay connected like a real client; FileResponse aborts on early disconnect
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
