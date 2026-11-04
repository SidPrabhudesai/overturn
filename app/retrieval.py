"""Retrieval over trusted health sources.

LocalCorpusRetriever: BM25 over corpus/*.jsonl (built by scripts/build_corpus.py). Tested.
WebSearchRetriever:   Meta Responses API `web_search` tool. UNTESTED against Meta's API: the
                      response shape below follows the OpenAI Responses format that Meta's docs
                      say it is compatible with. Verify before relying on it.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests

from .config import Settings, is_trusted_url

MAX_PASSAGE_CHARS = 900


@dataclass(frozen=True)
class Passage:
    id: str  # "S1", "S2", ... unique within one search
    url: str
    title: str
    text: str


class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[Passage]: ...


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "that the their there these they this to was were what when which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.lens):
            s = 0.0
            for t in set(query):
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1.0))
                s += self.idf[t] * f * (self.k1 + 1) / denom
            out.append(s)
        return out


class LocalCorpusRetriever:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries
        self._bm25 = BM25([tokenize(e["title"] + " " + e["text"]) for e in entries])

    @classmethod
    def from_dir(cls, corpus_dir: str | Path) -> "LocalCorpusRetriever":
        entries: list[dict] = []
        for path in sorted(Path(corpus_dir).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all(isinstance(e.get(f), str) and e[f] for f in ("url", "text")) and is_trusted_url(e["url"]):
                    e.setdefault("title", "")
                    entries.append(e)
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def search(self, query: str, k: int = 4) -> list[Passage]:
        q = tokenize(query)
        if not q or not self.entries:
            return []
        scored = sorted(zip(self._bm25.scores(q), range(len(self.entries))), reverse=True)
        hits = [(s, i) for s, i in scored[:k] if s > 0]
        return [
            Passage(
                id=f"S{n}",
                url=self.entries[i]["url"],
                title=self.entries[i]["title"],
                text=self.entries[i]["text"][:MAX_PASSAGE_CHARS],
            )
            for n, (_, i) in enumerate(hits, start=1)
        ]


class WebSearchRetriever:
    """UNTESTED against Meta's API. See module docstring."""

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self._s = settings
        self._http = session or requests.Session()

    def search(self, query: str, k: int = 4) -> list[Passage]:
        payload = {
            "model": self._s.model,
            "input": (
                "Find evidence about this health claim from official public-health sources "
                f"(WHO, CDC, NIH, NHS, Cochrane). Summarise what they say.\nClaim: {query}"
            ),
            "tools": [{"type": "web_search"}],
        }
        resp = self._http.post(
            f"{self._s.base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {self._s.api_key}"},
            json=payload,
            timeout=self._s.timeout,
        )
        resp.raise_for_status()
        passages: list[Passage] = []
        seen: set[str] = set()
        for item in resp.json().get("output", []):
            for part in item.get("content", []) if isinstance(item, dict) else []:
                text = part.get("text", "")
                for ann in part.get("annotations", []):
                    url = ann.get("url", "")
                    if ann.get("type") != "url_citation" or url in seen or not is_trusted_url(url):
                        continue
                    seen.add(url)
                    start = max(0, int(ann.get("start_index", 0)) - 400)
                    end = int(ann.get("end_index", 0)) + 100
                    passages.append(
                        Passage(
                            id=f"S{len(passages) + 1}",
                            url=url,
                            title=ann.get("title", ""),
                            text=text[start:end][:MAX_PASSAGE_CHARS],
                        )
                    )
                    if len(passages) >= k:
                        return passages
        return passages
