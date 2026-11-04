"""Download trusted public-health pages and split them into passages for retrieval.

  python scripts/build_corpus.py corpus/sources.txt          # one URL per line, '#' for comments

Only URLs on the trusted-domain allowlist (app/config.py) are fetched. Writes corpus/corpus.jsonl,
one {"url", "title", "text"} object per passage. Re-run to refresh. This needs normal internet
access, so run it on your own machine.

Choose pages deliberately: a source's own page can change over time, and your verdicts will follow
whatever the pages say on the day you fetch them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import is_trusted_url  # noqa: E402

UA = "healthcheck-bot-corpus-builder/0.1 (hackathon project)"
DROP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg")


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, body text with paragraphs separated by blank lines)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    title = (soup.title.get_text(" ", strip=True) if soup.title else "") or ""
    root = soup.find("main") or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in root.find_all(["p", "li", "h1", "h2", "h3"])]
    return title, "\n\n".join(p for p in paras if len(p) > 25)


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Pack whole paragraphs into chunks of at most max_chars (long paragraphs are split)."""
    chunks: list[str] = []
    cur = ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > 200 else max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if cur and len(cur) + len(para) + 2 > max_chars:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", help="text file with one URL per line")
    ap.add_argument("--out", default=str(ROOT / "corpus" / "corpus.jsonl"))
    args = ap.parse_args()

    urls = [
        line.strip()
        for line in Path(args.sources).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    total, failed = 0, []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for url in urls:
            if not is_trusted_url(url):
                print(f"SKIP (not on allowlist): {url}", file=sys.stderr)
                failed.append(url)
                continue
            try:
                resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"FAIL {url}: {type(exc).__name__}", file=sys.stderr)
                failed.append(url)
                continue
            title, text = html_to_text(resp.text)
            chunks = chunk_text(text)
            for c in chunks:
                fh.write(json.dumps({"url": url, "title": title, "text": c}) + "\n")
            total += len(chunks)
            print(f"ok   {url}: {len(chunks)} passages")
    print(f"\n{total} passages from {len(urls) - len(failed)}/{len(urls)} pages -> {out}")
    return 1 if failed and not total else 0


if __name__ == "__main__":
    raise SystemExit(main())
