import json

from app.retrieval import MAX_PASSAGE_CHARS, LocalCorpusRetriever
from scripts.build_corpus import chunk_text, html_to_text


def entries():
    return [
        {"url": "https://www.cdc.gov/measles", "title": "Measles", "text": "The MMR vaccine prevents measles, mumps and rubella."},
        {"url": "https://www.who.int/salt", "title": "Salt", "text": "Eating too much sodium raises blood pressure."},
        {"url": "https://www.nhs.uk/hands", "title": "Hands", "text": "Washing hands with soap reduces infections."},
    ]


def test_bm25_ranks_relevant_first_and_numbers_ids():
    r = LocalCorpusRetriever(entries())
    hits = r.search("Does the MMR vaccine prevent measles?", k=3)
    assert hits[0].url.endswith("/measles")
    assert [h.id for h in hits] == [f"S{i}" for i in range(1, len(hits) + 1)]


def test_no_overlap_or_empty_query_returns_nothing():
    r = LocalCorpusRetriever(entries())
    assert r.search("quantum chromodynamics") == []
    assert r.search("the of and") == []  # stopwords only
    assert LocalCorpusRetriever([]).search("measles") == []


def test_passage_text_is_truncated():
    r = LocalCorpusRetriever([{"url": "https://www.cdc.gov/a", "title": "t", "text": "measles " * 500}])
    assert len(r.search("measles")[0].text) == MAX_PASSAGE_CHARS


def test_from_dir_drops_untrusted_and_malformed_lines(tmp_path):
    rows = entries() + [
        {"url": "https://evil.example.com/measles", "title": "x", "text": "measles cures everything"},
        {"url": "https://www.cdc.gov/nourl", "title": "x"},
    ]
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\nnot json\n\n", encoding="utf-8")
    r = LocalCorpusRetriever.from_dir(tmp_path)
    assert len(r) == 3
    assert all("evil" not in h.url for h in r.search("measles"))


def test_html_to_text_strips_boilerplate():
    html = """<html><head><title>Vaccines | CDC</title><script>var x=1;</script></head><body>
    <nav><p>Home About Contact us today for more links</p></nav>
    <main><h1>MMR vaccine safety overview</h1><p>The MMR vaccine has been studied for decades and is safe.</p>
    <p>tiny</p></main><footer><p>Copyright notice that should be dropped entirely</p></footer></body></html>"""
    title, text = html_to_text(html)
    assert title == "Vaccines | CDC"
    assert "studied for decades" in text and "Copyright" not in text and "Home About" not in text and "var x" not in text


def test_chunk_text_respects_limit_and_loses_nothing():
    long_para = ". ".join(f"Sentence number {i} about vaccines" for i in range(60)) + "."
    text = "\n\n".join(["Short intro paragraph about health." * 3, long_para, "Closing paragraph about screening."])
    chunks = chunk_text(text, max_chars=300)
    assert chunks and all(len(c) <= 300 for c in chunks)
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())
