"""Does Meta's API accept images / PDFs, and in which request shape?

  python scripts/vision_test.py                 # makes a test PDF (and PNG if Pillow is installed)
  python scripts/vision_test.py --image my.png  # use your own image of a letter
  python scripts/vision_test.py --pdf my.pdf

The test document contains a marker string (CO-50 and 2026-12-15). A shape "passes" only if the model's
reply contains both, so a 200 response that ignored the attachment does not count.
It tries several request shapes because I could not verify Meta's exact format; paste me the output.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402

TEXT_LINES = ["CLAIM DENIED", "Denial code: CO-50", "Appeal deadline: 2026-12-15"]
PROMPT = "Read the attached document. Reply with the denial code and the appeal deadline only."
EXPECT = ("CO-50", "2026-12-15")


def make_pdf(path: Path) -> None:
    stream = "BT /F1 18 Tf 72 720 Td 24 TL " + " ".join(f"({t}) Tj T*" for t in TEXT_LINES) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def make_png(path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    img = Image.new("RGB", (900, 300), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    for i, t in enumerate(TEXT_LINES):
        d.text((30, 30 + i * 70), t, fill="black", font=font)
    img.save(path)
    return True


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def shapes(image: Path | None, pdf: Path | None):
    """(name, endpoint, body-builder) candidates. Body gets the model name."""
    out = []
    if image:
        img_uri = f"data:image/{image.suffix.lstrip('.').lower().replace('jpg', 'jpeg')};base64,{b64(image)}"
        out.append(("image: chat image_url data-URI", "/chat/completions", lambda m: {
            "model": m, "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": img_uri}}]}]}))
        out.append(("image: responses input_image", "/responses", lambda m: {
            "model": m, "input": [{"role": "user", "content": [
                {"type": "input_text", "text": PROMPT},
                {"type": "input_image", "image_url": img_uri}]}]}))
    if pdf:
        pdf_uri = f"data:application/pdf;base64,{b64(pdf)}"
        out.append(("pdf: chat file/file_data", "/chat/completions", lambda m: {
            "model": m, "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "file", "file": {"filename": pdf.name, "file_data": pdf_uri}}]}]}))
        out.append(("pdf: responses input_file", "/responses", lambda m: {
            "model": m, "input": [{"role": "user", "content": [
                {"type": "input_text", "text": PROMPT},
                {"type": "input_file", "filename": pdf.name, "file_data": pdf_uri}]}]}))
    return out


def reply_text(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    parts = []  # Responses API shape
    for item in data.get("output", []) if isinstance(data, dict) else []:
        for c in item.get("content", []) or []:
            parts.append(c.get("text", ""))
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--pdf")
    args = ap.parse_args()

    s = load_settings()
    if not s.api_key:
        print("MODEL_API_KEY is not set (copy .env.example to .env).")
        return 2

    work = Path(__file__).resolve().parent.parent / "evals" / "results"
    work.mkdir(parents=True, exist_ok=True)
    pdf = Path(args.pdf) if args.pdf else work / "test_letter.pdf"
    if not args.pdf:
        make_pdf(pdf)
    image = Path(args.image) if args.image else work / "test_letter.png"
    if not args.image and not make_png(image):
        print("Pillow not installed: skipping the generated image. Re-run with --image path/to/letter.png "
              "(a phone photo or screenshot of any text works; edit EXPECT or just read the reply).")
        image = None

    results = []
    for name, ep, build in shapes(image, pdf):
        url = s.base_url.rstrip("/") + ep
        try:
            r = requests.post(url, json=build(s.model), timeout=90,
                              headers={"Authorization": f"Bearer {s.api_key}"})
        except requests.RequestException as exc:
            print(f"[ERR ] {name}: {type(exc).__name__}")
            results.append((name, False))
            continue
        if not r.ok:
            print(f"[FAIL] {name}: HTTP {r.status_code} {r.text[:300]!r}")
            results.append((name, False))
            continue
        try:
            text = reply_text(r.json())
        except ValueError:
            text = ""
        ok = all(e in text for e in EXPECT)
        print(f"[{'PASS' if ok else 'WEAK'}] {name}: reply={text.strip()[:200]!r}")
        results.append((name, ok))

    print("\nSummary:", ", ".join(f"{n}={'ok' if ok else 'no'}" for n, ok in results))
    print("Paste this whole output to me. Any PASS line tells us which request shape to build on.")
    return 0 if any(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
