"""HTTP entrypoints: a web chat page, a JSON API, and the Twilio WhatsApp webhook.

Privacy: message text is never logged or stored. Logs carry a short hash of the sender only.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import time
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Any

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from .appeal import AppealResult, UnsupportedDocument, build_appeal
from .batch import MAX_FILES, review_flags, run_batch, summary_row, to_csv
from .config import Settings, load_settings
from .format import HELP, format_report
from .llm import LLM, LLMError, MetaLLM
from .pipeline import Report, check_message
from .retrieval import LocalCorpusRetriever, Retriever, WebSearchRetriever
from .twilio_utils import EMPTY_TWIML, send_whatsapp, twiml_message, valid_signature

log = logging.getLogger("healthcheck")

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
RATE_LIMIT = 10  # messages
RATE_WINDOW = 600  # seconds


class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT, window: float = RATE_WINDOW, clock=time.monotonic) -> None:
        self.limit, self.window, self._clock = limit, window, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self._clock()
        q = self._hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


class Services:
    """Builds the model client and retriever lazily so the app can start without a key."""

    def __init__(self, settings: Settings | None = None, llm: LLM | None = None, retriever: Retriever | None = None):
        self._settings = settings
        self._llm = llm
        self._retriever = retriever
        self.limiter = RateLimiter()

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = MetaLLM(self.settings)
        return self._llm

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            s = self.settings
            self._retriever = WebSearchRetriever(s) if s.retriever == "web" else LocalCorpusRetriever.from_dir(s.corpus_dir)
            if isinstance(self._retriever, LocalCorpusRetriever) and not len(self._retriever):
                log.warning("corpus is empty: every claim will come back 'not enough evidence'. Run scripts/build_corpus.py")
        return self._retriever

    def check(self, text: str) -> Report:
        s = self.settings
        return check_message(self.llm, self.retriever, text, s.max_claims, max_chars=s.max_input_chars)

    def appeal(self, data: bytes, eob: bytes | None = None) -> AppealResult:
        return build_appeal(self.llm, self.retriever, data, eob=eob)

    def batch(self, files: list[tuple[str, bytes]]):
        return run_batch(self.llm, self.retriever, files)

    def reply_text(self, text: str) -> str:
        if not text.strip():
            return HELP
        try:
            return format_report(self.check(text))
        except LLMError:
            log.exception("model call failed")
            return "Sorry, I couldn't reach my fact-checking model just now. Please try again in a minute."


def _sender_id(sender: str) -> str:
    return hashlib.sha256(sender.encode()).hexdigest()[:8]


def report_to_dict(report: Report) -> dict[str, Any]:
    return {
        "language": report.language,
        "safety_notes": list(report.safety_notes),
        "results": [
            {
                "claim": r.claim,
                "verdict": r.verdict,
                "explanation": r.explanation,
                "downgraded": r.downgraded,
                "sources": [{"url": p.url, "title": p.title} for p in r.sources],
            }
            for r in report.results
        ],
    }


def appeal_to_dict(r: AppealResult) -> dict[str, Any]:
    return {
        "facts": {k: {"value": f.value, "quote": f.quote} for k, f in r.facts.items()},
        "not_found": list(r.not_found),
        "rights": [{"statement": x.statement, "sources": [{"url": p.url, "title": p.title} for p in x.sources]} for x in r.rights],
        "letter": r.letter,
        "disclaimer": r.disclaimer,
        "deadline": {"text": r.deadline_text, "iso": r.deadline_iso, "days_left": r.days_left, "urgency": r.urgency},
        "checklist": {"kind": r.checklist_kind, "tips": list(r.checklist)},
        "mismatches": [{"field": m.field, "letter": m.letter_value, "eob": m.eob_value} for m in r.mismatches],
        "has_eob": r.eob_facts is not None,
        "suspicious": r.suspicious,
        "verified_against": r.verified_against,
        "trust": {
            "facts_proposed": r.facts_proposed, "facts_kept": len(r.facts),
            "rights_proposed": r.rights_proposed, "rights_after_citation_check": r.rights_after_citation_guard,
            "rights_final": len(r.rights),
        },
    }


BATCH_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Denial batch review</title>
<style>
:root{--bg:#f6f5f1;--fg:#1c1c1a;--mut:#6a6a64;--card:#fff;--bd:#dedcd3;--ac:#0b5cad;--ok:#1a7f4b;--warn:#b26a00;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ececea;--mut:#9b9b93;--card:#1f1e1c;--bd:#38372f;--ac:#6db3f2;--ok:#5fcf93;--warn:#f0b04d;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 64px}h1{font-size:28px;margin:0 0 4px}.mut{color:var(--mut);font-size:14px}
button{padding:10px 18px;font:inherit;border:0;border-radius:8px;background:var(--ac);color:#fff;cursor:pointer}button.alt{background:transparent;color:var(--ac);border:1px solid var(--ac)}
button:disabled{opacity:.55}.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px;margin-top:16px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:720px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--bd);font-size:14px;vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}tr.pick{cursor:pointer}tr.pick:hover,tr.sel{background:var(--bg)}
.b{font-weight:600}.urgent{color:var(--warn)}.ok{color:var(--ok)}.expired,.error{color:var(--bad)}
textarea{width:100%;min-height:300px;padding:12px;font:inherit;color:inherit;background:var(--bg);border:1px solid var(--bd);border-radius:8px}a{color:var(--ac)}
</style></head><body><main>
<h1>Denial batch review</h1>
<p class="mut">Upload up to 20 denial letters. Rows are sorted by deadline, soonest first. Every draft needs a person's review before it is sent. <a href="/appeal">Single letter</a></p>
<input type="file" id="f" multiple accept=".pdf,.png,.jpg,.jpeg,.webp,.gif">
<div class="row"><button id="go">Process batch</button><button id="dl" class="alt" disabled>Download summary CSV</button></div>
<div id="out" aria-live="polite"></div>
<p class="mut">General information, not legal or medical advice. Files are processed to answer this request and are not stored.</p>
<script>
const $=id=>document.getElementById(id);let csvText="";
function el(t,x,c){const e=document.createElement(t);if(x!=null)e.textContent=x;if(c)e.className=c;return e}
function render(d){const out=$("out");out.replaceChildren();csvText=d.csv;$("dl").disabled=false;
 const card=el("div",null,"card"),sc=el("div",null,"scroll"),tb=el("table"),hd=el("tr");
 ["File","Insurer","Code","Deadline","Days left","Needs review"].forEach(h=>hd.append(el("th",h)));tb.append(hd);
 const ta=el("textarea");ta.value="Select a row to see its draft letter.";
 d.items.forEach(it=>{const r=el("tr",null,"pick");
  r.append(el("td",it.file,"b"),el("td",it.insurer),el("td",it.denial_code),el("td",it.deadline),el("td",String(it.days_left),it.urgency),el("td",it.needs_review,it.status==="error"?"error":""));
  r.onclick=()=>{tb.querySelectorAll("tr").forEach(x=>x.classList.remove("sel"));r.classList.add("sel");ta.value=it.letter||"No letter was drafted for this file."};tb.append(r)});
 sc.append(tb);card.append(sc);out.append(card);const lc=el("div",null,"card");lc.append(el("h2","Draft letter"),ta);out.append(lc)}
$("go").onclick=async()=>{const out=$("out"),b=$("go"),fs=$("f").files;
 if(!fs.length){out.replaceChildren(el("p","Choose some files first.","mut"));return}
 b.disabled=true;out.replaceChildren(el("p","Processing "+fs.length+" file(s)… this can take a few minutes.","mut"));
 try{const fd=new FormData();for(const f of fs)fd.append("files",f);
  const r=await fetch("/api/batch",{method:"POST",body:fd});const d=await r.json();
  if(!r.ok){out.replaceChildren(el("p",d.error||"Something went wrong.","mut"));return}render(d)
 }catch(e){out.replaceChildren(el("p","Network error. Please try again.","mut"))}finally{b.disabled=false}};
$("dl").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csvText],{type:"text/csv"}));a.download="denial-summary.csv";a.click();URL.revokeObjectURL(a.href)};
</script></main></body></html>"""


APPEAL_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Denial appeal helper</title>
<style>
:root{--bg:#f6f5f1;--fg:#1c1c1a;--mut:#6a6a64;--card:#fff;--bd:#dedcd3;--ac:#0b5cad;--ok:#1a7f4b;--warn:#b26a00;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ececea;--mut:#9b9b93;--card:#1f1e1c;--bd:#38372f;--ac:#6db3f2;--ok:#5fcf93;--warn:#f0b04d;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:28px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:17px;margin:0 0 8px}
.mut{color:var(--mut);font-size:14px}.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px 18px;margin-top:16px}
.drops{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}@media(max-width:560px){.drops{grid-template-columns:1fr}}
.drop{border:2px dashed var(--bd);border-radius:12px;padding:22px 14px;text-align:center;background:var(--card);cursor:pointer;display:block}
.drop.over{border-color:var(--ac)}.drop strong{display:block}.drop input{display:none}
button{padding:10px 18px;font:inherit;border:0;border-radius:8px;background:var(--ac);color:#fff;cursor:pointer}
button.alt{background:transparent;color:var(--ac);border:1px solid var(--ac)}button:disabled{opacity:.55;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.banner{border-radius:12px;padding:14px 18px;margin-top:16px;font-weight:600;border:1px solid var(--bd);background:var(--card)}
.banner.ok{border-color:var(--ok);color:var(--ok)}.banner.urgent{border-color:var(--warn);color:var(--warn)}.banner.expired{border-color:var(--bad);color:var(--bad)}
.fact{padding:8px 0;border-top:1px solid var(--bd)}.fact:first-of-type{border-top:0}.fact .k{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.fact q{display:block;font-size:14px;color:var(--mut);font-style:italic}a{color:var(--ac);overflow-wrap:anywhere}
textarea{width:100%;min-height:340px;padding:12px;font:inherit;color:inherit;background:var(--bg);border:1px solid var(--bd);border-radius:8px;resize:vertical}
ul{margin:6px 0 0;padding-left:20px}.trust{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{background:var(--bg);border-radius:8px;padding:10px 12px}.stat b{display:block;font-size:20px}.warnbox{border-color:var(--warn)}
</style></head><body><main>
<h1>Denial appeal helper</h1>
<p class="mut">Upload a photo or PDF of a health-insurance denial letter. Every fact shown is backed by a quote from your letter, and every right is checked against an official source.</p>
<div class="drops">
 <label class="drop" id="d1"><input type="file" id="f1" accept=".pdf,.png,.jpg,.jpeg,.webp,.gif"><strong>Denial letter</strong><span class="mut" id="n1">Drop a file or click to choose (required)</span></label>
 <label class="drop" id="d2"><input type="file" id="f2" accept=".pdf,.png,.jpg,.jpeg,.webp,.gif"><strong>Explanation of benefits</strong><span class="mut" id="n2">Optional: lets me cross-check the two documents</span></label>
</div>
<div class="row"><button id="go">Build my appeal</button></div>
<p class="mut" id="samples" hidden>No letter handy? Try a fictional one: <span id="sl"></span></p>
<div id="out" aria-live="polite"></div>
<p class="mut">General information, not legal or medical advice. Your files are processed to answer this request and are not stored. <a href="/batch">Practices: process many letters at once</a></p>
<script>
const $=id=>document.getElementById(id);
function el(t,x,c){const e=document.createElement(t);if(x!=null)e.textContent=x;if(c)e.className=c;return e}
function card(title){const c=el("section",null,"card");if(title)c.append(el("h2",title));return c}
[["d1","f1","n1"],["d2","f2","n2"]].forEach(([d,f,n])=>{
 const box=$(d),inp=$(f),lab=$(n);
 const show=()=>{if(inp.files[0])lab.textContent=inp.files[0].name};
 inp.onchange=show;
 ["dragenter","dragover"].forEach(ev=>box.addEventListener(ev,e=>{e.preventDefault();box.classList.add("over")}));
 ["dragleave","drop"].forEach(ev=>box.addEventListener(ev,e=>{e.preventDefault();box.classList.remove("over")}));
 box.addEventListener("drop",e=>{if(e.dataTransfer.files[0]){inp.files=e.dataTransfer.files;show()}});
});
function render(d){const out=$("out");out.replaceChildren();
 const dl=d.deadline;
 if(dl.urgency!=="unknown"){const t=dl.urgency==="expired"?"This deadline appears to have passed ("+dl.text+"). Contact your plan right away and ask about late appeals.":
   dl.days_left+" day"+(dl.days_left===1?"":"s")+" left to appeal (deadline "+dl.text+")";
  out.append(el("div",t,"banner "+dl.urgency))}
 else out.append(el("div",dl.text?"I could not read a date from: "+dl.text+". Check your letter for the appeal deadline.":"I could not find an appeal deadline in your letter. Check it, or call your plan.","banner urgent"));
 if(d.suspicious){const w=card("Unusual text in your document");w.classList.add("warnbox");w.append(el("p","Part of this document reads like instructions to an AI system. I ignored it, and nothing it says was used. If you didn't expect that, treat the document with care."));out.append(w)}
 if(d.mismatches.length){const c=card("Your two documents disagree");c.classList.add("warnbox");
  d.mismatches.forEach(m=>c.append(el("p","• "+m.field.replaceAll("_"," ")+": letter says “"+m.letter+"”, EOB says “"+m.eob+"”")));
  c.append(el("p","Mismatches can be worth raising in your appeal. Confirm them against the originals.","mut"));out.append(c)}
 const t=d.trust,ts=card("What was checked");ts.append(el("p",d.verified_against==="pdf_text_layer"?"Quotes were checked against the text inside your PDF, not just the AI's reading of it.":"Quotes were checked against the AI's reading of your document. Compare them with your original.","mut"));const g=el("div",null,"trust");
 [["Facts matched to your letter",t.facts_kept+" of "+t.facts_proposed],["Rights with an official source",t.rights_after_citation_check+" of "+t.rights_proposed],["Rights confirmed by a second check",t.rights_final+" of "+t.rights_after_citation_check]]
  .forEach(([l,v])=>{const s=el("div",null,"stat");s.append(el("b",v),el("span",l,"mut"));g.append(s)});ts.append(g);out.append(ts);
 const fc=card("What your letter says");
 Object.entries(d.facts).forEach(([k,v])=>{const r=el("div",null,"fact");r.append(el("div",k.replaceAll("_"," "),"k"),el("div",v.value),el("q","“"+v.quote+"”"));fc.append(r)});
 if(d.not_found.length)fc.append(el("p","Not found in your letter: "+d.not_found.join(", ").replaceAll("_"," "),"mut"));out.append(fc);
 if(d.rights.length){const rc=card("Your appeal rights (official sources)");
  d.rights.forEach(x=>{rc.append(el("p",x.statement));x.sources.forEach(s=>{const a=el("a",s.title||s.url);a.href=s.url;a.target="_blank";a.rel="noopener noreferrer";const w=el("div");w.append(a);rc.append(w)})});out.append(rc)}
 const ck=card("What to gather");const ul=el("ul");d.checklist.tips.forEach(x=>ul.append(el("li",x)));ck.append(ul,el("p","General tips, not legal advice.","mut"));out.append(ck);
 const lc=card("Draft appeal letter (edit freely)");const ta=el("textarea");ta.value=d.letter;lc.append(ta);
 const row=el("div",null,"row"),cp=el("button","Copy letter"),dn=el("button","Download .txt","alt");
 cp.onclick=async()=>{try{await navigator.clipboard.writeText(ta.value);cp.textContent="Copied"}catch(e){ta.select();cp.textContent="Press Ctrl/Cmd+C"}setTimeout(()=>cp.textContent="Copy letter",1800)};
 dn.onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([ta.value],{type:"text/plain"}));a.download="appeal-letter.txt";a.click();URL.revokeObjectURL(a.href)};
 row.append(cp,dn);lc.append(row,el("p",d.disclaimer,"mut"));out.append(lc);
}
fetch("/api/samples").then(r=>r.json()).then(d=>{if(!d.samples||!d.samples.length)return;$("samples").hidden=false;
 d.samples.forEach(n=>{const b=el("button",n.replace(".pdf","").replaceAll("_"," "),"alt");b.style.margin="2px 6px 2px 0";b.style.padding="4px 10px";
  b.onclick=async()=>{const blob=await (await fetch("/api/samples/"+encodeURIComponent(n))).blob();const dt=new DataTransfer();dt.items.add(new File([blob],n,{type:"application/pdf"}));
   $("f1").files=dt.files;$("n1").textContent=n;$("go").click()};$("sl").append(b)})}).catch(()=>{});
$("go").onclick=async()=>{const out=$("out"),b=$("go");
 if(!$("f1").files[0]){out.replaceChildren(el("p","Choose your denial letter first.","mut"));return}
 b.disabled=true;out.replaceChildren(el("p","Reading your documents… this can take up to a minute.","mut"));
 try{const fd=new FormData();fd.append("file",$("f1").files[0]);if($("f2").files[0])fd.append("eob",$("f2").files[0]);
  const r=await fetch("/api/appeal",{method:"POST",body:fd});const d=await r.json();
  if(!r.ok){out.replaceChildren(el("p",d.error||"Something went wrong.","mut"));return}render(d)
 }catch(e){out.replaceChildren(el("p","Network error. Please try again.","mut"))}finally{b.disabled=false}};
</script></main></body></html>"""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Health claim checker</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#666;--card:#fff;--bd:#dcdcd4;--ac:#0b5cad}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--fg:#eee;--mut:#9a9a92;--card:#201f1d;--bd:#3a3935;--ac:#6db3f2}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:640px;margin:0 auto;padding:24px 16px}
textarea{width:100%;box-sizing:border-box;min-height:110px;padding:10px;font:inherit;color:inherit;background:var(--card);border:1px solid var(--bd);border-radius:8px}
button{margin-top:10px;padding:10px 18px;font:inherit;border:0;border-radius:8px;background:var(--ac);color:#fff;cursor:pointer}
button:disabled{opacity:.6}.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px 14px;margin-top:14px}
.v{font-weight:600}.mut{color:var(--mut);font-size:14px}a{color:var(--ac);overflow-wrap:anywhere}
</style></head><body><main>
<h1>Health claim checker</h1>
<p class="mut">Paste something you saw online. I check it against WHO, CDC, NIH, NHS and similar sources.</p>
<label for="t" class="mut">Claim or message</label>
<textarea id="t" maxlength="1000"></textarea><br><button id="b">Check it</button>
<div id="out" aria-live="polite"></div>
<p class="mut">General information only, not medical advice.</p>
<script>
const L={supported:"✅ Supported by evidence",contradicted:"❌ Contradicted by evidence",misleading:"⚠️ Misleading or missing context",insufficient_evidence:"❓ Not enough evidence in my sources"};
const out=document.getElementById("out"),b=document.getElementById("b"),t=document.getElementById("t");
function el(tag,txt,cls){const e=document.createElement(tag);if(txt)e.textContent=txt;if(cls)e.className=cls;return e}
b.onclick=async()=>{b.disabled=true;out.replaceChildren(el("p","Checking…","mut"));
 try{const r=await fetch("/api/check",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({text:t.value})});
  const d=await r.json();out.replaceChildren();
  if(!r.ok){out.append(el("p",d.error||"Something went wrong.","mut"));return}
  d.safety_notes.forEach(n=>{const c=el("div",null,"card");c.append(el("p",n));out.append(c)});
  if(!d.results.length)out.append(el("p","I didn't find a checkable health claim in that.","mut"));
  d.results.forEach(x=>{const c=el("div",null,"card");c.append(el("div","Claim: "+x.claim),el("div",L[x.verdict],"v"),el("div",x.explanation));
   x.sources.forEach(s=>{const a=el("a",s.title||s.url);a.href=s.url;a.rel="noopener noreferrer";a.target="_blank";const w=el("div");w.append(a);c.append(w)});out.append(c)})
 }catch(e){out.replaceChildren(el("p","Network error. Please try again.","mut"))}finally{b.disabled=false}};
</script></main></body></html>"""


def create_app(services: Services | None = None) -> Starlette:
    svc = services or Services()

    async def index(request: Request) -> Response:
        return HTMLResponse(PAGE)

    async def api_check(request: Request) -> Response:
        ip = request.client.host if request.client else "unknown"
        if not svc.limiter.allow("web:" + ip):
            return JSONResponse({"error": "Too many requests. Please wait a few minutes."}, status_code=429)
        try:
            body = await request.json()
            text = body.get("text", "") if isinstance(body, dict) else ""
        except ValueError:
            return JSONResponse({"error": "Send JSON like {\"text\": \"...\"}."}, status_code=400)
        if not isinstance(text, str) or not text.strip():
            return JSONResponse({"error": "Please enter a claim to check."}, status_code=400)
        try:
            report = await run_in_threadpool(svc.check, text)
        except LLMError:
            log.exception("model call failed")
            return JSONResponse({"error": "The fact-checking model is unavailable. Please try again."}, status_code=503)
        return JSONResponse(report_to_dict(report))

    async def appeal_page(request: Request) -> Response:
        return HTMLResponse(APPEAL_PAGE)

    async def api_appeal(request: Request) -> Response:
        ip = request.client.host if request.client else "unknown"
        if not svc.limiter.allow("appeal:" + ip):
            return JSONResponse({"error": "Too many requests. Please wait a few minutes."}, status_code=429)
        try:
            form = await request.form()
            upload, eob_up = form.get("file"), form.get("eob")
            data = await upload.read() if hasattr(upload, "read") else b""
            eob = (await eob_up.read() if hasattr(eob_up, "read") else b"") or None
        except Exception:
            return JSONResponse({"error": "Upload a file in the 'file' field."}, status_code=400)
        try:
            result = await run_in_threadpool(svc.appeal, data, eob)
        except UnsupportedDocument as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except LLMError:
            log.exception("model call failed")
            return JSONResponse({"error": "The model is unavailable. Please try again."}, status_code=503)
        return JSONResponse(appeal_to_dict(result))

    def _sample_names() -> list[str]:
        return sorted(p.name for p in SAMPLES_DIR.glob("*.pdf")) if SAMPLES_DIR.is_dir() else []

    async def api_samples(request: Request) -> Response:
        return JSONResponse({"samples": _sample_names()})

    async def api_sample(request: Request) -> Response:
        name = request.path_params["name"]
        if name not in _sample_names():  # exact match against the directory listing: no path tricks
            return JSONResponse({"error": "Unknown sample."}, status_code=404)
        return FileResponse(SAMPLES_DIR / name, media_type="application/pdf")

    async def batch_page(request: Request) -> Response:
        return HTMLResponse(BATCH_PAGE)

    async def api_batch(request: Request) -> Response:
        ip = request.client.host if request.client else "unknown"
        if not svc.limiter.allow("batch:" + ip):
            return JSONResponse({"error": "Too many requests. Please wait a few minutes."}, status_code=429)
        try:
            form = await request.form()
            files = []
            for up in form.getlist("files"):
                if hasattr(up, "read"):
                    files.append((str(getattr(up, "filename", "") or "file"), await up.read()))
        except Exception:
            return JSONResponse({"error": "Upload files in the 'files' field."}, status_code=400)
        if not files:
            return JSONResponse({"error": "Choose at least one file."}, status_code=400)
        if len(files) > MAX_FILES or sum(len(d) for _, d in files) > 40 * 1024 * 1024:
            return JSONResponse({"error": f"At most {MAX_FILES} files and 40 MB per batch."}, status_code=400)
        try:
            items = await run_in_threadpool(svc.batch, files)
        except LLMError:
            log.exception("model call failed")
            return JSONResponse({"error": "The model is unavailable. Please try again."}, status_code=503)
        rows = [summary_row(i) for i in items]
        return JSONResponse({"items": [{**r, "letter": i.result.letter if i.result else ""} for r, i in zip(rows, items)],
                             "csv": to_csv(rows)})

    async def _reply_async(sender: str, body: str) -> None:
        s = svc.settings
        text = await run_in_threadpool(svc.reply_text, body)
        try:
            await run_in_threadpool(
                send_whatsapp, s.twilio_account_sid, s.twilio_auth_token, s.twilio_whatsapp_from, sender, text
            )
        except Exception:
            log.exception("failed to send WhatsApp reply to %s", _sender_id(sender))

    async def twilio_webhook(request: Request) -> Response:
        s = svc.settings
        form = await request.form()
        params = {k: str(v) for k, v in form.items()}
        if s.twilio_auth_token and s.public_url:
            url = s.public_url.rstrip("/") + "/webhook/twilio"
            if not valid_signature(s.twilio_auth_token, url, params, request.headers.get("X-Twilio-Signature", "")):
                return Response("invalid signature", status_code=403)
        sender, body = params.get("From", ""), params.get("Body", "")
        log.info("message from %s (%d chars)", _sender_id(sender), len(body))
        if not svc.limiter.allow("wa:" + sender):
            return Response(twiml_message("You're sending messages quickly. Please wait a few minutes."), media_type="application/xml")
        if s.twilio_async and body.strip():
            return Response(EMPTY_TWIML, media_type="application/xml", background=BackgroundTask(_reply_async, sender, body))
        reply = await run_in_threadpool(svc.reply_text, body)
        return Response(twiml_message(reply), media_type="application/xml")

    async def health(request: Request) -> Response:
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/check", api_check, methods=["POST"]),
            Route("/appeal", appeal_page),
            Route("/api/appeal", api_appeal, methods=["POST"]),
            Route("/api/samples", api_samples),
            Route("/api/samples/{name}", api_sample),
            Route("/batch", batch_page),
            Route("/api/batch", api_batch, methods=["POST"]),
            Route("/webhook/twilio", twilio_webhook, methods=["POST"]),
            Route("/healthz", health),
        ]
    )


app = create_app()
