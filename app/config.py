"""Settings and the trusted-source allowlist."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

# Sources a verdict is allowed to rest on. The guard in pipeline.py downgrades any
# verdict whose citations fall outside this list, so edit it deliberately.
TRUSTED_DOMAINS: tuple[str, ...] = (
    "who.int",
    "cdc.gov",
    "nih.gov",
    "medlineplus.gov",
    "fda.gov",
    "nhs.uk",
    "cochrane.org",
    "cochranelibrary.com",
    "aap.org",
    "mayoclinic.org",
    "ecdc.europa.eu",
    # insurance-appeal rights (government sources only)
    "healthcare.gov",
    "cms.gov",
    "dol.gov",
    "hhs.gov",
    "medicare.gov",
)


def is_trusted_url(url: str, domains: tuple[str, ...] = TRUSTED_DOMAINS) -> bool:
    """True for http(s) URLs whose host is a trusted domain or a subdomain of one."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in domains)


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    base_url: str = "https://api.meta.ai/v1"
    model: str = "muse-spark-1.2"
    reasoning_effort: str = "low"
    timeout: float = 45.0
    retriever: str = "local"
    corpus_dir: str = "corpus"
    max_claims: int = 3
    max_input_chars: int = 1000
    twilio_auth_token: str = ""
    public_url: str = ""
    twilio_account_sid: str = ""
    twilio_whatsapp_from: str = "whatsapp:+14155238886"

    @property
    def twilio_async(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    if env is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # python-dotenv is optional
            pass
        env = os.environ

    model = env.get("MODEL", "muse-spark-1.2").strip()
    if "contributor" in model.lower() and env.get("ALLOW_CONTRIBUTOR_TIER") != "1":
        raise ValueError(
            f"{model!r} is the Contributor tier: Meta may train on prompts. "
            "Do not use it for real users' health messages. For synthetic eval claims only, "
            "set ALLOW_CONTRIBUTOR_TIER=1."
        )
    return Settings(
        api_key=(env.get("MODEL_API_KEY") or env.get("META_API_KEY") or "").strip(),
        base_url=env.get("META_BASE_URL", "https://api.meta.ai/v1").strip(),
        model=model,
        reasoning_effort=env.get("REASONING_EFFORT", "low").strip(),
        retriever=env.get("RETRIEVER", "local").strip().lower(),
        corpus_dir=env.get("CORPUS_DIR", "corpus").strip(),
        twilio_auth_token=env.get("TWILIO_AUTH_TOKEN", "").strip(),
        public_url=env.get("PUBLIC_URL", "").strip(),
        twilio_account_sid=env.get("TWILIO_ACCOUNT_SID", "").strip(),
        twilio_whatsapp_from=env.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip(),
    )
