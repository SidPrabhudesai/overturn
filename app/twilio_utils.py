"""Twilio helpers: request signature check, TwiML, and async REST replies."""
from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Mapping
from xml.sax.saxutils import escape

import requests


def valid_signature(auth_token: str, url: str, params: Mapping[str, str], signature: str) -> bool:
    """Twilio signs: HMAC-SHA1(token, url + each POST param as key+value, sorted by key), base64."""
    data = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature or "")


def twiml_message(text: str) -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{escape(text)}</Message></Response>'


EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


def send_whatsapp(account_sid: str, auth_token: str, from_: str, to: str, body: str) -> None:
    """Send a reply through Twilio's REST API (used when the model is too slow for the webhook)."""
    resp = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
        auth=(account_sid, auth_token),
        data={"From": from_, "To": to, "Body": body},
        timeout=20,
    )
    resp.raise_for_status()
