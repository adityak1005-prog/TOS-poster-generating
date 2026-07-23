"""
Opt-in email: sends the finished poster to a visitor's address if and when
they click "email me this poster" on the reveal screen (after they've
already seen the poster + QR -- nothing is emailed automatically during
capture). Uses Gmail SMTP with an existing Gmail account + an "App
Password" (not the account's normal login password -- Google requires
2-Step Verification enabled on the account before it will let you generate
one, at https://myaccount.google.com/apppasswords).

This module never blocks or breaks the main booth pipeline: app.py's
/booth/email/{job_id} endpoint runs the actual send in a background asyncio
task and wraps it in a try/except, so a bad address, Gmail's rate limit, or
a network hiccup here just means "no email went out" -- the poster is still
reachable via the QR/link for the storage retention window (see storage.py
/ the cleanup sweep in app.py).

Caveats worth knowing about Gmail SMTP specifically (vs. a dedicated
transactional email provider):
- Gmail rate-limits outbound mail to roughly 500 messages/day per account --
  fine for a single college-fest day, but don't assume this scales further.
- Attachments from a personal Gmail account are somewhat more likely to be
  flagged as spam by strict recipient mail servers than mail from a
  dedicated sender domain. Acceptable trade-off for "free and zero setup."
"""

import os
import re
import smtplib
import requests
from email.message import EmailMessage

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
EMAIL_FROM_NAME = os.environ.get("EMAIL_FROM_NAME", "AI Movie Booth")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_TIMEOUT_S = 15

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_plausible_email(address: str) -> bool:
    """Cheap sanity check before attempting to send -- not full RFC
    validation, just enough to skip obviously-bad input without wasting an
    SMTP round trip."""
    return bool(address) and bool(_EMAIL_RE.match(address.strip()))


def send_poster_email(to_address: str, character: str, caption: str, image_bytes: bytes) -> None:
    """
    Sends the composed poster (already in memory, no need to fetch it back
    from Supabase) as a JPEG attachment. Raises on any failure -- callers
    (app.py) are expected to catch and log rather than let this break the
    booth job.
    """
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        raise RuntimeError("GMAIL_ADDRESS / GMAIL_APP_PASSWORD not configured in .env")

    to_address = to_address.strip()
    if not is_plausible_email(to_address):
        raise ValueError(f"'{to_address}' doesn't look like a valid email address")

    msg = EmailMessage()
    msg["Subject"] = f"Your AI Movie Booth poster -- you matched {character}!"
    msg["From"] = f"{EMAIL_FROM_NAME} <{GMAIL_ADDRESS}>"
    msg["To"] = to_address
    msg.set_content(
        f"You matched: {character}\n\n"
        f"{caption}\n\n"
        "Your poster is attached. Thanks for stopping by the AI Movie Booth!"
    )
    msg.add_attachment(
        image_bytes, maintype="image", subtype="jpeg", filename="movie_booth_poster.jpg"
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_S) as smtp:
        smtp.starttls()
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)


def send_poster_email_from_url(
    to_address: str, character: str, caption: str, image_url: str, fetch_timeout_s: int = 15
) -> None:
    """
    Used by the post-reveal "email me this poster" flow: the composed image
    bytes aren't kept in memory past the original capture request, but the
    poster is already sitting at a public Supabase URL by the time someone
    clicks the mail button -- so fetch it back and attach it. Raises on any
    failure (network fetch or SMTP), same contract as send_poster_email;
    callers are expected to run this in a background task so a slow/failed
    send never blocks the booth or the next capture.
    """
    resp = requests.get(image_url, timeout=fetch_timeout_s)
    resp.raise_for_status()
    send_poster_email(to_address, character, caption, resp.content)
