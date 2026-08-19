"""Passwordless sign-in by one-time code, plus signed session tokens.

Design notes, since this is the security-sensitive part of the app:

* Codes are never stored — only an HMAC of the code, keyed by the server
  secret and bound to the identifier, compared in constant time.
* Sign-up and sign-in are the same flow, and the API answers identically
  whether or not the identifier is already registered, so the endpoint cannot
  be used to enumerate accounts.
* Every code expires, survives a limited number of guesses, and is rate
  limited per identifier both by a resend cooldown and an hourly cap.
* Session tokens are stateless HMACs with an expiry baked into the payload.
"""

import base64
import hashlib
import hmac
import logging
import re
import secrets
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CODE_LENGTH = 6
CODE_TTL_SECONDS = 600           # 10 minutes
RESEND_COOLDOWN_SECONDS = 60
MAX_SENDS_PER_WINDOW = 5
SEND_WINDOW_SECONDS = 3600
MAX_VERIFY_ATTEMPTS = 5
SESSION_TTL_SECONDS = 30 * 86400

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")


def normalize_email(raw: str) -> Optional[str]:
    """Lowercase and validate an email address."""
    value = (raw or "").strip().lower()
    if not value or len(value) > 254 or not _EMAIL_RE.match(value):
        return None
    return value


def normalize_phone(raw: str) -> Optional[str]:
    """
    Normalise a phone number to E.164-ish form ("+79991234567").

    A local Russian "8XXXXXXXXXX" is rewritten to "+7XXXXXXXXXX", which is the
    single most common way people type their number.
    """
    value = re.sub(r"[\s\-()./]", "", (raw or "").strip())
    if not value:
        return None

    plus = value.startswith("+")
    digits = value[1:] if plus else value
    if not digits.isdigit():
        return None

    if not plus and len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if not 8 <= len(digits) <= 15:
        return None
    return "+" + digits


def detect_channel(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Work out whether the user typed an email or a phone number.

    Returns (channel, normalized identifier) or (None, None).
    """
    value = (raw or "").strip()
    if "@" in value:
        email = normalize_email(value)
        return ("email", email) if email else (None, None)
    phone = normalize_phone(value)
    return ("phone", phone) if phone else (None, None)


def generate_code() -> str:
    """A cryptographically random numeric code."""
    return "".join(secrets.choice("0123456789") for _ in range(CODE_LENGTH))


def hash_code(code: str, identifier: str, secret: str) -> str:
    """HMAC of a code, bound to the identifier so codes are not portable."""
    message = f"{identifier}|{code}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def issue_token(user_id: int, secret: str, ttl: int = SESSION_TTL_SECONDS,
                now: Optional[float] = None) -> str:
    """Create a signed session token for `user_id`."""
    expires = int((now if now is not None else time.time()) + ttl)
    payload = f"{user_id}.{expires}"
    signature = hmac.new(
        secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).digest()
    return f"{payload}.{base64.urlsafe_b64encode(signature).decode().rstrip('=')}"


def verify_token(token: str, secret: str, now: Optional[float] = None) -> Optional[int]:
    """Return the user id carried by a valid, unexpired token."""
    if not token or token.count(".") != 2:
        return None
    user_part, expires_part, signature = token.split(".")

    expected = issue_token_signature(f"{user_part}.{expires_part}", secret)
    if not hmac.compare_digest(signature, expected):
        return None

    try:
        user_id, expires = int(user_part), int(expires_part)
    except ValueError:
        return None
    if expires < (now if now is not None else time.time()):
        return None
    return user_id


def issue_token_signature(payload: str, secret: str) -> str:
    signature = hmac.new(
        secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).digest()
    return base64.urlsafe_b64encode(signature).decode().rstrip("=")


@dataclass
class CodeRequest:
    """Outcome of asking for a verification code."""

    ok: bool
    channel: str = ""
    identifier: str = ""
    code: str = ""
    error: str = ""
    retry_after: int = 0


@dataclass
class VerifyResult:
    """Outcome of checking a verification code."""

    ok: bool
    user_id: Optional[int] = None
    created: bool = False
    error: str = ""
    attempts_left: int = 0


class AuthService:
    """Issues and checks one-time codes on top of `Storage`."""

    def __init__(self, storage, secret: str):
        self.storage = storage
        self.secret = secret

    def request_code(self, raw_identifier: str, now: Optional[float] = None) -> CodeRequest:
        """
        Produce a code for an identifier, enforcing the rate limits.

        The caller is responsible for delivering `code`; it is deliberately not
        persisted anywhere in plain text.
        """
        moment = now if now is not None else time.time()
        channel, identifier = detect_channel(raw_identifier)
        if not identifier:
            return CodeRequest(False, error="invalid_identifier")

        existing = self.storage.get_auth_code(identifier)
        sends = 1
        window_started = moment

        if existing:
            since_last = moment - existing["created_at"]
            if since_last < RESEND_COOLDOWN_SECONDS:
                return CodeRequest(
                    False,
                    channel=channel,
                    identifier=identifier,
                    error="cooldown",
                    retry_after=int(RESEND_COOLDOWN_SECONDS - since_last) + 1,
                )

            window_started = existing["window_started_at"]
            if moment - window_started < SEND_WINDOW_SECONDS:
                if existing["sends"] >= MAX_SENDS_PER_WINDOW:
                    return CodeRequest(
                        False,
                        channel=channel,
                        identifier=identifier,
                        error="too_many_requests",
                        retry_after=int(
                            SEND_WINDOW_SECONDS - (moment - window_started)
                        ) + 1,
                    )
                sends = existing["sends"] + 1
            else:
                window_started = moment

        code = generate_code()
        self.storage.save_auth_code(
            identifier=identifier,
            channel=channel,
            code_hash=hash_code(code, identifier, self.secret),
            expires_at=moment + CODE_TTL_SECONDS,
            window_started_at=window_started,
            sends=sends,
            now=moment,
        )
        return CodeRequest(True, channel=channel, identifier=identifier, code=code)

    def verify(
        self, raw_identifier: str, code: str, now: Optional[float] = None
    ) -> VerifyResult:
        """Check a code and return the account it belongs to, creating it if new."""
        moment = now if now is not None else time.time()
        channel, identifier = detect_channel(raw_identifier)
        if not identifier:
            return VerifyResult(False, error="invalid_identifier")

        record = self.storage.get_auth_code(identifier)
        if record is None:
            return VerifyResult(False, error="no_code")
        if record["expires_at"] < moment:
            self.storage.delete_auth_code(identifier)
            return VerifyResult(False, error="expired")
        if record["attempts"] >= MAX_VERIFY_ATTEMPTS:
            self.storage.delete_auth_code(identifier)
            return VerifyResult(False, error="too_many_attempts")

        submitted = (code or "").strip()
        if not hmac.compare_digest(
            hash_code(submitted, identifier, self.secret), record["code_hash"]
        ):
            attempts = self.storage.bump_auth_attempts(identifier)
            left = max(0, MAX_VERIFY_ATTEMPTS - attempts)
            if left == 0:
                self.storage.delete_auth_code(identifier)
            return VerifyResult(False, error="wrong_code", attempts_left=left)

        # Correct code: burn it so it cannot be replayed.
        self.storage.delete_auth_code(identifier)

        user = self.storage.get_user_by_identifier(channel, identifier)
        if user is not None:
            return VerifyResult(True, user_id=user.id, created=False)

        user = self.storage.create_web_user(channel, identifier)
        return VerifyResult(True, user_id=user.id, created=True)

    def issue_session(self, user_id: int) -> str:
        return issue_token(user_id, self.secret)

    def user_id_from_token(self, token: str) -> Optional[int]:
        return verify_token(token, self.secret)
