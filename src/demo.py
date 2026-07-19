"""Demo mode — an opt-in, public, locked-down chat showcase.

Off by default (``DEMO=false``) so a fresh fork gets the full authenticated
app. When ``DEMO=true``, ``AuthMiddleware`` mints a per-visitor synthetic owner
and lets an unauthenticated visitor reach ONLY the core chat surface, under a
least-privilege profile, rate-limited, with ephemeral (in-memory) history that
is never written to the deployer's disk.

Everything demo-specific lives here so the rest of the app calls into this
module rather than scattering ``if DEMO`` branches. When the flag is off, this
module is inert: ``DEMO_MODE`` is ``False`` and none of the hooks fire.

Security notes:
  * The pinned model + endpoint + API key are applied at read time
    (``sync_session_metadata``) and never persisted — the key stays env-only.
  * Demo owners are ``demo-<uuid>`` strings; ``is_demo_owner`` is a prefix
    check. The literal ``"demo"`` remains a RESERVED_USERNAME (a different
    string), so there is no collision with the account sentinel.
  * The route whitelist (``is_demo_allowed``) is the middleware boundary; the
    privilege profile (``DEMO_PRIVILEGES``) is the in-handler boundary. Both
    must hold for a capability to be reachable.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple


def _flag(name: str, default: str = "false") -> bool:
    """Parse a boolean env flag. true/1/yes (any case) is on; all else off."""
    return os.getenv(name, default).strip().lower() in ("true", "1", "yes")


def _int_env(name: str, default: int) -> int:
    """Parse a non-negative int env var. Unset/invalid falls back to `default`
    (a missing var must NEVER mean "unlimited"); only an explicit 0 disables a
    dimension. Negative values are treated as invalid → default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError:
        return default
    return val if val >= 0 else default


# --- The flag ---------------------------------------------------------------
DEMO_MODE: bool = _flag("DEMO", "false")

# --- Per-visitor identity ---------------------------------------------------
DEMO_COOKIE = "odysseus_demo"          # separate from the authed odysseus_session cookie
DEMO_OWNER_PREFIX = "demo-"            # owner ids look like demo-<32 hex uuid>
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")

# --- Pinned model + endpoint (env key, never persisted) ---------------------
DEMO_MODEL = (os.getenv("DEMO_MODEL", "").strip() or "gpt-5.6-luna")
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# --- Usage limits (only consulted when DEMO_MODE). 0 disables the dimension. -
DEMO_RATE_LIMIT_PER_MINUTE = _int_env("DEMO_RATE_LIMIT_PER_MINUTE", 10)
DEMO_MAX_MESSAGES_PER_SESSION = _int_env("DEMO_MAX_MESSAGES_PER_SESSION", 30)
DEMO_MAX_OUTPUT_TOKENS = _int_env("DEMO_MAX_OUTPUT_TOKENS", 512)
if DEMO_MAX_OUTPUT_TOKENS <= 0:
    # A 0/unset output cap would mean "no cap" downstream — keep a sane floor so
    # the demo can never be turned into an unbounded free generator.
    DEMO_MAX_OUTPUT_TOKENS = 512

LIMIT_MESSAGE = (
    "**Demo limit reached — deploy your own to keep going.**\n\n"
    "This is a public demo with usage caps so it stays affordable. Click "
    "**Deploy to Render** in the README to run your own private instance."
)

# --- Least-privilege profile ------------------------------------------------
# Consumed by AuthManager.get_privileges for demo owners; this drives the
# existing per-user enforcement in routes/chat_routes.py (which disables the
# matching tools) and _enforce_chat_privileges (allowed_models). Everything
# that writes, executes, spends extra, or reaches outward is OFF.
DEMO_PRIVILEGES: Dict[str, object] = {
    "can_use_agent": False,        # forces plain chat mode (no tool loop)
    "can_use_browser": False,      # no builtin browser
    "can_use_bash": False,         # no shell / python / file tools
    "can_use_documents": False,    # no document create/edit
    "can_use_research": False,     # no deep research
    "can_generate_images": False,  # no metered image spend
    "can_manage_memory": False,    # no memory/skills writes
    # Per-session cap is enforced in-memory (demo history isn't persisted, so a
    # DB-count daily cap would always read 0). Keep this at 0 here.
    "max_messages_per_day": 0,
    "allowed_models": [DEMO_MODEL],
    "allowed_models_restricted": True,
    "block_all_models": False,
}

# --- Route whitelist (the middleware boundary) ------------------------------
# The ONLY surface a demo visitor may reach. Auth-exempt routes (login, status,
# features, settings, version, /static) are handled by AuthMiddleware BEFORE the
# demo path runs, so they need not be repeated here.
_DEMO_ALLOWED_EXACT = {
    ("GET", "/"),                   # SPA shell
    ("GET", "/api/default-chat"),   # supplies endpoint+model so first send can create a session
    ("POST", "/api/session"),       # create the chat session (endpoint/model forced server-side)
    ("POST", "/api/chat_stream"),   # send a message + streamed reply (capabilities locked below)
}
_DEMO_ALLOWED_PREFIXES: Tuple[Tuple[str, str], ...] = (("GET", "/static"),)


def is_demo_owner(username: Optional[str]) -> bool:
    """True for a per-visitor demo owner id (demo-<uuid>). Prefix check — does
    NOT match the literal reserved username "demo"."""
    return bool(username) and str(username).startswith(DEMO_OWNER_PREFIX)


def is_demo_allowed(method: str, path: str) -> bool:
    """True if (method, path) is on the demo route whitelist."""
    if (method, path) in _DEMO_ALLOWED_EXACT:
        return True
    return any(method == m and path.startswith(p) for m, p in _DEMO_ALLOWED_PREFIXES)


# --- Per-visitor cookie / owner ---------------------------------------------
def resolve_demo_owner(request) -> Tuple[str, Optional[str]]:
    """Return ``(owner, new_cookie_value)`` for a demo visitor.

    Reuses the visitor's existing demo cookie when present and well-formed;
    otherwise mints a fresh unguessable id. ``new_cookie_value`` is the raw
    token to set on the response (or ``None`` when the cookie already existed).
    """
    tok = request.cookies.get(DEMO_COOKIE, "")
    if tok and _TOKEN_RE.match(tok):
        return DEMO_OWNER_PREFIX + tok, None
    new = uuid.uuid4().hex
    return DEMO_OWNER_PREFIX + new, new


def set_demo_cookie(response, token: str) -> None:
    """Set the per-visitor demo cookie: httponly, samesite=lax, secure per
    SECURE_COOKIES (true on Render), short-lived (history is ephemeral)."""
    response.set_cookie(
        key=DEMO_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=os.getenv("SECURE_COOKIES", "false").lower() == "true",
        max_age=60 * 60 * 24,  # 1 day; a returning visitor keeps their session cap within it
        path="/",
    )


# --- Session config (pinned model + env key, never persisted) ---------------
def apply_demo_session_config(session) -> None:
    """Force a demo session to talk to OpenAI with the pinned model and the
    server's env OPENAI_API_KEY. Called from sync_session_metadata so this is
    authoritative on every read — the key is never read from, or written to, the
    DB. No-op-safe when the env key is missing (the LLM call then fails cleanly
    as "server missing key" rather than leaking a partial config)."""
    key = os.getenv("OPENAI_API_KEY")
    session.endpoint_url = OPENAI_CHAT_URL
    session.model = DEMO_MODEL
    session.headers = {"Authorization": f"Bearer {key}"} if key else {}


# --- Rate + per-session message limits --------------------------------------
_rate_limiter = None
if DEMO_RATE_LIMIT_PER_MINUTE > 0:
    from src.rate_limiter import RateLimiter
    _rate_limiter = RateLimiter(max_requests=DEMO_RATE_LIMIT_PER_MINUTE, window_seconds=60)

# owner -> [message_count, last_touch_monotonic]
_session_counts: Dict[str, List[float]] = {}
_counts_lock = threading.Lock()
_last_purge = time.monotonic()
_PURGE_AFTER = 60 * 60 * 24  # forget a visitor's count a day after their last message


def _maybe_purge(now: float) -> None:
    """Drop stale per-visitor counters so the dict can't grow without bound."""
    global _last_purge
    if now - _last_purge < _PURGE_AFTER:
        return
    _last_purge = now
    stale = [k for k, v in _session_counts.items() if now - v[1] > _PURGE_AFTER]
    for k in stale:
        del _session_counts[k]


def check_demo_limits(owner: str, client_ip: str) -> Optional[str]:
    """Return a friendly limit message if the visitor is over a cap, else None.

    Call once per chat send, BEFORE spending the key. Enforces (a) a sliding
    per-minute rate limit keyed by visitor+IP and (b) a total per-session
    message cap. A tripped cap returns text, never an exception, so the caller
    can render it as a normal assistant turn instead of a 500/hang.
    """
    if _rate_limiter is not None:
        if not _rate_limiter.check(f"{owner}|{client_ip}"):
            return LIMIT_MESSAGE
    if DEMO_MAX_MESSAGES_PER_SESSION > 0:
        now = time.monotonic()
        with _counts_lock:
            _maybe_purge(now)
            entry = _session_counts.get(owner)
            used = entry[0] if entry else 0
            if used >= DEMO_MAX_MESSAGES_PER_SESSION:
                return LIMIT_MESSAGE
            _session_counts[owner] = [used + 1, now]
    return None


def demo_client_ip(request) -> str:
    """Best-effort client IP for rate limiting, honoring Render's proxy."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if getattr(request, "client", None) else ""


async def demo_limit_sse(message: str):
    """SSE generator that renders `message` as a single assistant turn and ends.
    Matches the chat_stream framing the frontend consumes (data: {delta} …
    data: [DONE]) so a tripped limit shows as a normal reply, not a broken
    stream."""
    yield f'data: {json.dumps({"delta": message})}\n\n'
    yield "data: [DONE]\n\n"


def clamp_demo_output_tokens(current: Optional[int]) -> int:
    """Return the max_tokens to use for a demo turn: the tighter of the
    request's value and DEMO_MAX_OUTPUT_TOKENS. Treats 0/None (which mean
    "no cap" downstream) as needing the demo cap applied."""
    if not current or current > DEMO_MAX_OUTPUT_TOKENS:
        return DEMO_MAX_OUTPUT_TOKENS
    return current


def log_startup_mode(logger) -> None:
    """Log which mode booted so a misconfigured deploy is obvious in the logs."""
    if DEMO_MODE:
        logger.warning(
            "[startup] DEMO mode ENABLED — public, no-signup, locked-down chat demo is live "
            "and spends OPENAI_API_KEY. model=%s rate=%s/min msgs/session=%s max_output_tokens=%s",
            DEMO_MODEL,
            DEMO_RATE_LIMIT_PER_MINUTE or "unlimited",
            DEMO_MAX_MESSAGES_PER_SESSION or "unlimited",
            DEMO_MAX_OUTPUT_TOKENS,
        )
    else:
        logger.info("[startup] normal (authenticated) mode — DEMO is off")
