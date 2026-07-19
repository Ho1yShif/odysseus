"""Demo mode — unit coverage for the DEMO-flag public chat showcase.

These exercise the choke points src/demo.py owns plus the two cross-module
hooks that must fire only for demo owners: the ephemeral-history persist skip
(SessionManager) and the least-privilege profile (AuthManager). Everything runs
with DEMO forced on via importlib.reload so the module-level flag reflects it.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def demo(monkeypatch):
    """Reload src.demo with DEMO=true so DEMO_MODE and the limiter are live."""
    monkeypatch.setenv("DEMO", "true")
    monkeypatch.setenv("DEMO_MODEL", "gpt-4.1-nano")
    monkeypatch.setenv("DEMO_RATE_LIMIT_PER_MINUTE", "10")
    monkeypatch.setenv("DEMO_MAX_MESSAGES_PER_SESSION", "3")
    monkeypatch.setenv("DEMO_MAX_OUTPUT_TOKENS", "512")
    import src.demo as d
    d = importlib.reload(d)
    yield d
    # Restore module to the ambient env for other tests.
    importlib.reload(d)


# --- identity ---------------------------------------------------------------
def test_is_demo_owner_matches_prefix_not_reserved_sentinel(demo):
    assert demo.is_demo_owner("demo-" + "a" * 32) is True
    # The literal reserved username must NOT be treated as a demo owner.
    assert demo.is_demo_owner("demo") is False
    assert demo.is_demo_owner("admin") is False
    assert demo.is_demo_owner("") is False
    assert demo.is_demo_owner(None) is False


# --- route whitelist --------------------------------------------------------
def test_route_whitelist_allows_only_the_chat_surface(demo):
    assert demo.is_demo_allowed("GET", "/") is True
    assert demo.is_demo_allowed("GET", "/api/default-chat") is True
    assert demo.is_demo_allowed("POST", "/api/session") is True
    assert demo.is_demo_allowed("POST", "/api/chat_stream") is True
    assert demo.is_demo_allowed("GET", "/static/app.js") is True
    # Everything dangerous stays off the whitelist.
    for method, path in [
        ("GET", "/api/tasks"),
        ("GET", "/api/assistant"),
        ("POST", "/api/mcp/servers"),
        ("POST", "/api/shell"),
        ("POST", "/api/upload"),
        ("GET", "/api/auth/settings"),  # handled as auth-exempt upstream, not here
        ("DELETE", "/api/session"),     # wrong method for a whitelisted path
    ]:
        assert demo.is_demo_allowed(method, path) is False, (method, path)


# --- cookie / owner minting -------------------------------------------------
def test_resolve_demo_owner_mints_then_reuses(demo):
    no_cookie = SimpleNamespace(cookies={})
    owner, new = demo.resolve_demo_owner(no_cookie)
    assert owner.startswith("demo-")
    assert new is not None and len(new) == 32
    # A returning visitor with a well-formed cookie keeps their owner, no re-mint.
    returning = SimpleNamespace(cookies={demo.DEMO_COOKIE: new})
    owner2, new2 = demo.resolve_demo_owner(returning)
    assert owner2 == owner
    assert new2 is None
    # A malformed cookie is rejected and a fresh id is minted.
    junk = SimpleNamespace(cookies={demo.DEMO_COOKIE: "not-a-valid-token"})
    owner3, new3 = demo.resolve_demo_owner(junk)
    assert new3 is not None and owner3 != owner


# --- output-token clamp -----------------------------------------------------
def test_clamp_output_tokens_never_uncapped(demo):
    cap = demo.DEMO_MAX_OUTPUT_TOKENS
    assert demo.clamp_demo_output_tokens(0) == cap       # 0 == "no cap" downstream
    assert demo.clamp_demo_output_tokens(None) == cap
    assert demo.clamp_demo_output_tokens(10_000) == cap  # over cap -> clamped
    assert demo.clamp_demo_output_tokens(100) == 100     # under cap -> kept


def test_output_token_floor_when_env_zero(monkeypatch):
    # An explicit 0 / negative output cap must fall back to a positive floor —
    # never "unlimited".
    monkeypatch.setenv("DEMO", "true")
    monkeypatch.setenv("DEMO_MAX_OUTPUT_TOKENS", "0")
    import src.demo as d
    d = importlib.reload(d)
    try:
        assert d.DEMO_MAX_OUTPUT_TOKENS > 0
    finally:
        importlib.reload(d)


# --- per-session message cap ------------------------------------------------
def test_message_cap_trips_after_limit(demo):
    owner = "demo-" + "b" * 32
    # Rate limit is 10/min so the first calls pass on that axis; cap is 3.
    seen = [demo.check_demo_limits(owner, "1.2.3.4") for _ in range(3)]
    assert seen == [None, None, None]
    tripped = demo.check_demo_limits(owner, "1.2.3.4")
    assert tripped == demo.LIMIT_MESSAGE


def test_rate_limit_trips_independently(monkeypatch):
    monkeypatch.setenv("DEMO", "true")
    monkeypatch.setenv("DEMO_RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setenv("DEMO_MAX_MESSAGES_PER_SESSION", "0")  # disable msg cap
    import src.demo as d
    d = importlib.reload(d)
    try:
        owner = "demo-" + "c" * 32
        assert d.check_demo_limits(owner, "9.9.9.9") is None
        assert d.check_demo_limits(owner, "9.9.9.9") is None
        assert d.check_demo_limits(owner, "9.9.9.9") == d.LIMIT_MESSAGE
    finally:
        importlib.reload(d)


# --- pinned session config (env key, never persisted) -----------------------
def test_apply_demo_session_config_pins_model_and_env_key(demo, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-xyz")
    sess = SimpleNamespace(endpoint_url="https://evil/x", model="gpt-4-turbo", headers={})
    demo.apply_demo_session_config(sess)
    assert sess.model == demo.DEMO_MODEL
    assert sess.endpoint_url == demo.OPENAI_CHAT_URL
    assert sess.headers == {"Authorization": "Bearer sk-live-xyz"}


def test_apply_demo_session_config_no_key_is_safe(demo, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sess = SimpleNamespace(endpoint_url="", model="", headers={"x": "y"})
    demo.apply_demo_session_config(sess)
    # No partial/leaked auth header when the server has no key.
    assert sess.headers == {}
    assert sess.model == demo.DEMO_MODEL


# --- privileges choke point (AuthManager.get_privileges) --------------------
def test_get_privileges_returns_locked_profile_for_demo_owner(demo):
    from core.auth import AuthManager
    # The demo branch returns before self.users is touched, so a bare instance
    # is enough — no config needed.
    am = AuthManager.__new__(AuthManager)
    privs = am.get_privileges("demo-" + "d" * 32)
    # Every spend / escalation surface is off.
    for off in (
        "can_use_agent", "can_use_browser", "can_use_bash", "can_use_documents",
        "can_use_research", "can_generate_images", "can_manage_memory",
    ):
        assert privs[off] is False, off
    assert privs["allowed_models"] == [demo.DEMO_MODEL]
    assert privs["allowed_models_restricted"] is True


# --- ephemeral history (SessionManager._persist_message skip) ---------------
def test_persist_message_skipped_for_demo_owner(demo, monkeypatch):
    import core.session_manager as SM
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
        owner="demo-" + "e" * 32
    )
    monkeypatch.setattr(SM, "SessionLocal", MagicMock(return_value=db))

    manager = SM.SessionManager.__new__(SM.SessionManager)
    manager.sessions = {"sid": SimpleNamespace(history=[])}
    from core.models import ChatMessage
    manager._persist_message("sid", ChatMessage("user", "secret demo chat"))

    # Nothing written to disk for a demo owner.
    db.add.assert_not_called()
    db.commit.assert_not_called()
