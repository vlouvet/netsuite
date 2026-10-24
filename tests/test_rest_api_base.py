"""Tests for `RestApiBase` — the shared HTTP plumbing under both
`NetSuiteRestApi` and `NetSuiteRestlet`."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from netsuite.exceptions import (
    NetsuiteAPIRequestError,
    NetsuiteAPIResponseParsingError,
)
from netsuite.rest_api_base import (
    DEFAULT_SIGNATURE_METHOD,
    RestApiBase,
    authlib_hmac_sha256_sign_method,
)


class _ConcreteApi(RestApiBase):
    """Minimal concretion so we can exercise `RestApiBase` directly."""

    def __init__(self, config):
        self._config = config
        self._default_timeout = 10
        self._concurrent_requests = 5

    def _make_url(self, subpath):
        return f"https://example.com{subpath}"


def _httpx_response(status_code, text, headers=None):
    request = httpx.Request("GET", "https://example.com/x")
    return httpx.Response(
        status_code,
        content=text.encode("utf-8"),
        headers=headers or {},
        request=request,
    )


# ---------------------------------------------------------------------------
# `_request` — wraps `_request_impl` with status checks and JSON decoding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_returns_decoded_json_on_2xx(dummy_config):
    api = _ConcreteApi(dummy_config)
    api._request_impl = AsyncMock(return_value=_httpx_response(200, '{"ok": true}'))
    assert await api._request("GET", "/x") == {"ok": True}


@pytest.mark.asyncio
async def test_request_returns_none_on_204(dummy_config):
    api = _ConcreteApi(dummy_config)
    api._request_impl = AsyncMock(return_value=_httpx_response(204, ""))
    assert await api._request("DELETE", "/x") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500, 503])
async def test_request_raises_on_non_2xx(dummy_config, status_code):
    api = _ConcreteApi(dummy_config)
    api._request_impl = AsyncMock(return_value=_httpx_response(status_code, "boom"))
    with pytest.raises(NetsuiteAPIRequestError) as excinfo:
        await api._request("GET", "/x")
    assert excinfo.value.status_code == status_code
    assert excinfo.value.response_text == "boom"
    assert str(status_code) in str(excinfo.value)


@pytest.mark.asyncio
async def test_request_raises_parse_error_on_invalid_json(dummy_config):
    api = _ConcreteApi(dummy_config)
    api._request_impl = AsyncMock(
        return_value=_httpx_response(200, "<html>not json</html>")
    )
    with pytest.raises(NetsuiteAPIResponseParsingError) as excinfo:
        await api._request("GET", "/x")
    # The parse error subclasses request error and carries the same payload.
    assert excinfo.value.status_code == 200
    assert excinfo.value.response_text == "<html>not json</html>"


# ---------------------------------------------------------------------------
# `_request_impl` — argument shaping into httpx
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_impl_uses_make_url_when_no_url_kwarg(dummy_config):
    api = _ConcreteApi(dummy_config)
    captured = {}

    class _FakeClient:
        async def request(self, **kw):
            captured.update(kw)
            return _httpx_response(200, "{}")

    with patch("netsuite.rest_api_base.httpx.AsyncClient", return_value=_FakeClient()):
        await api._request_impl("get", "/widgets", params={"a": 1})

    # `method` must be uppercased; URL comes from `_make_url`.
    assert captured["method"] == "GET"
    assert captured["url"] == "https://example.com/widgets"
    assert captured["params"] == {"a": 1}
    # Default headers must be merged in.
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["timeout"] == 10  # `_default_timeout`


@pytest.mark.asyncio
async def test_request_impl_url_kwarg_overrides_subpath(dummy_config):
    """Passing `url=` must bypass `_make_url` — used by SuiteQL pagination
    and `token_info`."""
    api = _ConcreteApi(dummy_config)
    captured = {}

    class _FakeClient:
        async def request(self, **kw):
            captured.update(kw)
            return _httpx_response(200, "{}")

    with patch("netsuite.rest_api_base.httpx.AsyncClient", return_value=_FakeClient()):
        await api._request_impl(
            "POST", "ignored", url="https://override.example.com/foo"
        )

    assert captured["url"] == "https://override.example.com/foo"


@pytest.mark.asyncio
async def test_request_impl_serializes_json_to_data(dummy_config):
    """`json=` must be popped, dumped to a string, and forwarded as `data=`.
    httpx would otherwise re-serialize and double-encode."""
    api = _ConcreteApi(dummy_config)
    captured = {}

    class _FakeClient:
        async def request(self, **kw):
            captured.update(kw)
            return _httpx_response(200, "{}")

    with patch("netsuite.rest_api_base.httpx.AsyncClient", return_value=_FakeClient()):
        await api._request_impl("POST", "/x", json={"q": "SELECT 1"})

    assert "json" not in captured
    assert (
        captured["data"] == '{"q": "SELECT 1"}'
        or captured["data"] == '{"q":"SELECT 1"}'
    )


@pytest.mark.asyncio
async def test_request_impl_caller_headers_override_defaults(dummy_config):
    api = _ConcreteApi(dummy_config)
    captured = {}

    class _FakeClient:
        async def request(self, **kw):
            captured.update(kw)
            return _httpx_response(200, "{}")

    with patch("netsuite.rest_api_base.httpx.AsyncClient", return_value=_FakeClient()):
        await api._request_impl(
            "GET",
            "/x",
            headers={"Content-Type": "application/schema+json", "X-Extra": "1"},
        )

    assert captured["headers"]["Content-Type"] == "application/schema+json"
    assert captured["headers"]["X-Extra"] == "1"


@pytest.mark.asyncio
async def test_request_impl_caller_timeout_overrides_default(dummy_config):
    api = _ConcreteApi(dummy_config)
    captured = {}

    class _FakeClient:
        async def request(self, **kw):
            captured.update(kw)
            return _httpx_response(200, "{}")

    with patch("netsuite.rest_api_base.httpx.AsyncClient", return_value=_FakeClient()):
        await api._request_impl("GET", "/x", timeout=99)

    assert captured["timeout"] == 99


@pytest.mark.asyncio
async def test_request_impl_logs_at_debug_level(dummy_config, caplog):
    api = _ConcreteApi(dummy_config)

    class _FakeClient:
        async def request(self, **kw):
            return _httpx_response(200, "{}", headers={"X-Foo": "bar"})

    with patch("netsuite.rest_api_base.httpx.AsyncClient", return_value=_FakeClient()):
        with caplog.at_level(logging.DEBUG, logger="netsuite.rest_api_base"):
            await api._request_impl("GET", "/x")

    debug_msgs = [r.message for r in caplog.records]
    assert any("Making GET request" in m for m in debug_msgs)
    assert any("response headers" in m for m in debug_msgs)


# ---------------------------------------------------------------------------
# `_make_url` is abstract; `_make_auth` and `_make_default_headers` defaults
# ---------------------------------------------------------------------------


def test_base_make_url_is_abstract(dummy_config):
    api = RestApiBase()
    api._config = dummy_config
    with pytest.raises(NotImplementedError):
        api._make_url("/x")


def test_base_default_headers(dummy_config):
    assert RestApiBase()._make_default_headers() == {"Content-Type": "application/json"}


def test_make_auth_passes_token_credentials(dummy_config):
    api = _ConcreteApi(dummy_config)
    api._signature_method = DEFAULT_SIGNATURE_METHOD
    auth = api._make_auth()
    # OAuth1Auth stores credentials directly as attributes on itself.
    assert auth.client_id == dummy_config.auth.consumer_key
    assert auth.client_secret == dummy_config.auth.consumer_secret
    assert auth.token == dummy_config.auth.token_id
    assert auth.token_secret == dummy_config.auth.token_secret
    assert auth.realm == dummy_config.account


# ---------------------------------------------------------------------------
# Concurrency control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_semaphore_is_lazily_created_and_caches(dummy_config):
    api = _ConcreteApi(dummy_config)
    sem = api._request_semaphore
    assert isinstance(sem, asyncio.Semaphore)
    # Cached: second access returns the same instance.
    assert api._request_semaphore is sem


# ---------------------------------------------------------------------------
# Client reuse & lifecycle
# ---------------------------------------------------------------------------


# Bound before any test patches `httpx.AsyncClient` — patching the attribute
# on the module reaches this module's `httpx` too, so the helper below would
# otherwise call the mock it exists to feed.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_transport_client(**kw):
    """A real `httpx.AsyncClient` that answers every request locally, so we
    exercise httpx's own pooling/closing rather than a stand-in."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    return _REAL_ASYNC_CLIENT(transport=transport, **kw)


@pytest.mark.asyncio
async def test_client_is_lazily_created_and_cached(dummy_config):
    api = _ConcreteApi(dummy_config)
    # Nothing built until first use — the client must bind to the running
    # loop, not to whatever loop existed at construction time.
    assert "_client" not in api.__dict__
    client = api._client
    assert isinstance(client, httpx.AsyncClient)
    assert api._client is client
    await api.aclose()


@pytest.mark.asyncio
async def test_client_is_reused_across_requests(dummy_config):
    """The regression this whole change exists for: one client for the object's
    lifetime, not one per request. A per-request client re-parses the certifi
    CA bundle and forgoes keep-alive."""
    api = _ConcreteApi(dummy_config)
    clients = []

    def _factory(**kw):
        clients.append(_mock_transport_client(**kw))
        return clients[-1]

    with patch("netsuite.rest_api_base.httpx.AsyncClient", side_effect=_factory):
        for _ in range(3):
            await api._request_impl("GET", "/x")

    assert len(clients) == 1, "expected a single client across 3 requests"
    assert not clients[0].is_closed, "client must stay open between requests"
    await api.aclose()


@pytest.mark.asyncio
async def test_client_pool_limits_follow_concurrent_requests(dummy_config):
    api = _ConcreteApi(dummy_config)  # `_concurrent_requests` == 5
    with patch(
        "netsuite.rest_api_base.httpx.AsyncClient", side_effect=_mock_transport_client
    ) as mock_client:
        api._client

    limits = mock_client.call_args.kwargs["limits"]
    assert limits.max_connections == 5
    assert limits.max_keepalive_connections == 5
    await api.aclose()


@pytest.mark.asyncio
async def test_aclose_closes_the_client(dummy_config):
    api = _ConcreteApi(dummy_config)
    with patch(
        "netsuite.rest_api_base.httpx.AsyncClient", side_effect=_mock_transport_client
    ):
        await api._request_impl("GET", "/x")
    client = api._client
    await api.aclose()
    assert client.is_closed


@pytest.mark.asyncio
async def test_aclose_is_idempotent_and_safe_before_any_request(dummy_config):
    api = _ConcreteApi(dummy_config)
    # Never used: must not construct a client just to close it.
    await api.aclose()
    assert "_client" not in api.__dict__

    with patch(
        "netsuite.rest_api_base.httpx.AsyncClient", side_effect=_mock_transport_client
    ):
        await api._request_impl("GET", "/x")
    await api.aclose()
    await api.aclose()  # second close must not raise


@pytest.mark.asyncio
async def test_async_context_manager_closes_client(dummy_config):
    with patch(
        "netsuite.rest_api_base.httpx.AsyncClient", side_effect=_mock_transport_client
    ):
        async with _ConcreteApi(dummy_config) as api:
            await api._request_impl("GET", "/x")
            client = api._client
            assert not client.is_closed
    assert client.is_closed


@pytest.mark.asyncio
async def test_injected_client_is_used_and_never_closed(dummy_config):
    """A caller-supplied client belongs to the caller: we must use it and
    leave it open."""
    injected = _mock_transport_client()
    api = _ConcreteApi(dummy_config)
    api._provided_client = injected

    with patch("netsuite.rest_api_base.httpx.AsyncClient") as mock_client:
        resp = await api._request_impl("GET", "/x")

    assert resp.status_code == 200
    assert api._client is injected
    mock_client.assert_not_called()  # we must not build one of our own

    await api.aclose()
    assert not injected.is_closed, "aclose() must not close a client we don't own"
    await injected.aclose()


@pytest.mark.asyncio
async def test_del_warns_about_an_unclosed_client(dummy_config):
    api = _ConcreteApi(dummy_config)
    with patch(
        "netsuite.rest_api_base.httpx.AsyncClient", side_effect=_mock_transport_client
    ):
        await api._request_impl("GET", "/x")
    client = api._client

    with pytest.warns(ResourceWarning, match="Unclosed _ConcreteApi"):
        api.__del__()

    await client.aclose()


@pytest.mark.asyncio
async def test_del_stays_quiet_once_closed(dummy_config, recwarn):
    api = _ConcreteApi(dummy_config)
    with patch(
        "netsuite.rest_api_base.httpx.AsyncClient", side_effect=_mock_transport_client
    ):
        await api._request_impl("GET", "/x")
    await api.aclose()

    api.__del__()
    assert not [w for w in recwarn if issubclass(w.category, ResourceWarning)]


def test_del_is_quiet_when_no_client_was_ever_built(dummy_config, recwarn):
    _ConcreteApi(dummy_config).__del__()
    assert not [w for w in recwarn if issubclass(w.category, ResourceWarning)]


def test_object_survives_a_second_event_loop(dummy_config):
    """A cached client and semaphore bind to the loop they're first used on.
    Reusing one object across `asyncio.run()` calls — a module-level client in
    a worker, say — worked when every request built its own client, so it must
    keep working. Deliberately not an `asyncio` test: it drives its own loops.
    """
    api = _ConcreteApi(dummy_config)

    with patch(
        "netsuite.rest_api_base.httpx.AsyncClient", side_effect=_mock_transport_client
    ):
        asyncio.run(api._request_impl("GET", "/x"))
        first, first_loop = api._client, api._loop

        # Second `asyncio.run` — a brand new loop. Old code raised
        # "Event loop is closed" here.
        asyncio.run(api._request_impl("GET", "/x"))

    assert api._client is not first, "client must be rebuilt for the new loop"
    assert api._loop is not first_loop
    asyncio.run(api.aclose())


def test_same_loop_keeps_one_client_across_many_requests(dummy_config):
    """The rebind check must not fire within a single loop — that would put us
    back to a client per request."""
    api = _ConcreteApi(dummy_config)
    clients = []

    def _factory(**kw):
        clients.append(_mock_transport_client(**kw))
        return clients[-1]

    async def _drive():
        for _ in range(5):
            await api._request_impl("GET", "/x")
        await api.aclose()

    with patch("netsuite.rest_api_base.httpx.AsyncClient", side_effect=_factory):
        asyncio.run(_drive())

    assert len(clients) == 1


# ---------------------------------------------------------------------------
# HMAC-SHA256 signing
# ---------------------------------------------------------------------------


def test_authlib_hmac_sha256_signs_via_oauthlib():
    """The custom signing method should defer to oauthlib's `sign_hmac_sha256`
    using authlib's signature base string."""
    fake_request = MagicMock()
    fake_client = MagicMock(client_secret="secret", token_secret="ts")
    with patch(
        "netsuite.rest_api_base.generate_signature_base_string",
        return_value="base",
    ) as mock_base, patch(
        "netsuite.rest_api_base.sign_hmac_sha256",
        return_value="signed",
    ) as mock_sign:
        sig = authlib_hmac_sha256_sign_method(fake_client, fake_request)

    mock_base.assert_called_once_with(fake_request)
    mock_sign.assert_called_once_with("base", "secret", "ts")
    assert sig == "signed"


def test_hmac_sha256_method_is_registered_on_authlib():
    """Importing `rest_api_base` should register HMAC-SHA256 with authlib's
    `ClientAuth`. This is what lets `_make_auth` produce SHA256 signatures."""
    from authlib.oauth1.rfc5849.client_auth import ClientAuth

    assert "HMAC-SHA256" in ClientAuth.SIGNATURE_METHODS
