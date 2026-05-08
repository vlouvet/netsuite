import logging
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from netsuite import NetSuiteRestApi


def test_expected_hostname(dummy_config):
    rest_api = NetSuiteRestApi(dummy_config)
    assert rest_api.hostname == "123456-sb1.suitetalk.api.netsuite.com"


def _location_response(location: str) -> Mock:
    """Mock a NetSuite POST-create response: HTTP 204 with a Location header."""
    resp = Mock(spec=httpx.Response)
    resp.status_code = 204
    resp.headers = {"location": location}
    resp.text = ""
    return resp


@pytest.mark.asyncio
async def test_post_returns_numeric_record_id_from_location_header(dummy_config):
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request_impl = AsyncMock(  # type: ignore[method-assign]
        return_value=_location_response(
            "https://123456-sb1.suitetalk.api.netsuite.com/services/rest/record/v1/customer/647"
        )
    )
    result = await rest_api.post("/record/v1/customer", json={"entityid": "Acme"})
    assert result == 647
    assert isinstance(result, int)


@pytest.mark.asyncio
async def test_post_returns_string_external_id_from_location_header(dummy_config):
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request_impl = AsyncMock(  # type: ignore[method-assign]
        return_value=_location_response(
            "https://123456-sb1.suitetalk.api.netsuite.com/services/rest/record/v1/customer/eid:CUST001"
        )
    )
    result = await rest_api.post("/record/v1/customer", json={"entityid": "Acme"})
    assert result == "eid:CUST001"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_post_returns_none_when_no_location_header(dummy_config):
    """Some POST endpoints (e.g. SuiteQL) return 204 without a Location.
    Those should still return None — the new behavior only kicks in for
    record creates."""
    rest_api = NetSuiteRestApi(dummy_config)
    resp = Mock(spec=httpx.Response)
    resp.status_code = 204
    resp.headers = {}
    resp.text = ""
    rest_api._request_impl = AsyncMock(return_value=resp)  # type: ignore[method-assign]
    assert await rest_api.post("/record/v1/customer", json={}) is None


@pytest.mark.asyncio
async def test_create_record_helper_posts_to_record_endpoint(dummy_config):
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request_impl = AsyncMock(  # type: ignore[method-assign]
        return_value=_location_response(
            "https://123456-sb1.suitetalk.api.netsuite.com/services/rest/record/v1/customer/123"
        )
    )
    result = await rest_api.create_record(
        "customer",
        {"entityid": "Acme", "subsidiary": {"id": "1"}},
    )
    assert result == 123
    # Verify it routed through POST /record/v1/customer.
    rest_api._request_impl.assert_awaited_once()
    args, kwargs = rest_api._request_impl.await_args
    assert args == ("POST", "/record/v1/customer")
    assert kwargs["json"] == {"entityid": "Acme", "subsidiary": {"id": "1"}}


@pytest.mark.asyncio
async def test_suiteql_posts_to_query_endpoint(dummy_config):
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request = AsyncMock(return_value={"items": []})  # type: ignore[method-assign]
    await rest_api.suiteql(q="SELECT id FROM customer", limit=50, offset=10)
    rest_api._request.assert_awaited_once()
    args, kwargs = rest_api._request.await_args
    assert args == ("POST", "/query/v1/suiteql")
    assert kwargs["json"] == {"q": "SELECT id FROM customer"}
    assert kwargs["params"] == {"limit": 50, "offset": 10}
    assert kwargs["headers"]["Prefer"] == "transient"


def _page(items, *, has_more, next_url=None):
    page = {"items": items, "hasMore": has_more, "links": []}
    if next_url is not None:
        page["links"].append({"rel": "next", "href": next_url})
    return page


@pytest.mark.asyncio
async def test_suiteql_paginated_follows_next_link_until_exhausted(dummy_config):
    """Regression test for jacobsvante/netsuite#42 — pagination must walk the
    `next` link until `hasMore` is False, without re-sending the original
    `params` (which would double-encode offset/limit)."""
    rest_api = NetSuiteRestApi(dummy_config)
    pages = [
        _page([1, 2], has_more=True, next_url="https://example.com/page2"),
        _page([3, 4], has_more=True, next_url="https://example.com/page3"),
        _page([5], has_more=False),
    ]
    rest_api._request = AsyncMock(side_effect=pages)  # type: ignore[method-assign]

    collected = []
    async for page in rest_api.suiteql_paginated(
        q="SELECT id FROM transaction", limit=2
    ):
        collected.extend(page["items"])

    assert collected == [1, 2, 3, 4, 5]
    assert rest_api._request.await_count == 3

    # Subsequent calls must use the absolute `url` from the next link, and
    # must NOT re-pass `params` (NetSuite's next URL already encodes them).
    second_call_kwargs = rest_api._request.await_args_list[1].kwargs
    assert second_call_kwargs.get("url") == "https://example.com/page2"
    assert "params" not in second_call_kwargs


@pytest.mark.asyncio
async def test_suiteql_paginated_stops_when_hasmore_false_on_first_page(dummy_config):
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request = AsyncMock(  # type: ignore[method-assign]
        return_value=_page([1], has_more=False)
    )
    pages = [page async for page in rest_api.suiteql_paginated(q="SELECT 1")]
    assert len(pages) == 1
    rest_api._request.assert_awaited_once()


@pytest.mark.asyncio
async def test_suiteql_paginated_stops_when_no_next_link(dummy_config):
    """`hasMore=true` but no `rel=next` link should still terminate cleanly."""
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request = AsyncMock(  # type: ignore[method-assign]
        return_value=_page([1], has_more=True)  # no next_url
    )
    pages = [page async for page in rest_api.suiteql_paginated(q="SELECT 1")]
    assert len(pages) == 1


@pytest.mark.asyncio
async def test_suiteql_warns_on_order_by_with_small_limit(dummy_config, caplog):
    """Regression test for jacobsvante/netsuite#29: warn when an `ORDER BY`
    SuiteQL query is combined with a limit that may trigger NetSuite's
    zero-row quirk."""
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request = AsyncMock(return_value={"items": []})  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="netsuite.rest_api"):
        await rest_api.suiteql(q="SELECT id FROM subsidiary ORDER BY id")
    assert any("ORDER BY" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_suiteql_quiet_when_order_by_with_safe_limit(dummy_config, caplog):
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request = AsyncMock(return_value={"items": []})  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="netsuite.rest_api"):
        await rest_api.suiteql(q="SELECT id FROM subsidiary ORDER BY id", limit=1000)
    assert not any("ORDER BY" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_suiteql_quiet_when_no_order_by(dummy_config, caplog):
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request = AsyncMock(return_value={"items": []})  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="netsuite.rest_api"):
        await rest_api.suiteql(q="SELECT id FROM subsidiary")
    assert not any("ORDER BY" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_suiteql_order_by_detection_is_case_insensitive(dummy_config, caplog):
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request = AsyncMock(return_value={"items": []})  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="netsuite.rest_api"):
        await rest_api.suiteql(q="select id from subsidiary order by id")
    assert any("ORDER BY" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_suiteql_order_by_detection_ignores_substrings(dummy_config, caplog):
    """`order_by_id` as a column name shouldn't trigger the warning."""
    rest_api = NetSuiteRestApi(dummy_config)
    rest_api._request = AsyncMock(return_value={"items": []})  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="netsuite.rest_api"):
        await rest_api.suiteql(q="SELECT order_by_id FROM custom_table")
    assert not any("ORDER BY" in r.message for r in caplog.records)
