"""Tests for the NetSuite facade — the cached_property accessors that lazily
construct the REST, Restlet, and SOAP sub-clients with their options.
"""

import warnings

import pytest

from netsuite import NetSuite
from netsuite.rest_api import NetSuiteRestApi
from netsuite.restlet import NetSuiteRestlet
from netsuite.soap_api.zeep import ZEEP_INSTALLED


def test_init_defaults_empty_option_dicts(dummy_config):
    ns = NetSuite(dummy_config)
    assert ns._rest_api_options == {}
    assert ns._soap_api_options == {}
    assert ns._restlet_options == {}


def test_rest_api_is_cached_and_receives_options(dummy_config):
    ns = NetSuite(dummy_config, rest_api_options={"default_timeout": 5})
    api = ns.rest_api
    assert isinstance(api, NetSuiteRestApi)
    assert api._default_timeout == 5
    # cached_property: same instance on second access
    assert ns.rest_api is api


def test_restlet_is_cached(dummy_config):
    ns = NetSuite(dummy_config)
    restlet = ns.restlet
    assert isinstance(restlet, NetSuiteRestlet)
    assert ns.restlet is restlet


@pytest.mark.skipif(not ZEEP_INSTALLED, reason="Requires zeep")
def test_soap_api_is_constructed(dummy_config):
    ns = NetSuite(dummy_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        soap = ns.soap_api
    assert ns.soap_api is soap


# ---------------------------------------------------------------------------
# Lifecycle — closing the sub-clients' connection pools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_closes_the_sub_apis_that_were_used(dummy_config):
    ns = NetSuite(dummy_config)
    ns.rest_api._client  # instantiate both, and their clients
    ns.restlet._client

    await ns.aclose()

    assert ns.rest_api._client.is_closed
    assert ns.restlet._client.is_closed


@pytest.mark.asyncio
async def test_aclose_does_not_construct_unused_sub_apis(dummy_config):
    """Reaching for `ns.rest_api` here would build the very thing we're
    trying to avoid — a client just to close it."""
    ns = NetSuite(dummy_config)
    await ns.aclose()
    assert "rest_api" not in ns.__dict__
    assert "restlet" not in ns.__dict__


@pytest.mark.asyncio
async def test_async_context_manager_closes_sub_apis(dummy_config):
    async with NetSuite(dummy_config) as ns:
        client = ns.rest_api._client
        assert not client.is_closed
    assert client.is_closed
