import logging
import re
from functools import cached_property
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Union

from . import rest_api_base
from .config import Config

logger = logging.getLogger(__name__)

__all__ = ("NetSuiteRestApi",)


def _next_link(suiteql_response: Dict[str, Any]) -> Optional[str]:
    """Return the absolute URL of the `next` link from a SuiteQL response, or None."""
    if not suiteql_response.get("hasMore"):
        return None
    for link in suiteql_response.get("links") or ():
        if link.get("rel") == "next":
            return link.get("href")
    return None


# Matches an `ORDER BY` clause (case-insensitive). Word boundaries prevent
# false positives on column names like `order_by_id`. The heuristic
# deliberately fires on subquery ORDER BY's too, since those can also
# surface the NetSuite empty-result quirk.
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)

# Below this threshold, `ORDER BY` is at risk of NetSuite's
# zero-row quirk (jacobsvante/netsuite#29).
_ORDER_BY_SAFE_LIMIT = 1000


class NetSuiteRestApi(rest_api_base.RestApiBase):
    def __init__(
        self,
        config: Config,
        *,
        default_timeout: int = 60,
        concurrent_requests: int = 10,
        signature_method: str = rest_api_base.DEFAULT_SIGNATURE_METHOD,
    ):
        self._config = config
        self._default_timeout = default_timeout
        self._concurrent_requests = concurrent_requests
        self._signature_method = signature_method

    @cached_property
    def hostname(self) -> str:
        return self._make_hostname()

    async def request(self, method: str, subpath: str, **request_kw):
        return await self._request_impl(method, subpath, **request_kw)

    async def get(self, subpath: str, **request_kw):
        return await self._request("GET", subpath, **request_kw)

    async def post(self, subpath: str, **request_kw):
        return await self._request(
            "POST",
            subpath,
            **request_kw,
        )

    async def put(self, subpath: str, **request_kw):
        return await self._request("PUT", subpath, **request_kw)

    async def patch(self, subpath: str, **request_kw):
        return await self._request("PATCH", subpath, **request_kw)

    async def delete(self, subpath: str, **request_kw):
        return await self._request("DELETE", subpath, **request_kw)

    async def create_record(
        self,
        record_type: str,
        record_data: Dict[str, Any],
        **request_kw,
    ) -> Union[int, str]:
        """Create a new record and return its ID.

        NetSuite's REST API responds to a successful POST create with HTTP
        204 plus a `Location` header pointing at the new record. This
        helper POSTs to ``/record/v1/{record_type}`` and returns the ID
        from that header (numeric IDs as ``int``, external IDs like
        ``eid:CUST001`` as ``str``).

        Example:
            >>> customer_id = await rest_api.create_record(
            ...     "customer",
            ...     {"entityid": "Acme", "subsidiary": {"id": "1"}},
            ... )

        Documentation:
            https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_1545141395.html
        """
        return await self.post(
            f"/record/v1/{record_type}", json=record_data, **request_kw
        )

    # TODO maybe break out params vs poping?
    async def suiteql(self, q: str, limit: int = 10, offset: int = 0, **request_kw):
        """
        Run a single SuiteQL query.

        Example:
        >>> suiteql(q="SELECT * FROM Transaction", limit=10, offset=0)

        Note on `ORDER BY`: NetSuite has a known quirk where a SuiteQL query
        with `ORDER BY` and a small `limit` (the default 10) can return zero
        items. If you hit this, request a larger page (`limit=1000`) or sort
        client-side after fetching. This method also logs a warning when it
        detects the pattern. See jacobsvante/netsuite#29.

        Note on pagination: NetSuite caps `limit` at 1000. To stream every
        page until exhaustion, use `suiteql_paginated` instead.

        Documentation:

        - https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_156257799794.html#Using-SuiteQL
        """
        if _ORDER_BY_RE.search(q) and limit < _ORDER_BY_SAFE_LIMIT:
            logger.warning(
                "SuiteQL query contains `ORDER BY` with limit=%d. NetSuite "
                "has a known quirk where this combination can return zero "
                "rows. Consider raising limit to %d or sorting client-side. "
                "See jacobsvante/netsuite#29.",
                limit,
                _ORDER_BY_SAFE_LIMIT,
            )
        return await self._request(
            "POST",
            "/query/v1/suiteql",
            headers={"Prefer": "transient", **request_kw.pop("headers", {})},
            json={"q": q, **request_kw.pop("json", {})},
            # limit & offset look like the only available params
            params={"limit": limit, "offset": offset, **request_kw.pop("params", {})},
            **request_kw,
        )

    async def suiteql_paginated(
        self,
        q: str,
        *,
        limit: int = 1000,
        offset: int = 0,
        **request_kw,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator yielding each page of a SuiteQL query, following the
        `next` link in NetSuite's response until `hasMore` is False.

        Yields the raw page dict (with `items`, `count`, `hasMore`, `links`,
        etc.). Use `limit=1000` (the NetSuite max) to minimize round trips.

        Example:
        >>> async for page in rest_api.suiteql_paginated(q="SELECT id FROM transaction"):
        ...     for row in page["items"]:
        ...         ...

        Caveat: a single SuiteQL query can return at most 100,000 rows in
        total — that is a NetSuite-side cap, not a library limitation. To
        retrieve more, partition the query with a WHERE clause (e.g. on
        `id` ranges or date windows) and run several paginated queries.
        See jacobsvante/netsuite#42.
        """
        # First page goes through the normal `suiteql` path so users get
        # consistent header/param handling. Subsequent pages follow the
        # absolute `next` link from each response.
        page = await self.suiteql(q, limit=limit, offset=offset, **request_kw)
        yield page

        next_url = _next_link(page)
        # Subsequent pages reuse the same body; only the URL (with offset)
        # changes. We forward `**request_kw` so callers' headers/params
        # still apply.
        body_kw = {
            "headers": {"Prefer": "transient", **request_kw.pop("headers", {})},
            "json": {"q": q, **request_kw.pop("json", {})},
        }
        # `params` are encoded in the next URL, so we must not also pass
        # them here — that would double-up offset/limit.
        request_kw.pop("params", None)

        while next_url is not None:
            page = await self._request(
                "POST",
                # `subpath` is ignored when `url` is provided, but
                # `_request_impl` still requires the parameter.
                "/query/v1/suiteql",
                url=next_url,
                **body_kw,
                **request_kw,
            )
            yield page
            next_url = _next_link(page)

    async def jsonschema(self, record_type: str, **request_kw):
        headers = {
            "Accept": "application/schema+json",
            **request_kw.pop("headers", {}),
        }
        return await self._request(
            "GET",
            f"/record/v1/metadata-catalog/{record_type}",
            headers=headers,
            **request_kw,
        )

    async def token_info(self, **request_kw):
        """
        Retrieves metadata about the current token. Role, company, etc.

        https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/chapter_157017286140.html#Issue-Token-and-Revoke-Token-REST-Services-for-Token-based-Authentication
        """

        # this overrides the default URL generation: this specific endpoint hits a completely different host
        request_kw["url"] = (
            f"https://{self._config.account_slugified}.restlets.api.netsuite.com/rest/tokeninfo"
        )

        return await self._request(
            method="GET",
            # useless, but required by _request
            subpath="ignored",
            **request_kw,
        )

    async def openapi(self, record_types: Sequence[str] = (), **request_kw):
        """
        Retrieves the OpenAPI specification (metadata catalog) for the Netsuite REST API. This is the best way to
        introspect the NetSuite account and return the record structure.

        https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_1545126526.html

        Args:
            record_types (Sequence[str]): Optional. List of record types to include in the OpenAPI specification.
            **request_kw: Optional keyword arguments to be passed to the underlying request.

        Returns:
            The OpenAPI specification as a JSON object.
        """

        headers = {
            "Accept": "application/swagger+json",
            **request_kw.pop("headers", {}),
        }
        params = request_kw.pop("params", {})

        if len(record_types) > 0:
            params["select"] = ",".join(record_types)

        return await self._request(
            "GET",
            "/record/v1/metadata-catalog",
            headers=headers,
            params=params,
            **request_kw,
        )

    def _make_hostname(self):
        return f"{self._config.account_slugified}.suitetalk.api.netsuite.com"

    def _make_url(self, subpath: str):
        return f"https://{self.hostname}/services/rest{subpath}"

    def _make_default_headers(self):
        return {
            "Content-Type": "application/json",
            "X-NetSuite-PropertyNameValidation": "error",
        }
