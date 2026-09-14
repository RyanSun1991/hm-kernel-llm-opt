"""Bounded operator-endpoint HTTP, with redirects disabled to preserve credential scope."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .store import canonical_json


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(url, *, method="GET", body=None, headers=None, timeout=15, cap=2_000_000):
    request = Request(
        url,
        data=canonical_json(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(cap + 1)
            if len(raw) > cap:
                raise ValueError("HTTP response exceeded configured byte limit")
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        # Response bodies/URLs can contain credentials; callers persist only this class/code.
        raise OSError(f"HTTP status {exc.code}; delivery may be uncertain") from None
    except (URLError, TimeoutError) as exc:
        raise OSError(f"HTTP transport failed ({type(exc).__name__}); delivery uncertain") from None
