"""Reject DNS-rebinding and cross-site requests to the local Web UI server."""

from __future__ import annotations

import functools
import ipaddress
import os
from collections.abc import Iterable
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

ALLOWED_HOSTS_ENV_VAR = "ZIV_UI_ALLOWED_HOSTS"

_LOOPBACK_HOSTNAMES = frozenset({"localhost"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_CROSS_SITE_FETCH_VALUES = frozenset({"cross-site", "same-site"})

# Hostnames the launcher derived from --host (set in-process; never exported to child processes).
_configured_hostnames: frozenset[str] = frozenset()


def configure_allowed_hostnames(names: Iterable[str]) -> None:
    """Set the hostnames (besides IPs, ``localhost`` and ``ZIV_UI_ALLOWED_HOSTS``) this server answers to."""
    global _configured_hostnames
    _configured_hostnames = frozenset(name.strip().lower() for name in names if name.strip())


@functools.lru_cache(maxsize=8)
def _parse_hostnames(value: str) -> frozenset[str]:
    return frozenset(name.strip().lower() for name in value.split(",") if name.strip())


def _allowed_hostnames() -> frozenset[str]:
    # The env lookup is a dict read; parsing is cached per distinct value.
    return _LOOPBACK_HOSTNAMES | _configured_hostnames | _parse_hostnames(os.environ.get(ALLOWED_HOSTS_ENV_VAR, ""))


def _hostname(host_header: str) -> str:
    """Return the hostname part of a Host header value, without port or IPv6 brackets."""
    return (urlsplit(f"//{host_header}").hostname or "").lower()


def is_allowed_host(host_header: str | None) -> bool:
    """Return whether a Host header names this server rather than a rebound domain.

    IP literals cannot be the target of DNS rebinding, so any IP is accepted;
    hostnames must be ``localhost`` or listed in ``ZIV_UI_ALLOWED_HOSTS``.
    """
    if not host_header:
        return False
    try:
        hostname = _hostname(host_header)
    except ValueError:
        return False
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return hostname in _allowed_hostnames()
    return True


def is_same_origin_request(method: str, host_header: str | None, origin: str | None, fetch_site: str | None) -> bool:
    """Return whether a state-changing request originates from this server's own pages."""
    if method.upper() in _SAFE_METHODS:
        return True
    if origin is not None:
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and bool(host_header) and parsed.netloc.lower() == host_header.lower()
    # Browsers always send Origin on cross-origin writes; this covers older ones that only send Fetch Metadata.
    return (fetch_site or "").lower() not in _CROSS_SITE_FETCH_VALUES


class LocalRequestGuardMiddleware:
    """ASGI middleware enforcing Host allow-listing and same-origin writes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        host_header = headers.get("host")
        if not is_allowed_host(host_header):
            response = PlainTextResponse(f"Invalid Host header. Add the hostname to {ALLOWED_HOSTS_ENV_VAR} to allow it.", status_code=400)
            await response(scope, receive, send)
            return
        if not is_same_origin_request(scope["method"], host_header, headers.get("origin"), headers.get("sec-fetch-site")):
            response = PlainTextResponse("Cross-origin requests are not allowed.", status_code=403)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
