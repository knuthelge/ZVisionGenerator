"""Tests for the Web UI Host / Origin request guard (DNS rebinding and CSRF protection)."""

from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient
import pytest

from zvisiongenerator.web import request_guard
from zvisiongenerator.web import server as web_server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_server, "_persist_web_config", lambda _payload: pytest.fail("config must not be persisted"))
    with TestClient(web_server.app, base_url="http://127.0.0.1:8080") as test_client:
        yield test_client


@pytest.mark.parametrize("host", ["127.0.0.1:8080", "localhost:8080", "[::1]:8080", "192.168.1.20:8080", "LOCALHOST"])
def test_allowed_hosts(host):
    assert request_guard.is_allowed_host(host)


@pytest.mark.parametrize("host", [None, "", "attacker.example", "attacker.example:8080", "127.0.0.1.nip.io:8080", "localhost.attacker.example"])
def test_rejected_hosts(host):
    assert not request_guard.is_allowed_host(host)


def test_extra_hosts_from_env(monkeypatch):
    monkeypatch.setenv(request_guard.ALLOWED_HOSTS_ENV_VAR, "my-mac.local, other")
    assert request_guard.is_allowed_host("my-mac.local:8080")
    assert request_guard.is_allowed_host("OTHER")


def test_rebound_host_is_rejected_on_reads(client):
    response = client.get("/api/config", headers={"Host": "attacker.example:8080"})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example", "Content-Type": "text/plain"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
    ],
)
def test_cross_origin_writes_are_rejected(client, headers):
    response = client.post("/api/config", content='{"ui": {"output_dir": "~"}}', headers=headers)
    assert response.status_code == 403


def test_same_origin_write_is_allowed(monkeypatch):
    persisted: list[object] = []
    monkeypatch.setattr(web_server, "_persist_web_config", persisted.append)
    monkeypatch.setattr(web_server, "load_web_config", lambda: object())
    monkeypatch.setattr(web_server, "build_api_config_response", lambda _cfg: {"ok": True})
    with TestClient(web_server.app, base_url="http://127.0.0.1:8080") as client:
        response = client.post("/api/config", json={"ui": {}}, headers={"Origin": "http://127.0.0.1:8080", "Sec-Fetch-Site": "same-origin"})
    assert response.status_code == 200
    assert persisted == [{"ui": {}}]


@pytest.fixture
def reset_configured_hostnames():
    yield
    request_guard.configure_allowed_hostnames(())


@pytest.mark.parametrize(
    ("host", "machine", "expected"),
    [
        ("127.0.0.1", "MyMac.local", []),
        ("::1", "MyMac.local", []),
        ("my-box.lan", "MyMac.local", ["my-box.lan"]),
        ("0.0.0.0", "MyMac.local", ["mymac.local", "mymac"]),
        ("::", "mymac", ["mymac", "mymac.local"]),
        ("0.0.0.0", "MyMac.lan", ["mymac.lan", "mymac", "mymac.local"]),
        ("192.168.1.20", "MyMac.local", ["mymac.local", "mymac"]),
    ],
)
def test_bound_hostnames(monkeypatch, host, machine, expected):
    from zvisiongenerator import web

    monkeypatch.setattr(web.socket, "gethostname", lambda: machine)
    assert web._bound_hostnames(host) == expected


def test_configured_hostnames_are_allowed_without_touching_env(monkeypatch, reset_configured_hostnames):
    monkeypatch.delenv(request_guard.ALLOWED_HOSTS_ENV_VAR, raising=False)
    request_guard.configure_allowed_hostnames(["mymac", "mymac.local"])
    assert request_guard.is_allowed_host("mymac:8080")
    assert request_guard.is_allowed_host("MyMac.local:8080")
    assert not request_guard.is_allowed_host("attacker.example:8080")
    assert request_guard.ALLOWED_HOSTS_ENV_VAR not in os.environ


@pytest.mark.skipif(sys.platform == "win32", reason="TIME_WAIT reuse semantics are POSIX-specific")
def test_port_probe_keeps_port_with_only_time_wait_connections():
    """A quick restart must reuse the same port; lingering TIME_WAIT sockets from the old server must not move it."""
    import socket
    import time

    from zvisiongenerator import web

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen()
    client = socket.create_connection(("127.0.0.1", port))
    accepted, _ = server.accept()
    accepted.close()  # server closes first -> server-side TIME_WAIT on `port`, like a killed ziv-ui
    server.close()
    time.sleep(0.05)
    client.close()

    assert web._find_available_port("127.0.0.1", port) == port


def test_port_probe_skips_port_with_active_listener():
    import socket

    from zvisiongenerator import web

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert web._find_available_port("127.0.0.1", port) != port


def test_port_probe_skips_port_held_by_wildcard_listener():
    """Another process on 0.0.0.0:P (e.g. a Docker-published port) must not be shadowed by binding 127.0.0.1:P."""
    import socket

    from zvisiongenerator import web

    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert web._find_available_port("127.0.0.1", port) != port


def test_port_probe_supports_ipv6_hosts():
    """--host :: (and other IPv6 addresses) must not make every port look taken."""
    import socket

    from zvisiongenerator import web

    if not socket.has_ipv6:
        pytest.skip("IPv6 unavailable")
    with socket.socket(socket.AF_INET6) as finder:
        finder.bind(("::1", 0))
        port = finder.getsockname()[1]
    assert web._find_available_port("::1", port) == port
    assert web._find_available_port("::", port) == port

    with socket.socket(socket.AF_INET6) as listener:
        listener.bind(("::1", 0))
        listener.listen()
        busy = listener.getsockname()[1]
        assert web._find_available_port("::1", busy) != busy
