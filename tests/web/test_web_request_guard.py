"""Tests for the Web UI Host / Origin request guard (DNS rebinding and CSRF protection)."""

from __future__ import annotations

import os

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


def test_configured_hostnames_are_allowed_without_touching_env(monkeypatch, reset_configured_hostnames):
    monkeypatch.delenv(request_guard.ALLOWED_HOSTS_ENV_VAR, raising=False)
    request_guard.configure_allowed_hostnames(["mymac", "mymac.local"])
    assert request_guard.is_allowed_host("mymac:8080")
    assert request_guard.is_allowed_host("MyMac.local:8080")
    assert not request_guard.is_allowed_host("attacker.example:8080")
    assert request_guard.ALLOWED_HOSTS_ENV_VAR not in os.environ
