"""Tests for saving a Hugging Face token from the Web UI."""

from __future__ import annotations

from fastapi.testclient import TestClient
import huggingface_hub
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import httpx
import pytest

from zvisiongenerator.web import server as web_server

_SECRET = "hf_secret_value"
_ROUTE = "/api/huggingface/token"


def _post(body: object):
    with TestClient(web_server.app) as client:
        return client.post(_ROUTE, json=body)


def _hub_error(status: int) -> HfHubHTTPError:
    return HfHubHTTPError("Invalid user token.", response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/whoami-v2")))


def test_a_valid_token_is_saved_and_the_new_source_returned(monkeypatch):
    saved: dict[str, str] = {}

    def login(*, token: str, add_to_git_credential: bool) -> None:
        assert add_to_git_credential is False
        saved["token"] = token

    monkeypatch.setattr(huggingface_hub, "login", login)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: saved.get("token"))

    response = _post({"token": f"  {_SECRET}\n"})

    assert response.status_code == 200
    assert response.json() == {"huggingface_token_source": "saved"}
    assert saved == {"token": _SECRET}


@pytest.mark.parametrize("body", [{"token": ""}, {"token": "   "}, {}, {"token": 5}, ["hf_x"]])
def test_a_missing_token_is_refused_without_asking_the_hub(monkeypatch, body):
    def login(**_kwargs) -> None:
        raise AssertionError("login must not be called")

    monkeypatch.setattr(huggingface_hub, "login", login)

    assert _post(body).status_code == 400


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (_hub_error(401), 400),
        (httpx.ConnectError("unreachable"), 502),
        (httpx.LocalProtocolError(f"Illegal header value b'Bearer {_SECRET}'"), 502),
    ],
)
def test_a_refused_or_unchecked_token_gives_one_line_without_the_token(monkeypatch, error, status):
    def login(**_kwargs) -> None:
        raise error

    monkeypatch.setattr(huggingface_hub, "login", login)

    response = _post({"token": _SECRET})

    assert response.status_code == status
    assert response.json()["detail"]
    assert _SECRET not in response.text
