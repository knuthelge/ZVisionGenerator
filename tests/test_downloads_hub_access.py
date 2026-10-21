"""Tests for Hugging Face token detection and Hub refusal messages."""

from __future__ import annotations

import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

from zvisiongenerator.downloads.hub_access import LOGIN_SOURCE, describe_refusal, refusal_message, token_source

_REPO = "owner/repo"
_LINK = f"https://huggingface.co/{_REPO}"


def _error(cls, status: int = 401):
    return cls("boom", response=httpx.Response(status, request=httpx.Request("GET", f"https://huggingface.co/api/models/{_REPO}")))


def _caused_by(inner: BaseException) -> RuntimeError:
    outer = RuntimeError("wrapped")
    outer.__cause__ = inner
    return outer


_REFUSALS = {
    "gated": lambda: _error(GatedRepoError, 403),
    "not_found": lambda: _error(RepositoryNotFoundError, 401),
    "http_401": lambda: _error(HfHubHTTPError, 401),
    "http_403": lambda: _error(HfHubHTTPError, 403),
}


@pytest.mark.parametrize("kind", list(_REFUSALS))
@pytest.mark.parametrize("wrapped", [False, True])
def test_refusal_without_token_is_one_line_with_link_and_login(kind, wrapped):
    exc = _REFUSALS[kind]()
    message = refusal_message(_caused_by(exc) if wrapped else exc, token_present=False, repo_id=_REPO)

    assert message is not None
    assert "\n" not in message
    assert _LINK in message
    assert "hf auth login" in message


@pytest.mark.parametrize("kind", ["gated", "not_found", "http_403"])
def test_refusal_with_token_links_the_repo_and_skips_login(kind):
    message = refusal_message(_REFUSALS[kind](), token_present=True, repo_id=_REPO)

    assert message is not None
    assert "\n" not in message
    assert _LINK in message
    assert "hf auth login" not in message


def test_rejected_token_says_to_log_in_again():
    message = refusal_message(_error(HfHubHTTPError, 401), token_present=True, repo_id=_REPO)

    assert message is not None
    assert "\n" not in message
    assert "hf auth login" in message
    assert "HF_TOKEN" in message


def test_messages_differ_by_cause():
    gated = refusal_message(_REFUSALS["gated"](), token_present=False, repo_id=_REPO)
    missing = refusal_message(_REFUSALS["not_found"](), token_present=False, repo_id=_REPO)

    assert gated != missing


def test_repo_comes_from_the_error_when_not_given():
    error = _error(GatedRepoError, 403)
    error.repo_id = "other/model"

    message = refusal_message(error, token_present=False)

    assert message is not None
    assert "https://huggingface.co/other/model" in message


def test_repo_id_argument_wins_over_the_error():
    error = _error(GatedRepoError, 403)
    error.repo_id = "other/model"

    message = refusal_message(error, token_present=False, repo_id=_REPO)

    assert message is not None
    assert _LINK in message
    assert "other/model" not in message


def test_unknown_repo_links_the_hub_root():
    message = refusal_message(_error(GatedRepoError, 403), token_present=False)

    assert message is not None
    assert "\n" not in message
    assert "https://huggingface.co" in message
    assert "hf auth login" in message


@pytest.mark.parametrize(
    "exc",
    [
        _error(HfHubHTTPError, 500),
        _error(HfHubHTTPError, 404),
        _error(RevisionNotFoundError, 404),
        ValueError("bad"),
        httpx.ConnectError("offline"),
        _caused_by(ValueError("bad")),
    ],
)
@pytest.mark.parametrize("token_present", [False, True])
def test_other_errors_are_not_refusals(exc, token_present):
    assert refusal_message(exc, token_present=token_present, repo_id=_REPO) is None


def test_token_source_names_the_environment_variable_holding_the_token():
    assert token_source(read_token=lambda: "tok", environ={"HF_TOKEN": "tok"}) == "HF_TOKEN"
    assert token_source(read_token=lambda: "tok", environ={"HUGGING_FACE_HUB_TOKEN": "tok"}) == "HUGGING_FACE_HUB_TOKEN"


def test_token_source_is_login_for_a_saved_token():
    assert token_source(read_token=lambda: "tok", environ={}) == LOGIN_SOURCE
    assert LOGIN_SOURCE == "hf auth login"


def test_token_source_is_none_without_a_token():
    assert token_source(read_token=lambda: None, environ={"HF_TOKEN": "tok"}) is None
    assert token_source(read_token=lambda: "", environ={}) is None


def test_token_source_is_none_when_reading_the_token_raises():
    def broken() -> str:
        raise RuntimeError("OIDC unavailable")

    assert token_source(read_token=broken, environ={}) is None


def test_token_source_reads_hub_token_by_default(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "saved")
    monkeypatch.setenv("HF_TOKEN", "other")

    assert token_source() == LOGIN_SOURCE

    monkeypatch.setenv("HF_TOKEN", "saved")

    assert token_source() == "HF_TOKEN"


def test_describe_refusal_uses_the_token_state(monkeypatch):
    without = describe_refusal(_error(GatedRepoError, 403), _REPO)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "saved")
    with_token = describe_refusal(_error(GatedRepoError, 403), _REPO)

    assert without is not None
    assert with_token is not None
    assert "hf auth login" in without
    assert "hf auth login" not in with_token


def test_describe_refusal_of_a_non_refusal_never_reads_the_token(monkeypatch):
    def fail() -> str:
        raise AssertionError("the token must not be read")

    monkeypatch.setattr(huggingface_hub, "get_token", fail)

    assert describe_refusal(ValueError("bad"), _REPO) is None
    assert describe_refusal(_error(HfHubHTTPError, 500), _REPO) is None
