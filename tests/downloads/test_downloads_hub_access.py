"""Tests for Hugging Face token detection and Hub refusal messages."""

from __future__ import annotations

from huggingface_hub.utils import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

from zvisiongenerator.downloads.hub_access import SAVED_SOURCE, describe_refusal, refusal_message, save_token, token_source

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


@pytest.mark.parametrize(
    ("kind", "wrapped"),
    [("gated", False), ("not_found", False), ("http_401", False), ("http_403", False), ("gated", True)],
)
def test_refusal_without_token_is_one_line_with_link_and_login(kind, wrapped):
    exc = _REFUSALS[kind]()
    message = refusal_message(_caused_by(exc) if wrapped else exc, token_present=False, repo_id=_REPO)

    assert message is not None
    assert "\n" not in message
    assert _LINK in message
    assert "uvx hf auth login" in message


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
def test_other_errors_are_not_refusals(exc):
    assert refusal_message(exc, token_present=False, repo_id=_REPO) is None


def test_token_source_names_the_environment_variable_holding_the_token():
    assert token_source(read_token=lambda: "tok", environ={"HF_TOKEN": "tok"}) == "HF_TOKEN"
    assert token_source(read_token=lambda: "tok", environ={"HUGGING_FACE_HUB_TOKEN": "tok"}) == "HUGGING_FACE_HUB_TOKEN"


def test_token_source_is_saved_for_a_saved_token():
    assert token_source(read_token=lambda: "tok", environ={}) == SAVED_SOURCE
    assert SAVED_SOURCE == "saved"


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

    assert token_source() == SAVED_SOURCE

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


def _login_raising(exc: BaseException):
    def login(**_kwargs) -> None:
        raise exc

    return login


def test_save_token_trims_and_saves_without_git_credentials():
    calls = []

    save_token("  hf_abc\n", login=lambda **kwargs: calls.append(kwargs))

    assert calls == [{"token": "hf_abc", "add_to_git_credential": False}]


@pytest.mark.parametrize("token", ["", "  \n", "hf_ab\u200bcd", "hf_ab cd", "\u201chf_abcd\u201d", "api_org_abcd"])
def test_save_token_refuses_a_paste_that_is_no_personal_token_without_asking_the_hub(token):
    def login(**_kwargs) -> None:
        raise AssertionError("login must not be called")

    with pytest.raises(ValueError):
        save_token(token, login=login)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_error(HfHubHTTPError, 401), ValueError),
        (ValueError("JSON decode error"), RuntimeError),
        (PermissionError("token file"), RuntimeError),
        (_error(HfHubHTTPError, 500), RuntimeError),
        (httpx.ConnectError("unreachable"), RuntimeError),
        (httpx.LocalProtocolError("Illegal header value b'Bearer hf_secret_value'"), RuntimeError),
    ],
)
def test_save_token_errors_never_quote_the_token(exc, expected):
    with pytest.raises(expected) as raised:
        save_token("hf_secret_value", login=_login_raising(exc))

    assert "hf_secret_value" not in str(raised.value)
    assert raised.value.__cause__ is None
