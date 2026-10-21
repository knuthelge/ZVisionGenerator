"""Tests for the dry-run target check."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import (
    DryRunError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from zvisiongenerator.downloads.check import check_target
from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck

_TARGET = DownloadTarget("owner/repo", revision="v1", allow_patterns=("vae/*",), ignore_patterns=("*.bin",), local_dir="/tmp/x")


def _info(size: int | None, *, will_download: bool = True):
    return SimpleNamespace(file_size=size, will_download=will_download)


def _raising(exc: BaseException):
    def fake(*args, **kwargs):
        raise exc

    return fake


def _dry_run_error(cause: BaseException | None) -> DryRunError:
    error = DryRunError("dry run failed")
    error.__cause__ = cause
    return error


def _http_error(cls):
    return cls("boom", response=httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co")))


def test_counts_only_files_that_will_download():
    calls: list = []

    def fake(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return [_info(10), _info(5), _info(99, will_download=False)]

    assert check_target(_TARGET, snapshot_download=fake) == TargetCheck(2, 15)
    assert calls == [
        (
            "owner/repo",
            {"revision": "v1", "allow_patterns": ["vae/*"], "ignore_patterns": ["*.bin"], "local_dir": "/tmp/x", "dry_run": True},
        )
    ]


def test_unknown_size_makes_missing_bytes_none():
    assert check_target(_TARGET, snapshot_download=lambda *a, **k: [_info(10), _info(None)]) == TargetCheck(2, None)


def test_nothing_missing():
    assert check_target(DownloadTarget("o/r"), snapshot_download=lambda *a, **k: [_info(10, will_download=False)]) == TargetCheck(0, 0)


@pytest.mark.parametrize(
    "exc",
    [
        OfflineModeIsEnabled("offline"),
        LocalEntryNotFoundError("no connection"),
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("slow"),
        _dry_run_error(None),
        _dry_run_error(httpx.ConnectError("refused")),
        _dry_run_error(httpx.ReadTimeout("slow")),
        _dry_run_error(OfflineModeIsEnabled("offline")),
    ],
)
def test_unreachable_hub_returns_none(exc):
    assert check_target(_TARGET, snapshot_download=_raising(exc)) is None


@pytest.mark.parametrize(
    "exc",
    [
        _dry_run_error(_http_error(GatedRepoError)),
        _dry_run_error(_http_error(RepositoryNotFoundError)),
        _dry_run_error(_http_error(HfHubHTTPError)),
        _http_error(RevisionNotFoundError),
        _http_error(GatedRepoError),
    ],
)
def test_hub_refusal_raises_runtime_error_naming_the_repo(exc):
    with pytest.raises(RuntimeError, match="owner/repo"):
        check_target(_TARGET, snapshot_download=_raising(exc))


@pytest.mark.parametrize("exc", [_dry_run_error(_http_error(GatedRepoError)), _http_error(GatedRepoError)])
def test_gated_refusal_is_one_line_with_the_next_step(exc):
    with pytest.raises(RuntimeError) as raised:
        check_target(_TARGET, snapshot_download=_raising(exc))

    message = str(raised.value)
    assert "\n" not in message
    assert "https://huggingface.co/owner/repo" in message
    assert "hf auth login" in message


def test_connection_error_in_a_dry_run_is_not_a_refusal():
    assert check_target(_TARGET, snapshot_download=_raising(_dry_run_error(httpx.ConnectError("offline")))) is None


def test_unrelated_errors_propagate():
    with pytest.raises(ValueError, match="bad"):
        check_target(_TARGET, snapshot_download=_raising(ValueError("bad")))
    with pytest.raises(DryRunError):
        check_target(_TARGET, snapshot_download=_raising(_dry_run_error(ValueError("other"))))
