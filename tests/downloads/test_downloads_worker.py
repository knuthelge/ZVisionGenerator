"""Tests for the download worker's protocol output."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any

from huggingface_hub.utils import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from tests._paths import REPO_ROOT

from zvisiongenerator.downloads import worker
from zvisiongenerator.downloads.target import DownloadTarget

_TARGET = DownloadTarget(
    "owner/repo",
    revision="v1",
    allow_patterns=("*.safetensors", "config.json"),
    ignore_patterns=("*.md",),
    local_dir="/some/dir",
)


def _protocol(output: str) -> list[dict[str, Any]]:
    return [json.loads(line[len(worker.PROTOCOL_PREFIX) :]) for line in output.splitlines() if line.startswith(worker.PROTOCOL_PREFIX)]


def _patch_download(monkeypatch: pytest.MonkeyPatch, fake) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)


def test_main_prints_progress_then_done(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(worker, "REPORT_INTERVAL_SECS", 0.01)
    reported = threading.Event()
    emit = worker._emit

    def spy_emit(payload: dict[str, object], lock: threading.Lock) -> None:
        emit(payload, lock)
        reported.set()

    monkeypatch.setattr(worker, "_emit", spy_emit)

    def fake(repo_id: str, **kwargs: Any) -> str:
        wire = kwargs["tqdm_class"](total=100, desc="Downloading bytes", unit="B")
        written = kwargs["tqdm_class"](total=100, desc="Reconstructing", unit="B")
        wire.update(60)
        assert reported.wait(timeout=10)  # the reporter printed the first counts
        wire.update(34)
        written.update(100)
        return "/cache"

    _patch_download(monkeypatch, fake)
    assert worker.main([_TARGET.to_json()]) == 0
    lines = _protocol(capsys.readouterr().out)
    assert lines[-1] == {"done": True, "wire": 94, "written": 100}
    progress = [line for line in lines if "done" not in line]
    assert progress
    assert progress[0] == {"wire": 60, "written": 0}
    assert all(set(line) == {"wire", "written"} for line in progress)


def test_main_forwards_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(repo_id: str, **kwargs: Any) -> str:
        seen["repo_id"] = repo_id
        seen.update(kwargs)
        return "/cache"

    _patch_download(monkeypatch, fake)
    assert worker.main([_TARGET.to_json()]) == 0
    assert seen["repo_id"] == "owner/repo"
    assert seen["revision"] == "v1"
    assert seen["allow_patterns"] == ["*.safetensors", "config.json"]
    assert seen["ignore_patterns"] == ["*.md"]
    assert seen["local_dir"] == "/some/dir"
    assert callable(seen["tqdm_class"])

    seen.clear()
    assert worker.main([DownloadTarget("owner/repo").to_json()]) == 0
    assert (seen["allow_patterns"], seen["ignore_patterns"], seen["revision"], seen["local_dir"]) == (None, None, None, None)


def test_main_reports_a_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake(repo_id: str, **kwargs: Any) -> str:
        raise OSError("No space left on device")

    _patch_download(monkeypatch, fake)
    assert worker.main([_TARGET.to_json()]) == 1
    lines = _protocol(capsys.readouterr().out)
    assert lines[-1] == {"error": "OSError: No space left on device"}
    assert not any(line.get("done") for line in lines)


def test_main_reports_a_hub_refusal_through_the_refusal_message(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake(repo_id: str, **kwargs: Any) -> str:
        raise GatedRepoError("boom", response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/api/models/owner/repo")))

    _patch_download(monkeypatch, fake)
    assert worker.main([_TARGET.to_json()]) == 1
    error = _protocol(capsys.readouterr().out)[-1]["error"]
    assert "https://huggingface.co/owner/repo" in error


@pytest.mark.parametrize("argv", [["{not json"], ["a", "b"]])
def test_main_rejects_bad_arguments(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert worker.main(argv) == 1
    assert "error" in _protocol(capsys.readouterr().out)[-1]


def test_lifeline_ends_the_process_when_stdin_closes() -> None:
    code = (
        "import sys, time, huggingface_hub\n"
        "from zvisiongenerator.downloads import worker\n"
        "def fake(*a, **k):\n"
        "    print('ready', flush=True)\n"
        "    time.sleep(60)\n"
        "huggingface_hub.snapshot_download = fake\n"
        "raise SystemExit(worker.main([sys.argv[1]], start_lifeline=True))\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, DownloadTarget("owner/repo").to_json()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=REPO_ROOT,
    )
    try:
        assert process.stdin is not None and process.stdout is not None
        assert process.stdout.readline().strip() == b"ready"
        assert process.poll() is None
        process.stdin.close()
        assert process.wait(timeout=20) == 3
    finally:
        if process.poll() is None:
            process.kill()
        if process.stdout is not None:
            process.stdout.close()


def test_running_the_module_emits_no_runtime_warning() -> None:
    # stdin stays open so the lifeline does not end the child before it reports the bad argument.
    process = subprocess.Popen(
        [sys.executable, "-W", "error::RuntimeWarning", "-m", "zvisiongenerator.downloads.worker", "--bad"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )
    try:
        assert process.stdout is not None and process.stderr is not None
        returncode = process.wait(timeout=60)
        out, err = process.stdout.read(), process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
        if process.stdin is not None:
            process.stdin.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    assert "RuntimeWarning" not in err
    assert returncode == 1
    assert "error" in _protocol(out)[-1]


@pytest.mark.parametrize(("anonymous", "expected"), [(True, {"token": False}), (False, {})])
def test_main_sends_no_token_only_for_an_anonymous_target(monkeypatch: pytest.MonkeyPatch, anonymous: bool, expected: dict[str, Any]) -> None:
    seen: dict[str, Any] = {}

    def fake(repo_id: str, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "/cache"

    _patch_download(monkeypatch, fake)
    target = DownloadTarget("owner/repo", local_dir="/some/dir", anonymous=anonymous)

    assert worker.main([target.to_json()]) == 0
    assert {key: seen[key] for key in expected} == expected
    assert ("token" in seen) is anonymous
