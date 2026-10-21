"""Tests for the download worker's protocol output."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from zvisiongenerator.downloads import worker
from zvisiongenerator.downloads.target import DownloadTarget

_ROOT = Path(__file__).resolve().parent.parent
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


def test_protocol_prefix_is_stable() -> None:
    assert worker.PROTOCOL_PREFIX == "ZIV-DOWNLOAD "


def test_main_prints_progress_then_done(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(worker, "REPORT_INTERVAL_SECS", 0.01)

    def fake(repo_id: str, **kwargs: Any) -> str:
        wire = kwargs["tqdm_class"](total=100, desc="Downloading bytes", unit="B")
        written = kwargs["tqdm_class"](total=100, desc="Reconstructing", unit="B")
        wire.update(60)
        time.sleep(0.2)
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


def test_main_forwards_unset_patterns_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    _patch_download(monkeypatch, lambda repo_id, **kwargs: seen.update(kwargs))
    assert worker.main([DownloadTarget("owner/repo").to_json()]) == 0
    assert seen["allow_patterns"] is None
    assert seen["ignore_patterns"] is None
    assert seen["revision"] is None
    assert seen["local_dir"] is None


def test_main_reports_a_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake(repo_id: str, **kwargs: Any) -> str:
        raise OSError("No space left on device")

    _patch_download(monkeypatch, fake)
    assert worker.main([_TARGET.to_json()]) == 1
    lines = _protocol(capsys.readouterr().out)
    assert lines[-1] == {"error": "OSError: No space left on device"}
    assert not any(line.get("done") for line in lines)


def test_main_reports_a_hub_refusal_as_one_line_with_the_next_step(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake(repo_id: str, **kwargs: Any) -> str:
        raise GatedRepoError("boom", response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/api/models/owner/repo")))

    _patch_download(monkeypatch, fake)
    assert worker.main([_TARGET.to_json()]) == 1
    error = _protocol(capsys.readouterr().out)[-1]["error"]
    assert "\n" not in error
    assert "https://huggingface.co/owner/repo" in error
    assert "hf auth login" in error


@pytest.mark.parametrize("argv", [[], ["{not json"], ["{}"], ["a", "b"]])
def test_main_rejects_bad_arguments(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert worker.main(argv) == 1
    assert "error" in _protocol(capsys.readouterr().out)[-1]


def test_lifeline_ends_the_process_when_stdin_closes() -> None:
    code = (
        "import sys, time, huggingface_hub\n"
        "from zvisiongenerator.downloads import worker\n"
        "huggingface_hub.snapshot_download = lambda *a, **k: time.sleep(60)\n"
        "raise SystemExit(worker.main([sys.argv[1]], start_lifeline=True))\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, DownloadTarget("owner/repo").to_json()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=_ROOT,
    )
    try:
        assert process.stdin is not None
        time.sleep(1.0)
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
        cwd=_ROOT,
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
