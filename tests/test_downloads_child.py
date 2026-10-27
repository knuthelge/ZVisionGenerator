"""Tests for running a download in a child process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._paths import REPO_ROOT
from tests._server_helpers import FakeProcess, Launcher

from zvisiongenerator.downloads import child
from zvisiongenerator.downloads.child import ChildResult, child_env, run_in_child
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.downloads.protocol import PROTOCOL_PREFIX

_TARGET = DownloadTarget("owner/repo", allow_patterns=("*.safetensors",))


_FakeProcess = FakeProcess
_Launcher = Launcher


def _line(text: str) -> str:
    return PROTOCOL_PREFIX + text + "\n"


def _run(process: _FakeProcess, *, cancelled=lambda: False, cleanup=None, counts: list | None = None, **kwargs: Any) -> tuple[ChildResult, _Launcher, list]:
    launcher = _Launcher(process)
    calls: list = []
    result = run_in_child(
        _TARGET,
        on_counts=lambda wire, written: (counts if counts is not None else []).append((wire, written)),
        cancelled=cancelled,
        popen=launcher,
        poll_secs=0,
        grace_secs=0.01,
        cleanup=cleanup or calls.append,
        **kwargs,
    )
    return result, launcher, calls


def test_progress_then_done() -> None:
    process = _FakeProcess([_line('{"wire": 10, "written": 0}'), "noise\n", _line('{"wire": 90, "written": 100}'), _line('{"done": true, "wire": 94, "written": 100}')])
    counts: list = []
    result, _, cleaned = _run(process, counts=counts)
    assert result == ChildResult("done", None)
    assert counts == [(10, 0), (90, 100), (94, 100)]
    assert cleaned == []
    assert process.stdin.closed


def test_launch_arguments() -> None:
    result, launcher, _ = _run(_FakeProcess([_line('{"done": true, "wire": 0, "written": 0}')]))
    assert result.status == "done"
    assert launcher.args[0] == [sys.executable, "-m", "zvisiongenerator.downloads.worker", _TARGET.to_json()]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX session option")
def test_child_gets_its_own_session_on_posix() -> None:
    _, launcher, _ = _run(_FakeProcess([_line('{"done": true, "wire": 0, "written": 0}')]))
    assert launcher.kwargs["start_new_session"] is True
    assert "creationflags" not in launcher.kwargs


def test_child_gets_its_own_process_group_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(child.sys, "platform", "win32")
    monkeypatch.setattr(child.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    _, launcher, _ = _run(_FakeProcess([_line('{"done": true, "wire": 0, "written": 0}')]))
    assert launcher.kwargs["creationflags"] == 512
    assert "start_new_session" not in launcher.kwargs


def test_env_is_unbuffered_and_not_tqdm_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TQDM_DISABLE", "1")
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    _, launcher, _ = _run(_FakeProcess([_line('{"done": true, "wire": 0, "written": 0}')]))
    env = launcher.kwargs["env"]
    assert "TQDM_DISABLE" not in env
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert "TQDM_DISABLE" not in child_env()


def test_cancel_terminates_then_kills_and_cleans_up_once() -> None:
    process = _FakeProcess(running_polls=None, wait_timeouts=1)
    cancel_after = iter([False, False, True])
    result, _, cleaned = _run(process, cancelled=lambda: next(cancel_after, True))
    assert result == ChildResult("stopped", None)
    assert process.calls[:2] == ["terminate", "wait"]
    assert "kill" in process.calls
    assert process.calls.index("kill") > process.calls.index("terminate")
    assert cleaned == [_TARGET]
    assert process.stdin.closed


def test_cancel_does_not_kill_a_child_that_exits_in_time() -> None:
    process = _FakeProcess(running_polls=None)
    result, _, cleaned = _run(process, cancelled=lambda: True)
    assert result.status == "stopped"
    assert "kill" not in process.calls
    assert cleaned == [_TARGET]


def test_crash_cleans_up_and_reports_the_output_tail() -> None:
    process = _FakeProcess(["first\n", "Traceback...\n", "MemoryError\n"], exit_code=1)
    result, _, cleaned = _run(process)
    assert result.status == "failed"
    assert result.message == "first\nTraceback...\nMemoryError"
    assert cleaned == [_TARGET]


def test_tail_keeps_only_the_last_twenty_lines() -> None:
    process = _FakeProcess([f"line {i}\n" for i in range(50)], exit_code=1)
    result, _, _ = _run(process)
    assert result.message is not None
    assert result.message.splitlines() == [f"line {i}" for i in range(30, 50)]


def test_error_line_is_preferred_over_the_tail() -> None:
    process = _FakeProcess(["noise\n", _line('{"error": "OSError: No space left on device"}')], exit_code=1)
    result, _, cleaned = _run(process)
    assert result == ChildResult("failed", "OSError: No space left on device")
    assert cleaned == [_TARGET]


def test_silent_crash_reports_the_exit_code() -> None:
    result, _, _ = _run(_FakeProcess(exit_code=137))
    assert result.status == "failed"
    assert "137" in (result.message or "")


def test_exit_zero_without_a_done_line_is_a_failure() -> None:
    result, _, cleaned = _run(_FakeProcess(["hello\n"], exit_code=0))
    assert result.status == "failed"
    assert cleaned == [_TARGET]


@pytest.mark.parametrize(
    ("process", "cancelled", "status"),
    [
        (lambda: _FakeProcess(exit_code=1), False, "failed"),
        (lambda: _FakeProcess(running_polls=None), True, "stopped"),
    ],
    ids=["after-a-crash", "after-a-cancel"],
)
def test_cleanup_oserror_only_warns(process, cancelled: bool, status: str) -> None:
    def cleanup(target: DownloadTarget) -> None:
        raise OSError("locked")

    with pytest.warns(UserWarning, match="owner/repo"):
        result, _, _ = _run(process(), cancelled=lambda: cancelled, cleanup=cleanup)
    assert result.status == status


def test_malformed_protocol_line_is_kept_as_output() -> None:
    result, _, _ = _run(_FakeProcess([_line("{broken"), "x\n"], exit_code=1))
    assert result.status == "failed"
    assert PROTOCOL_PREFIX.strip() in (result.message or "")


def test_launch_failure_is_a_failed_result() -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("no python")

    result = run_in_child(_TARGET, on_counts=lambda *_: None, cancelled=lambda: False, popen=broken, poll_secs=0)
    assert result.status == "failed"
    assert "no python" in (result.message or "")


def test_unexpected_error_in_the_callback_still_stops_the_child() -> None:
    process = _FakeProcess([_line('{"wire": 1, "written": 0}')], running_polls=None)

    def boom(wire: int, written: int) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        run_in_child(_TARGET, on_counts=boom, cancelled=lambda: False, popen=_Launcher(process), poll_secs=0, grace_secs=0.01, cleanup=lambda t: None)
    assert "terminate" in process.calls


# Real worker process (SC-6): the patched download only finishes after a progress line reached the parent.
_WRAPPER = """
import sys, time
from pathlib import Path
import huggingface_hub
from zvisiongenerator.downloads import worker

release = Path(sys.argv[2])
mode = sys.argv[3]

def fake_snapshot_download(repo_id, **kwargs):
    sink = kwargs["tqdm_class"](total=100, desc="Downloading bytes", unit="B")
    sink.update(40)
    deadline = time.monotonic() + (120 if mode == "hang" else 30)
    while not release.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("release file never appeared")
        time.sleep(0.05)
    sink.update(60)

huggingface_hub.snapshot_download = fake_snapshot_download
raise SystemExit(worker.main([sys.argv[1]], start_lifeline=True))
"""


def _real_popen(release: Path, mode: str):
    def popen(command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        return subprocess.Popen([sys.executable, "-c", _WRAPPER, command[-1], str(release), mode], cwd=REPO_ROOT, **kwargs)

    return popen


def test_real_child_streams_counts_before_it_finishes(tmp_path: Path) -> None:
    release = tmp_path / "release"
    counts: list[tuple[int, int]] = []

    def on_counts(wire: int, written: int) -> None:
        counts.append((wire, written))
        release.touch()

    result = run_in_child(_TARGET, on_counts=on_counts, cancelled=lambda: False, popen=_real_popen(release, "finish"), poll_secs=0.05, cleanup=lambda t: None)
    assert result.status == "done", result.message
    assert len(counts) >= 1
    assert counts[0] == (40, 0)
    assert counts[-1] == (100, 0)


def test_real_child_is_stopped_by_cancel(tmp_path: Path) -> None:
    release = tmp_path / "never"
    seen: list[tuple[int, int]] = []
    cleaned: list[DownloadTarget] = []
    result = run_in_child(
        _TARGET,
        on_counts=lambda wire, written: seen.append((wire, written)),
        cancelled=lambda: bool(seen),
        popen=_real_popen(release, "hang"),
        poll_secs=0.05,
        grace_secs=10.0,
        cleanup=cleaned.append,
    )
    assert result == ChildResult("stopped", None)
    assert seen
    assert cleaned == [_TARGET]


def test_env_bounds_the_xet_download_buffer_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE", raising=False)
    assert child_env()["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"] == "512mb"


def test_env_keeps_a_user_set_xet_download_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE", "2gb")
    assert child_env()["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"] == "2gb"
