"""A scripted ``Popen`` and its launcher, for the tests that drive the download child process and the web runner."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from typing import Any


class _Stdin:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Stdout:
    def __init__(self, lines: Iterable[str]) -> None:
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        pass


class FakeProcess:
    """A scripted ``Popen``: prints *lines*, runs for *running_polls* polls, then exits with *exit_code*."""

    def __init__(self, lines: Iterable[str] = (), *, running_polls: int | None = 0, exit_code: int = 0, wait_timeouts: int = 0) -> None:
        self.stdin = _Stdin()
        self.stdout = _Stdout(lines)
        self._running_polls = running_polls
        self._exit_code = exit_code
        self._wait_timeouts = wait_timeouts
        self.returncode: int | None = None
        self.calls: list[str] = []

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._running_polls is None:
            return None
        if self._running_polls == 0:
            self.returncode = self._exit_code
            return self.returncode
        self._running_polls -= 1
        return None

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append("wait")
        if self._wait_timeouts:
            self._wait_timeouts -= 1
            raise subprocess.TimeoutExpired("fake", timeout or 0)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class Launcher:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> FakeProcess:
        self.args, self.kwargs = args, kwargs
        return self.process
