"""Stand-ins for the ffmpeg and ffprobe processes, so no test starts a real one.

``FakeProcess`` replaces a ``subprocess.Popen`` whose stdout is given bytes, ``completed`` replaces the result of a
``subprocess.run``, and ``f32le`` turns audio into the raw PCM bytes ffmpeg would print.
"""

from __future__ import annotations

import io
import subprocess
from typing import Any
from unittest.mock import MagicMock

import numpy as np


def completed(stdout: bytes | str = b"", *, stderr: bytes | str = b"", returncode: int = 0) -> subprocess.CompletedProcess:
    """Return the result of a finished ``subprocess.run``."""
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def f32le(audio: np.ndarray) -> bytes:
    """Return stereo ``(2, samples)`` float32 audio as the interleaved ``f32le`` bytes ffmpeg writes."""
    return np.ascontiguousarray(np.asarray(audio, dtype=np.float32).T).tobytes()


def raw_frames(frames: np.ndarray) -> bytes:
    """Return ``(count, height, width, 3)`` uint8 frames as the raw ``rgb24`` bytes ffmpeg writes."""
    return np.ascontiguousarray(frames, dtype=np.uint8).tobytes()


class Sink:
    """A process's stdin that keeps what is written to it."""

    def __init__(self, *, broken: bool = False) -> None:
        self.chunks: list[bytes] = []
        self.closed = False
        self._broken = broken

    def write(self, data: bytes) -> int:
        if self._broken:
            raise BrokenPipeError
        self.chunks.append(bytes(data))
        return len(data)

    def close(self) -> None:
        if self._broken and not self.closed:
            self.closed = True
            raise BrokenPipeError
        self.closed = True

    @property
    def written(self) -> bytes:
        return b"".join(self.chunks)


class FakeProcess:
    """A stand-in for a ``subprocess.Popen`` that serves ``stdout``, records what is written to ``stdin`` and exits with ``returncode``.

    ``stderr`` is written to the file object the code passed as ``stderr=``, where a real process would write it.
    """

    def __init__(self, stdout: bytes = b"", *, returncode: int | None = 0, stderr: bytes = b"", stderr_file: Any = None, broken_stdin: bool = False) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stdin = Sink(broken=broken_stdin)
        self.returncode = returncode
        if stderr_file is not None:
            stderr_file.write(stderr)

    def __enter__(self) -> FakeProcess:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.stdin.close()
        return False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def popen_serving(*outputs: bytes, returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    """Return a ``Popen`` replacement that serves each output in turn as a process's stdout and records the commands."""
    queue = list(outputs)

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        return FakeProcess(queue.pop(0), returncode=returncode, stderr=stderr, stderr_file=kwargs.get("stderr"))

    return MagicMock(side_effect=popen)
