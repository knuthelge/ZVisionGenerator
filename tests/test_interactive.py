"""Tests for zvisiongenerator.utils.interactive — SkipSignal EOF handling."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch


def test_listen_unix_eof_breaks_loop():
    """_listen_unix should stop when stdin returns empty string (EOF)."""
    from zvisiongenerator.utils.interactive import SkipSignal

    signal = SkipSignal()
    signal._running = True

    # Mock stdin.read to return "" (EOF) immediately
    mock_stdin = MagicMock()
    mock_stdin.fileno.return_value = 0
    mock_stdin.read.return_value = ""

    with (
        patch.object(sys, "stdin", mock_stdin),
        patch("select.select", return_value=([mock_stdin], [], [])),
        patch("termios.tcgetattr", return_value=[]),
        patch("termios.tcsetattr"),
        patch("tty.setcbreak"),
    ):
        signal._listen_unix()

    # No keys should have been dispatched (EOF is not a real key)
    assert signal.consume() is None


# ---------------------------------------------------------------------------
# SkipSignal isatty guard
# ---------------------------------------------------------------------------


class TestSkipSignalIsattyGuard:
    """SkipSignal.start() must check isatty() and handle fileno() failures."""

    def test_start_noop_when_not_tty(self):
        """start() should return early (no thread) when stdin is not a tty."""
        from zvisiongenerator.utils.interactive import SkipSignal

        sig = SkipSignal()
        with patch.object(sys, "stdin", new=io.StringIO("test")):
            sig.start()
            assert sig._thread is None, "Thread should NOT be started for non-tty"

    def test_start_noop_when_no_isatty(self):
        """start() should handle stdin with no isatty attribute."""
        from zvisiongenerator.utils.interactive import SkipSignal

        sig = SkipSignal()
        mock_stdin = MagicMock(spec=[])  # no isatty
        with patch.object(sys, "stdin", mock_stdin):
            sig.start()
            assert sig._thread is None

    def test_listen_unix_handles_fileno_failure(self):
        """_listen_unix should return gracefully if fileno() raises."""
        from zvisiongenerator.utils.interactive import SkipSignal

        sig = SkipSignal()
        sig._running = True
        mock_stdin = MagicMock()
        mock_stdin.fileno.side_effect = OSError("no fd")
        with patch.object(sys, "stdin", mock_stdin):
            # Should not raise
            sig._listen_unix()
