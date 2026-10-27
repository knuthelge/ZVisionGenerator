"""Tests for zvisiongenerator.utils.interactive — SkipSignal EOF handling."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

import pytest


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


class TestSkipSignalPending:
    def test_pending_reads_without_consuming(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        assert signal.pending() is None
        signal.queue_action("quit")
        assert signal.pending() == "quit"
        assert signal.pending() == "quit"
        assert signal.consume() == "quit"
        assert signal.pending() is None

    @pytest.mark.parametrize("later", ["skip", "pause", "repeat"])
    def test_a_queued_quit_is_never_replaced(self, later):
        """A stopped job may hold no model; a key pressed after Quit must not turn the stop into a generation."""
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action("quit")
        signal.queue_action(later)
        signal._on_key(later[0])

        assert signal.consume() == "quit"
        signal.queue_action(later)
        assert signal.consume() == later


class TestSkipSignalReview:
    """The review state: its own decision slot, separate from Pause."""

    @staticmethod
    def _wait_in_thread(signal):
        """Start wait_for_review on a thread; return (thread, result list) once the review is waiting."""
        import threading
        import time

        result: list[str] = []
        thread = threading.Thread(target=lambda: result.append(signal.wait_for_review()), daemon=True)
        thread.start()
        deadline = time.monotonic() + 1.0
        while not signal._reviewing and time.monotonic() < deadline:
            time.sleep(0.001)
        assert signal._reviewing
        return thread, result

    @pytest.mark.parametrize(("key", "decision"), [("y", "approve"), ("N", "discard")])
    def test_keys_decide_during_a_review(self, key, decision):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        thread, result = self._wait_in_thread(signal)
        signal._on_key(key)
        thread.join(timeout=1.0)

        assert result == [decision]
        assert signal.pending() is None

    def test_q_quits_during_a_review_and_stays_queued(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        thread, result = self._wait_in_thread(signal)
        signal._on_key("q")
        thread.join(timeout=1.0)

        assert result == ["quit"]
        assert signal.pending() == "quit"

    @pytest.mark.parametrize("key", ["p", "\r"])
    def test_other_keys_are_ignored_during_a_review(self, key):
        """Enter and the command keys neither decide nor queue anything, so a stray key cannot discard a render."""
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        thread, result = self._wait_in_thread(signal)
        signal._on_key(key)
        thread.join(timeout=0.1)

        assert thread.is_alive()
        assert result == []
        assert signal.pending() is None
        signal.set_review_decision("discard")
        thread.join(timeout=1.0)

    def test_first_decision_wins(self):
        """A decision set before the wait starts is not lost (the web can answer as soon as job_paused is published)."""
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.set_review_decision("approve")
        signal.set_review_decision("discard")
        signal._on_key("n")

        assert signal.wait_for_review() == "approve"

    @pytest.mark.parametrize("decision", ["approve", "discard"])
    def test_web_decision_wakes_the_waiter(self, decision):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        thread, result = self._wait_in_thread(signal)
        signal.set_review_decision(decision)
        thread.join(timeout=1.0)

        assert result == [decision]

    def test_unknown_web_decision_is_refused(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        with pytest.raises(ValueError, match="Unknown review decision"):
            SkipSignal().set_review_decision("later")

    def test_queue_action_quit_wakes_the_waiter(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        thread, result = self._wait_in_thread(signal)
        signal.queue_action("quit")
        thread.join(timeout=1.0)

        assert result == ["quit"]
        assert signal.pending() == "quit"

    def test_a_queued_quit_returns_at_once(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action("quit")

        assert signal.wait_for_review() == "quit"
        assert signal.pending() == "quit"

    def test_a_queued_skip_ends_the_review_without_an_upscale_and_stays_queued(self):
        """A Next queued before the review skips the image, as Next always does."""
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action("skip")

        assert signal.wait_for_review() == "skip"
        assert signal.pending() == "skip"

    def test_a_decision_already_made_wins_over_a_queued_skip(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action("skip")
        signal.set_review_decision("approve")

        assert signal.wait_for_review() == "approve"

    @pytest.mark.parametrize("action", ["pause", "repeat"])
    def test_a_queued_pause_or_repeat_survives(self, action):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action(action)
        signal.set_review_decision("approve")

        assert signal.wait_for_review() == "approve"
        assert signal.pending() == action

    def test_review_does_not_use_the_pause_state(self):
        """Resume and any key end a Pause, never a review."""
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        thread, result = self._wait_in_thread(signal)
        signal.resume()
        signal._on_key("p")
        thread.join(timeout=0.1)

        assert thread.is_alive()
        assert not signal.is_waiting_for_resume()
        signal.set_review_decision("approve")
        thread.join(timeout=1.0)

    def test_a_decision_does_not_leak_into_the_next_review(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.set_review_decision("approve")
        assert signal.wait_for_review() == "approve"

        thread, result = self._wait_in_thread(signal)
        thread.join(timeout=0.1)
        assert thread.is_alive()
        signal.set_review_decision("discard")
        thread.join(timeout=1.0)
        assert result == ["discard"]


class TestSkipSignalStaleDecisions:
    """A decision that arrives after a review ended must not decide the next one."""

    @pytest.mark.parametrize("queued", ["skip", "quit"])
    def test_a_decision_after_an_early_return_does_not_decide_the_next_review(self, queued):
        import threading

        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action(queued)
        assert signal.wait_for_review() == queued

        signal.set_review_decision("approve")  # the web answered the review that just ended
        signal.consume()

        signal.begin_review()
        result: list[str] = []
        thread = threading.Thread(target=lambda: result.append(signal.wait_for_review()), daemon=True)
        thread.start()
        thread.join(timeout=0.05)

        assert thread.is_alive()
        assert result == []
        signal.set_review_decision("discard")
        thread.join(timeout=1.0)
        assert result == ["discard"]

    def test_an_answer_to_the_announcement_is_kept_once_the_review_begins(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action("quit")
        signal.wait_for_review()
        signal.consume()

        signal.begin_review()
        signal.set_review_decision("approve")

        assert signal.wait_for_review() == "approve"


class TestSkipSignalPauseAndQuit:
    """A Quit ends a pause, also one that starts after the Quit was queued."""

    def test_a_quit_queued_before_the_pause_returns_at_once(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action("quit")

        signal.wait_for_key()

        assert not signal.is_waiting_for_resume()
        assert signal.pending() == "quit"

    def test_a_quit_queued_during_the_pause_wakes_it(self):
        import threading
        import time

        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        paused = threading.Thread(target=signal.wait_for_key, daemon=True)
        paused.start()
        deadline = time.monotonic() + 1.0
        while not signal.is_waiting_for_resume() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert signal.is_waiting_for_resume()

        signal.queue_action("quit")
        paused.join(timeout=1.0)

        assert not paused.is_alive()
        assert signal.pending() == "quit"

    def test_other_queued_actions_do_not_end_a_pause(self):
        import threading
        import time

        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.queue_action("skip")
        paused = threading.Thread(target=signal.wait_for_key, daemon=True)
        paused.start()
        deadline = time.monotonic() + 1.0
        while not signal.is_waiting_for_resume() and time.monotonic() < deadline:
            time.sleep(0.001)
        paused.join(timeout=0.2)

        assert paused.is_alive()
        signal.resume()
        paused.join(timeout=1.0)
        assert not paused.is_alive()


class TestStdinIsInteractive:
    def test_follows_isatty(self, monkeypatch):
        from types import SimpleNamespace

        from zvisiongenerator.utils.interactive import stdin_is_interactive

        monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
        assert stdin_is_interactive() is True
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
        assert stdin_is_interactive() is False

    def test_false_without_isatty(self, monkeypatch):
        from zvisiongenerator.utils.interactive import stdin_is_interactive

        monkeypatch.setattr(sys, "stdin", MagicMock(spec=[]))
        assert stdin_is_interactive() is False


class TestSkipSignalReviewEdges:
    """Races and loop behaviour of the review state that the happy-path tests do not reach."""

    def test_a_quit_before_the_wait_starts_wins_over_a_decision(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal.set_review_decision("approve")
        signal.queue_action("quit")

        assert signal.wait_for_review() == "quit"
        assert signal.pending() == "quit"
        assert signal._review_decision is None

    def test_the_wait_polls_with_a_timeout_so_ctrl_c_stays_responsive(self):
        """The waiter must never block without a timeout: Windows only delivers Ctrl-C between waits."""
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        timeouts: list[float | None] = []
        real_wait = signal._review_event.wait

        def _spy(timeout=None):
            timeouts.append(timeout)
            if len(timeouts) == 3:
                signal.set_review_decision("approve")
            return real_wait(timeout=0.001)

        signal._review_event = MagicMock(wait=_spy, set=signal._review_event.set, clear=signal._review_event.clear)

        assert signal.wait_for_review() == "approve"
        assert len(timeouts) >= 3
        assert all(t is not None and 0 < t <= 1.0 for t in timeouts)

    def test_an_interrupt_during_the_wait_leaves_the_signal_reusable(self):
        import threading

        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal._review_event = MagicMock(wait=MagicMock(side_effect=KeyboardInterrupt), set=MagicMock(), clear=MagicMock())

        with pytest.raises(KeyboardInterrupt):
            signal.wait_for_review()

        assert signal._reviewing is False
        signal._review_event = threading.Event()
        signal.begin_review()
        signal.set_review_decision("approve")
        assert signal.wait_for_review() == "approve"

    @pytest.mark.parametrize("key", ["y", "Y", "\r"])
    def test_review_keys_do_nothing_outside_a_review(self, key):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        signal._on_key(key)

        assert signal.pending() is None
        assert signal._review_decision is None

    def test_command_keys_still_queue_outside_a_review(self):
        """The review branch must not change Next, Quit, Pause and Repeat."""
        from zvisiongenerator.utils.interactive import SkipSignal

        for key, action in (("n", "skip"), ("p", "pause"), ("r", "repeat"), ("q", "quit")):
            signal = SkipSignal()
            signal._on_key(key)
            assert signal.consume() == action

    def test_a_key_press_ends_a_pause_but_not_a_review(self):
        """Pause keeps its any-key resume, and a review keeps its own decision slot beside it."""
        import threading
        import time

        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal()
        paused = threading.Thread(target=signal.wait_for_key, daemon=True)
        paused.start()
        deadline = time.monotonic() + 1.0
        while not signal.is_waiting_for_resume() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert signal.is_waiting_for_resume()
        signal._on_key("x")
        paused.join(timeout=1.0)
        assert not paused.is_alive()
        assert signal._review_decision is None


class TestQuitOnlySignal:
    """A signal for video: outside a review only q queues an action."""

    @pytest.mark.parametrize("key", ["n", "p", "r", "x"])
    def test_other_command_keys_queue_nothing_and_print_nothing(self, key, capsys):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal(quit_only=True)
        signal._on_key(key)

        assert signal.pending() is None
        assert capsys.readouterr().err == ""

    def test_q_still_queues_quit(self):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal(quit_only=True)
        signal._on_key("q")

        assert signal.pending() == "quit"

    @pytest.mark.parametrize(("key", "decision"), [("y", "approve"), ("n", "discard")])
    def test_a_review_still_takes_its_keys(self, key, decision):
        from zvisiongenerator.utils.interactive import SkipSignal

        signal = SkipSignal(quit_only=True)
        thread, result = TestSkipSignalReview._wait_in_thread(signal)
        signal._on_key(key)
        thread.join(timeout=1.0)

        assert result == [decision]
        assert signal.pending() is None
