"""Handle interactive skip/quit signals during generation."""

from __future__ import annotations

import sys
import threading


def stdin_is_interactive() -> bool:
    """Return whether stdin is a terminal that can deliver key presses."""
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


class SkipSignal:
    _INTERRUPT_ACTIONS = frozenset({"skip", "quit"})
    _KNOWN_ACTIONS = frozenset({"skip", "quit", "pause", "repeat"})
    _REVIEW_DECISIONS = frozenset({"approve", "discard"})
    _REVIEW_POLL_SECONDS = 0.2

    def __init__(self, *, quit_only: bool = False):
        """Create the signal.

        Args:
            quit_only: Whether the keys ``n``, ``p`` and ``r`` are ignored outside a review, so that only ``q`` queues
                an action. For video, where Next, Pause and Repeat do not exist.
        """
        self._quit_only = quit_only
        self._action = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._old_settings = None
        self._terminal_lock = threading.Lock()
        self._waiting_for_resume = False
        self._resume_event = threading.Event()
        self._reviewing = False
        self._review_decision: str | None = None
        self._review_open = True
        self._review_event = threading.Event()

    def start(self):
        if not stdin_is_interactive():
            return  # Don't start listener in non-interactive contexts
        self._running = True
        with self._lock:
            self._action = None
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._restore_terminal()

    def reset(self):
        with self._lock:
            self._action = None

    def check(self) -> bool:
        with self._lock:
            return self._action in self._INTERRUPT_ACTIONS

    def pending(self) -> str | None:
        """Return the queued action without consuming it."""
        with self._lock:
            return self._action

    def consume(self) -> str | None:
        with self._lock:
            action = self._action
            self._action = None
            return action

    def queue_action(self, action: str) -> None:
        normalized = action.strip().lower()
        if normalized not in self._KNOWN_ACTIONS:
            raise ValueError(f"Unknown action '{action}'.")
        with self._lock:
            self._set_action(normalized)
        if normalized == "quit":
            self._review_event.set()
            self._resume_event.set()

    def _set_action(self, action: str) -> None:
        """Queue *action*, unless a quit is already queued: a requested quit is never replaced (call with the lock held)."""
        if self._action != "quit":
            self._action = action

    def resume(self) -> None:
        with self._lock:
            if not self._waiting_for_resume:
                return
            self._resume_event.set()

    def is_waiting_for_resume(self) -> bool:
        with self._lock:
            return self._waiting_for_resume

    def wait_for_key(self):
        """Block until a key press or :meth:`resume`; return at once when a Quit is queued, and wake on one (it stays queued)."""
        with self._lock:
            if self._action == "quit":
                return
            self._waiting_for_resume = True
            self._resume_event.clear()
        self._resume_event.wait()
        with self._lock:
            self._waiting_for_resume = False

    def set_review_decision(self, decision: str) -> None:
        """Record the decision of a review (``"approve"`` or ``"discard"``) and wake the waiter; the first decision wins.

        Raises:
            ValueError: If *decision* is neither ``"approve"`` nor ``"discard"``.
        """
        if decision not in self._REVIEW_DECISIONS:
            raise ValueError(f"Unknown review decision '{decision}'.")
        with self._lock:
            if not self._review_open or self._review_decision is not None:
                return
            self._review_decision = decision
        self._review_event.set()

    def begin_review(self) -> None:
        """Open the review to decisions; call it before announcing the review, so an answer to the announcement is kept.

        A review closes when :meth:`wait_for_review` ends, so a decision arriving after that cannot decide the next one.
        """
        with self._lock:
            self._review_open = True

    def wait_for_review(self) -> str:
        """Block until the review is decided and return ``"approve"``, ``"discard"``, ``"skip"`` or ``"quit"``.

        A Quit or Skip (Next) queued before the review returns at once and stays queued for the runner to consume (a
        decision already made wins over a Skip); a Skip ends the review without an upscale. A queued Pause or Repeat stays queued and takes effect after the
        image, whichever way the review ends. The wait polls with a short timeout so Ctrl-C stays responsive on every
        platform.
        """
        with self._lock:
            self._review_open = True
        try:
            with self._lock:
                if self._action == "quit":
                    return "quit"
                if self._action == "skip" and self._review_decision is None:
                    return "skip"
                self._reviewing = True
            while True:
                with self._lock:
                    if self._action == "quit":
                        return "quit"
                    if self._review_decision is not None:
                        return self._review_decision
                    self._review_event.clear()
                self._review_event.wait(timeout=self._REVIEW_POLL_SECONDS)
        finally:
            with self._lock:
                self._reviewing = False
                self._review_decision = None
                self._review_open = False
            self._review_event.clear()

    def _on_review_key(self, ch: str) -> str | None:
        """Map a key pressed during a review to a decision or Quit; return the message to show (call with the lock held)."""
        if self._review_decision is not None:
            return None
        if ch == "y":
            self._review_decision = "approve"
            self._review_event.set()
            return "✔  [y] approved — upscaling."
        if ch == "n":
            self._review_decision = "discard"
            self._review_event.set()
            return "✖  [n] discarded."
        if ch == "q":
            self._set_action("quit")
            self._review_event.set()
            return "⏹  [q] quit."
        return None

    def _on_key(self, ch):
        msg = None
        with self._lock:
            if self._reviewing:
                msg = self._on_review_key(ch.lower())
            elif self._waiting_for_resume:
                self._resume_event.set()
                return
            else:
                msg = self._on_command_key(ch.lower())

        if msg:
            print(msg, file=sys.stderr, flush=True)

    def _on_command_key(self, ch: str) -> str | None:
        """Queue the action for a command key and return the message to show (call with the lock held)."""
        if self._quit_only and ch != "q":
            return None
        if self._action == "quit" and ch in ("n", "p", "r"):
            return "⏹  quit already queued."
        if ch == "n":
            self._set_action("skip")
            return "⏭  [n] skip — queued, takes effect after current step."
        if ch == "q":
            self._set_action("quit")
            return "⏹  [q] quit — queued, takes effect after current step."
        if ch == "p":
            self._set_action("pause")
            return "⏸  [p] pause — queued, takes effect after current image."
        if ch == "r":
            self._set_action("repeat")
            return "♻  [r] repeat — queued, takes effect after current image."
        return None

    def _listen(self):
        if sys.platform == "win32":
            self._listen_windows()
        else:
            self._listen_unix()

    def _listen_unix(self):
        import select
        import termios
        import tty

        try:
            fd = sys.stdin.fileno()
        except AttributeError, OSError:
            return  # stdin not usable (e.g., redirected or unavailable)

        try:
            self._old_settings = termios.tcgetattr(fd)
        except termios.error:
            return

        try:
            tty.setcbreak(fd)
            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    self._on_key(ch)
        finally:
            self._restore_terminal()

    def _listen_windows(self):
        import msvcrt
        import time

        while self._running:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode("utf-8", errors="ignore")
                self._on_key(ch)
            else:
                time.sleep(0.05)

    def _restore_terminal(self):
        with self._terminal_lock:
            if self._old_settings is not None:
                import termios

                fd = sys.stdin.fileno()
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)
                self._old_settings = None
