"""Handle interactive skip/quit signals during generation."""

from __future__ import annotations

import sys
import threading


class SkipSignal:
    _INTERRUPT_ACTIONS = frozenset({"skip", "quit"})
    _KNOWN_ACTIONS = frozenset({"skip", "quit", "pause", "repeat"})

    def __init__(self):
        self._action = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._old_settings = None
        self._terminal_lock = threading.Lock()
        self._waiting_for_resume = False
        self._resume_event = threading.Event()

    def start(self):
        if not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
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
            self._action = normalized

    def resume(self) -> None:
        with self._lock:
            if not self._waiting_for_resume:
                return
            self._resume_event.set()

    def is_waiting_for_resume(self) -> bool:
        with self._lock:
            return self._waiting_for_resume

    def wait_for_key(self):
        with self._lock:
            self._waiting_for_resume = True
            self._resume_event.clear()
        self._resume_event.wait()
        with self._lock:
            self._waiting_for_resume = False

    def _on_key(self, ch):
        msg = None
        with self._lock:
            if self._waiting_for_resume:
                self._resume_event.set()
                return

            ch = ch.lower()
            if ch == "n":
                self._action = "skip"
                msg = "⏭  [n] skip — queued, takes effect after current step."
            elif ch == "q":
                self._action = "quit"
                msg = "⏹  [q] quit — queued, takes effect after current step."
            elif ch == "p":
                self._action = "pause"
                msg = "⏸  [p] pause — queued, takes effect after current image."
            elif ch == "r":
                self._action = "repeat"
                msg = "♻  [r] repeat — queued, takes effect after current image."

        if msg:
            print(msg, file=sys.stderr, flush=True)

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
