"""Own the single resident prompt-enhancer model: exclusive use, idle release, and availability checks."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.config import model_reference, resolve_enhancer_model
from zvisiongenerator.utils.model_files import find_local_model_dir
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, enhancement_requested

if TYPE_CHECKING:
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer

type EnhancerFactory = Callable[[str, str | None], "PromptEnhancer"]
type Scheduler = Callable[[float, Callable[[], None]], None]
type PhaseCallback = Callable[[str], None]


def is_model_downloaded(repo: str, revision: str | None) -> bool:
    """Return whether the enhancer weights are a local directory or fully in the Hugging Face cache."""
    return find_local_model_dir(model_reference(repo, revision)) is not None


def runs_on_cpu(enhancer: PromptEnhancer) -> bool:
    """Return whether *enhancer* fell back to the CPU (transformers without CUDA), where rewrites take minutes."""
    return getattr(enhancer, "on_cuda", True) is False


def _hub_offline() -> bool:
    from huggingface_hub import constants

    return bool(constants.HF_HUB_OFFLINE)


def ensure_available(repo: str, revision: str | None, *, downloaded: Callable[[str, str | None], bool] = is_model_downloaded, offline: Callable[[], bool] = _hub_offline) -> bool:
    """Fail fast when the model is neither downloaded nor downloadable; return whether it is downloaded.

    Raises:
        RuntimeError: When the weights are missing and Hugging Face is offline.
    """
    if downloaded(repo, revision):
        return True
    if offline():
        raise RuntimeError(f"Enhancer model {model_reference(repo, revision)} is not downloaded and Hugging Face is offline.")
    return False


def _thread_scheduler(delay: float, callback: Callable[[], None]) -> None:
    timer = threading.Timer(delay, callback)
    timer.daemon = True
    timer.start()


class PromptEnhancerSession:
    """Serialize enhancer use and keep at most one model resident.

    Args:
        factory: Builds a loaded enhancer for ``(repo, revision)``.
        clock: Monotonic time source (injectable for tests).
        scheduler: Runs a callback after a delay (injectable for tests).
        downloaded: Reports whether a model is already downloaded.
    """

    def __init__(
        self,
        factory: EnhancerFactory,
        *,
        clock: Callable[[], float] = time.monotonic,
        scheduler: Scheduler = _thread_scheduler,
        downloaded: Callable[[str, str | None], bool] = is_model_downloaded,
    ) -> None:
        self._factory = factory
        self._clock = clock
        self._scheduler = scheduler
        self._downloaded = downloaded
        self._use_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._reservation: object | None = None
        self._enhancer: PromptEnhancer | None = None
        self._key: tuple[str, str | None] | None = None
        self._busy = False
        self._last_used = 0.0

    def busy(self) -> bool:
        """Return whether an enhancement (or model load/download) is in progress or reserved."""
        return self._busy or self._reservation is not None

    def reserve(self) -> object | None:
        """Claim the next :meth:`acquire` for an on-demand enhancement; return a token, or None when busy/claimed.

        Lets the caller admit work atomically (e.g. under the web job lock) before its worker thread starts.
        """
        with self._state_lock:
            if self._busy or self._reservation is not None:
                return None
            self._reservation = object()
            return self._reservation

    def cancel_reservation(self, token: object) -> None:
        """Drop *token*'s reservation if :meth:`acquire` never consumed it; other reservations are untouched."""
        with self._state_lock:
            if self._reservation is token:
                self._reservation = None

    def resident(self) -> tuple[str, str | None] | None:
        """Return the ``(repo, revision)`` of the loaded model, if any."""
        return self._key

    @contextmanager
    def acquire(self, repo: str, revision: str | None, *, idle_seconds: float | None, on_phase: PhaseCallback | None = None) -> Iterator[PromptEnhancer]:
        """Hold the enhancer exclusively, loading (or swapping) the model as needed.

        Args:
            idle_seconds: After use, release the model once idle this long; ``None`` keeps it
                resident until :meth:`release` (job use), ``0`` releases immediately.
            on_phase: Receives ``"downloading"`` or ``"loading"`` before a model load.
        """
        with self._use_lock:
            with self._state_lock:
                self._busy = True
                self._reservation = None
            try:
                if self._enhancer is not None and self._key != (repo, revision):
                    self._close()
                if self._enhancer is None:
                    if on_phase is not None:
                        on_phase("loading" if self._downloaded(repo, revision) else "downloading")
                    self._enhancer = self._factory(repo, revision)
                    self._key = (repo, revision)
                yield self._enhancer
            finally:
                self._busy = False
                self._last_used = self._clock()
                if idle_seconds is not None:
                    if idle_seconds <= 0:
                        self._close()
                    else:
                        self._scheduler(idle_seconds, lambda: self.release_if_idle(idle_seconds))

    def release(self) -> None:
        """Unload the resident model (waits for an in-progress enhancement to finish)."""
        with self._use_lock:
            self._close()

    def release_if_idle(self, idle_seconds: float) -> bool:
        """Unload the model when unused for *idle_seconds*; never waits on an active user."""
        if not self._use_lock.acquire(blocking=False):
            return False
        try:
            if self._enhancer is None or self._clock() - self._last_used < idle_seconds:
                return False
            self._close()
            return True
        finally:
            self._use_lock.release()

    def _close(self) -> None:
        enhancer, self._enhancer, self._key = self._enhancer, None, None
        if enhancer is not None:
            enhancer.close()


_SESSION: PromptEnhancerSession | None = None
_SESSION_LOCK = threading.Lock()


def get_prompt_enhancer_session() -> PromptEnhancerSession:
    """Return the process-wide enhancer session (platform adapter chosen in ``backends``)."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            from zvisiongenerator.backends import create_prompt_enhancer

            _SESSION = PromptEnhancerSession(create_prompt_enhancer)
        return _SESSION


def plan_job_enhancer(
    config: dict[str, Any],
    *,
    platform_key: str,
    disabled: bool,
    override: EnhanceSettings | None,
    enhance_by_set: dict[str, list[EnhanceSettings | None]] | None,
    cli_model: str | None = None,
) -> tuple[str, str | None] | None:
    """Return the enhancer ``(repo, revision)`` when any prompt in the job is enhanced, after a preflight.

    Raises:
        ValueError: When no enhancer model is configured or *cli_model* is malformed.
        RuntimeError: When the model is missing and Hugging Face is offline.
    """
    if not enhancement_requested(disabled=disabled, override=override, enhance_by_set=enhance_by_set):
        return None
    repo, revision = resolve_enhancer_model(config, platform_key=platform_key, cli_model=cli_model)
    ensure_available(repo, revision)
    return repo, revision


@contextmanager
def job_enhancer(repo: str, revision: str | None, *, on_phase: PhaseCallback | None = None, session: PromptEnhancerSession | None = None) -> Iterator[PromptEnhancer]:
    """Hold the enhancer for a whole auto-enhance job and release it when the job ends (even on failure)."""
    active = session or get_prompt_enhancer_session()
    try:
        with active.acquire(repo, revision, idle_seconds=None, on_phase=on_phase) as enhancer:
            yield enhancer
    finally:
        active.release()


def release_resident_enhancer() -> None:
    """Unload an idle enhancer (e.g. before a job that does not enhance loads its model)."""
    if _SESSION is not None:
        _SESSION.release()
