"""YuE2 audio backend: songs and instrumentals from the vendored kit, with a CUDA weight swap.

A take runs the kit stage by stage: ``plan`` (the ABC score) and ``generate_semantic`` (the song tokens) are the
autoregressive phases, ``CachedNAR`` prefills and solves the acoustic latents, and ``decode`` turns them into a stereo
48 kHz waveform. With ``weight_swap`` (CUDA) the model is held in pinned CPU memory by :class:`WeightSwap`, which puts
only the running phase's weights on the GPU. Without it (MPS) the whole model sits on the device. Either way the weights are
streamed in tensor by tensor (see :mod:`yue2_load`) and the acoustic part runs in one loop with the same progress.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.backends.audio.build_cancel import stop_while_building
from zvisiongenerator.backends.audio.yue2_load import load_model, size_checked_weights
from zvisiongenerator.backends.audio.yue2_swap import WeightSwap
from zvisiongenerator.core.audio_backend import DECODE_PHASE, AudioCancelled, raise_if_cancelled
from zvisiongenerator.core.audio_types import AudioClip
from zvisiongenerator.utils.audio_fade import fade_out
from zvisiongenerator.utils.music_mode import INSTRUMENTAL, music_mode

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioPreset

__all__ = ["SAMPLE_RATE", "YuE2Backend"]

SAMPLE_RATE: Final = 48000
"""The sample rate of the VAE's waveform."""

_TOKENS_PER_SECOND: Final = 25
"""Semantic tokens (codec frames) per second of audio."""

_TOKEN_STRIDE: Final = 10
"""Report every this many tokens, so a long song does not flood the progress callback."""


class _CancelProbe:
    """Wrap the runner's cancel check so the backend knows its own stop, not some other ``InterruptedError``."""

    def __init__(self, cancelled: CancelCheck) -> None:
        self._cancelled = cancelled
        self.requested = False

    def __call__(self) -> bool:
        """Return ``True`` once the user asked to stop, and keep returning it."""
        if not self.requested and self._cancelled():
            self.requested = True
        return self.requested

    def raise_if_requested(self) -> None:
        """Raise :class:`AudioCancelled` if the user asked to stop."""
        if self():
            raise AudioCancelled("The take was stopped.")


class YuE2Backend:
    """Audio backend of the ``yue2`` family. One instance serves one loaded preset; call :meth:`close` to free it."""

    def __init__(self) -> None:
        self._pipe: Any = None
        self._swap: WeightSwap | None = None
        self._device = ""
        self._settings: dict[str, Any] = {}
        self._max_seconds = 0.0

    def load(self, preset: AudioPreset, device: str, cancelled: CancelCheck | None = None) -> None:
        """Load YuE2 for *preset* on *device*.

        The model is streamed in and handed to the pipeline. With the ``weight_swap`` setting (on by default on CUDA, off on
        MPS) it is pinned in CPU memory and wrapped in a :class:`WeightSwap`; without it, it is placed on *device*. The process memory fraction the kit caps is
        set back to 1.0 on CUDA either way.

        Args:
            preset: The ``yue2`` preset: its model and VAE files, its limits and its settings for *device*.
            device: ``cuda`` or ``mps``.
            cancelled: Checked while the pipeline is built, before each tensor is read, while the model is built from them and
                between the steps that place it; ``True`` stops the load.

        Raises:
            AudioCancelled: If *cancelled* returned ``True``; everything loaded so far is freed.
        """
        settings = preset.settings_for(device)
        weight_swap = bool(settings.get("weight_swap", device == "cuda"))
        model_files, vae_files = preset.files
        self._device = device
        if weight_swap and device == "cuda":
            _stop_pin_rounding()
        try:
            raise_if_cancelled(cancelled)
            with stop_while_building(cancelled):
                pipe = _build_pipeline(str(snapshot_dir(model_files)), str(snapshot_dir(vae_files)), device, settings)
            self._pipe = pipe
            raise_if_cancelled(cancelled)
            if device == "cuda":
                _lift_memory_fraction()
            self._swap = _install_model(pipe, device, weight_swap=weight_swap, cancelled=cancelled)
            self._settings = settings
            self._max_seconds = float(preset.limits.max_seconds)
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Make one song or instrumental for *request*.

        Args:
            request: A ``music`` take: the style in ``prompt`` and the lyrics, which decide song or instrumental.
            on_progress: Called as ``(phase, step, total, 1, 1)`` in the phases Plan, Song tokens, Synthesis and Decode.
            cancelled: Called between steps; ``True`` stops the take.

        Returns:
            A stereo 48 kHz clip. A take that reached the length limit ends in a fade-out and is marked ``cut``.

        Raises:
            AudioCancelled: If *cancelled* returned ``True``.
        """
        settings = self._settings
        lyrics, kind, _ = music_mode(request.lyrics, settings["instrumental_sections"])
        cot = "melody" if kind == INSTRUMENTAL else "full"
        if self._swap is not None:
            self._swap.restore_resident()
        stop = _CancelProbe(cancelled)
        try:
            return self._take(request.prompt, lyrics, cot, request.seed, on_progress, stop)
        except InterruptedError as error:
            if stop.requested:
                raise AudioCancelled(str(error)) from error
            raise

    def close(self) -> None:
        """Free the model and the pinned weights and lift the process memory fraction the kit capped.

        The runner returns the cached accelerator memory and the torch flags the kit changed. Safe to call twice, after a
        failed load and after a failed take.
        """
        pipe, swap, device = self._pipe, self._swap, self._device
        self._pipe = self._swap = None
        self._device = ""
        self._settings = {}
        try:
            if pipe is not None:
                pipe.close()
        finally:
            if swap is not None:
                swap.release()
            if device == "cuda":
                _lift_memory_fraction()

    def _take(self, style: str, lyrics: str, cot: str, seed: int, on_progress: ProgressCallback, stop: _CancelProbe) -> AudioClip:
        pipe, swap, settings = self._pipe, self._swap, self._settings
        if swap is not None:
            swap.phase("ar")
        on_progress("Plan", 0, None, 1, 1)
        plan = pipe.plan(
            style,
            lyrics,
            cot=cot,
            seed=seed,
            cfg_scale=settings["cfg"],
            abc_sampling={"max_tokens": settings["abc_max_tokens"]},
            cancelled=stop,
            on_token=_count_tokens("Plan", on_progress),
        )
        on_progress("Plan", len(plan.abc_ids), None, 1, 1)
        on_progress("Song tokens", 0, None, 1, 1)
        cap = {"max_tokens": int(round(_TOKENS_PER_SECOND * self._max_seconds))}
        semantic = pipe.generate_semantic(plan, sampling=cap, cancelled=stop, on_token=_count_tokens("Song tokens", on_progress))
        on_progress("Song tokens", len(semantic.tokens), None, 1, 1)
        latents = _synthesize(pipe, swap, semantic, on_progress, stop)
        stop.raise_if_requested()
        on_progress(DECODE_PHASE, 0, 1, 1, 1)
        if swap is not None:
            swap.phase("none")
        audio = pipe.decode(latents)
        on_progress(DECODE_PHASE, 1, 1, 1, 1)
        samples = np.ascontiguousarray(audio.T, dtype=np.float32)
        if semantic.truncated:
            return AudioClip(fade_out(samples, SAMPLE_RATE, float(settings["fade_seconds"])), SAMPLE_RATE, cut=True)
        return AudioClip(samples, SAMPLE_RATE)


def _count_tokens(label: str, on_progress: ProgressCallback) -> Callable[[str, int], None]:
    """Return a kit ``on_token`` callback that reports the token count under *label*, every few tokens."""
    count = 0

    def on_token(_phase: str, _token: int) -> None:
        nonlocal count
        count += 1
        if count == 1 or count % _TOKEN_STRIDE == 0:
            on_progress(label, count, None, 1, 1)

    return on_token


def _synthesize(pipe: Any, swap: WeightSwap | None, semantic: Any, on_progress: ProgressCallback, stop: _CancelProbe) -> Any:
    """Run the kit's ``synthesize`` loop chunk by chunk, reporting every ODE step; return the latents.

    With a *swap*, the prefill and the ODE solve each run in their own swap phase.
    """
    import torch

    from yue2.nar import CachedNAR, song_chunks

    config = pipe.generation_config
    chunks = song_chunks(semantic.plan.prefix, semantic.tokens, semantic.plan.request.seed, config.context)
    steps = config.ode_steps
    total = steps * len(chunks)
    on_progress("Synthesis", 0, total, 1, 1)
    outputs = []
    with torch.inference_mode():
        model = pipe._load_model(for_nar=True)
        for index, chunk in enumerate(chunks):
            stop.raise_if_requested()
            if swap is not None:
                swap.phase("prefill")
            engine = CachedNAR(model, chunk, "sdpa", None)
            try:
                if swap is not None:
                    swap.phase("ode")
                done_before = index * steps
                outputs.append(engine.solve(steps, stop, on_progress=lambda done, _total, base=done_before: on_progress("Synthesis", base + done, total, 1, 1)))
            finally:
                engine.close()
            del engine
            if swap is not None:
                torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0).detach().float().cpu().numpy()


def _build_pipeline(model_dir: str, vae_dir: str, device: str, settings: dict[str, Any]) -> Any:
    """Construct the kit's pipeline with the preset's settings."""
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    with size_checked_weights():  # the kit would sha256-hash about 8 GB of weights on every build
        return YuE2Pipeline.from_pretrained(
            model_dir,
            vae=vae_dir,
            device=device,
            progress=False,
            memory_budget_gib=settings["memory_budget_gib"],
            vae_core_frames=settings["vae_core_frames"],
            generation_config=GenerationConfig(ode_steps=settings["ode_steps"]),
        )


def _install_model(pipe: Any, device: str, *, weight_swap: bool, cancelled: CancelCheck | None = None) -> WeightSwap | None:
    """Stream the model in and make the pipeline use it; return the :class:`WeightSwap` that owns its placement, if any.

    With *weight_swap* the tensors are pinned in CPU memory as they are read, so the swap has nothing left to copy.
    Without it the model moves to *device* parameter by parameter, each CPU copy freed as it goes.
    """
    model = load_model(pipe.model_dir, pin=weight_swap, cancelled=cancelled)
    pipe._model = model
    raise_if_cancelled(cancelled)
    if weight_swap:
        swap = WeightSwap(model, device)
        raise_if_cancelled(cancelled)  # pinning the weights takes seconds
        # The stock loader would move the whole model to the GPU; the swap owns placement.
        pipe._load_model = lambda for_nar=False: model
        return swap
    model.to(device)
    raise_if_cancelled(cancelled)
    # The kit's decode moves the model to the CPU to make room for the VAE; bring it back for the next take.
    pipe._load_model = lambda for_nar=False: model.to(device)
    return None


def _stop_pin_rounding() -> None:
    """Keep PyTorch from rounding the pinned weights up to powers of two, which would add about a third to their size.

    PyTorch reads the allocator options when CUDA first starts, so this only helps when nothing in the process has used
    CUDA yet; the other CUDA backends make the same call at their start.
    """
    from zvisiongenerator.backends.cuda.memory import configure_allocator

    configure_allocator()


def _lift_memory_fraction() -> None:
    """Set the process memory fraction back to 1.0; the kit's constructor caps it at its budget."""
    import torch

    torch.cuda.set_per_process_memory_fraction(1.0)
