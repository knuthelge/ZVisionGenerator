"""YuE2 audio backend: songs and instrumentals from the vendored kit, with a CUDA weight swap.

A take runs the kit stage by stage: ``plan`` (the ABC score) and ``generate_semantic`` (the song tokens) are the
autoregressive phases, ``CachedNAR`` prefills and solves the acoustic latents, and ``decode`` turns them into a stereo
48 kHz waveform. With ``weight_swap`` (CUDA) the model is held in pinned CPU memory by :class:`WeightSwap`, which puts
only the running phase's weights on the GPU. Without it (MPS) the kit's own ``synthesize`` runs the acoustic part.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from zvisiongenerator.backends.audio.yue2_swap import WeightSwap
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioClip
from zvisiongenerator.utils.audio_fade import fade_out
from zvisiongenerator.utils.hf_snapshot import snapshot_download, snapshot_with
from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.music_mode import INSTRUMENTAL, music_mode
from zvisiongenerator.utils.torch_flags import TorchFlags, capture_torch_flags, restore_torch_flags

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioFiles, AudioPlatform, AudioPreset

__all__ = ["SAMPLE_RATE", "YuE2Backend"]

SAMPLE_RATE: Final = 48000
"""The sample rate of the VAE's waveform."""

_TOKENS_PER_SECOND: Final = 25
"""Semantic tokens (codec frames) per second of audio."""

_TOKEN_STRIDE: Final = 10
"""Report every this many tokens, so a long song does not flood the progress callback."""


@dataclass(frozen=True)
class _PipeInputs:
    """What one take hands the pipeline's ``plan``.

    Attributes:
        style: The style tags.
        lyrics: The lyrics, or the section list of an instrumental.
        cot: The planning mode (``full`` or ``melody``).
    """

    style: str
    lyrics: str
    cot: str


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
        self._flags: TorchFlags | None = None

    def load(self, preset: AudioPreset, platform_entry: AudioPlatform, device: str) -> None:
        """Load YuE2 for *preset* on *device*.

        With the ``weight_swap`` setting the model is loaded on the CPU, wrapped in a :class:`WeightSwap` and handed to
        the pipeline; the process memory fraction the kit caps is set back to 1.0 on CUDA either way.

        Args:
            preset: The ``yue2`` preset: its limits and shared settings.
            platform_entry: The preset's entry for *device*: the model and VAE files and the platform's settings.
            device: ``cuda`` or ``mps``.

        Raises:
            RuntimeError: If a model is already loaded.
            ValueError: If the preset has no ``max_seconds`` limit, the entry does not list the model and the VAE, or
                ``weight_swap`` is set off CUDA.
        """
        if self._pipe is not None:
            raise RuntimeError("The YuE2 backend already holds a model; close it before loading another.")
        settings = {**preset.settings, **platform_entry.settings}
        weight_swap = bool(settings["weight_swap"])
        if weight_swap and device != "cuda":
            raise ValueError(f"YuE2 weight_swap needs CUDA; the device is '{device}'.")
        if preset.limits.max_seconds is None:
            raise ValueError("The YuE2 preset must set limits.max_seconds.")
        if len(platform_entry.files) != 2:
            raise ValueError("The YuE2 platform entry must list two repositories: the model, then the VAE.")
        model_files, vae_files = platform_entry.files
        self._device = device
        self._flags = capture_torch_flags()  # the kit sets global numerics flags; close() puts them back
        try:
            pipe = _build_pipeline(_snapshot_dir(model_files), _snapshot_dir(vae_files), device, settings)
            self._pipe = pipe
            if device == "cuda":
                _lift_memory_fraction()
            if weight_swap:
                self._swap = _install_swap(pipe, device)
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
            RuntimeError: If no model is loaded.
            ValueError: If the request is not a music take.
            AudioCancelled: If *cancelled* returned ``True``.
        """
        if self._pipe is None:
            raise RuntimeError("The YuE2 backend has no model loaded.")
        if request.kind != "music":
            raise ValueError(f"The YuE2 backend makes music, not '{request.kind}'.")
        settings = self._settings
        cot, lyrics, kind, _ = music_mode(request.lyrics, settings["instrumental_sections"], cot_song=settings["cot_song"], cot_instrumental=settings["cot_instrumental"])
        inputs = _instrumental_inputs(request.prompt, lyrics, cot) if kind == INSTRUMENTAL else _song_inputs(request.prompt, lyrics, cot)
        if self._swap is not None:
            self._swap.restore_resident()
        stop = _CancelProbe(cancelled)
        try:
            return self._take(inputs, request.seed, on_progress, stop)
        except InterruptedError as error:
            if stop.requested:
                raise AudioCancelled(str(error)) from error
            raise

    def close(self) -> None:
        """Free the model, the pinned weights and the memory limit, and restore the torch flags the kit changed.

        Safe to call twice, after a failed load and after a failed take.
        """
        pipe, swap, device, flags = self._pipe, self._swap, self._device, self._flags
        self._flags = None
        self._pipe = self._swap = None
        self._device = ""
        self._settings = {}
        try:
            if pipe is not None:
                pipe.close()
        finally:
            if swap is not None:
                swap.release()
            del pipe, swap
            try:
                if device == "cuda":
                    _free_cuda()
            finally:
                if flags is not None:
                    restore_torch_flags(flags)

    def _take(self, inputs: _PipeInputs, seed: int, on_progress: ProgressCallback, stop: _CancelProbe) -> AudioClip:
        pipe, swap, settings = self._pipe, self._swap, self._settings
        if swap is not None:
            swap.phase("ar")
        on_progress("Plan", 0, None, 1, 1)
        plan = pipe.plan(
            inputs.style,
            inputs.lyrics,
            cot=inputs.cot,
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
        if swap is None:
            on_progress("Synthesis", 0, None, 1, 1)
            latents = pipe.synthesize(semantic, cancelled=stop)
        else:
            latents = _synthesize_swapped(pipe, swap, semantic, on_progress, stop)
        stop.raise_if_requested()
        on_progress("Decode", 0, 1, 1, 1)
        if swap is not None:
            swap.phase("none")
        audio = pipe.decode(latents)
        on_progress("Decode", 1, 1, 1, 1)
        samples = np.ascontiguousarray(audio.T, dtype=np.float32)
        if semantic.truncated:
            return AudioClip(fade_out(samples, SAMPLE_RATE, float(settings["fade_seconds"])), SAMPLE_RATE, cut=True)
        return AudioClip(samples, SAMPLE_RATE)


def _song_inputs(style: str, lyrics: str, cot: str) -> _PipeInputs:
    """Return the pipeline inputs of a song: the style and the lyrics as given."""
    return _PipeInputs(style, lyrics, cot)


def _instrumental_inputs(style: str, section_list: str, cot: str) -> _PipeInputs:
    """Return the pipeline inputs of an instrumental: the style and the section list that sets its length."""
    return _PipeInputs(style, section_list, cot)


def _count_tokens(label: str, on_progress: ProgressCallback) -> Callable[[str, int], None]:
    """Return a kit ``on_token`` callback that reports the token count under *label*, every few tokens."""
    count = 0

    def on_token(_phase: str, _token: int) -> None:
        nonlocal count
        count += 1
        if count == 1 or count % _TOKEN_STRIDE == 0:
            on_progress(label, count, None, 1, 1)

    return on_token


def _synthesize_swapped(pipe: Any, swap: WeightSwap, semantic: Any, on_progress: ProgressCallback, stop: _CancelProbe) -> Any:
    """Run the kit's ``synthesize`` loop with the prefill and the ODE solve in their own swap phases; return the latents."""
    import torch

    from yue2.nar import CachedNAR, song_chunks

    config = pipe.generation_config
    chunks = song_chunks(semantic.plan.prefix, semantic.tokens, semantic.plan.request.seed, config.context)
    steps = config.ode_steps
    total = steps * len(chunks)
    on_progress("Synthesis", 0, total, 1, 1)
    outputs = []
    with torch.inference_mode():
        for index, chunk in enumerate(chunks):
            stop.raise_if_requested()
            swap.phase("prefill")
            engine = CachedNAR(pipe._model, chunk, "sdpa", None)
            try:
                swap.phase("ode")
                done_before = index * steps
                outputs.append(engine.solve(steps, stop, on_progress=lambda done, _total, base=done_before: on_progress("Synthesis", base + done, total, 1, 1)))
            finally:
                engine.close()
            del engine
            torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0).detach().float().cpu().numpy()


def _snapshot_dir(files: AudioFiles) -> str:
    """Return the folder of the pinned snapshot that holds *files* (the download step has already fetched it)."""
    return str(snapshot_with(HubFiles(files.repo_id, files.revision, files.patterns), snapshot_download))


def _build_pipeline(model_dir: str, vae_dir: str, device: str, settings: dict[str, Any]) -> Any:
    """Construct the kit's pipeline with the preset's settings."""
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    return YuE2Pipeline.from_pretrained(
        model_dir,
        vae=vae_dir,
        device=device,
        progress=False,
        memory_budget_gib=settings["memory_budget_gib"],
        vae_core_frames=settings["vae_core_frames"],
        generation_config=GenerationConfig(ode_steps=settings["ode_steps"]),
    )


def _install_swap(pipe: Any, device: str) -> WeightSwap:
    """Load the model on the CPU, wrap it in a :class:`WeightSwap` and make the pipeline use it as its model."""
    model = _load_cpu_model(pipe.model_dir)
    swap = WeightSwap(model, device)
    pipe._model = model
    # The stock loader would move the whole model to the GPU; the swap owns placement.
    pipe._load_model = lambda for_nar=False: model
    return swap


def _load_cpu_model(model_dir: Any) -> Any:
    """Load ``YuE2ForCausalLM`` from *model_dir* on the CPU in bf16, as the kit's own loader does before it moves it."""
    import torch

    from yue2.modeling_yue2 import YuE2ForCausalLM

    return YuE2ForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()


def _lift_memory_fraction() -> None:
    """Set the process memory fraction back to 1.0; the kit's constructor caps it at its budget."""
    import torch

    torch.cuda.set_per_process_memory_fraction(1.0)


def _free_cuda() -> None:
    """Collect garbage, return cached blocks to the driver and lift the process memory fraction."""
    import torch

    gc.collect()
    try:
        torch.cuda.empty_cache()
    finally:
        _lift_memory_fraction()
