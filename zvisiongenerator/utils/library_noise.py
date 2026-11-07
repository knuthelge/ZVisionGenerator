"""Silence the known, harmless console noise of the libraries the audio models use, and nothing else."""

from __future__ import annotations

import importlib
import io
import logging
import os
import shutil
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace

__all__ = ["MUTED_ERROR_TEXT", "mute_library_warnings", "quiet_flash_attn_probe", "quiet_qwen_graph_capture", "quiet_qwen_import", "silence_autotune_report"]

MUTED_ERROR_TEXT = ("`use_return_dict` is deprecated",)
"""Text of the log lines the Hugging Face library writes to the error stream on every YuE2 take, with nothing for the user to act on."""

_MUTED_WARNINGS = ((FutureWarning, r"`torch\.nn\.utils\.weight_norm` is deprecated"),)
"""The (category, message pattern) of the warnings the audio models' own code raises and users cannot act on."""

_FLASH_ATTN_PROBE = (
    "No module named 'flash_attn'",
    "flash_attn not installed, disabling Flash Attention",
    "flash_attn varlen/bert_padding not available, disabling varlen attention",
)
"""What Stable Audio 3 prints when it probes for the optional ``flash_attn`` package, which is not needed."""

_QWEN_IMPORT_STDOUT = (
    "********",
    "Warning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.",
)
"""What ``qwen_tts`` prints to standard output when it probes for the optional ``flash_attn`` package (between blank lines)."""

_QWEN_GRAPH_STDOUT = (
    "Warming up predictor (",
    "Warming up talker graph (",
    "Capturing CUDA graph for predictor...",
    "Capturing CUDA graph for talker decode...",
    "CUDA graph captured!",
    "Talker CUDA graph captured!",
)
"""What ``faster_qwen3_tts`` prints to standard output while it warms up and captures its CUDA graphs (the start of each line)."""


def mute_library_warnings() -> None:
    """Ignore the known warnings of :data:`_MUTED_WARNINGS`; every other warning is untouched."""
    for category, pattern in _MUTED_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern, category=category)


def _only_errors(record: logging.LogRecord) -> bool:
    """Keep a log record only when it is an error."""
    return record.levelno >= logging.ERROR


def _no_probe_popen(*_args: object, **_kwargs: object) -> SimpleNamespace:
    """Stand in for ``os.popen`` without starting a shell: return a pipe of a program that printed nothing."""
    return SimpleNamespace(readlines=lambda: [], close=lambda: None)


@contextmanager
def _without_sox_probe() -> Iterator[None]:
    """Make ``os.popen`` start no shell for the block when the ``sox`` program is not installed, so no shell says "sox: not found".

    ``sox`` runs ``os.popen("sox -h")`` as it is imported and only checks whether that printed anything. The original is put back even when the block raises.
    """
    if shutil.which("sox") is not None:
        yield
        return
    popen = os.popen
    os.popen = _no_probe_popen  # type: ignore[assignment]
    try:
        yield
    finally:
        os.popen = popen


@contextmanager
def _quiet_stdout(is_noise: Callable[[str], bool]) -> Iterator[None]:
    """Hold back what the block prints and show it again, except the lines for which *is_noise* is true."""
    held = io.StringIO()
    try:
        with redirect_stdout(held):
            yield
    finally:
        for line in held.getvalue().splitlines():
            if not is_noise(line):
                print(line)


@contextmanager
def quiet_flash_attn_probe() -> Iterator[None]:
    """Hold back what the block prints and show it again, except the ``flash_attn`` probe lines; for an import that probes."""
    with _quiet_stdout(lambda line: line.strip() in _FLASH_ATTN_PROBE):
        yield


@contextmanager
def quiet_qwen_import() -> Iterator[None]:
    """Hold back the known, harmless output of importing ``qwen_tts`` and show everything else.

    Importing ``qwen_tts`` pulls in ``sox``, which probes for its command-line program and logs a warning when it is missing, and the
    library prints a banner to standard output about the optional ``flash_attn``. For the block, a missing ``sox`` program is not
    probed through a shell (see :func:`_without_sox_probe`), the ``sox`` logger drops everything below ERROR, and standard output
    is held and written again except the banner and its blank padding lines. The error stream is not touched.
    """
    sox_logger = logging.getLogger("sox")
    sox_logger.addFilter(_only_errors)  # a filter, not a level: ``sox`` sets its own level to WARNING when it is imported
    try:
        with _without_sox_probe(), _quiet_stdout(lambda line: not line.strip() or line.strip() in _QWEN_IMPORT_STDOUT):
            yield
    finally:
        sox_logger.removeFilter(_only_errors)


@contextmanager
def quiet_qwen_graph_capture() -> Iterator[None]:
    """Hold back the progress lines ``faster_qwen3_tts`` prints while it warms up and captures its CUDA graphs and show everything else."""
    with _quiet_stdout(lambda line: line.strip().startswith(_QWEN_GRAPH_STDOUT)):
        yield


def silence_autotune_report() -> None:
    """Stop PyTorch's inductor from writing its kernel autotune tables to the error stream; do nothing without the inductor."""
    try:
        config = importlib.import_module("torch._inductor.config")
    except ImportError:
        return
    # The report is on by default and written by the compile step on the first take; a setting this PyTorch lacks stays unset.
    for name, value in (("max_autotune_report_choices_stats", False), ("autotune_num_choices_displayed", 0)):
        if hasattr(config, name):
            setattr(config, name, value)
