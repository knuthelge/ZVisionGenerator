"""Silence the known, harmless console noise of the libraries the audio models use, and nothing else."""

from __future__ import annotations

import importlib
import io
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout

__all__ = ["MUTED_ERROR_TEXT", "mute_library_warnings", "quiet_flash_attn_probe", "silence_autotune_report"]

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


def mute_library_warnings() -> None:
    """Ignore the known warnings of :data:`_MUTED_WARNINGS`; every other warning is untouched."""
    for category, pattern in _MUTED_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern, category=category)


@contextmanager
def quiet_flash_attn_probe() -> Iterator[None]:
    """Hold back what the block prints and show it again, except the ``flash_attn`` probe lines; for an import that probes."""
    held = io.StringIO()
    try:
        with redirect_stdout(held):
            yield
    finally:
        for line in held.getvalue().splitlines():
            if line.strip() not in _FLASH_ATTN_PROBE:
                print(line)


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
