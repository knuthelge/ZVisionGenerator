"""The memory bench's media: each is a module with ``prepare``, ``case_argv``, ``ensure_sources``, ``run_cli``, ``install_probes``, ``estimate`` and ``OUTPUT_SUFFIXES``."""

from __future__ import annotations

from types import ModuleType

from bench.memory import audio, image, video

MEDIA_MODULES: dict[str, ModuleType] = {"image": image, "video": video, "audio": audio}


def medium(name: str) -> ModuleType:
    """Return the module of *name* (``image``, ``video`` or ``audio``)."""
    return MEDIA_MODULES[name]
