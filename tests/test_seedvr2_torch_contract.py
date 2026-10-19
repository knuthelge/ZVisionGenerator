"""Pin the vendored SeedVR2 PyTorch entry points that the CUDA SeedVR2 upscaler calls.

Introspection only: no model is built and no weights are loaded. A failure here after ``make update-seedvr2`` means
upstream changed a name, a signature or a bundled file the upscaler relies on.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

pytest.importorskip("torchvision")
pytest.importorskip("seedvr2_torch")

from seedvr2_torch.src.core import generation_phases, generation_utils  # noqa: E402
from seedvr2_torch.src.utils.constants import get_script_directory  # noqa: E402
from seedvr2_torch.src.utils.debug import Debug  # noqa: E402


def _parameters(function: object) -> set[str]:
    return set(inspect.signature(function).parameters)


def test_imports_without_comfyui() -> None:
    assert not [name for name in sys.modules if name.split(".")[0] in {"comfy", "comfy_api", "folder_paths", "comfyui_version"}]


def test_bundled_files_resolve_from_the_script_directory() -> None:
    root = Path(get_script_directory())
    for name in ("pos_emb.pt", "neg_emb.pt", "configs_3b/main.yaml", "configs_7b/main.yaml"):
        assert (root / name).is_file(), name


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (generation_utils.setup_generation_context, {"dit_device", "vae_device", "dit_offload_device", "vae_offload_device", "tensor_offload_device", "debug"}),
        (
            generation_utils.prepare_runner,
            {
                "dit_model",
                "vae_model",
                "model_dir",
                "debug",
                "ctx",
                "dit_cache",
                "vae_cache",
                "block_swap_config",
                "encode_tiled",
                "encode_tile_size",
                "encode_tile_overlap",
                "decode_tiled",
                "decode_tile_size",
                "decode_tile_overlap",
                "attention_mode",
            },
        ),
        (generation_utils.load_text_embeddings, {"script_directory", "device", "dtype", "debug"}),
        (generation_utils.compute_generation_info, {"ctx", "images", "resolution", "max_resolution", "batch_size", "seed", "debug"}),
        (generation_phases.encode_all_batches, {"runner", "ctx", "images", "debug", "batch_size", "seed", "resolution", "max_resolution", "input_noise_scale", "color_correction"}),
        (generation_phases.upscale_all_batches, {"runner", "ctx", "debug", "seed", "latent_noise_scale", "cache_model"}),
        (generation_phases.decode_all_batches, {"runner", "ctx", "debug", "cache_model"}),
        (generation_phases.postprocess_all_batches, {"ctx", "debug", "color_correction", "batch_size"}),
    ],
)
def test_entry_point_accepts_the_keywords_the_upscaler_passes(function: object, expected: set[str]) -> None:
    assert expected <= _parameters(function)


def test_debug_takes_an_enabled_flag() -> None:
    assert "enabled" in _parameters(Debug)
