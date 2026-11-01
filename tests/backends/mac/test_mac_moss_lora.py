"""Tests for the MOSS-Image LoRA mapping: every layout the trainers write applies, unmerged, through mflux's loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")

_DIM = 64
_FFN = 128
_RANK = 4


@pytest.fixture
def mx():
    pytest.importorskip("mflux")
    return pytest.importorskip("mlx.core")


@pytest.fixture
def transformer(mx):
    from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer

    return MOSSImageTransformer(dim=_DIM, ffn_dim=_FFN, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=8)


def _apply(mx, transformer, tmp_path: Path, tensors: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """Write a LoRA file with random *tensors* of the given shapes, apply it, and return the file's arrays."""
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader

    from zvisiongenerator.backends.mac.moss_lora import moss_lora_mapping

    rng = np.random.default_rng(0)
    arrays = {name: rng.standard_normal(shape).astype(np.float32) for name, shape in tensors.items()}
    path = tmp_path / "lora.safetensors"
    mx.save_safetensors(str(path), {name: mx.array(value) for name, value in arrays.items()})
    LoRALoader.load_and_apply_lora(lora_mapping=moss_lora_mapping(), transformer=transformer, lora_paths=[str(path)], lora_scales=[1.0], bake_lora=False)
    return arrays


def _is_lora(module) -> bool:
    return type(module).__name__ in {"LoRALinear", "FusedLoRALinear", "LoKrLinear"}


@pytest.mark.parametrize(
    ("prefix", "down", "up"),
    [
        ("diffusion_model.layers.{n}.proj.q", "lora_down.weight", "lora_up.weight"),
        ("model.diffusion_model.layers.{n}.proj.q", "lora_A.weight", "lora_B.weight"),
        ("base_model.model.layers.{n}.proj.q", "lora_A.default.weight", "lora_B.default.weight"),
        ("lora_unet_layers_{n}_proj_q", "lora_down.weight", "lora_up.weight"),
    ],
)
def test_each_naming_of_a_layer_projection_applies_unmerged(mx, transformer, tmp_path, prefix, down, up):
    _apply(mx, transformer, tmp_path, {f"{prefix.format(n=1)}.{down}": (_RANK, _DIM), f"{prefix.format(n=1)}.{up}": (_DIM, _RANK)})

    assert _is_lora(transformer.layers[1].proj["q"])
    assert not _is_lora(transformer.layers[0].proj["q"])


def test_a_fused_qkv_lora_splits_its_up_rows_over_q_k_and_v(mx, transformer, tmp_path):
    arrays = _apply(
        mx,
        transformer,
        tmp_path,
        {"diffusion_model.layers.0.proj.qkv.lora_A.weight": (_RANK, _DIM), "diffusion_model.layers.0.proj.qkv.lora_B.weight": (3 * _DIM, _RANK)},
    )

    up = arrays["diffusion_model.layers.0.proj.qkv.lora_B.weight"]
    for index, name in enumerate(("q", "k", "v")):
        layer = transformer.layers[0].proj[name]
        assert _is_lora(layer)
        assert np.allclose(np.array(layer.lora_B), up[index * _DIM : (index + 1) * _DIM].T)
        assert np.allclose(np.array(layer.lora_A), arrays["diffusion_model.layers.0.proj.qkv.lora_A.weight"].T)


def test_text_embedder_and_global_layers_apply(mx, transformer, tmp_path):
    tensors = {}
    for path, (rows, cols) in {
        "text_embedder.blocks.1.ffn_gate": (_FFN, _DIM),
        "text_embedder.proj": (_DIM, 32),
        "time_embedding.2": (_DIM, _DIM),
        "video_embedder": (_DIM, 8),
    }.items():
        tensors[f"diffusion_model.{path}.lora_A.weight"] = (_RANK, cols)
        tensors[f"diffusion_model.{path}.lora_B.weight"] = (rows, _RANK)

    _apply(mx, transformer, tmp_path, tensors)

    assert _is_lora(transformer.text_embedder.blocks[1].ffn_gate)
    assert _is_lora(transformer.text_embedder.proj)
    assert _is_lora(transformer.time_embedding[2])
    assert _is_lora(transformer.video_embedder)


def test_a_lokr_file_applies_through_the_same_targets(mx, transformer, tmp_path):
    _apply(mx, transformer, tmp_path, {"diffusion_model.layers.0.proj.o.lokr_w1": (8, 8), "diffusion_model.layers.0.proj.o.lokr_w2": (8, 8)})

    assert _is_lora(transformer.layers[0].proj["o"])


def test_a_fused_qkv_lokr_adds_each_projections_rows_of_the_product_once_per_layer(mx, transformer):
    from zvisiongenerator.backends.mac.moss_lora import apply_fused_lokr, prepare_lora

    rng = np.random.default_rng(1)
    w1, w2 = rng.standard_normal((3, 4)).astype(np.float32), rng.standard_normal((_DIM, 16)).astype(np.float32)
    weights = {
        "diffusion_model.layers.1.proj.qkv.lokr_w1": mx.array(w1),
        "diffusion_model.layers.1.proj.qkv.lokr_w2": mx.array(w2),
        "diffusion_model.layers.1.proj.q.alpha": mx.array(4.0),
        "diffusion_model.layers.1.proj.o.lora_down.weight": mx.zeros((_RANK, _DIM)),
        "diffusion_model.layers.1.proj.o.lora_up.weight": mx.zeros((_DIM, _RANK)),
        "diffusion_model.layers.1.proj.o.alpha": mx.array(4.0),
    }

    prepared = prepare_lora(weights, "test.safetensors")
    x = mx.array(rng.standard_normal((1, 2, _DIM)).astype(np.float32))
    with mx.stream(mx.cpu):  # exact float32; the GPU's matmuls round these random products more loosely
        before = {name: np.array(transformer.layers[1].proj[name](x)) for name in ("q", "k", "v")}
        apply_fused_lokr(transformer, prepared.fused_lokr, 0.5)
        after = {name: np.array(transformer.layers[1].proj[name](x)) for name in ("q", "k", "v")}

    assert sorted(prepared.rest) == ["diffusion_model.layers.1.proj.o.alpha", "diffusion_model.layers.1.proj.o.lora_down.weight", "diffusion_model.layers.1.proj.o.lora_up.weight"]
    update = np.array(x).reshape(-1, _DIM) @ np.kron(w1, w2).T * 0.5
    for index, name in enumerate(("q", "k", "v")):
        assert np.allclose(after[name].reshape(-1, _DIM) - before[name].reshape(-1, _DIM), update[:, index * _DIM : (index + 1) * _DIM], atol=1e-4)


def test_the_fused_product_is_computed_once_for_q_k_and_v(mx, transformer, monkeypatch):
    from zvisiongenerator.backends.mac import moss_lora

    calls = []
    real_matmul = mx.matmul
    monkeypatch.setattr(mx, "matmul", lambda a, b: calls.append(1) or real_matmul(a, b))
    fused = {0: {"lokr_w1": mx.ones((3, 4)), "lokr_w2": mx.ones((_DIM, 16))}}
    moss_lora.apply_fused_lokr(transformer, fused, 1.0)
    x = mx.ones((1, 2, _DIM))

    for name in ("q", "k", "v"):
        transformer.layers[0].proj[name](x)

    assert len(calls) == 2  # w1 @ X, then @ w2ᵀ: once for the three projections


def test_unsupported_tensors_are_skipped_with_a_warning(mx):
    from zvisiongenerator.backends.mac.moss_lora import prepare_lora

    weights = {
        "diffusion_model.layers.0.proj.q.hada_w1_a": mx.zeros((2, 2)),
        "diffusion_model.layers.0.proj.k.diff": mx.zeros((2, 2)),
        "diffusion_model.layers.0.proj.v.lora_A.weight": mx.zeros((2, 2)),
        "diffusion_model.layers.0.proj.v.lora_B.weight": mx.zeros((2, 2)),
    }

    with pytest.warns(UserWarning, match="skipped 2 LoHa, diff or DoRA tensors"):
        prepared = prepare_lora(weights, "mixed.safetensors")

    assert sorted(prepared.rest) == ["diffusion_model.layers.0.proj.v.lora_A.weight", "diffusion_model.layers.0.proj.v.lora_B.weight"]


def test_only_what_mflux_cannot_apply_needs_preparing():
    from zvisiongenerator.backends.mac.moss_lora import needs_preparing

    assert needs_preparing(["diffusion_model.layers.0.proj.qkv.lokr_w1", "diffusion_model.layers.0.proj.qkv.lokr_w2"])
    assert needs_preparing(["diffusion_model.layers.0.proj.q.alpha", "diffusion_model.layers.0.proj.o.lokr_w1"])
    assert needs_preparing(["diffusion_model.layers.0.proj.q.hada_w1_a"])
    assert not needs_preparing(["diffusion_model.layers.0.proj.q.lora_down.weight", "diffusion_model.layers.0.proj.q.lora_up.weight", "diffusion_model.layers.0.proj.q.alpha"])
    assert not needs_preparing(["diffusion_model.layers.0.proj.qkv.lora_A.weight", "diffusion_model.layers.0.proj.qkv.lora_B.weight"])
