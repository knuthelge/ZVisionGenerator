"""Tests for reading MiniMax H3 fine-tunes for conversion: layouts, the native mapping, un-pruning and refusals, on a tiny config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests._mac_helpers import _TINY, _diffusers_state, _native

from zvisiongenerator.converters import h3_checkpoint
from zvisiongenerator.converters.h3_checkpoint import diffusers_tensors, inspect_h3


@pytest.fixture(autouse=True)
def _tiny_config(monkeypatch):
    monkeypatch.setattr(h3_checkpoint, "STOCK_CONFIG", _TINY)


def _equal(found: dict[str, torch.Tensor], state: dict[str, torch.Tensor]) -> bool:
    return found.keys() == state.keys() and all(torch.equal(found[name].float(), state[name].float()) for name in state)


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model.", "diffusion_model."])
def test_a_comfyui_file_maps_to_diffusers_names_with_contiguous_qkv(tmp_path, prefix):
    state = _diffusers_state(torch.Generator().manual_seed(0))
    path = tmp_path / "ft.safetensors"
    save_file({prefix + name: tensor for name, tensor in _native(state, interleave=False).items()}, str(path))

    checkpoint = inspect_h3(path)

    assert (checkpoint.layout, checkpoint.curve_rank) == ("comfyui", None)
    assert _equal(dict(diffusers_tensors(checkpoint)), state)


def test_a_native_shard_set_is_read_whole_from_any_shard_with_per_head_qkv(tmp_path):
    state = _diffusers_state(torch.Generator().manual_seed(1))
    native = _native(state, interleave=True)
    names = sorted(native)
    halves = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2 :]}
    for file, part in halves.items():
        save_file({name: native[name] for name in part}, str(tmp_path / file))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: file for file, part in halves.items() for name in part}}))

    checkpoint = inspect_h3(tmp_path / "model-00002-of-00002.safetensors")

    assert checkpoint.layout == "native"
    assert len(checkpoint.files) == 2
    assert _equal(dict(diffusers_tensors(checkpoint)), state)


def test_a_diffusers_file_passes_through(tmp_path):
    state = _diffusers_state(torch.Generator().manual_seed(2))
    save_file(state, str(tmp_path / "ft.safetensors"))

    checkpoint = inspect_h3(tmp_path / "ft.safetensors")

    assert checkpoint.layout == "diffusers"
    assert _equal(dict(diffusers_tensors(checkpoint)), state)


def _pruned(tmp_path: Path, generator: torch.Generator, rank: int = 3) -> tuple[Path, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Save a pruned tiny checkpoint whose curve is a rank-*rank* projection of the stock embedding; return it, its curve readers and stock's embedder."""
    state = _diffusers_state(generator)
    embedder = {name.removeprefix("time_embedder."): tensor for name, tensor in state.items() if name.startswith("time_embedder.")}
    grid = torch.linspace(0, 1, 65, dtype=torch.float64)
    embedded = torch.nn.functional.silu(h3_checkpoint._time_embedding(grid, embedder))
    basis = torch.linalg.svd(embedded - embedded.mean(0), full_matrices=False).Vh[:rank]
    curve = ((embedded - embedded.mean(0)) @ basis.T).float()
    native = _native(state, interleave=False)
    for name in [name for name in native if name.startswith("time_embedder.")]:
        del native[name]
    readers: dict[str, torch.Tensor] = {}
    for name in [name for name in native if name.endswith("adaln_proj.linear.weight")]:
        native[name] = torch.randn(native[name].shape[0], rank, generator=generator)
        readers[name] = native[name]
    native["adaln_t_table"] = curve
    path = tmp_path / "pruned.safetensors"
    save_file(native, str(path))
    return path, readers, embedder


def test_a_pruned_checkpoint_is_unpruned_to_stock_shapes_with_the_same_output(tmp_path):
    path, readers, embedder = _pruned(tmp_path, torch.Generator().manual_seed(3))
    checkpoint = inspect_h3(path)

    found = dict(diffusers_tensors(checkpoint, time_embedder=embedder))

    assert checkpoint.curve_rank == 3
    assert {name: list(tensor.shape) for name, tensor in found.items()} == h3_checkpoint._expected_shapes(None)
    table = dict(h3_checkpoint.decoded_tensors([path], {"adaln_t_table"}.__contains__))["adaln_t_table"].double()
    native_bias = dict(h3_checkpoint.decoded_tensors([path], {"blocks.1.adaln_proj.linear.bias"}.__contains__))["blocks.1.adaln_proj.linear.bias"].double()
    timesteps = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
    position = timesteps * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    coordinates = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    curve_output = coordinates @ readers["blocks.1.adaln_proj.linear.weight"].double().T + native_bias
    dense_output = (
        torch.nn.functional.silu(h3_checkpoint._time_embedding(timesteps, embedder)) @ found["transformer_blocks.1.adaln_proj.linear.weight"].double().T
        + found["transformer_blocks.1.adaln_proj.linear.bias"].double()
    )
    assert torch.allclose(dense_output, curve_output, rtol=1e-3, atol=1e-3)
    assert torch.equal(found["time_embedder.linear_1.weight"], embedder["linear_1.weight"])


def test_a_pruned_checkpoint_needs_stocks_time_embedder(tmp_path):
    path, _readers, _embedder = _pruned(tmp_path, torch.Generator().manual_seed(4))

    with pytest.raises(ValueError):
        list(diffusers_tensors(inspect_h3(path)))


def test_a_curve_that_is_no_projection_of_the_embedding_is_refused(tmp_path):
    path, _readers, embedder = _pruned(tmp_path, torch.Generator().manual_seed(5))
    other = {name: torch.zeros_like(tensor) for name, tensor in embedder.items()}  # a constant embedding: no curve is its projection

    with pytest.raises(ValueError):
        list(diffusers_tensors(inspect_h3(path), time_embedder=other))


def _save_native(tmp_path: Path, extra: dict[str, torch.Tensor] | None = None, drop: str | None = None, metadata: dict[str, str] | None = None, name: str = "ft.safetensors") -> Path:
    native = _native(_diffusers_state(torch.Generator().manual_seed(6)), interleave=False)
    native.update(extra or {})
    native.pop(drop or "", None)
    save_file(native, str(tmp_path / name), metadata=metadata)
    return tmp_path / name


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extra": {"blocks.0.attn.to_gate_compress.weight": torch.ones(2, 2)}},
        {"drop": "blocks.1.mlp.fc2.weight"},
        {"extra": {"final_layer.video_out.weight": torch.ones(16, 8)}},
        {"metadata": {"config": json.dumps({"transformer": {"pdd_steps": 4}})}},
        {"extra": {"blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(1, 8)}},
    ],
)
def test_checkpoints_the_app_does_not_run_are_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        inspect_h3(_save_native(tmp_path, **kwargs))


def test_a_file_of_another_model_is_refused(tmp_path):
    save_file({"model.diffusion_model.patchify_proj.weight": torch.ones(2, 2)}, str(tmp_path / "ltx.safetensors"))

    with pytest.raises(ValueError):
        inspect_h3(tmp_path / "ltx.safetensors")


def test_a_shard_set_missing_a_shard_is_refused(tmp_path):
    path = _save_native(tmp_path, name="model-00001-of-00002.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": path.name, "b": "model-00002-of-00002.safetensors"}}))

    with pytest.raises(ValueError):
        inspect_h3(path)


def test_turbo_hints_come_from_the_name_and_metadata_and_decide_nothing(tmp_path):
    plain = inspect_h3(_save_native(tmp_path, name="my-finetune.safetensors"))
    hinted = inspect_h3(_save_native(tmp_path, name="h3_lightning_4step.safetensors", metadata={"modelspec.title": "Turbo merge", "video_shift": "12.0"}))

    assert plain.turbo_hints == ()
    assert any("lightning" in hint for hint in hinted.turbo_hints)
    assert any("video_shift" in hint for hint in hinted.turbo_hints)


def test_quantized_weights_are_counted_and_decoded(tmp_path):
    state = _diffusers_state(torch.Generator().manual_seed(7))
    native = _native(state, interleave=False)
    weight = native["blocks.0.attn.out_proj.weight"]
    scale = weight.abs().amax(dim=1, keepdim=True) / 127
    native["blocks.0.attn.out_proj.weight"] = (weight / scale).round().to(torch.int8)
    native["blocks.0.attn.out_proj.weight_scale"] = scale
    native["blocks.0.attn.out_proj.comfy_quant"] = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
    save_file(native, str(tmp_path / "int8.safetensors"))

    checkpoint = inspect_h3(tmp_path / "int8.safetensors")
    found = dict(diffusers_tensors(checkpoint, {"transformer_blocks.0.attn.to_out.0.weight"}.__contains__))

    assert checkpoint.precision == {"int8_tensorwise": 1}
    assert torch.allclose(found["transformer_blocks.0.attn.to_out.0.weight"].float(), weight, atol=0.05)


def test_the_older_scaled_fp8_marker_is_ignored(tmp_path):
    path = _save_native(tmp_path, extra={"scaled_fp8": torch.zeros(0, dtype=torch.float8_e4m3fn)})

    assert "scaled_fp8" not in inspect_h3(path).shapes
