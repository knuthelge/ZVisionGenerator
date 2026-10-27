"""Tests for writing converted LTX-2.3 folders on macOS (the MLX layout), with a stand-in for ``mx.quantize``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from zvisiongenerator import backends
from zvisiongenerator.backends.mac import ltx_converted
from zvisiongenerator.backends.mac.ltx_converted import embedded_config, mlx_name, transformer_entries, write_converted
from zvisiongenerator.converters import ltx_checkpoint
from zvisiongenerator.converters.ltx_checkpoint import inspect_checkpoint
from zvisiongenerator.utils.ltx_converted import read_converted_ltx
from zvisiongenerator.utils.model_files import is_ltx_model_dir, ltx_mlx_transformer_file
from zvisiongenerator.utils.safetensors_stream import read_header

P = "model.diffusion_model."
_STOCK_CONFIG = {"transformer": {"num_layers": 48, "text_encoder_norm_type": "per_token_rms"}, "vae": {"a": 1}}
# The files of the stand-in MLX pack (its transformer included) and the ones of them a converted folder takes.
_PACK_FILES = (
    "transformer-distilled-1.1.safetensors",
    "connector.safetensors",
    "vae_decoder.safetensors",
    "vocoder.safetensors",
    "spatial_upscaler_x2_v1_1.safetensors",
    "embedded_config.json",
    "config.json",
)
_BASE_FILES = tuple(name for name in _PACK_FILES if not name.startswith("transformer"))


def _tensors() -> dict[str, torch.Tensor]:
    """Return a tiny checkpoint shaped like LTX-2.3 where the converter checks it, with block Linears of 64 inputs."""
    tensors = {
        "patchify_proj.weight": torch.zeros(4096, 128, dtype=torch.bfloat16),
        "audio_patchify_proj.weight": torch.zeros(2048, 128, dtype=torch.bfloat16),
        "prompt_adaln_single.linear_1.weight": torch.ones(2, 2, dtype=torch.float32),
        "scale_shift_table": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    for block in range(48):
        tensors[f"transformer_blocks.{block}.attn1.to_q.weight"] = torch.full((2, 64), float(block), dtype=torch.bfloat16)
        tensors[f"transformer_blocks.{block}.attn1.to_gate_logits.weight"] = torch.ones(2, 64, dtype=torch.bfloat16)
        tensors[f"transformer_blocks.{block}.attn1.to_out.0.bias"] = torch.ones(2, dtype=torch.bfloat16)
        tensors[f"transformer_blocks.{block}.ff.net.0.proj.bias"] = torch.ones(2, dtype=torch.bfloat16)
        tensors[f"transformer_blocks.{block}.scale_shift_table"] = torch.ones(6, 2, dtype=torch.bfloat16)
    return {f"{P}{name}": tensor for name, tensor in tensors.items()}


def _connectors(value: float = 1.0) -> dict[str, torch.Tensor]:
    return {f"{P}video_embeddings_connector.weight": torch.full((2,), value, dtype=torch.bfloat16), "text_embedding_projection.weight": torch.ones(2, dtype=torch.bfloat16)}


def _checkpoint(tmp_path: Path, *, connectors: dict[str, torch.Tensor] | None = None, config: dict | None = None) -> ltx_checkpoint.LtxCheckpoint:
    path = tmp_path / "ft.safetensors"
    save_file({**_tensors(), **(connectors or {})}, str(path), metadata={"config": json.dumps(config)} if config is not None else None)
    return inspect_checkpoint(path)


def _base(tmp_path: Path) -> Path:
    """Return a stand-in MLX folder: its components, and files outside the pack's list (a transformer, a LoRA, a second upsampler) that must never be linked, and stock's config."""
    base = tmp_path / "base"
    base.mkdir()
    for name in ("vae_decoder.safetensors", "vocoder.safetensors", "config.json", "spatial_upscaler_x2_v1_1.safetensors"):
        (base / name).write_bytes(b"stock " + name.encode())
    save_file(
        {f"connector.{name}": torch.full((2,), 9.0, dtype=torch.bfloat16) for name in ("video_embeddings_connector.weight", "text_embedding_projection.weight")}, str(base / "connector.safetensors")
    )
    for name in (
        "transformer-distilled-1.1.safetensors",
        "transformer-dev.safetensors",
        "ltx-2.3-22b-distilled-lora-384.safetensors",
        "spatial_upscaler_x1_5_v1_0.safetensors",
        "vae_encoder_av.safetensors",
    ):
        (base / name).write_bytes(b"never linked")
    (base / "embedded_config.json").write_text(json.dumps(_STOCK_CONFIG))
    return base


def _quantize(weight: torch.Tensor, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stand in for mx.quantize: the right shapes, the packed words holding the weight's first value."""
    rows, columns = weight.shape
    packed = np.full((rows, columns * bits // 32), int(weight[0, 0].item()), dtype=np.uint32)
    groups = np.zeros((rows, columns // 64), dtype=np.uint16)
    return packed, groups, groups


def _write(tmp_path: Path, checkpoint: ltx_checkpoint.LtxCheckpoint, *, bits: int = 4, own_connectors: bool = False) -> Path:
    target = tmp_path / "out"
    target.mkdir()
    write_converted(checkpoint, target, bits, base_dir=_base(tmp_path), base_files=_BASE_FILES, own_connectors=own_connectors, lora=None, quantize=_quantize, merge_device=None, report=lambda *_: None)
    return target


def test_the_folder_has_the_mlx_layout_with_one_transformer_file(tmp_path):
    target = _write(tmp_path, _checkpoint(tmp_path))

    assert sorted(path.name for path in target.glob("transformer*")) == ["transformer-distilled.safetensors"]
    assert ltx_mlx_transformer_file(target) == target / "transformer-distilled.safetensors"
    assert is_ltx_model_dir(target)
    assert (target / "connector.safetensors").read_bytes() == (tmp_path / "base" / "connector.safetensors").read_bytes()  # linked, not written
    assert {path.name for path in target.iterdir()} == {
        "transformer-distilled.safetensors",
        "connector.safetensors",
        "embedded_config.json",
        "vae_decoder.safetensors",
        "vocoder.safetensors",
        "config.json",
        "spatial_upscaler_x2_v1_1.safetensors",
        "ziv-ltx.json",
    }
    manifest = read_converted_ltx(target)
    assert manifest is not None and manifest["bits"] == 4 and manifest["storage"] == "mlx-affine-g64"


@pytest.mark.parametrize("bits", [4, 8])
def test_block_linears_are_quantized_and_the_rest_kept_as_stock_stores_it(tmp_path, bits):
    target = _write(tmp_path, _checkpoint(tmp_path), bits=bits)

    header = read_header(target / "transformer-distilled.safetensors")[1]
    block = "transformer.transformer_blocks.3"
    assert header[f"{block}.attn1.to_q.weight"]["dtype"] == "U32"
    assert header[f"{block}.attn1.to_q.weight"]["shape"] == [2, 64 * bits // 32]
    assert header[f"{block}.attn1.to_q.scales"] == {**header[f"{block}.attn1.to_q.scales"], "dtype": "BF16", "shape": [2, 1]}
    assert header[f"{block}.attn1.to_q.biases"]["shape"] == [2, 1]
    assert header[f"{block}.attn1.to_out.bias"]["dtype"] == "BF16"
    # Stock keeps the modulation tables in float32 and everything else in bfloat16, whatever the checkpoint used.
    assert header[f"{block}.scale_shift_table"]["dtype"] == "F32"
    assert header["transformer.scale_shift_table"]["dtype"] == "F32"
    assert header["transformer.prompt_adaln_single.linear1.weight"]["dtype"] == "BF16"
    assert header["transformer.patchify_proj.weight"]["dtype"] == "BF16"
    written = load_file(str(target / "transformer-distilled.safetensors"))
    assert written[f"{block}.attn1.to_q.weight"].view(torch.int32).unique().tolist() == [3]
    assert torch.equal(written[f"{block}.scale_shift_table"], torch.ones(6, 2))


def test_names_follow_the_vendored_mlx_renames():
    assert mlx_name(f"{P}transformer_blocks.0.attn1.to_out.0.weight") == "transformer.transformer_blocks.0.attn1.to_out.weight"
    assert mlx_name(f"{P}transformer_blocks.0.ff.net.0.proj.weight") == "transformer.transformer_blocks.0.ff.proj_in.weight"
    assert mlx_name(f"{P}transformer_blocks.0.audio_ff.net.2.bias") == "transformer.transformer_blocks.0.audio_ff.proj_out.bias"
    assert mlx_name(f"{P}adaln_single.emb.timestep_embedder.linear_2.weight") == "transformer.adaln_single.emb.timestep_embedder.linear2.weight"


def test_a_block_weight_that_does_not_split_into_groups_of_64_is_refused(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    shapes = {**checkpoint.shapes, f"{P}transformer_blocks.0.attn1.to_q.weight": [2, 48]}

    with pytest.raises(ValueError, match="groups of 64"):
        transformer_entries(ltx_checkpoint.LtxCheckpoint(**{**checkpoint.__dict__, "shapes": shapes}), [f"{P}transformer_blocks.0.attn1.to_q.weight"], 4)


def test_changed_connectors_are_written_from_the_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(ltx_checkpoint, "_pinned", lambda version: {"connectors": dict.fromkeys(_connectors(), "stock hash"), "distillation": {}})
    target = _write(tmp_path, _checkpoint(tmp_path, connectors=_connectors(5.0)), own_connectors=True)

    written = load_file(str(target / "connector.safetensors"))
    assert set(written) == {"connector.video_embeddings_connector.weight", "connector.text_embedding_projection.weight"}
    assert written["connector.video_embeddings_connector.weight"].tolist() == [5.0, 5.0]


def test_a_missing_connector_group_is_taken_from_stock(tmp_path):
    connectors = {f"{P}video_embeddings_connector.weight": torch.full((2,), 5.0, dtype=torch.bfloat16)}  # no text_embedding_projection, as many exports
    target = _write(tmp_path, _checkpoint(tmp_path, connectors=connectors), own_connectors=True)

    written = load_file(str(target / "connector.safetensors"))
    assert written["connector.video_embeddings_connector.weight"].tolist() == [5.0, 5.0]
    assert written["connector.text_embedding_projection.weight"].tolist() == [9.0, 9.0]


def test_connectors_stock_lacks_are_refused(tmp_path):
    connectors = {**_connectors(), f"{P}video_embeddings_connector.extra.weight": torch.ones(2, dtype=torch.bfloat16)}

    with pytest.raises(ValueError):
        _write(tmp_path, _checkpoint(tmp_path, connectors=connectors), own_connectors=True)


def test_the_checkpoints_own_config_is_used_in_stock_spelling(tmp_path):
    own = {"transformer": {"num_layers": 48, "text_encoder_norm_type": "PER_TOKEN_RMS", "extra": "Kept"}, "vae": {"a": 2}}
    target = _write(tmp_path, _checkpoint(tmp_path, config=own))

    assert json.loads((target / "embedded_config.json").read_text()) == {"transformer": {"num_layers": 48, "text_encoder_norm_type": "per_token_rms", "extra": "Kept"}, "vae": {"a": 2}}


def test_stock_keys_the_checkpoints_section_lacks_are_kept():
    merged = embedded_config({"transformer": {"num_layers": 40}}, _STOCK_CONFIG)

    assert merged == {"transformer": {"num_layers": 40, "text_encoder_norm_type": "per_token_rms"}, "vae": {"a": 1}}


def test_stocks_config_is_used_when_the_checkpoint_has_none(tmp_path):
    target = _write(tmp_path, _checkpoint(tmp_path))

    assert json.loads((target / "embedded_config.json").read_text()) == _STOCK_CONFIG
    assert embedded_config(None, _STOCK_CONFIG) == _STOCK_CONFIG


_PIN = "f" * 40


def _pinned_config(**extra) -> dict:
    return {"video_model_presets": {"ltx": {"mlx_base": {"q4": f"org/base@{_PIN}"}, "mlx_files": list(_PACK_FILES), **extra}}}


@pytest.fixture
def convert_with_seams(monkeypatch, tmp_path):
    """Replace the base download, the connector check and the writer; return what they saw and a runner taking the config."""
    import contextlib
    from types import SimpleNamespace

    seen: dict = {"status": "stock"}
    monkeypatch.setattr(ltx_converted, "connectors_status", lambda checkpoint: seen["status"])
    monkeypatch.setattr(ltx_converted, "_base_dir", lambda base: (seen.update(base=base), tmp_path)[1])
    monkeypatch.setattr(ltx_converted, "write_converted", lambda *args, **kwargs: seen.update(own=kwargs["own_connectors"], files=kwargs["base_files"]))
    monkeypatch.setattr("zvisiongenerator.backends.mac.memory.mlx_cache_limit", lambda limit: contextlib.nullcontext())

    def run(config: dict, status: str = "stock") -> dict:
        seen["status"] = status
        ltx_converted.convert(SimpleNamespace(version="2.3"), tmp_path, 4, config=config, version=None, models_dir=tmp_path, lora=None, report=lambda *_: None)
        return seen

    return run


@pytest.mark.parametrize(("status", "own"), [("changed", True), ("unknown", True), ("stock", False), ("absent", False)])
def test_changed_and_quantized_connectors_are_kept_and_stock_ones_linked(convert_with_seams, status, own):
    assert convert_with_seams(_pinned_config(), status)["own"] is own


def test_the_base_is_the_pinned_packs_files_but_its_transformer(convert_with_seams):
    seen = convert_with_seams(_pinned_config())

    assert (seen["base"].repo_id, seen["base"].revision, seen["base"].files) == ("org/base", _PIN, _BASE_FILES)
    assert seen["files"] == _BASE_FILES


def test_an_unpinned_base_of_a_repo_pinned_elsewhere_converts_from_that_pack(convert_with_seams):
    config = _pinned_config()
    config["video_model_presets"]["ltx"]["mlx_base"] = {"q4": "Org/Base", "q8": f"org/base@{_PIN}"}

    seen = convert_with_seams(config)

    assert (seen["base"].repo_id, seen["base"].revision, seen["base"].files) == ("org/base", _PIN, _BASE_FILES)
    assert seen["files"] == _BASE_FILES


def test_an_unpinned_base_cannot_convert(tmp_path):
    from types import SimpleNamespace

    config = {"video_model_presets": {"ltx": {"mlx_base": {"q4": "org/base"}}}}

    with pytest.raises(ValueError, match="pinned"):
        ltx_converted.convert(SimpleNamespace(version="2.3"), tmp_path, 4, config=config, version=None, models_dir=tmp_path, lora=None, report=lambda *_: None)


def test_the_packaged_packs_give_a_base_without_a_transformer_for_both_versions():
    from zvisiongenerator.utils.config import load_config, ltx_mlx_base_repo, ltx_mlx_packs

    config = load_config()
    for version in (None, "ltx25"):
        pack = ltx_mlx_packs(config)[ltx_mlx_base_repo(config, 4, version)]
        base = ltx_converted._base_files(pack)

        assert base and not any(name.startswith("transformer") for name in base)
        assert {"embedded_config.json", "connector.safetensors"} <= set(base)  # read from the base, written or linked by the converter
        assert set(base) == set(pack.files) - {name for name in pack.files if name.startswith("transformer")}


def test_the_base_is_downloaded_as_just_those_files_at_the_pin(monkeypatch, tmp_path):
    from zvisiongenerator.utils.h3_preset import HubFiles

    calls: list[dict] = []

    def download(**kwargs):
        calls.append(kwargs)
        for name in kwargs["allow_patterns"]:
            (tmp_path / name).write_bytes(b"x")
        return str(tmp_path)

    monkeypatch.setattr(ltx_converted, "snapshot_download", download)
    monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: [])

    folder = ltx_converted._base_dir(HubFiles("org/base", _PIN, _BASE_FILES))

    assert folder == tmp_path
    assert calls == [{"repo_id": "org/base", "revision": _PIN, "allow_patterns": list(_BASE_FILES)}]


def test_a_base_file_the_revision_lacks_is_an_error(monkeypatch, tmp_path):
    from zvisiongenerator.utils.h3_preset import HubFiles

    monkeypatch.setattr(ltx_converted, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: [])

    with pytest.raises(RuntimeError, match="vocoder.safetensors"):
        ltx_converted._base_dir(HubFiles("org/base", _PIN, ("vocoder.safetensors",)))


def test_a_failed_base_download_is_one_runtime_error(monkeypatch):
    from zvisiongenerator.utils.h3_preset import HubFiles

    def fail(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr(ltx_converted, "snapshot_download", fail)
    monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: [])

    with pytest.raises(RuntimeError, match="offline"):
        ltx_converted._base_dir(HubFiles("org/base", _PIN, ("vocoder.safetensors",)))


def test_macos_dispatches_to_the_mlx_writer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(backends.sys, "platform", "darwin")
    monkeypatch.setattr(ltx_converted, "convert", lambda *args, **kwargs: calls.append((args, kwargs)))

    backends.convert_ltx_checkpoint("checkpoint", tmp_path, 8, config={}, version=None, models_dir=tmp_path, lora=None, report=print)  # type: ignore[arg-type]

    assert calls and calls[0][0][2] == 8
