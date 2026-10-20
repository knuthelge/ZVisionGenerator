"""Tests for converting LTX-2.3 checkpoints: reading the checkpoint, the CLI conversion, detection and the folder manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from zvisiongenerator.converters import ltx_checkpoint, ltx_convert
from zvisiongenerator.converters.convert_checkpoint import _build_model_parser, _cmd_model
from zvisiongenerator.converters.list_assets import list_video_models
from zvisiongenerator.converters.ltx_checkpoint import check_lora, connectors_status, distillation, inspect_checkpoint, transformer_tensors
from zvisiongenerator.utils.ltx_converted import read_converted_ltx, write_converted_ltx
from zvisiongenerator.utils.video_model_detect import detect_video_model

P = "model.diffusion_model."
_DISTILL_NAMES = [f"{P}{name}.{part}" for name in ("patchify_proj", "audio_patchify_proj", "proj_out", "audio_proj_out") for part in ("weight", "bias")]


def _ltx_tensors(*, blocks: int = 48, gate: bool = True, ff_bias: bool = True) -> dict[str, torch.Tensor]:
    """Return the tensors of a tiny checkpoint shaped like LTX-2.3 where the converter checks it, under native names."""
    tensors = {
        "patchify_proj.weight": torch.zeros(4096, 128, dtype=torch.bfloat16),
        "patchify_proj.bias": torch.zeros(4096, dtype=torch.bfloat16),
        "audio_patchify_proj.weight": torch.zeros(2048, 128, dtype=torch.bfloat16),
        "audio_patchify_proj.bias": torch.zeros(2048, dtype=torch.bfloat16),
        "proj_out.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "proj_out.bias": torch.ones(2, dtype=torch.bfloat16),
        "audio_proj_out.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "audio_proj_out.bias": torch.ones(2, dtype=torch.bfloat16),
        "prompt_adaln_single.linear.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "scale_shift_table": torch.ones(2, 2, dtype=torch.float32),
    }
    for block in range(blocks):
        tensors[f"transformer_blocks.{block}.attn1.to_q.weight"] = torch.full((2, 2), float(block), dtype=torch.bfloat16)
        if gate:
            tensors[f"transformer_blocks.{block}.attn1.to_gate_logits.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
        if ff_bias:
            tensors[f"transformer_blocks.{block}.ff.net.0.proj.bias"] = torch.ones(2, dtype=torch.bfloat16)
    return tensors


def _save(path: Path, tensors: dict[str, torch.Tensor], prefix: str = P) -> Path:
    """Save *tensors* under *prefix* to *path*."""
    save_file({f"{prefix}{name}": tensor for name, tensor in tensors.items()}, str(path))
    return path


def _connectors() -> dict[str, torch.Tensor]:
    return {f"{P}video_embeddings_connector.weight": torch.ones(2, dtype=torch.bfloat16), "text_embedding_projection.weight": torch.ones(2, dtype=torch.bfloat16)}


@pytest.mark.parametrize("prefix", [P, "diffusion_model.", ""])
def test_transformer_names_are_normalized_from_any_export_prefix(tmp_path, prefix):
    checkpoint = inspect_checkpoint(_save(tmp_path / "ft.safetensors", _ltx_tensors(), prefix))

    assert f"{P}transformer_blocks.47.attn1.to_q.weight" in checkpoint.transformer
    assert checkpoint.transformer[f"{P}patchify_proj.weight"] == f"{prefix}patchify_proj.weight"


def test_full_checkpoints_keep_the_projection_unprefixed_and_ignore_the_bundled_components(tmp_path):
    path = tmp_path / "full.safetensors"
    save_file({**{f"{P}{name}": t for name, t in _ltx_tensors().items()}, **_connectors(), "vae.decoder.weight": torch.ones(2)}, str(path))

    checkpoint = inspect_checkpoint(path)

    assert set(checkpoint.connectors) == set(_connectors())
    assert not any(name.startswith("vae.") for name in checkpoint.transformer)
    assert checkpoint.bundles_components


@pytest.mark.parametrize(
    ("tensors", "message"),
    [
        (_ltx_tensors(gate=False), "LTX-2.0"),
        (_ltx_tensors(ff_bias=False), "LTX-2.5"),
        (_ltx_tensors(blocks=40), "40 blocks"),
    ],
)
def test_checkpoints_that_are_not_ltx23_are_refused(tmp_path, tensors, message):
    with pytest.raises(ValueError, match=message):
        inspect_checkpoint(_save(tmp_path / "other.safetensors", tensors))


def test_scaled_fp8_connectors_are_read_without_their_scales(tmp_path):
    path = tmp_path / "fp8.safetensors"
    connector = f"{P}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q"
    save_file(
        {
            **{f"{P}{name}": t for name, t in _ltx_tensors().items()},
            f"{connector}.weight": torch.full((2, 2), 2.0).to(torch.float8_e4m3fn),
            f"{connector}.weight_scale": torch.tensor(0.5),
            f"{connector}.comfy_quant": torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8),
        },
        str(path),
    )

    checkpoint = inspect_checkpoint(path)

    assert set(checkpoint.connectors) == {f"{connector}.weight"}


def test_a_diffusers_layout_file_is_refused(tmp_path):
    path = _save(tmp_path / "diffusers.safetensors", {"proj_in.weight": torch.ones(2, 2), "time_embed.linear.weight": torch.ones(2, 2)}, prefix="")

    with pytest.raises(ValueError, match="diffusers' layout"):
        inspect_checkpoint(path)


def test_scaled_fp8_weights_are_dequantized_per_tensor_in_a_mixed_file(tmp_path):
    tensors = _ltx_tensors()
    tensors["transformer_blocks.3.attn1.to_q.weight"] = torch.full((2, 2), 2.0).to(torch.float8_e4m3fn)
    tensors["transformer_blocks.3.attn1.to_q.weight_scale"] = torch.tensor(0.5)
    tensors["transformer_blocks.3.attn1.to_q.comfy_quant"] = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)
    checkpoint = inspect_checkpoint(_save(tmp_path / "fp8.safetensors", tensors))

    out = dict(transformer_tensors(checkpoint))

    assert out[f"{P}transformer_blocks.3.attn1.to_q.weight"].dtype == torch.bfloat16
    assert torch.equal(out[f"{P}transformer_blocks.3.attn1.to_q.weight"], torch.ones(2, 2, dtype=torch.bfloat16))
    assert torch.equal(out[f"{P}transformer_blocks.4.attn1.to_q.weight"], torch.full((2, 2), 4.0, dtype=torch.bfloat16))
    assert not any(name.endswith(("weight_scale", "comfy_quant")) for name in out)
    assert checkpoint.fp8_weights == 1


def test_other_quantization_formats_are_refused(tmp_path):
    tensors = _ltx_tensors()
    tensors["transformer_blocks.3.attn1.to_q.comfy_quant"] = torch.tensor(list(json.dumps({"format": "int4"}).encode()), dtype=torch.uint8)

    with pytest.raises(ValueError, match="int4"):
        inspect_checkpoint(_save(tmp_path / "int4.safetensors", tensors))


def _lora(path: Path, target: str, down: torch.Tensor, up: torch.Tensor) -> Path:
    save_file({f"diffusion_model.{target}.lora_A.weight": down, f"diffusion_model.{target}.lora_B.weight": up}, str(path))
    return path


def test_distill_merges_the_lora_delta_before_the_tensor_is_yielded(tmp_path):
    checkpoint = inspect_checkpoint(_save(tmp_path / "dev.safetensors", _ltx_tensors()))
    lora = _lora(tmp_path / "lora.safetensors", "transformer_blocks.1.attn1.to_q", torch.ones(1, 2, dtype=torch.bfloat16), torch.full((2, 1), 3.0, dtype=torch.bfloat16))

    merged = dict(transformer_tensors(checkpoint, lora=lora))
    plain = dict(transformer_tensors(checkpoint))

    assert torch.equal(merged[f"{P}transformer_blocks.1.attn1.to_q.weight"], torch.full((2, 2), 4.0, dtype=torch.bfloat16))
    assert torch.equal(plain[f"{P}transformer_blocks.1.attn1.to_q.weight"], torch.ones(2, 2, dtype=torch.bfloat16))
    assert torch.equal(merged[f"{P}transformer_blocks.2.attn1.to_q.weight"], plain[f"{P}transformer_blocks.2.attn1.to_q.weight"])


def test_a_lora_for_weights_the_checkpoint_lacks_is_refused(tmp_path):
    checkpoint = inspect_checkpoint(_save(tmp_path / "dev.safetensors", _ltx_tensors()))
    lora = _lora(tmp_path / "lora.safetensors", "transformer_blocks.1.attn9.to_q", torch.ones(1, 2), torch.ones(2, 1))

    with pytest.raises(ValueError, match="does not have"):
        check_lora(checkpoint, lora)


def _pin(monkeypatch, tmp_path, *, distillation_from: dict[str, dict[str, torch.Tensor]] | None = None, connectors: dict[str, torch.Tensor] | None = None) -> None:
    """Pin the hashes of the tensors in each labelled checkpoint of *distillation_from*, and of *connectors*."""
    pinned: dict[str, dict] = {"distillation": {}, "connectors": {}}
    for label, tensors in (distillation_from or {}).items():
        reference = inspect_checkpoint(_save(tmp_path / f"{label}.safetensors", tensors))
        pinned["distillation"][label] = ltx_checkpoint._hashes(reference, {name: name for name in _DISTILL_NAMES})
    if connectors is not None:
        path = tmp_path / "stock-connectors.safetensors"
        save_file({**{f"{P}{name}": t for name, t in _ltx_tensors().items()}, **connectors}, str(path))
        pinned["connectors"] = ltx_checkpoint._hashes(inspect_checkpoint(path), {name: name for name in connectors})
    monkeypatch.setattr(ltx_checkpoint, "_pinned", lambda: pinned)


def _distilled(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {**tensors, "proj_out.weight": torch.full((2, 2), 7.0, dtype=torch.bfloat16)}


@pytest.mark.parametrize(("variant", "expected"), [("dev", "dev"), ("distilled", "distilled"), ("other", "unknown")])
def test_the_distillation_check_compares_the_four_tensors_with_the_pinned_hashes(monkeypatch, tmp_path, variant, expected):
    dev = _ltx_tensors()
    _pin(monkeypatch, tmp_path, distillation_from={"dev": dev, "distilled-1.1": _distilled(dev)})
    tensors = {"dev": dev, "distilled": _distilled(dev), "other": {**dev, "audio_proj_out.bias": torch.zeros(2, dtype=torch.bfloat16)}}[variant]

    assert distillation(inspect_checkpoint(_save(tmp_path / "ft.safetensors", tensors))) == expected


def test_only_a_plain_dev_checkpoint_gets_the_distill_advice(monkeypatch, tmp_path):
    dev = _ltx_tensors()
    _pin(monkeypatch, tmp_path, distillation_from={"dev": dev, "distilled-1.1": _distilled(dev)})

    assert "--distill" in ltx_convert.inspect_ltx(_save(tmp_path / "ft.safetensors", dev)).advice
    assert "--distill" not in ltx_convert.inspect_ltx(_save(tmp_path / "d.safetensors", _distilled(dev))).advice


def _stock_connectors() -> dict[str, torch.Tensor]:
    """Return stand-in stock connectors: two tensors of the video connector and the projection."""
    return {
        f"{P}video_embeddings_connector.a.weight": torch.ones(2, dtype=torch.bfloat16),
        f"{P}video_embeddings_connector.b.weight": torch.full((2,), 2.0, dtype=torch.bfloat16),
        "text_embedding_projection.weight": torch.ones(2, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("none", "absent"),
        ("stock", "stock"),
        ("changed", "changed"),
        ("no projection", "stock"),
        ("only the projection", "stock"),
        ("half the video connector", "changed"),
        ("float32", "stock"),
        ("an extra tensor", "changed"),
    ],
)
def test_connectors_are_absent_equal_to_stock_or_changed(monkeypatch, tmp_path, variant, expected):
    stock = _stock_connectors()
    _pin(monkeypatch, tmp_path, connectors=stock)
    video_a, video_b, projection = stock
    found = {
        "none": {},
        "stock": stock,
        "changed": {**stock, projection: torch.zeros(2, dtype=torch.bfloat16)},
        "no projection": {video_a: stock[video_a], video_b: stock[video_b]},
        "only the projection": {projection: stock[projection]},
        "half the video connector": {video_a: stock[video_a], projection: stock[projection]},
        "float32": {name: tensor.float() for name, tensor in stock.items()},
        "an extra tensor": {**stock, f"{P}audio_embeddings_connector.c.weight": torch.ones(2, dtype=torch.bfloat16)},
    }[variant]
    path = tmp_path / "ft.safetensors"
    save_file({**{f"{P}{name}": t for name, t in _ltx_tensors().items()}, **found}, str(path))

    assert connectors_status(inspect_checkpoint(path)) == expected


# ── CLI ──────────────────────────────────────────────────────────────────────


def _run_cli(argv: list[str]) -> None:
    _cmd_model(_build_model_parser().parse_args(["model", *argv]))


@pytest.fixture
def ltx_cli(monkeypatch, tmp_path):
    """A data dir, a tiny dev checkpoint and a recorded platform conversion that writes a complete folder."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr("zvisiongenerator.converters.convert_checkpoint.get_ziv_data_dir", lambda: data_dir)
    monkeypatch.setattr("zvisiongenerator.utils.config.load_config", lambda: {"model_aliases": {"ltx-4": "x"}})
    monkeypatch.setattr(ltx_checkpoint, "_pinned", lambda: {"distillation": {"dev": dict.fromkeys(_DISTILL_NAMES, "0")}, "connectors": {}})
    calls: list[dict] = []

    def _convert(checkpoint, target, bits, **kwargs):
        calls.append({"target": target, "bits": bits, **kwargs})
        (target / "transformer").mkdir()
        write_converted_ltx(target, {"bits": bits, "storage": "nf4", "block_matrices": 1, "transformer_rest": 1})

    monkeypatch.setattr("zvisiongenerator.backends.convert_ltx_checkpoint", _convert)
    monkeypatch.setattr("zvisiongenerator.converters.convert_checkpoint._store_quantized_copy", lambda *a: pytest.fail("LTX stores no extra copy"))
    checkpoint = _save(tmp_path / "ft.safetensors", _ltx_tensors())
    return data_dir / "models", checkpoint, calls


def test_ltx_needs_a_quantize_level(ltx_cli, capsys):
    _models, checkpoint, calls = ltx_cli

    with pytest.raises(SystemExit):
        _run_cli(["-i", str(checkpoint), "--model-type", "ltx-2.3"])

    assert "--quantize 4 or 8" in capsys.readouterr().err
    assert calls == []


def test_ltx_converts_at_the_one_level_given_and_stores_no_extra_copy(ltx_cli):
    models, checkpoint, calls = ltx_cli

    _run_cli(["-i", str(checkpoint), "--model-type", "ltx-2.3", "--quantize", "8", "--name", "myft"])

    assert calls[0]["bits"] == 8 and calls[0]["lora"] is None
    assert read_converted_ltx(models / "myft")["bits"] == 8
    assert sorted(path.name for path in models.iterdir()) == ["myft"]  # no partial left, no @q8 copy


def test_a_configured_alias_is_refused_as_the_name(ltx_cli):
    models, checkpoint, calls = ltx_cli

    with pytest.raises(SystemExit):
        _run_cli(["-i", str(checkpoint), "--model-type", "ltx-2.3", "--quantize", "4", "--name", "ltx-4"])

    assert calls == []


def test_an_existing_name_is_refused_and_left_alone_without_force(ltx_cli):
    models, checkpoint, calls = ltx_cli
    (models / "myft").mkdir(parents=True)
    (models / "myft" / "keep.txt").write_text("old")

    with pytest.raises(SystemExit):
        _run_cli(["-i", str(checkpoint), "--model-type", "ltx-2.3", "--quantize", "4", "--name", "myft"])

    assert (models / "myft" / "keep.txt").read_text() == "old"
    assert calls == []


def test_force_replaces_the_old_folder_only_once_the_new_one_is_complete(ltx_cli, monkeypatch):
    models, checkpoint, calls = ltx_cli
    (models / "myft").mkdir(parents=True)
    (models / "myft" / "keep.txt").write_text("old")
    seen_during: list[bool] = []
    original = __import__("zvisiongenerator.backends", fromlist=["convert_ltx_checkpoint"]).convert_ltx_checkpoint

    def _convert(checkpoint, target, bits, **kwargs):
        seen_during.append((models / "myft" / "keep.txt").exists())
        original(checkpoint, target, bits, **kwargs)

    monkeypatch.setattr("zvisiongenerator.backends.convert_ltx_checkpoint", _convert)

    _run_cli(["-i", str(checkpoint), "--model-type", "ltx-2.3", "--quantize", "4", "--name", "myft", "--force"])

    assert seen_during == [True]
    assert not (models / "myft" / "keep.txt").exists()
    assert read_converted_ltx(models / "myft") is not None


def test_an_interrupted_conversion_leaves_no_folder(ltx_cli, monkeypatch):
    models, checkpoint, _calls = ltx_cli

    def _interrupted(checkpoint, target, bits, **kwargs):
        (target / "transformer").mkdir()
        raise KeyboardInterrupt

    monkeypatch.setattr("zvisiongenerator.backends.convert_ltx_checkpoint", _interrupted)

    with pytest.raises(KeyboardInterrupt):
        _run_cli(["-i", str(checkpoint), "--model-type", "ltx-2.3", "--quantize", "4", "--name", "myft"])

    assert list(models.iterdir()) == []


def test_distill_fails_before_converting_when_the_lora_cannot_be_downloaded(ltx_cli, monkeypatch, capsys):
    _models, checkpoint, calls = ltx_cli
    monkeypatch.setattr("zvisiongenerator.utils.config.load_config", lambda: {"video_model_presets": {"ltx": {"distilled_lora": {"repo": "o/r", "filename": "l.safetensors"}}}})

    def _offline(**_kwargs):
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _offline)

    with pytest.raises(SystemExit):
        _run_cli(["-i", str(checkpoint), "--model-type", "ltx-2.3", "--quantize", "4", "--distill"])

    assert "couldn't be downloaded" in capsys.readouterr().err
    assert calls == []


def test_distill_and_force_are_refused_for_image_types(ltx_cli):
    _models, checkpoint, _calls = ltx_cli

    with pytest.raises(SystemExit):
        _run_cli(["-i", str(checkpoint), "--model-type", "zimage", "--distill"])


# ── Detection and manifest ───────────────────────────────────────────────────


@pytest.mark.parametrize("class_name", ["LTX2Pipeline", "LTX2ImageToVideoPipeline"])
def test_ltx_folders_are_detected_by_their_contents_whatever_their_name(tmp_path, class_name):
    folder = tmp_path / "models" / "myfinetune"
    folder.mkdir(parents=True)
    (folder / "model_index.json").write_text(json.dumps({"_class_name": class_name}))

    assert detect_video_model(str(folder)).family == "ltx"
    assert [entry.name for entry in list_video_models(tmp_path)] == ["myfinetune"]


def test_partial_folders_are_not_listed(tmp_path):
    folder = tmp_path / "models" / ".myft.1234abcd.partial"
    folder.mkdir(parents=True)
    (folder / "model_index.json").write_text(json.dumps({"_class_name": "LTX2Pipeline"}))

    assert list_video_models(tmp_path) == []


def test_the_manifest_reads_back_only_when_complete_and_current(tmp_path):
    write_converted_ltx(tmp_path, {"bits": 4, "storage": "nf4", "block_matrices": 10, "transformer_rest": 2})
    assert read_converted_ltx(tmp_path)["bits"] == 4

    (tmp_path / "ziv-ltx.json").write_text(json.dumps({"kind": "ltx-2.3", "layout": 99, "bits": 4, "block_matrices": 1, "transformer_rest": 1}))
    assert read_converted_ltx(tmp_path) is None
    (tmp_path / "ziv-ltx.json").write_text("{")
    assert read_converted_ltx(tmp_path) is None
