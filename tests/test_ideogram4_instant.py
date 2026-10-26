"""Tests for building Ideogram 4 Instant: settings, the transformer conversion and the first-use build."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402

from zvisiongenerator.converters.ideogram4_instant import convert_tensor, fp8_rows, transformer_config, write_transformer  # noqa: E402
from zvisiongenerator.ideogram4_instant_builder import build_instant_model, ensure_instant_model, instant_model_index  # noqa: E402
from zvisiongenerator.image_model_loader import LOADING_PHASE  # noqa: E402
from zvisiongenerator.built_models import ensure_built_model  # noqa: E402
from zvisiongenerator.utils.built_model import BUILD_RECORD_NAME, BUILDING_PHASE  # noqa: E402
from zvisiongenerator.utils.ideogram4_instant import InstantSource, instant_source, is_built, is_instant_dir  # noqa: E402
from zvisiongenerator.utils.image_model_detect import detect_image_model  # noqa: E402

_CONFIG = {
    "ideogram4_instant": {
        "transformer": {"repo": "owner/instant", "revision": "a" * 40, "filename": "instant.safetensors"},
        "components": {"linux": "owner/base-nf4@" + "b" * 40, "darwin": "owner/base-fp8@" + "c" * 40},
        "transformer_format": {"linux": "bf16", "darwin": "fp8"},
    }
}


def _source(transformer_format: str = "bf16") -> InstantSource:
    return InstantSource("owner/instant", "a" * 40, "instant.safetensors", "owner/base@" + "b" * 40, transformer_format)


class TestInstantSource:
    def test_each_platform_gets_its_components_and_format(self):
        linux, darwin = instant_source(_CONFIG, "linux"), instant_source(_CONFIG, "darwin")

        assert (linux.components_repo, linux.components_revision, linux.transformer_format) == ("owner/base-nf4", "b" * 40, "bf16")
        assert (darwin.components_repo, darwin.transformer_format) == ("owner/base-fp8", "fp8")
        assert (linux.transformer_repo, linux.transformer_revision, linux.transformer_file) == ("owner/instant", "a" * 40, "instant.safetensors")

    def test_a_platform_without_an_entry_is_refused(self):
        with pytest.raises(ValueError, match="win32"):
            instant_source(_CONFIG, "win32")

    def test_an_unknown_format_is_refused(self):
        config = json.loads(json.dumps(_CONFIG))
        config["ideogram4_instant"]["transformer_format"]["linux"] = "int4"

        with pytest.raises(ValueError, match="transformer_format"):
            instant_source(config, "linux")


class TestInstantDir:
    def test_only_the_ideo_folder_in_the_models_directory_counts(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))

        assert is_instant_dir(tmp_path / "models" / "ideo")
        assert not is_instant_dir(tmp_path / "elsewhere" / "ideo")
        assert not is_instant_dir(tmp_path / "models" / "zit")

    def test_the_folder_is_detected_as_ideogram4_before_it_is_built(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))

        assert detect_image_model(str(tmp_path / "models" / "ideo")).family == "ideogram4"


class TestConvertTensor:
    def test_bf16_splits_the_fused_attention_by_rows_into_diffusers_names(self):
        qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2)

        result = convert_tensor("layers.3.attention.qkv.weight", qkv, "bf16")

        assert list(result) == ["layers.3.attention.to_q.weight", "layers.3.attention.to_k.weight", "layers.3.attention.to_v.weight"]
        assert torch.equal(torch.cat(list(result.values())), qkv.to(torch.bfloat16))
        assert all(tensor.dtype == torch.bfloat16 for tensor in result.values())

    def test_bf16_renames_the_output_projection_and_keeps_other_tensors(self):
        weight = torch.ones(2, 2)

        assert list(convert_tensor("layers.0.attention.o.weight", weight, "bf16")) == ["layers.0.attention.to_out.0.weight"]
        assert list(convert_tensor("layers.0.feed_forward.w1.weight", weight, "bf16")) == ["layers.0.feed_forward.w1.weight"]

    def test_fp8_stores_linear_weights_as_codes_and_a_row_scale(self):
        weight = torch.tensor([[1.0, -2.0], [0.5, 0.25]])

        result = convert_tensor("layers.0.attention.qkv.weight", weight, "fp8")

        assert result["layers.0.attention.qkv.weight"].dtype == torch.float8_e4m3fn
        assert result["layers.0.attention.qkv.weight_scale"].dtype == torch.float32
        assert result["layers.0.attention.qkv.weight_scale"].shape == (2,)

    @pytest.mark.parametrize("name", ["embed_image_indicator.weight", "layers.0.attention.norm_q.weight", "input_proj.bias"])
    def test_fp8_keeps_embeddings_norms_and_biases_in_bf16(self, name):
        tensor = torch.ones(2, 4) if name.startswith("embed") else torch.ones(4)

        result = convert_tensor(name, tensor, "fp8")

        assert list(result) == [name] and result[name].dtype == torch.bfloat16

    def test_an_unknown_format_raises(self):
        with pytest.raises(ValueError, match="int4"):
            convert_tensor("input_proj.weight", torch.ones(2, 2), "int4")


class TestFp8Rows:
    def test_codes_times_scale_restore_the_weight_closely(self):
        weight = torch.randn(16, 64)

        codes, scale = fp8_rows(weight)
        restored = codes.to(torch.float32) * scale[:, None]

        assert torch.allclose(restored, weight, rtol=0.07, atol=1e-3)
        assert torch.allclose(scale, weight.abs().amax(dim=1) / 448.0)

    def test_a_zero_row_gets_scale_one(self):
        codes, scale = fp8_rows(torch.zeros(1, 4))

        assert scale.tolist() == [1.0] and codes.to(torch.float32).abs().sum() == 0


def _write_source(path):
    save_file(
        {
            "layers.0.attention.qkv.weight": torch.randn(6, 2, dtype=torch.bfloat16),
            "layers.0.attention.o.weight": torch.randn(2, 2, dtype=torch.bfloat16),
            "layers.0.attention_norm1.weight": torch.ones(2, dtype=torch.bfloat16),
            "embed_image_indicator.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        },
        str(path),
    )
    return path


class TestWriteTransformer:
    def test_it_writes_sharded_tensors_an_index_and_the_config_without_storage_keys(self, tmp_path):
        source = _write_source(tmp_path / "instant.safetensors")
        base = {"_class_name": "Ideogram4Transformer2DModel", "num_layers": 34, "quantization_config": {"quant_method": "bitsandbytes"}, "_name_or_path": "/somewhere"}

        assert write_transformer(source, tmp_path / "transformer", "bf16", base) is True

        index = json.loads((tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == {
            "layers.0.attention.to_q.weight",
            "layers.0.attention.to_k.weight",
            "layers.0.attention.to_v.weight",
            "layers.0.attention.to_out.0.weight",
            "layers.0.attention_norm1.weight",
            "embed_image_indicator.weight",
        }
        assert json.loads((tmp_path / "transformer" / "config.json").read_text()) == {"_class_name": "Ideogram4Transformer2DModel", "num_layers": 34}

    def test_fp8_writes_each_weight_with_its_scale(self, tmp_path):
        source = _write_source(tmp_path / "instant.safetensors")

        write_transformer(source, tmp_path / "transformer", "fp8", {})

        tensors = {}
        for shard in (tmp_path / "transformer").glob("*.safetensors"):
            tensors.update(load_file(str(shard)))
        assert tensors["layers.0.attention.qkv.weight"].dtype == torch.float8_e4m3fn
        assert "layers.0.attention.qkv.weight_scale" in tensors and "layers.0.attention.o.weight_scale" in tensors
        assert tensors["embed_image_indicator.weight"].dtype == torch.bfloat16

    def test_a_cancelled_write_stops_and_reports_it(self, tmp_path):
        source = _write_source(tmp_path / "instant.safetensors")

        assert write_transformer(source, tmp_path / "transformer", "bf16", {}, cancelled=lambda: True) is False
        assert not (tmp_path / "transformer" / "config.json").exists()


def test_transformer_config_drops_only_storage_keys():
    assert transformer_config({"num_layers": 34, "quantization_config": {}, "_name_or_path": "x"}) == {"num_layers": 34}


def test_the_model_index_has_no_unconditional_transformer():
    index = instant_model_index({"_class_name": "Ideogram4Pipeline", "_name_or_path": "x", "transformer": ["diffusers", "T"], "unconditional_transformer": ["diffusers", "T"]})

    assert index == {"_class_name": "Ideogram4Pipeline", "transformer": ["diffusers", "T"], "unconditional_transformer": [None, None]}


def _fake_components(root):
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "Ideogram4Pipeline", "unconditional_transformer": ["diffusers", "T"]}))
    (root / "transformer").mkdir()
    (root / "transformer" / "config.json").write_text(json.dumps({"num_layers": 34}))
    (root / "LICENSE.md").write_text("Ideogram 4 Non-Commercial Model Agreement")
    for name in ("scheduler", "text_encoder", "tokenizer", "vae"):
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps({"name": name}))
    return root


class TestBuildInstantModel:
    def _downloaders(self, tmp_path, calls):
        components = _fake_components(tmp_path / "cache" / "base")

        def download_components(source, downloads):
            calls.append("components")
            return components

        def download_transformer(source, downloads):
            calls.append("transformer")
            downloads.mkdir(parents=True, exist_ok=True)
            return _write_source(downloads / source.transformer_file)

        return {"download_components": download_components, "download_transformer": download_transformer}

    def test_it_assembles_the_folder_records_the_source_and_removes_the_download(self, tmp_path):
        calls: list[str] = []
        target = tmp_path / "models" / "ideo"

        assert build_instant_model(_source(), target, **self._downloaders(tmp_path, calls)) is True

        assert calls == ["components", "transformer"]
        assert is_built(target, _source())
        assert json.loads((target / "model_index.json").read_text())["unconditional_transformer"] == [None, None]
        assert json.loads((target / "text_encoder" / "config.json").read_text()) == {"name": "text_encoder"}
        assert (target / "transformer" / "diffusion_pytorch_model.safetensors.index.json").is_file()
        assert (target / "LICENSE.md").read_text().startswith("Ideogram 4")
        assert not (tmp_path / "models" / ".ideo.download").exists()
        assert [path.name for path in (tmp_path / "models").iterdir()] == ["ideo"]

    def test_a_cancelled_build_leaves_no_folder_but_keeps_the_download_to_resume(self, tmp_path):
        target = tmp_path / "models" / "ideo"

        assert build_instant_model(_source(), target, cancelled=lambda: True, **self._downloaders(tmp_path, [])) is False

        assert not target.exists()
        assert (tmp_path / "models" / ".ideo.download" / "instant.safetensors").is_file()
        assert not list((tmp_path / "models").glob(".ideo.*.partial"))

    def test_a_failed_build_discards_the_partial_folder(self, tmp_path):
        target = tmp_path / "models" / "ideo"
        downloaders = self._downloaders(tmp_path, [])
        downloaders["download_transformer"] = lambda source, downloads: (_ for _ in ()).throw(OSError("network"))

        with pytest.raises(OSError, match="network"):
            build_instant_model(_source(), target, **downloaders)

        assert not target.exists() and not list((tmp_path / "models").glob(".ideo.*.partial"))


class TestForeignFolder:
    def test_a_non_empty_folder_without_a_build_record_is_never_replaced_and_nothing_downloads(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        calls: list[str] = []
        other = tmp_path / "models" / "ideo"
        other.mkdir(parents=True)
        (other / "weights.safetensors").write_bytes(b"someone else's model")

        with pytest.raises(RuntimeError, match="Rename or move"):
            build_instant_model(_source(), other, **TestBuildInstantModel()._downloaders(tmp_path, calls))

        assert calls == []
        assert (other / "weights.safetensors").read_bytes() == b"someone else's model"
        assert not (tmp_path / "models" / ".ideo.download").exists()

    def test_a_folder_built_by_v0_13_0_is_still_ours_and_is_rebuilt_when_outdated(self, tmp_path):
        target = tmp_path / "models" / "ideo"
        target.mkdir(parents=True)
        (target / "model_index.json").write_text("{}")
        # The record exactly as v0.13.0 wrote it, with a revision that is no longer the pinned one.
        record = {
            "version": 1,
            "transformer_repo": "Hippotes/Ideogram4-Fal-ComfyUI",
            "transformer_revision": "0" * 40,
            "transformer_file": "Ideogram4-instant_bf16.safetensors",
            "components": "unsloth/ideogram-4-nf4-diffusers@cb5b5dda5e0406c04160f3555c48256ec0cfc899",
            "transformer_format": "bf16",
        }
        (target / BUILD_RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        assert build_instant_model(_source(), target, **TestBuildInstantModel()._downloaders(tmp_path, [])) is True
        assert is_built(target, _source())

    def test_an_empty_folder_is_built_in(self, tmp_path):
        target = tmp_path / "models" / "ideo"
        target.mkdir(parents=True)

        assert build_instant_model(_source(), target, **TestBuildInstantModel()._downloaders(tmp_path, [])) is True
        assert is_built(target, _source())

    def test_a_folder_with_an_outdated_record_is_rebuilt(self, tmp_path):
        target = tmp_path / "models" / "ideo"
        downloaders = TestBuildInstantModel()._downloaders(tmp_path, [])
        build_instant_model(_source(), target, **downloaders)

        assert build_instant_model(_source("fp8"), target, **downloaders) is True
        assert is_built(target, _source("fp8"))


class TestEnsureBuiltModel:
    def test_an_ideo_folder_goes_to_the_instant_builder_with_the_cancel_check(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        seen: list = []
        monkeypatch.setattr("zvisiongenerator.ideogram4_instant_builder.build_instant_model", lambda source, target, cancelled=None: seen.append(cancelled) or False)
        stop = lambda: True  # noqa: E731
        phases: list[str] = []

        assert ensure_built_model(str(tmp_path / "models" / "ideo"), _CONFIG, platform_key="linux", on_phase=phases.append, cancelled=stop) is False

        assert seen == [stop] and phases == [BUILDING_PHASE]


class TestEnsureInstantModel:
    def test_other_models_are_left_alone(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        monkeypatch.setattr("zvisiongenerator.ideogram4_instant_builder.build_instant_model", lambda *a, **k: pytest.fail("built"))

        assert ensure_instant_model(str(tmp_path / "models" / "zit"), _CONFIG, platform_key="linux") is True

    def test_a_missing_folder_is_built_and_reports_its_phases(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        built: list[tuple] = []
        monkeypatch.setattr("zvisiongenerator.ideogram4_instant_builder.build_instant_model", lambda source, target, cancelled=None: built.append((source, target)) or True)
        phases: list[str] = []

        assert ensure_instant_model(str(tmp_path / "models" / "ideo"), _CONFIG, platform_key="linux", on_phase=phases.append) is True

        assert built == [(instant_source(_CONFIG, "linux"), tmp_path / "models" / "ideo")]
        assert phases == [BUILDING_PHASE, LOADING_PHASE]

    def test_a_current_build_is_reused_and_a_pin_change_rebuilds(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        target = tmp_path / "models" / "ideo"
        target.mkdir(parents=True)
        from zvisiongenerator.utils.ideogram4_instant import write_build_record

        write_build_record(target, instant_source(_CONFIG, "linux"))
        built: list[str] = []
        monkeypatch.setattr("zvisiongenerator.ideogram4_instant_builder.build_instant_model", lambda source, target, cancelled=None: built.append(source.transformer_revision) or True)

        ensure_instant_model(str(target), _CONFIG, platform_key="linux")
        changed = json.loads(json.dumps(_CONFIG))
        changed["ideogram4_instant"]["transformer"]["revision"] = "d" * 40
        ensure_instant_model(str(target), changed, platform_key="linux")

        assert built == ["d" * 40]
        assert (target / BUILD_RECORD_NAME).is_file()
