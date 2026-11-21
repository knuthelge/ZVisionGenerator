"""Tests for the command that regenerates the shipped weight table: which downloads feed it, how rows are read and how the file is rewritten."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from zvisiongenerator.utils import weight_totals_table
from zvisiongenerator.utils.weight_totals_table import (
    IMAGE,
    LTX_CUDA_REPO,
    LTX_MLX_GEMMA,
    LTX_MLX_PACK,
    LTX_TRANSFORMER_FILE,
    memory_sources,
    regenerate,
    render_table,
    write_table,
)


def _kinds(model_id: str) -> dict[tuple[str, str], tuple[str, ...]]:
    return {(source.target.repo_id, source.kind): source.files for source in memory_sources(model_id)}


class TestMemorySources:
    def test_an_image_model_reads_its_repo_as_components(self) -> None:
        assert _kinds("klein9b") == {("unsloth/FLUX.2-klein-9B", IMAGE): ()}

    def test_ideogram_reads_the_base_repos_and_the_transformer_file(self) -> None:
        sources = memory_sources("ideo")

        assert {source.target.repo_id for source in sources} == {"unsloth/ideogram-4-fp8", "unsloth/ideogram-4-nf4-diffusers", "Hippotes/Ideogram4-Fal-ComfyUI"}
        assert next(source.files for source in sources if source.target.repo_id.startswith("Hippotes")) == ("Ideogram4-instant_bf16.safetensors",)

    def test_moss_turbo_reads_the_lora_file_and_sft_does_not(self) -> None:
        turbo, sft = _kinds("moss"), _kinds("moss-sft")

        assert ("akhaliq/MOSS-Image-SFT-4-steps", IMAGE) in turbo and ("akhaliq/MOSS-Image-SFT-4-steps", IMAGE) not in sft
        assert turbo[("akhaliq/MOSS-Image-SFT-4-steps", IMAGE)] == ("moss_image_sft_4step_v2.0.safetensors",)

    def test_ltx_reads_the_diffusers_repo_and_the_transformer_file_on_cuda_and_the_packs_on_mac(self) -> None:
        kinds = {kind for _, kind in _kinds("ltx-4")}

        assert kinds == {LTX_CUDA_REPO, LTX_TRANSFORMER_FILE, LTX_MLX_PACK, LTX_MLX_GEMMA}
        assert {kind for _, kind in _kinds("ltx25-4")} == {LTX_CUDA_REPO, LTX_MLX_PACK}  # LTX-2.5 carries its own transformer and Gemma

    def test_each_download_appears_once(self) -> None:
        sources = memory_sources("ltx-4")

        assert len({(source.key, source.kind, source.files) for source in sources}) == len(sources)

    @pytest.mark.parametrize("model_id", ["h3", "yue2", "seedvr2-7b", "enhancer-fast"])
    def test_a_model_without_a_weight_estimate_is_refused(self, model_id: str) -> None:
        with pytest.raises(ValueError, match="neither"):
            memory_sources(model_id)


class TestRegenerate:
    @pytest.fixture
    def reads(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        read: list[str] = []

        def read_row(_reader: object, sources: list[Any]) -> dict[str, Any]:
            key = sources[0].key
            read.append(key)
            if key.startswith("Tongyi"):
                raise PermissionError("gated")
            return {"read": key}

        monkeypatch.setattr(weight_totals_table, "_read_row", read_row)
        return read

    def test_a_repo_that_cannot_be_read_keeps_its_old_row_and_is_reported(self, reads: list[str]) -> None:
        old = {"Tongyi-MAI/Z-Image-Turbo": {"old": True}, "gone/repo": {"old": True}}

        table, failures = regenerate(object(), old)  # type: ignore[arg-type]

        assert table["Tongyi-MAI/Z-Image-Turbo"] == {"old": True} and list(failures) == ["Tongyi-MAI/Z-Image-Turbo"]
        assert "gated" in failures["Tongyi-MAI/Z-Image-Turbo"]
        assert "gone/repo" not in table  # a repo no pinned model uses any more drops out
        assert table["unsloth/FLUX.2-klein-9B"] == {"read": "unsloth/FLUX.2-klein-9B"}

    def test_a_repo_that_cannot_be_read_and_has_no_old_row_has_none(self, reads: list[str]) -> None:
        table, failures = regenerate(object(), {})  # type: ignore[arg-type]

        assert "Tongyi-MAI/Z-Image-Turbo" not in table and "Tongyi-MAI/Z-Image-Turbo" in failures

    def test_a_filter_reads_only_the_repos_it_accepts_and_keeps_the_other_rows(self, reads: list[str]) -> None:
        old = {"unsloth/FLUX.2-klein-9B": {"old": True}}

        table, _ = regenerate(object(), old, lambda key: "Krea" in key)  # type: ignore[arg-type]

        assert reads == ["unsloth/Krea-2-Turbo"] and table["unsloth/FLUX.2-klein-9B"] == {"old": True}

    def test_the_rows_come_out_in_key_order(self, reads: list[str]) -> None:
        table, _ = regenerate(object(), {})  # type: ignore[arg-type]

        assert list(table) == sorted(table)


class TestTableFile:
    _TABLE = {
        "owner/repo@" + "a" * 40: {"family": "zimage", "size": "9b", "components": {"vae": {"float_matrix_elements": 5, "prequantized": True}}, "files": {"x.safetensors": {"bytes": 3}}},
        "owner/unpinned": {"revision": "455f" + "0" * 36, "ltx_cuda": {"connectors": 1, "text_encoder_bytes": 2}},
    }

    def test_the_rendered_table_reads_back_as_it_was_written(self) -> None:
        assert yaml.safe_load(render_table(self._TABLE)) == {"weight_totals": self._TABLE}

    def test_an_empty_table_renders_as_an_empty_mapping(self) -> None:
        assert yaml.safe_load(render_table({})) == {"weight_totals": {}}

    def test_a_table_replaces_only_the_last_section(self, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text("# weight_totals: described here\nmodels:\n  a: 1\n\nweight_totals:\n  old: { x: 1 }\n", encoding="utf-8")

        write_table(path, self._TABLE)

        text = path.read_text(encoding="utf-8")
        assert text.startswith("# weight_totals: described here\nmodels:\n  a: 1\n\nweight_totals:\n")
        assert yaml.safe_load(text) == {"models": {"a": 1}, "weight_totals": self._TABLE}

    def test_a_file_without_the_section_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text("models: {}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="weight_totals"):
            write_table(path, {})


class _FakeReader:
    """A ``HubReader`` for one repo: its file listing and the tensors of each file."""

    def __init__(self, tensors: dict[str, dict[str, tuple[str, list[int]]]]) -> None:
        self._tensors = tensors

    def listing(self, repo_id: str, revision: str | None) -> tuple[str, list[str]]:
        return "f" * 40, [*self._tensors, "config.json"]

    def headers(self, repo_id: str, revision: str | None, names: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
        return {name: {tensor: {"dtype": dtype, "shape": shape, "data_offsets": [0, 1]} for tensor, (dtype, shape) in self._tensors[name].items()} for name in names}


def _target(files: tuple[str, ...] | None, ignore: tuple[str, ...] | None = None) -> Any:
    from zvisiongenerator.downloads.target import DownloadTarget

    return weight_totals_table.RowSource(DownloadTarget("owner/pack", "a" * 40, allow_patterns=files, ignore_patterns=ignore), LTX_MLX_PACK)


class TestLtxRows:
    def test_a_pack_with_its_own_text_encoder_holds_it_and_one_without_leaves_it_to_the_gemma_repo(self) -> None:
        tensors = {
            "transformer-distilled.safetensors": {"w": ("BF16", [4, 4])},
            "connector.safetensors": {"w": ("BF16", [2, 2])},
            "vae_decoder_conv.safetensors": {"w": ("BF16", [3])},
            "audio_vae.safetensors": {"w": ("BF16", [1])},
            "vocoder.safetensors": {"w": ("BF16", [1])},
            "spatial_upscaler_x2_v1_0.safetensors": {"w": ("BF16", [5])},
            "text_encoder.safetensors": {"w": ("BF16", [6, 6])},
        }
        names = (*tensors, "text_encoder_config.json", "embedded_config.json")

        bundled = weight_totals_table._ltx_mlx_pack(_FakeReader(tensors), _target(names))  # type: ignore[arg-type]
        without = weight_totals_table._ltx_mlx_pack(_FakeReader(tensors), _target(tuple(name for name in names if not name.startswith("text_encoder"))))  # type: ignore[arg-type]

        assert bundled["ltx_mlx"] == {"connector": 8, "transformer": 32, "decoders": 10, "upsampler": 10, "gemma": 72}
        assert "gemma" not in without["ltx_mlx"] and without["ltx_mlx"]["transformer"] == 32

    def test_a_listed_weight_file_the_repo_lacks_is_an_error(self) -> None:
        with pytest.raises(FileNotFoundError, match="connector.safetensors"):
            weight_totals_table._ltx_mlx_pack(_FakeReader({"transformer.safetensors": {"w": ("BF16", [1])}}), _target(("transformer.safetensors", "connector.safetensors")))  # type: ignore[arg-type]

    def test_the_gemma_repo_holds_the_bytes_of_all_its_weights(self) -> None:
        tensors = {"model-00001-of-00002.safetensors": {"w": ("BF16", [10])}, "model-00002-of-00002.safetensors": {"w": ("BF16", [5])}}
        source = weight_totals_table.RowSource(_target(None).target, LTX_MLX_GEMMA)

        row = weight_totals_table._ltx_mlx_gemma(_FakeReader(tensors), source)  # type: ignore[arg-type]

        assert row["ltx_mlx"] == {"gemma": 30}

    def test_a_diffusers_repo_counts_the_connectors_the_fp8_text_encoder_and_its_own_transformer(self) -> None:
        tensors = {
            "connectors/model.safetensors": {"w": ("BF16", [4, 4])},
            "text_encoder/model.safetensors": {"model.layers.0.self_attn.q_proj.weight": ("BF16", [2, 3]), "model.embed_tokens.weight": ("BF16", [5, 2])},
            "transformer/diffusion_pytorch_model.safetensors": {"transformer_blocks.0.attn.weight": ("BF16", [4, 4]), "proj_out.weight": ("BF16", [3, 2]), "scale": ("BF16", [7])},
            "vae/diffusion_pytorch_model.safetensors": {"w": ("BF16", [99])},
        }
        source = weight_totals_table.RowSource(_target(None).target, LTX_CUDA_REPO)

        row = weight_totals_table._ltx_cuda_repo(_FakeReader(tensors), source)  # type: ignore[arg-type]

        assert row["ltx_cuda"]["connectors"] == 16
        assert 16 <= row["ltx_cuda"]["text_encoder_bytes"] <= 32  # each element is one byte in FP8, two where the layer stays bfloat16
        assert row["transformers"] == {"transformer": {"block_matrices": 16, "transformer_rest": 13}}

    def test_a_repo_that_leaves_its_transformer_out_has_no_transformer_row(self) -> None:
        tensors = {
            "connectors/model.safetensors": {"w": ("BF16", [2, 2])},
            "text_encoder/model.safetensors": {"w": ("BF16", [2, 2])},
            "transformer/diffusion_pytorch_model.safetensors": {"w": ("BF16", [2, 2])},
        }
        source = weight_totals_table.RowSource(_target(None, ("transformer/*.safetensors",)).target, LTX_CUDA_REPO)

        row = weight_totals_table._ltx_cuda_repo(_FakeReader(tensors), source)  # type: ignore[arg-type]

        assert "transformers" not in row and "revision" not in row

    def test_a_single_file_checkpoint_counts_only_its_transformer(self) -> None:
        prefix = "model.diffusion_model."
        tensors = {
            "ltx.safetensors": {
                f"{prefix}transformer_blocks.0.w": ("BF16", [4, 4]),
                f"{prefix}proj_out.weight": ("BF16", [3, 2]),
                f"{prefix}video_embeddings_connector.w": ("BF16", [50, 50]),
                "vae.decoder.w": ("BF16", [99, 99]),
            }
        }
        source = weight_totals_table.RowSource(_target(("ltx.safetensors",)).target, LTX_TRANSFORMER_FILE, ("ltx.safetensors",))

        row = weight_totals_table._ltx_transformer_file(_FakeReader(tensors), source)  # type: ignore[arg-type]

        assert row["transformers"] == {"ltx.safetensors": {"block_matrices": 16, "transformer_rest": 6}}
