"""Tests for LoRA family detection from synthetic safetensors headers."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.lora_family import LORA_FAMILY_VERSION, LoraFamily, detect_lora_family


def _entry(*shape: int) -> dict:
    return {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, 0]}


def _lora(base: str, in_features: int, out_features: int, rank: int = 8, layout: str = "peft") -> dict:
    """Return the two header entries of one LoRA module, shaped like the given trainer layout."""
    down, up = ("lora_A", "lora_B") if layout == "peft" else ("lora_down", "lora_up")
    return {f"{base}.{down}.weight": _entry(rank, in_features), f"{base}.{up}.weight": _entry(out_features, rank)}


def _lokr(base: str, w1: tuple[int, int], w2: tuple[int, int]) -> dict:
    return {f"{base}.lokr_w1": _entry(*w1), f"{base}.lokr_w2": _entry(*w2)}


def _detect(header: dict, metadata: dict | None = None) -> LoraFamily:
    return detect_lora_family(header, metadata or {})


def _double_single(linear1_out: int, hidden: int, prefix: str = "") -> dict:
    return {
        **_lora(f"{prefix}double_blocks.0.img_attn.qkv", hidden, hidden * 3),
        **_lora(f"{prefix}single_blocks.0.linear1", hidden, linear1_out),
    }


class TestMetadata:
    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            ({"ss_base_model_version": "ideogram4"}, ("ideogram4", None)),
            ({"release_base_model": "FLUX.2 klein_9b"}, ("flux2_klein", "9b")),
            ({"modelspec.architecture": "flux2-klein-9b"}, ("flux2_klein", "9b")),
            ({"ss_base_model_version": "klein_4b"}, ("flux2_klein", "4b")),
            ({"release_base_model": "klein-4b"}, ("flux2_klein", "4b")),
            ({"ss_base_model_version": "Krea2"}, ("krea2", None)),
            ({"modelspec.architecture": "zimage"}, ("zimage", None)),
            ({"ss_base_model_version": "Z-Image-Turbo"}, ("zimage", None)),
            ({"modelspec.architecture": "ltx2/lora"}, ("ltx", None)),
            ({"ss_base_model_version": "flux1_dev"}, ("flux1", None)),
            ({"ss_base_model_version": "flux1"}, ("flux1", None)),
            ({"release_base_model": "FLUX.1 [dev]"}, ("flux1", None)),
            ({"modelspec.architecture": "flux-1-dev/lora"}, ("flux1", None)),
            ({"ss_base_model_version": "flux2_klein_9b"}, ("flux2_klein", "9b")),
            ({"release_base_model": "FLUX.2 klein_4b"}, ("flux2_klein", "4b")),
        ],
    )
    def test_label_names_the_family(self, metadata, expected):
        assert _detect({}, metadata) == LoraFamily(expected[0], expected[1], "metadata")

    def test_unrecognised_label_falls_through_to_the_fingerprint(self):
        header = _lora("layers.0.adaln_modulation", 1152, 4608)

        assert _detect(header, {"ss_base_model_version": "sdxl_base_v1-0"}) == LoraFamily("ideogram4", None, "fingerprint")

    def test_metadata_beats_the_fingerprint(self):
        header = _lora("transformer_blocks.0.attn1.to_q", 4096, 4096)

        assert _detect(header, {"modelspec.architecture": "krea2"}) == LoraFamily("krea2", None, "metadata")

    def test_non_string_values_do_not_raise(self):
        assert _detect({}, {"ss_base_model_version": 3}) == LoraFamily(None, None, "none")  # type: ignore[dict-item]


class TestFingerprintDoubleSingleBlocks:
    def test_linear1_out_36864_is_klein_9b(self):
        assert _detect(_double_single(36864, 4096)) == LoraFamily("flux2_klein", "9b", "fingerprint")

    def test_linear1_out_27648_is_klein_4b(self):
        assert _detect(_double_single(27648, 3072)) == LoraFamily("flux2_klein", "4b", "fingerprint")

    def test_linear1_out_21504_is_flux1(self):
        assert _detect(_double_single(21504, 3072)) == LoraFamily("flux1", None, "fingerprint")

    def test_trainer_prefix_is_stripped(self):
        assert _detect(_double_single(27648, 3072, prefix="diffusion_model.")) == LoraFamily("flux2_klein", "4b", "fingerprint")

    def test_lora_down_up_layout_gives_the_same_dims(self):
        header = _lora("single_blocks.0.linear1", 3072, 21504, layout="kohya")
        header.update(_lora("double_blocks.0.img_attn.qkv", 3072, 9216, layout="kohya"))

        assert _detect(header) == LoraFamily("flux1", None, "fingerprint")

    def test_lokr_dims_are_the_kronecker_product(self):
        header = _lokr("single_blocks.0.linear1", (216, 24), (128, 128))  # out 27648, in 3072
        header.update(_lora("double_blocks.0.img_attn.qkv", 3072, 9216))

        assert _detect(header) == LoraFamily("flux2_klein", "4b", "fingerprint")

    def test_unknown_linear1_width_is_not_detected(self):
        assert _detect(_double_single(12345, 3072)) == LoraFamily(None, None, "none")

    def test_missing_linear1_is_not_detected(self):
        header = {**_lora("double_blocks.0.img_attn.qkv", 3072, 9216), **_lora("single_blocks.0.linear2", 3072, 3072)}

        assert _detect(header) == LoraFamily(None, None, "none")


class TestFingerprintKohyaUnderscoreBlocks:
    def test_flux1_single_blocks(self):
        header = _lora("lora_unet_single_blocks_0_linear1", 3072, 21504, layout="kohya")
        header.update(_lora("lora_unet_double_blocks_0_img_attn_qkv", 3072, 9216, layout="kohya"))

        assert _detect(header) == LoraFamily("flux1", None, "fingerprint")

    @pytest.mark.parametrize(("linear1_out", "hidden", "variant"), [(36864, 4096, "9b"), (27648, 3072, "4b")])
    def test_klein_single_blocks(self, linear1_out, hidden, variant):
        header = _lora("lora_unet_single_blocks_3_linear1", hidden, linear1_out, layout="kohya")
        header.update(_lora("lora_unet_double_blocks_3_img_attn_qkv", hidden, hidden * 3, layout="kohya"))

        assert _detect(header) == LoraFamily("flux2_klein", variant, "fingerprint")

    def test_text_encoder_keys_beside_the_blocks_are_ignored(self):
        header = _lora("lora_unet_single_blocks_0_linear1", 3072, 21504, layout="kohya")
        header.update(_lora("lora_unet_double_blocks_0_img_attn_qkv", 3072, 9216, layout="kohya"))
        header.update(_lora("lora_te1_text_model_encoder_layers_0_mlp_fc1", 768, 3072, layout="kohya"))

        assert _detect(header) == LoraFamily("flux1", None, "fingerprint")

    def test_a_single_block_without_double_blocks_is_not_detected(self):
        assert _detect(_lora("lora_unet_single_blocks_0_linear1", 3072, 21504, layout="kohya")) == LoraFamily(None, None, "none")


class TestFingerprintDiffusersBlocks:
    @staticmethod
    def _fused(out: int, hidden: int, prefix: str = "transformer.") -> dict:
        return {
            **_lora(f"{prefix}single_transformer_blocks.0.attn.to_qkv_mlp_proj", hidden, out),
            **_lora(f"{prefix}transformer_blocks.0.attn.to_q", hidden, hidden),
        }

    def test_fused_projection_out_36864_is_klein_9b(self):
        assert _detect(self._fused(36864, 4096)) == LoraFamily("flux2_klein", "9b", "fingerprint")

    def test_fused_projection_out_27648_is_klein_4b(self):
        assert _detect(self._fused(27648, 3072)) == LoraFamily("flux2_klein", "4b", "fingerprint")

    def test_separate_projections_with_proj_mlp_are_flux1(self):
        header = {
            **_lora("transformer.single_transformer_blocks.0.attn.to_q", 3072, 3072),
            **_lora("transformer.single_transformer_blocks.0.attn.to_k", 3072, 3072),
            **_lora("transformer.single_transformer_blocks.0.proj_mlp", 3072, 12288),
            **_lora("transformer.transformer_blocks.0.attn.to_q", 3072, 3072),
        }

        assert _detect(header) == LoraFamily("flux1", None, "fingerprint")

    def test_lora_down_up_layout_gives_the_same_answer(self):
        header = {
            **_lora("single_transformer_blocks.0.proj_mlp", 3072, 12288, layout="kohya"),
            **_lora("transformer_blocks.0.attn.to_q", 3072, 3072, layout="kohya"),
        }

        assert _detect(header) == LoraFamily("flux1", None, "fingerprint")

    def test_single_blocks_without_a_known_projection_are_not_detected(self):
        header = {**_lora("single_transformer_blocks.0.attn.to_q", 3072, 3072), **_lora("transformer_blocks.0.attn.to_q", 3072, 3072)}

        assert _detect(header) == LoraFamily(None, None, "none")

    def test_hidden_size_guard_rejects_a_mismatched_width(self):
        assert _detect(self._fused(36864, 3072)) == LoraFamily(None, None, "none")

    def test_ltx_style_header_still_classifies_as_ltx(self):
        header = {**_lora("transformer_blocks.0.attn1.to_q", 4096, 4096), **_lora("transformer_blocks.0.attn2.to_q", 4096, 4096)}

        assert _detect(header) == LoraFamily("ltx", None, "fingerprint")

    def test_krea_style_header_still_classifies_as_krea2(self):
        header = {**_lora("transformer_blocks.0.attn.to_gate", 6144, 6144), **_lora("text_fusion.0.proj", 6144, 6144)}

        assert _detect(header) == LoraFamily("krea2", None, "fingerprint")

    def test_transformer_blocks_alone_do_not_make_flux(self):
        header = {**_lora("transformer_blocks.0.attn.to_q", 3072, 3072), **_lora("transformer_blocks.0.ff.net.0", 3072, 12288)}

        assert _detect(header) == LoraFamily(None, None, "none")


class TestFingerprintOtherFamilies:
    def test_krea2_gate_and_text_fusion(self):
        header = {**_lora("blocks.0.attn.gate", 6144, 6144), **_lora("txtfusion.layers.0.proj", 6144, 6144)}

        assert _detect(header) == LoraFamily("krea2", None, "fingerprint")

    def test_krea2_diffusers_names(self):
        header = {**_lora("transformer_blocks.0.attn.to_gate", 6144, 6144), **_lora("text_fusion.0.proj", 6144, 6144)}

        assert _detect(header) == LoraFamily("krea2", None, "fingerprint")

    def test_gate_with_attn1_is_not_krea2(self):
        header = {**_lora("transformer_blocks.0.attn.gate", 4096, 4096), **_lora("transformer_blocks.0.attn1.to_q", 4096, 4096)}
        header.update(_lora("txtfusion.layers.0.proj", 4096, 4096))

        assert _detect(header) == LoraFamily("ltx", None, "fingerprint")

    @pytest.mark.parametrize("attn", ["attn1", "attn2"])
    def test_ltx_transformer_blocks_with_attention(self, attn):
        header = _lora(f"transformer_blocks.3.{attn}.to_q", 4096, 4096)

        assert _detect(header) == LoraFamily("ltx", None, "fingerprint")

    @pytest.mark.parametrize("refiner", ["noise_refiner", "context_refiner", "cap_embedder"])
    def test_zimage_refiner_keys(self, refiner):
        header = {**_lora("layers.0.attention.to_q", 3840, 3840), **_lora(f"{refiner}.0.linear", 3840, 3840)}

        assert _detect(header) == LoraFamily("zimage", None, "fingerprint")

    def test_zimage_adaLN_modulation(self):
        header = _lora("layers.0.adaLN_modulation.1", 256, 15360)
        header.update(_lora("layers.0.attention.to_q", 3840, 3840))

        assert _detect(header) == LoraFamily("zimage", None, "fingerprint")

    def test_ideogram4_lowercase_adaln_modulation_with_lokr(self):
        header = _lokr("layers.0.adaln_modulation", (1152, 64), (4, 72))  # in 64*72=4608, out 1152*4=4608

        assert _detect(header) == LoraFamily("ideogram4", None, "fingerprint")

    @pytest.mark.parametrize(
        ("width", "expected"),
        [(3840, LoraFamily("zimage", None, "fingerprint")), (4608, LoraFamily("ideogram4", None, "fingerprint")), (2048, LoraFamily(None, None, "none"))],
    )
    def test_attention_only_layers_are_told_apart_by_width(self, width, expected):
        header = {**_lora("layers.0.attention.to_q", width, width), **_lora("layers.0.feed_forward.w1", width, 4 * width)}

        assert _detect(header) == expected

    def test_prefixed_keys_are_stripped_before_matching(self):
        header = _lora("base_model.model.layers.0.adaln_modulation", 1152, 4608)

        assert _detect(header) == LoraFamily("ideogram4", None, "fingerprint")


class TestHiddenSizeGuard:
    def test_guard_rejects_a_fingerprint_without_the_hidden_size(self):
        header = _lora("transformer_blocks.0.attn1.to_q", 2048, 2048)

        assert _detect(header) == LoraFamily(None, None, "none")

    def test_factorized_lokr_has_no_dims_so_the_guard_rejects(self):
        header = {"transformer_blocks.0.attn1.to_q.lokr_w1_a": _entry(4, 4), "transformer_blocks.0.attn1.to_q.lokr_w2_a": _entry(4, 4)}

        assert _detect(header) == LoraFamily(None, None, "none")

    def test_guard_accepts_the_hidden_size_on_either_side(self):
        header = {**_lora("transformer_blocks.0.attn1.to_q", 4096, 64), **_lora("transformer_blocks.0.ff.net.0", 64, 99)}

        assert _detect(header).family == "ltx"

    def test_guard_does_not_apply_to_metadata(self):
        assert _detect(_lora("x.y", 8, 8), {"ss_base_model_version": "zimage"}) == LoraFamily("zimage", None, "metadata")

    def test_malformed_shapes_do_not_raise(self):
        header = {"transformer_blocks.0.attn1.to_q.lora_A.weight": {"dtype": "BF16"}, "transformer_blocks.0.attn1.to_q.lora_B.weight": _entry(4)}

        assert _detect(header) == LoraFamily(None, None, "none")


class TestUnknown:
    def test_empty_input_is_unknown(self):
        assert _detect({}) == LoraFamily(None, None, "none")

    def test_unrelated_keys_are_unknown(self):
        assert _detect(_lora("lora_unet_down_blocks_0_attentions_0", 320, 320)) == LoraFamily(None, None, "none")

    def test_result_is_immutable(self):
        with pytest.raises(AttributeError):
            _detect({}).family = "x"  # type: ignore[misc]

    def test_version_is_an_int(self):
        assert isinstance(LORA_FAMILY_VERSION, int)


class TestPrefixesAndPurity:
    @pytest.mark.parametrize("prefix", ["diffusion_model.", "transformer.", "model.diffusion_model.", "base_model.model."])
    def test_each_trainer_prefix_is_stripped_before_the_start_of_key_checks(self, prefix):
        header = _lora(f"{prefix}layers.0.adaln_modulation", 1152, 4608)

        assert _detect(header) == LoraFamily("ideogram4", None, "fingerprint")

    @pytest.mark.parametrize(
        ("header", "family"),
        [
            (_lora("layers.0.adaLN_modulation.1", 256, 1024), "zimage"),
            (_lora("layers.0.adaln_modulation", 256, 1024), "ideogram4"),
            (_lora("blocks.0.attn.gate", 512, 512) | _lora("txtfusion.0.proj", 512, 512), "krea2"),
            (_lora("transformer_blocks.0.attn2.to_q", 512, 512), "ltx"),
        ],
    )
    def test_guard_rejects_each_family_without_its_hidden_size(self, header, family):
        assert _detect(header) == LoraFamily(None, None, "none"), family

    @pytest.mark.parametrize(("linear1_out", "hidden"), [(36864, 3072), (27648, 4096), (21504, 4096)])
    def test_guard_rejects_double_single_blocks_with_a_mismatched_hidden_size(self, linear1_out, hidden):
        header = _lora("double_blocks.0.img_attn.qkv", 64, 64) | _lora("single_blocks.0.linear1", 64, linear1_out)

        assert _detect(header) == LoraFamily(None, None, "none")

    def test_importing_the_detector_does_not_import_torch(self):
        import subprocess
        import sys

        code = "import sys; import zvisiongenerator.utils.lora_family; sys.exit(1 if 'torch' in sys.modules else 0)"
        assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
