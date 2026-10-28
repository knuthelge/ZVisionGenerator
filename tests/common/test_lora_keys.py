"""Tests for renaming a LoRA's tensors to the names each loader reads for each model family."""

from __future__ import annotations

import re

import pytest

from zvisiongenerator.utils.lora_keys import LORA_MODULES, folds_alpha, normalise_lora_keys

_FAMILIES = sorted(LORA_MODULES)


def _loader(family: str | None) -> str:
    """Return the loader that reads *family*'s LoRAs on CUDA: its own for MOSS-Image and MiniMax H3, else diffusers."""
    return family if family in ("moss", "h3") else "diffusers"


def _normalise(names, family: str | None = None, loader: str | None = None) -> dict[str, str]:
    return normalise_lora_keys(names, family, loader=loader or _loader(family))


def _rename(key: str, family: str | None = None, loader: str | None = None) -> str:
    """Return *key* as the normaliser spells it (itself when it needs no change)."""
    return _normalise([key], family, loader).get(key, key)


def _fill(template: str, number: int = 7) -> str:
    return template.replace("{n}", str(number))


def _flat(template: str, number: int = 7) -> str:
    return "lora_unet_" + _fill(template, number).replace(".", "_")


class TestAdapterName:
    @pytest.mark.parametrize(("family", "adapter"), [("zimage", "default"), ("ltx", "lora_0"), ("h3", "lora_0"), (None, "default"), (None, "my-adapter")])
    def test_any_adapter_name_is_dropped(self, family, adapter):
        names = [f"diffusion_model.layers.3.proj.q.lora_A.{adapter}.weight", f"diffusion_model.layers.3.proj.q.lora_B.{adapter}.weight"]

        renames = _normalise(names, family)

        assert renames == {names[0]: "diffusion_model.layers.3.proj.q.lora_A.weight", names[1]: "diffusion_model.layers.3.proj.q.lora_B.weight"}

    def test_the_adapter_name_of_a_kohya_pair_is_dropped_too(self):
        assert _rename("diffusion_model.a.b.lora_down.default.weight") == "diffusion_model.a.b.lora_down.weight"

    def test_two_adapters_on_one_module_keep_their_names(self):
        names = ["m.lora_A.0.weight", "m.lora_A.1.weight", "m.lora_B.0.weight", "m.lora_B.1.weight"]

        assert _normalise(names, None) == {}

    def test_a_name_another_tensor_already_has_is_not_taken(self):
        names = ["a.lora_A.default.weight", "a.lora_A.weight"]

        assert _normalise(names, "moss") == {}

    @pytest.mark.parametrize("key", ["a.lora_A.weight", "a.lora_B.weight", "a.alpha", "a.weight", "a.lora.down.weight"])
    def test_a_plain_key_is_left_alone(self, key):
        assert _rename(key, "moss") == key


class TestMiniMaxH3:
    """diffusers' H3 converter reads a ``default`` adapter name as the mark of a DiffSynth file, whose fused qkv is laid out per head."""

    def test_the_default_adapter_name_stays(self):
        assert _normalise(["blocks.0.attn.qkv_proj.lora_A.default.weight"], "h3") == {}

    def test_kohya_names_are_left_to_the_converter(self):
        assert _normalise(["lora_unet_blocks_0_attn_qkv_proj.lora_down.weight"], "h3") == {}


class TestKohyaNames:
    @pytest.mark.parametrize(
        ("flat", "dotted"),
        [
            ("blocks_3_attn_wq", "blocks.3.attn.wq"),
            ("tmlp_0", "tmlp.0"),  # the first time-embedding layer, not block 0 of a ``tmlp`` list
        ],
    )
    def test_krea2_module_names_keep_their_underscores(self, flat, dotted):
        names = [f"lora_unet_{flat}.lora_down.weight", f"lora_unet_{flat}.lora_up.weight", f"lora_unet_{flat}.alpha"]

        assert _normalise(names, "krea2") == {
            names[0]: f"diffusion_model.{dotted}.lora_A.weight",
            names[1]: f"diffusion_model.{dotted}.lora_B.weight",
            names[2]: f"diffusion_model.{dotted}.alpha",
        }

    @pytest.mark.parametrize(
        ("flat", "dotted"),
        [
            ("layers_4_attention_to_q", "layers.4.attention.to_q"),
            ("layers_4_attention_to_out_0", "layers.4.attention.to_out.0"),
        ],
    )
    def test_zimage_names_are_dotted_for_mflux(self, flat, dotted):
        assert _rename(f"lora_unet_{flat}.lora_down.weight", "zimage", "mflux") == f"diffusion_model.{dotted}.lora_down.weight"

    def test_zimage_names_stay_flat_for_diffusers_whose_converter_reads_them(self):
        # diffusers' Z-Image converter drops the redundant fused qkv and out layers of a Kohya file by name; dotted names lose that.
        assert _normalise(["lora_unet_layers_4_attention_qkv.lora_down.weight", "lora_unet_layers_4_attention_out.alpha"], "zimage") == {}

    def test_ideogram4_names_are_dotted(self):
        assert _rename("lora_unet_layers_9_adaln_modulation.lora_down.weight", "ideogram4") == "diffusion_model.layers.9.adaln_modulation.lora_down.weight"
        assert _rename("lora_unet_layers_9_attention_o.alpha", "ideogram4") == "diffusion_model.layers.9.attention.o.alpha"

    def test_a_flat_name_that_names_no_module_of_the_family_is_left_alone(self):
        key = "lora_unet_blocks_3_attn_qkv_proj.lora_down.weight"  # an H3 module, not a Krea 2 one

        assert _normalise([key], "krea2") == {}

    def test_the_names_of_a_family_that_reads_them_stay_flat(self):
        assert _normalise(["lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"], "flux2_klein") == {}
        assert _normalise(["lora_unet_layers_0_proj_q.lora_down.weight"], "moss") == {}
        assert _normalise(["lora_unet_layers_0_attn_wq.lora_down.weight"], "krea2", "mflux") == {}

    def test_a_flat_name_the_tables_do_not_know_is_left_alone(self):
        assert normalise_lora_keys(["lora_unet_input_blocks_1_0_proj_in.lora_down.weight"]) == {}

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_every_module_of_a_family_survives_flattening(self, family):
        for template in LORA_MODULES[family]:
            for number in (0, 41):
                key = f"{_flat(template, number)}.lora_down.weight"
                expected = f"diffusion_model.{_fill(template, number)}.lora_down.weight"

                assert _rename(key, None) == expected

    def test_no_two_modules_flatten_to_the_same_name(self):
        seen: dict[str, str] = {}
        for templates in LORA_MODULES.values():
            for template in templates:
                flat = template.replace(".", "_")
                assert seen.setdefault(flat, template) == template, f"{template} and {seen[flat]} both flatten to {flat}"


class TestPrefixes:
    @pytest.mark.parametrize(
        ("family", "key", "expected"),
        [
            ("zimage", "layers.0.attention.to_q.lora_A.weight", "diffusion_model.layers.0.attention.to_q.lora_A.weight"),
            ("zimage", "noise_refiner.0.attention.to_q.lora_A.weight", "diffusion_model.noise_refiner.0.attention.to_q.lora_A.weight"),
            ("ideogram4", "layers.0.attention.qkv.lora_A.weight", "diffusion_model.layers.0.attention.qkv.lora_A.weight"),
            ("flux2_klein", "transformer_blocks.0.attn.to_q.lora_A.weight", "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"),
            ("flux2_klein", "single_transformer_blocks.0.attn.to_out.lora_A.weight", "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight"),
            ("flux2_klein", "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight", "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"),
            ("krea2", "blocks.0.attn.wq.lora_A.weight", "diffusion_model.blocks.0.attn.wq.lora_A.weight"),
            ("krea2", "transformer_blocks.0.attn.to_q.lora_A.weight", "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"),
            ("ltx", "transformer_blocks.0.attn1.to_q.lora_A.weight", "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"),
        ],
    )
    def test_a_key_without_a_prefix_gets_the_one_the_converter_reads(self, family, key, expected):
        assert _rename(key, family) == expected

    @pytest.mark.parametrize(
        ("family", "key"),
        [
            ("zimage", "diffusion_model.layers.0.attention.to_q.lora_A.weight"),
            ("zimage", "transformer.layers.0.attention.to_q.lora_A.weight"),
            ("flux2_klein", "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight"),
            ("flux2_klein", "base_model.model.single_blocks.0.linear1.lora_A.weight"),
            ("flux2_klein", "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"),
            ("krea2", "base_model.model.blocks.0.attn.wq.lora_A.weight"),
            ("ltx", "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"),
            ("h3", "blocks.0.attn.qkv_proj.lora_A.weight"),
            ("moss", "layers.0.proj.q.lora_A.weight"),
        ],
    )
    def test_a_key_the_converter_already_reads_is_left_alone(self, family, key):
        assert _normalise([key], family) == {}

    def test_a_module_both_namespaces_name_goes_where_the_rest_of_the_file_goes(self):
        original = ["base_model.model.single_blocks.0.linear1.lora_A.weight", "base_model.model.double_stream_modulation_img.lin.lora_A.weight"]
        diffusers = ["base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight", "base_model.model.double_stream_modulation_img.linear.lora_A.weight"]

        assert _normalise(original, "flux2_klein") == {}
        assert set(_normalise(diffusers, "flux2_klein").values()) == {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight", "transformer.double_stream_modulation_img.linear.lora_A.weight"}

    def test_a_file_in_both_namespaces_is_left_alone(self):
        names = ["single_blocks.0.linear1.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_A.weight"]

        assert _normalise(names, "flux2_klein") == {}

    def test_the_family_agnostic_view_keeps_prefixes(self):
        assert normalise_lora_keys(["layers.0.attention.to_q.lora_A.weight"]) == {}

    def test_a_prefix_is_added_to_a_whole_module_alike(self):
        names = ["layers.0.attention.to_q.lora_A.default.weight", "layers.0.attention.to_q.lora_B.default.weight", "layers.0.attention.to_q.alpha"]

        assert set(_normalise(names, "zimage").values()) == {
            "diffusion_model.layers.0.attention.to_q.lora_A.weight",
            "diffusion_model.layers.0.attention.to_q.lora_B.weight",
            "diffusion_model.layers.0.attention.to_q.alpha",
        }


class TestKrea2Spelling:
    def test_down_and_up_are_written_as_a_and_b(self):
        names = ["diffusion_model.blocks.0.attn.wq.lora_down.weight", "diffusion_model.blocks.0.attn.wq.lora_up.weight"]

        assert _normalise(names, "krea2") == {names[0]: "diffusion_model.blocks.0.attn.wq.lora_A.weight", names[1]: "diffusion_model.blocks.0.attn.wq.lora_B.weight"}

    @pytest.mark.parametrize("family", ["zimage", "ideogram4", "moss", "ltx", "h3"])
    def test_other_families_keep_down_and_up(self, family):
        assert _normalise(["diffusion_model.layers.0.attention.o.lora_down.weight"], family) == {}

    def test_only_krea2_folds_alpha_on_diffusers(self):
        assert folds_alpha("krea2")
        assert not any(folds_alpha(family) for family in ("zimage", "flux2_klein", "ideogram4", "ltx", "moss", "h3"))
        assert not folds_alpha("krea2", loader="mflux")


class TestMacOSLoaders:
    """mflux and MLX read only files; a file they already match is never renamed, so it is never copied."""

    @pytest.mark.parametrize(
        ("family", "key"),
        [
            ("zimage", "layers.0.attention.to_q.lora_A.default.weight"),
            ("zimage", "diffusion_model.layers.0.attention.to_q.lora_A.default.weight"),
            ("flux2_klein", "transformer_blocks.0.attn.to_q.lora_A.default.weight"),
            ("flux2_klein", "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight"),
            ("krea2", "blocks.0.attn.wq.lora_A.default.weight"),
            ("krea2", "lora_unet_blocks_0_attn_wq.lora_down.weight"),
            ("krea2", "lora_unet_blocks_0_attn_wq.alpha"),
        ],
    )
    def test_mflux_keeps_what_it_reads(self, family, key):
        assert _normalise([key], family, "mflux") == {}

    def test_mflux_ideogram4_reads_no_adapter_name_and_no_bare_key(self):
        assert _rename("layers.0.attention.qkv.lora_A.default.weight", "ideogram4", "mflux") == "diffusion_model.layers.0.attention.qkv.lora_A.weight"

    def test_mflux_klein_reads_an_adapter_name_on_diffusers_names_only(self):
        assert _normalise(["transformer_blocks.0.attn.to_q.lora_A.default.weight"], "flux2_klein", "mflux") == {}
        assert _rename("base_model.model.single_blocks.0.linear1.lora_A.default.weight", "flux2_klein", "mflux") == "base_model.model.single_blocks.0.linear1.lora_A.weight"

    def test_mflux_klein_reads_a_peft_prefix_only_on_original_names(self):
        assert _rename("base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight", "flux2_klein", "mflux") == "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
        assert _normalise(["base_model.model.single_blocks.0.linear1.lora_A.weight"], "flux2_klein", "mflux") == {}

    def test_mlx_reads_no_adapter_name_but_reads_a_bare_key(self):
        assert _rename("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.default.weight", "ltx", "mlx") == "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"
        assert _rename("transformer_blocks.0.attn1.to_q.lora_A.default.weight", "ltx", "mlx") == "transformer_blocks.0.attn1.to_q.lora_A.weight"


class TestLtxNamespace:
    _BLOCK = "transformer_blocks.0.attn1.to_q"

    def test_a_peft_file_of_blocks_alone_goes_to_the_trainer_prefix(self):
        keys = [f"base_model.model.{self._BLOCK}.lora_A.default.weight", f"base_model.model.{self._BLOCK}.lora_B.default.weight"]

        assert _normalise(keys, "ltx") == {keys[0]: f"diffusion_model.{self._BLOCK}.lora_A.weight", keys[1]: f"diffusion_model.{self._BLOCK}.lora_B.weight"}

    def test_a_peft_file_with_the_diffusers_layers_goes_to_the_diffusers_prefix(self):
        keys = [f"base_model.model.{self._BLOCK}.lora_A.default.weight", "base_model.model.time_embed.linear.lora_A.default.weight", "base_model.model.proj_in.lora_A.default.weight"]

        assert set(_normalise(keys, "ltx").values()) == {f"transformer.{self._BLOCK}.lora_A.weight", "transformer.time_embed.linear.lora_A.weight", "transformer.proj_in.lora_A.weight"}

    def test_a_peft_file_with_the_trainers_layers_goes_to_the_trainer_prefix(self):
        keys = [f"base_model.model.{self._BLOCK}.lora_A.weight", "base_model.model.patchify_proj.lora_A.weight", "base_model.model.adaln_single.linear.lora_A.weight"]

        assert set(_normalise(keys, "ltx").values()) == {
            f"diffusion_model.{self._BLOCK}.lora_A.weight",
            "diffusion_model.patchify_proj.lora_A.weight",
            "diffusion_model.adaln_single.linear.lora_A.weight",
        }

    @pytest.mark.parametrize("root", ["patchify_proj", "adaln_single.linear"])
    def test_every_layer_of_a_bare_file_with_the_trainers_names_gets_the_one_prefix(self, root):
        keys = [f"{self._BLOCK}.lora_A.weight", f"{root}.lora_A.weight", "patchify_proj.lora_A.weight"]

        assert set(_normalise(keys, "ltx").values()) == {f"diffusion_model.{self._BLOCK}.lora_A.weight", f"diffusion_model.{root}.lora_A.weight", "diffusion_model.patchify_proj.lora_A.weight"}

    @pytest.mark.parametrize("root", ["proj_in", "time_embed.linear"])
    def test_every_layer_of_a_bare_file_with_the_diffusers_names_gets_the_one_prefix(self, root):
        keys = [f"{self._BLOCK}.lora_A.weight", f"{root}.lora_A.weight", "proj_out.lora_A.weight"]

        assert set(_normalise(keys, "ltx").values()) == {f"transformer.{self._BLOCK}.lora_A.weight", f"transformer.{root}.lora_A.weight", "transformer.proj_out.lora_A.weight"}

    def test_a_file_that_already_has_a_prefix_keeps_it(self):
        keys = [f"diffusion_model.{self._BLOCK}.lora_A.weight", "diffusion_model.patchify_proj.lora_A.weight", "transformer.time_embed.linear.lora_A.weight"]

        assert _normalise(keys, "ltx") == {}

    def test_a_file_that_names_both_namespaces_is_left_alone(self):
        keys = ["patchify_proj.lora_A.weight", "time_embed.linear.lora_A.weight", f"{self._BLOCK}.lora_A.weight"]

        assert _normalise(keys, "ltx") == {}

    def test_a_layer_neither_namespace_names_is_left_without_a_prefix(self):
        keys = [f"{self._BLOCK}.lora_A.weight", "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.lora_A.weight"]

        assert _normalise(keys, "ltx") == {keys[0]: f"diffusion_model.{self._BLOCK}.lora_A.weight"}

    def test_the_mlx_loader_gets_no_prefix_added(self):
        assert _normalise([f"{self._BLOCK}.lora_A.weight", "patchify_proj.lora_A.weight"], "ltx", "mlx") == {}


class TestLtxOutputProjection:
    def test_a_bare_to_out_is_written_with_its_index_for_diffusers(self):
        assert _rename("diffusion_model.transformer_blocks.3.audio_attn1.to_out.lora_A.weight", "ltx") == "diffusion_model.transformer_blocks.3.audio_attn1.to_out.0.lora_A.weight"

    def test_an_indexed_to_out_stays(self):
        assert _normalise(["diffusion_model.transformer_blocks.3.attn1.to_out.0.lora_A.weight"], "ltx") == {}

    def test_mlx_keeps_the_bare_name_its_model_has(self):
        assert _normalise(["diffusion_model.transformer_blocks.3.attn1.to_out.lora_A.weight"], "ltx", "mlx") == {}


class TestWhatIsLeftAlone:
    @pytest.mark.parametrize("family", ["zimage", "ltx"])
    def test_a_file_with_lycoris_tensors_keeps_every_name(self, family):
        names = ["layers.0.attention.to_q.lokr_w1", "layers.0.attention.to_q.lokr_w2", "layers.1.attention.to_q.lora_A.default.weight", "layers.1.attention.to_q.lora_B.default.weight"]

        assert _normalise(names, family) == {}

    def test_an_unknown_family_changes_nothing(self):
        assert _normalise(["a.lora_A.default.weight"], "sdxl") == {}

    def test_a_key_that_is_not_a_lora_tensor_is_left_alone(self):
        assert _normalise(["layers.0.norm.weight", "rh_aux_data_v4"], "zimage") == {}

    @pytest.mark.parametrize("family", [*_FAMILIES, None])
    def test_a_normalised_file_needs_no_second_pass(self, family):
        templates = LORA_MODULES.get(family or "krea2", ())
        keys = [_flat(template) + suffix for template in templates for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")]
        keys += [re.sub(r"\.lora_A\.weight$", ".lora_A.default.weight", f"{_fill(template)}.lora_A.weight") for template in templates]

        once = _normalise(keys, family)
        again = _normalise([once.get(key, key) for key in keys], family)

        assert again == {}
