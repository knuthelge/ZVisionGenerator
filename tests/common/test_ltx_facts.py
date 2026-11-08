"""Tests for the LTX facts read from the packaged model metadata: versions, companion files, converter bases and load options."""

from __future__ import annotations

import pytest

from tests._factories import metadata_copy, use_model_metadata
from zvisiongenerator.downloads.target import HubFiles
from zvisiongenerator.utils.ltx_facts import (
    SEPARATE_TRANSFORMER_IGNORE,
    ltx_companion_file,
    ltx_converter_base,
    ltx_converter_repo_ignore,
    ltx_diffusers_quantize,
    ltx_distilled_lora_licence,
    ltx_distilled_lora_source,
    ltx_gemma_repo_id,
    ltx_load_options,
    ltx_version,
    ltx_version_for_type,
)
from zvisiongenerator.utils.model_metadata import alias_map, builtin_aliases, family_block, model_entry
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference, resolve_model_path

_LTX25_PIN = "unsloth/LTX-2.5-FP8@7b40724fc39bbb01d9b4d72638c0d263531c88c2"
_LTX25_UPSAMPLER = (_LTX25_PIN, "latent_upsampler/diffusion_pytorch_model.safetensors")
_LTX25_MAC_PIN = "MarceloFSO/ltx-2.5-mlx-q4-fast@281bf05bd23207d6feef708fc2e270f84ba992b5"
_LTX23_REPO = "diffusers/LTX-2.3-Distilled-Diffusers@432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
_LTX23_REPO_IGNORE = ("transformer/*.safetensors", "transformer/*.safetensors.index.json")
_LTX23_LIGHTRICKS = "Lightricks/LTX-2.3@c1521179de480464f366a98a9d9b2acc4cc097b9"
_LTX23_TRANSFORMER = (_LTX23_LIGHTRICKS, "ltx-2.3-22b-distilled-1.1.safetensors")
_LTX23_UPSAMPLER = (_LTX23_LIGHTRICKS, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
_GEMMA = "mlx-community/gemma-3-12b-it-4bit@86cc6a8dedbc456dd0e4af01a9d09f396f77e558"


class TestLtxCompanionFile:
    """The single-file transformer and latent upsampler a model loads on Windows and Linux come from the family's companions."""

    @pytest.mark.parametrize("platform_key", ["win32", "linux"])
    @pytest.mark.parametrize("name", [None, "ltx-4", "ltx-8", "unknown", "org/some-ltx-repo"])
    def test_ltx_2_3_models_share_the_family_files(self, name, platform_key):
        assert ltx_companion_file(name, platform_key, "transformer") == _LTX23_TRANSFORMER
        assert ltx_companion_file(name, platform_key, "latent_upsampler") == _LTX23_UPSAMPLER

    @pytest.mark.parametrize("platform_key", ["win32", "linux"])
    def test_ltx25_4_uses_its_repos_own_transformer_and_its_own_upsampler(self, platform_key):
        assert ltx_companion_file("ltx25-4", platform_key, "transformer") is None
        assert ltx_companion_file("ltx25-4", platform_key, "latent_upsampler") == _LTX25_UPSAMPLER

    def test_macos_declares_no_single_file_companions(self):
        for name in (None, "ltx-4", "ltx25-4"):
            assert ltx_companion_file(name, "darwin", "transformer") is None
            assert ltx_companion_file(name, "darwin", "latent_upsampler") is None


class TestLtxDiffusersQuantize:
    """The quantize level of the diffusers LTX backend is the model entry's."""

    def test_the_packaged_models_quantize_ltx_4_ltx_8_and_ltx25_4_only(self):
        assert [ltx_diffusers_quantize(name) for name in ("ltx-4", "ltx-8", "ltx25-4", "other", None)] == [4, 8, 4, None, None]

    def test_a_model_of_another_family_or_kind_has_none(self):
        assert ltx_diffusers_quantize("h3") is None
        assert ltx_diffusers_quantize("zit") is None

    @pytest.mark.parametrize("platform_key", ["win32", "linux"])
    def test_packaged_ltx_aliases_resolve_to_the_diffusers_repo_off_macos(self, platform_key):
        aliases = alias_map({})

        assert "ltx-2.3" not in aliases
        for alias in ("ltx-4", "ltx-8"):
            assert resolve_model_path(alias, aliases=aliases, platform_key=platform_key) == _LTX23_REPO


class TestLtxLoadOptions:
    """The load options both the CLI and the Web UI pass to a video backend."""

    def test_options_carry_the_level_and_the_transformer_source(self):
        options = ltx_load_options("ltx-4", "linux", upscale=False)

        assert options == {
            "quantize": 4,
            "transformer_source": _LTX23_TRANSFORMER,
            "model_name": "ltx-4",
            "repo_ignore": _LTX23_REPO_IGNORE,
            "mlx_packs": options["mlx_packs"],
            "mlx_text_encoder": _GEMMA,
        }

    def test_upscale_adds_the_upsampler(self):
        options = ltx_load_options("other", "linux", upscale=True)

        assert options["quantize"] is None
        assert options["upscale"] is True
        assert options["latent_upsampler"] == _LTX23_UPSAMPLER

    def test_ltx25_4_load_options_with_upscale(self):
        options = ltx_load_options("ltx25-4", "linux", upscale=True)

        assert options == {
            "quantize": 4,
            "transformer_source": None,
            "model_name": "ltx25-4",
            "repo_ignore": ("LTX-2.5-*.safetensors", "samples/*"),
            "sampler": "euler_ancestral",
            "image_crf": 18,
            "upscale": True,
            "latent_upsampler": _LTX25_UPSAMPLER,
            "mlx_packs": options["mlx_packs"],
        }

    def test_ltx25_4_load_options_without_upscale_have_no_upsampler(self):
        options = ltx_load_options("ltx25-4", "linux", upscale=False)

        assert "upscale" not in options and "latent_upsampler" not in options
        assert options["transformer_source"] is None

    @pytest.mark.parametrize("name", ["ltx-4", "ltx-8"])
    def test_ltx_2_3_load_options_carry_no_version_key(self, name):
        """2.3 guard: the options of ``ltx-4`` and ``ltx-8`` carry none of the version-only keys."""
        plain = ltx_load_options(name, "linux", upscale=False)

        assert not {"sampler", "image_crf"} & set(plain)
        assert ltx_load_options(name, "linux", upscale=True) == {**plain, "upscale": True, "latent_upsampler": _LTX23_UPSAMPLER}

    def test_macos_options_carry_no_diffusers_companions_and_the_packs(self):
        options = ltx_load_options("ltx-4", "darwin", upscale=True)

        assert options["transformer_source"] is None and options["latent_upsampler"] is None
        assert set(options["mlx_packs"]) == {builtin_aliases()["ltx-4"]["darwin"], builtin_aliases()["ltx-8"]["darwin"], _LTX25_MAC_PIN}


class TestVersionOverlay:
    """R1: a version's block is overlaid on the family block of the models that name it."""

    @staticmethod
    def _serve(monkeypatch, tmp_path, **version):
        """Serve metadata with a ``v`` version of the ``ltx`` family holding *version*, and the model ``alias-v`` that names it."""
        data = metadata_copy()
        data["video_families"]["ltx"]["versions"]["v"] = {"model_type": "ltx-9", **version}
        data["models"]["alias-v"] = {"kind": "video", "family": "ltx", "version": "v", "name": "Alias", "downloads": {}}
        use_model_metadata(monkeypatch, tmp_path, data)

    def test_version_merges_nested_dicts_and_replaces_scalars_lists_and_none(self, monkeypatch, tmp_path):
        self._serve(monkeypatch, tmp_path, max_steps=4, label=None, converter_base={"q8": "ltx-4"}, step_options=[2], companions={"win32": []})

        preset = family_block("ltx", ltx_version("alias-v"))

        assert preset["max_steps"] == 4 and preset["label"] is None
        assert preset["converter_base"] == {"q4": "ltx-4", "q8": "ltx-4"}
        assert preset["step_options"] == [2]
        assert preset["companions"]["win32"] == [] and len(preset["companions"]["darwin"]) == 1
        assert "versions" not in preset and "model_type" not in preset

    @pytest.mark.parametrize("model_name", [None, "ltx-4", "unknown"])
    def test_an_unlisted_name_or_none_returns_the_family_block(self, monkeypatch, tmp_path, model_name):
        self._serve(monkeypatch, tmp_path, max_steps=4)

        preset = family_block("ltx", ltx_version(model_name))

        assert preset["max_steps"] == 8 and "versions" not in preset


class TestPackagedLtx25:
    """R2: the packaged ``ltx25-4`` entries, and the LTX-2.3 entries left as they were."""

    @pytest.mark.parametrize("platform_key", ["win32", "linux"])
    def test_ltx25_4_resolves_to_the_pinned_repo(self, platform_key):
        assert resolve_model_path("ltx25-4", aliases=alias_map({}), platform_key=platform_key) == _LTX25_PIN

    def test_ltx25_4_resolves_to_the_pinned_mirror_on_macos(self):
        assert resolve_model_path("ltx25-4", aliases=alias_map({}), platform_key="darwin") == _LTX25_MAC_PIN

    def test_the_cuda_upsampler_lives_in_the_models_own_repo(self):
        aliases = builtin_aliases()

        assert aliases["ltx25-4"]["win32"] == aliases["ltx25-4"]["linux"] == _LTX25_PIN
        assert aliases["ltx25-4"]["darwin"] == _LTX25_MAC_PIN
        assert ltx_companion_file("ltx25-4", "linux", "latent_upsampler")[0] == _LTX25_PIN


class TestLtxVersion:
    """R3: the version a model is, and the sources each version has."""

    def test_ltx_version_reports_version_membership(self):
        assert ltx_version("ltx25-4") == "ltx25"
        assert ltx_version("ltx-4") is None
        assert ltx_version(None) is None
        assert ltx_version("org/some-ltx-repo") is None

    def test_a_converted_ltx25_folder_takes_its_versions_settings(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils.ltx_converted import write_converted_ltx

        data = tmp_path / "data"
        for name, version in (("my25", "2.5"), ("my23", "2.3")):
            (data / "models" / name).mkdir(parents=True)
            write_converted_ltx(data / "models" / name, {"bits": 4, "storage": "nf4", "block_matrices": 1, "transformer_rest": 1}, version=version)
        monkeypatch.setattr("zvisiongenerator.utils.ltx_facts.get_ziv_data_dir", lambda: data)

        assert ltx_version("my25") == "ltx25"
        assert ltx_version(str(data / "models" / "my25")) == "ltx25"
        assert ltx_version("my23") is None
        assert ltx_load_options("my25", "linux", upscale=False)["sampler"] == "euler_ancestral"
        assert ltx_load_options("my25", "linux", upscale=False)["repo_ignore"] == ()  # LTX-2.5 declares no single-file transformer

    def test_the_converter_types_map_to_versions_with_their_own_sources(self):
        assert ltx_version_for_type("ltx-2.3") is None
        assert ltx_version_for_type("ltx-2.5") == "ltx25"
        repo, filename = ltx_distilled_lora_source("ltx25")
        assert repo.partition("@")[2] and "2.5" in filename  # pinned at a revision
        assert ltx_distilled_lora_source(None) != ltx_distilled_lora_source("ltx25")
        assert ltx_converter_base(4, "ltx25", "linux") != ltx_converter_base(4, None, "linux")
        assert ltx_converter_base(8, "ltx25", "darwin") != ltx_converter_base(8, None, "darwin")
        with pytest.raises(ValueError, match="ltx-9"):
            ltx_version_for_type("ltx-9")

    def test_a_built_in_models_repo_ignore_is_the_ignore_of_its_target(self):
        assert ltx_load_options("ltx25-4", "linux", upscale=False)["repo_ignore"] == ("LTX-2.5-*.safetensors", "samples/*")
        assert ltx_load_options("ltx-4", "win32", upscale=False)["repo_ignore"] == ltx_load_options("ltx-8", "linux", upscale=True)["repo_ignore"] == _LTX23_REPO_IGNORE
        assert ltx_load_options("ltx-4", "darwin", upscale=False)["repo_ignore"] == ()

    @pytest.mark.parametrize("model_name", [None, "owner/ltx", "unknown"])
    def test_any_other_repo_skips_the_transformer_weights_a_single_file_transformer_replaces(self, model_name):
        assert ltx_load_options(model_name, "linux", upscale=False)["repo_ignore"] == SEPARATE_TRANSFORMER_IGNORE
        assert ltx_load_options(model_name, "darwin", upscale=False)["repo_ignore"] == ()

    def test_the_converter_base_skips_the_transformer_and_what_its_target_declares_beyond_it(self):
        assert ltx_converter_repo_ignore(4, None, "linux") == ltx_converter_repo_ignore(8, None, "win32") == _LTX23_REPO_IGNORE
        assert ltx_converter_repo_ignore(4, "ltx25", "linux") == (*SEPARATE_TRANSFORMER_IGNORE, "LTX-2.5-*.safetensors", "samples/*")
        assert ltx_converter_repo_ignore(8, None, "darwin") == SEPARATE_TRANSFORMER_IGNORE

    def test_the_distilled_lora_and_converter_bases_follow_the_version(self):
        assert ltx_distilled_lora_source() == (_LTX23_LIGHTRICKS, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
        assert ltx_distilled_lora_licence() == "ltx-2-community" and ltx_distilled_lora_licence("ltx25") == "ltx-2x-community"
        assert ltx_converter_base(4, None, "linux") == ltx_converter_base(8, None, "linux") == _LTX23_REPO
        assert ltx_converter_base(8, "ltx25", "darwin") == _LTX25_MAC_PIN


class TestLtxMlxPacks:
    """The pinned MLX packs and the Gemma 3 reference macOS LTX downloads and loads from."""

    def test_the_packaged_packs_are_pinned_with_their_files(self):
        packs = ltx_load_options("ltx-4", "darwin", upscale=False)["mlx_packs"]
        aliases = builtin_aliases()

        assert set(packs) == {aliases["ltx-4"]["darwin"], aliases["ltx-8"]["darwin"], aliases["ltx25-4"]["darwin"]}
        assert all(len(pack.revision) == 40 for pack in packs.values())
        for name in ("ltx-4", "ltx-8", "ltx25-4"):
            assert packs[aliases[name]["darwin"]].files == tuple(model_entry(name)["downloads"]["darwin"][0]["files"])

    def test_a_pack_is_pinned_to_the_revision_of_its_model_target(self):
        for pack in ltx_load_options(None, "darwin", upscale=False)["mlx_packs"].values():
            assert pack == HubFiles(pack.repo_id, pack.revision, pack.files)
            assert parse_huggingface_repo_reference(f"{pack.repo_id}@{pack.revision}").revision == pack.revision  # type: ignore[union-attr]

    def test_the_text_encoder_is_the_pinned_gemma_3_of_the_2_3_family_and_the_2_5_packs_bundle_theirs(self):
        assert ltx_load_options("ltx-4", "darwin", upscale=False)["mlx_text_encoder"] == _GEMMA
        assert "mlx_text_encoder" not in ltx_load_options("ltx25-4", "darwin", upscale=False)
        assert ltx_gemma_repo_id() == "mlx-community/gemma-3-12b-it-4bit"

    def test_the_macos_aliases_are_the_pinned_bases(self):
        """The aliases are the packs' references, so an alias, the plan and the run cannot name different revisions."""
        aliases = builtin_aliases()
        packs = ltx_load_options(None, "darwin", upscale=False)["mlx_packs"]

        assert ltx_converter_base(4, None, "darwin") == aliases["ltx-4"]["darwin"]
        assert ltx_converter_base(8, None, "darwin") == aliases["ltx-8"]["darwin"]
        assert ltx_converter_base(4, "ltx25", "darwin") == aliases["ltx25-4"]["darwin"]
        assert all(aliases[name]["darwin"] in packs for name in ("ltx-4", "ltx-8", "ltx25-4"))
