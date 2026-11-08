"""Pin the shape of the packaged model metadata: what every entry declares, which refs are pinned, and what its builders link."""

from __future__ import annotations

import re

import pytest

from tests._factories import video_config
from zvisiongenerator.downloads.declaration import companion_targets, declared_repo_ids, model_targets, role_targets
from zvisiongenerator.utils.ltx_facts import EULER_ANCESTRAL, LTX_SAMPLERS
from zvisiongenerator.utils.config import video_family
from zvisiongenerator.utils.h3_preset import h3_mlx_preset, h3_preset
from zvisiongenerator.utils.ideogram4_instant import LINKED_COMPONENTS as INSTANT_LINKED
from zvisiongenerator.utils.ideogram4_instant import LICENSE_FILES, TRANSFORMER_FORMATS
from zvisiongenerator.utils.model_metadata import PLATFORM_KEYS, builtin_aliases, family_block, load_model_metadata, model_entry, model_ids
from zvisiongenerator.utils.matte import matte_pins
from zvisiongenerator.utils.moss_build import LINKED_COMPONENTS as MOSS_LINKED
from zvisiongenerator.utils.seedvr2 import seedvr2_files, seedvr2_tiers
from zvisiongenerator.utils.video_model_detect import detect_video_model

_PLATFORMS = sorted(PLATFORM_KEYS)
_PIN = re.compile(r"[0-9a-f]{40}")
# The two SeedVR2 weight repos stay unpinned: their files are chosen by name, not by revision.
_SEEDVR2_REPOS = {"numz/SeedVR2_comfyUI", "AInVFX/SeedVR2_comfyUI"}
_LTX_IDS = [model_id for model_id in model_ids("video") if model_entry(model_id)["family"] == "ltx"]
_ROLES = {"model", "transformer", "latent_upsampler", "text_encoder", "lora", "components", "sft", "turbo_lora", "voice", "birefnet", "vitmatte"}


@pytest.mark.parametrize("platform_key", _PLATFORMS)
@pytest.mark.parametrize("model_id", model_ids())
def test_every_model_declares_downloads_on_every_platform(model_id: str, platform_key: str) -> None:
    pairs = model_targets(model_id, platform_key)

    assert pairs, f"{model_id} declares no downloads on {platform_key}"
    assert {role for role, _ in pairs} <= _ROLES


def test_every_target_is_pinned_to_a_commit_unless_the_loader_resolves_it() -> None:
    """Only an image target with ``files_from: loader`` (its loader resolves the revision) and the SeedVR2 repos may be unpinned."""
    unpinned: set[tuple[str, str]] = set()
    for model_id in model_ids():
        kind = model_entry(model_id)["kind"]
        for platform_key in _PLATFORMS:
            for _, target in model_targets(model_id, platform_key) or ():
                if target.revision is not None:
                    assert _PIN.fullmatch(target.revision), f"{model_id}: {target.repo_id}@{target.revision} is not a commit"
                elif not (kind == "image" and target.allow_patterns is None) and target.repo_id not in _SEEDVR2_REPOS:
                    unpinned.add((model_id, target.repo_id))

    assert not unpinned, f"unpinned targets that are not image targets with files_from: loader: {sorted(unpinned)}"


def test_every_distilled_lora_is_pinned_and_names_a_licence_record() -> None:
    metadata = load_model_metadata()
    blocks = [metadata["video_families"]["ltx"], *metadata["video_families"]["ltx"]["versions"].values()]

    for block in blocks:
        lora = block["distilled_lora"]
        assert _PIN.search(lora["ref"].rpartition("@")[2])
        assert lora["licence"] in metadata["licences"]


@pytest.mark.parametrize("model_id", model_ids("image") + model_ids("video"))
def test_every_alias_has_a_model_target_or_is_built(model_id: str) -> None:
    entry = model_entry(model_id)

    if "build" in entry:
        assert builtin_aliases()[model_id] == model_id
        return
    for platform_key in _PLATFORMS:
        own = role_targets(model_targets(model_id, platform_key), "model")
        assert own, f"{model_id} has no role: model target on {platform_key}"


@pytest.mark.parametrize("platform_key", _PLATFORMS)
def test_every_video_source_detects_to_its_family(platform_key: str) -> None:
    for model_id in model_ids("video"):
        source = builtin_aliases()[model_id]
        source = source[platform_key] if isinstance(source, dict) else source

        assert detect_video_model(source).family == model_entry(model_id)["family"], model_id


@pytest.mark.parametrize("model_id", [model_id for model_id in model_ids() if "family" in model_entry(model_id) and model_entry(model_id)["kind"] == "video"])
def test_every_video_family_and_version_is_defined(model_id: str) -> None:
    entry = model_entry(model_id)
    family = load_model_metadata()["video_families"][entry["family"]]

    assert entry.get("version") is None or entry["version"] in family["versions"]


@pytest.mark.parametrize("family", sorted(load_model_metadata()["video_families"]))
def test_every_video_family_describes_itself_with_the_shipped_settings(family: str) -> None:
    described = video_family(video_config(), family)
    rule = described.frame_rule

    assert described.default_frames is None or (described.default_frames - rule.offset) % rule.alignment == 0
    assert described.step_options is None or list(described.step_options) == sorted(set(described.step_options))
    assert described.max_steps is None or described.step_options is None or described.step_options[-1] <= described.max_steps
    assert described.carry_frames >= 1 and described.extend_frame_step >= 1


def test_every_ltx_version_names_a_sampler_the_backend_knows_and_a_valid_image_crf() -> None:
    for version, block in load_model_metadata()["video_families"]["ltx"]["versions"].items():
        merged = family_block("ltx", version)
        assert merged.get("sampler", next(iter(LTX_SAMPLERS))) in LTX_SAMPLERS, version
        assert 0 <= merged.get("image_crf", 0) <= 51, version
        assert block["model_type"], version
    family = family_block("ltx", None)
    assert family.get("sampler", next(iter(LTX_SAMPLERS))) in LTX_SAMPLERS
    assert 0 <= family.get("image_crf", 0) <= 51


def test_the_ltx_family_block_does_not_use_the_ltx_25_sampler() -> None:
    """The euler-ancestral sampler belongs to the LTX-2.5 versions; the family block every other version inherits must not set it."""
    assert family_block("ltx", None).get("sampler") != EULER_ANCESTRAL


@pytest.mark.parametrize("model_id", _LTX_IDS)
def test_the_ltx_mlx_pack_target_is_pinned_and_lists_its_files(model_id: str) -> None:
    (target,) = role_targets(model_targets(model_id, "darwin"), "model")

    assert target.revision is not None and _PIN.fullmatch(target.revision)
    assert target.allow_patterns is not None


@pytest.mark.parametrize("model_id", _LTX_IDS)
def test_every_ltx_model_quantizes_at_4_or_8_bits(model_id: str) -> None:
    assert model_entry(model_id)["quantize"] in (4, 8)


def test_every_ltx_block_names_a_converter_base_for_both_levels() -> None:
    metadata = load_model_metadata()

    for block in [metadata["video_families"]["ltx"], *metadata["video_families"]["ltx"]["versions"].values()]:
        assert set(block["converter_base"]) == {"q4", "q8"}


@pytest.mark.parametrize("platform_key", _PLATFORMS)
def test_the_single_file_ltx_companions_list_exactly_their_file(platform_key: str) -> None:
    metadata = load_model_metadata()

    for version in (None, *metadata["video_families"]["ltx"]["versions"]):
        for role, target in companion_targets("ltx", version, platform_key):
            if role in ("transformer", "latent_upsampler"):
                assert target.allow_patterns is not None and len(target.allow_patterns) == 1, (version, role)


@pytest.mark.parametrize("platform_key", _PLATFORMS)
def test_every_upscaler_declares_its_memory_and_its_two_files_in_order(platform_key: str) -> None:
    upscalers = model_ids("upscaler")

    assert [tier.model for tier in seedvr2_tiers(platform_key)] == list(upscalers)
    for model_id in upscalers:
        assert len(seedvr2_files(model_id, platform_key)) == 2, model_id  # the loaders read them by position: the transformer, then the VAE


def test_the_background_removal_pins_list_their_files() -> None:
    for pin in matte_pins():
        assert pin.files


def test_the_h3_presets_build_on_every_platform_that_declares_their_files() -> None:
    for platform_key in ("win32", "linux"):
        assert h3_preset(video_config(), platform_key).schedules.keys() == {4, 8}
    assert h3_mlx_preset().schedules.keys() == {4, 8}


@pytest.mark.parametrize("platform_key", _PLATFORMS)
def test_the_instant_components_cover_the_folders_the_build_links(platform_key: str) -> None:
    (components,) = role_targets(model_targets("ideo", platform_key), "components")

    assert {f"{name}/*" for name in INSTANT_LINKED} <= set(components.allow_patterns)


@pytest.mark.parametrize("model_id", ["moss", "moss-sft"])
def test_the_moss_sft_target_covers_the_folders_the_build_links(model_id: str) -> None:
    (sft,) = role_targets(model_targets(model_id, "linux"), "sft")

    assert {f"{name}/*" for name in MOSS_LINKED} <= set(sft.allow_patterns)


def test_the_moss_models_share_one_sft_target() -> None:
    assert role_targets(model_targets("moss", "darwin"), "sft") == role_targets(model_targets("moss-sft", "darwin"), "sft")


def test_converter_bases_name_models_that_declare_a_pack_on_macos() -> None:
    metadata = load_model_metadata()
    blocks = [metadata["video_families"]["ltx"], *metadata["video_families"]["ltx"]["versions"].values()]

    for block in blocks:
        for model_id in block["converter_base"].values():
            assert role_targets(model_targets(model_id, "darwin"), "model")


def test_a_repo_declared_by_a_model_and_its_companions_is_listed_once() -> None:
    assert declared_repo_ids("ltx-4", "win32").count("Lightricks/LTX-2.3") == 1


@pytest.mark.parametrize("platform_key", _PLATFORMS)
def test_ideo_declares_one_components_and_one_single_file_transformer_in_a_format_the_converter_writes(platform_key: str) -> None:
    pairs = model_targets("ideo", platform_key)
    (components,) = role_targets(pairs, "components")
    (transformer,) = role_targets(pairs, "transformer")

    assert model_entry("ideo")["transformer_format"][platform_key] in TRANSFORMER_FORMATS
    assert len(transformer.allow_patterns) == 1
    assert {"model_index.json", "transformer/config.json", *LICENSE_FILES} <= set(components.allow_patterns)
    assert {f"{name}/*" for name in INSTANT_LINKED} <= set(components.allow_patterns)
