"""Pin the schema and the invariants of the bundled curated list, offline.

The list is data that people read and that downloads are verified against, so a duplicate name, a malformed commit or hash,
or a family the app does not run fails here. Whether the pinned
files still match the Hub is not checked here; ``make check-curated`` does that over the network.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from urllib.parse import urlsplit

import pytest

from tests._factories import load_packaged_config
from zvisiongenerator.utils.curated_loras import (
    LICENCES,
    LTX_VERSIONS,
    CURATED_FAMILIES,
    PREVIEW_HOSTS,
    SECTIONS,
    SIZE_VARIANTS,
    WORKFLOWS,
    CuratedLora,
    load_curated_loras,
)
from zvisiongenerator.web.workspace_contract import CANONICAL_WORKFLOW_VALUES

_ENTRIES = load_curated_loras()
_NON_COMMERCIAL_LICENCES = {"cc-by-nc-4.0", "flux-non-commercial", "flux-2-research"}
_GENERIC_NAMES = {"pytorch_lora_weights", "adapter_model", "lora", "model", "weights", "diffusion_pytorch_model", "model_lora"}
_PLATFORMS = {"darwin", "win32", "linux"}


def _ids(entries: tuple[CuratedLora, ...]) -> list[str]:
    return [entry.id for entry in entries]


class TestIdentity:
    def test_list_is_not_empty_and_covers_every_family(self):
        assert {entry.family for entry in _ENTRIES} == set(CURATED_FAMILIES)

    def test_every_ltx_version_has_entries(self):
        assert {entry.version for entry in _ENTRIES if entry.family == "ltx"} == set(LTX_VERSIONS["ltx"])

    def test_ids_are_unique(self):
        assert [key for key, count in Counter(_ids(_ENTRIES)).items() if count > 1] == []

    def test_local_names_are_unique(self):
        assert [key for key, count in Counter(entry.name for entry in _ENTRIES).items() if count > 1] == []

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_name_is_a_readable_slug_that_is_not_generic(self, entry: CuratedLora):
        assert re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", entry.name)
        assert 3 <= len(entry.name) <= 48
        assert entry.name not in _GENERIC_NAMES

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_id_is_a_stable_slug(self, entry: CuratedLora):
        assert re.fullmatch(r"[a-z0-9._-]+", entry.id)


class TestFamily:
    def test_every_family_is_a_model_family_of_the_packaged_config(self):
        config = load_packaged_config()
        known = (set(config["model_presets"]) | set(config["video_model_presets"])) - {"unknown"}
        assert set(CURATED_FAMILIES) <= known

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_size_and_version_are_ones_the_family_has(self, entry: CuratedLora):
        assert entry.family in CURATED_FAMILIES
        sizes = SIZE_VARIANTS.get(entry.family, ())
        versions = LTX_VERSIONS.get(entry.family, ())
        assert (entry.size_variant in sizes) if sizes else entry.size_variant is None
        assert (entry.version in versions) if versions else entry.version is None

    def test_moss_is_tested_on_windows_and_linux_only(self):
        moss = [entry for entry in _ENTRIES if entry.family == "moss"]
        assert moss
        assert all(set(entry.platforms) == {"win32", "linux"} for entry in moss)

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_platforms_are_known(self, entry: CuratedLora):
        assert entry.platforms
        assert set(entry.platforms) <= _PLATFORMS


class TestSource:
    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_revision_is_a_full_commit_and_sha256_is_a_full_hash(self, entry: CuratedLora):
        assert re.fullmatch(r"[0-9a-f]{40}", entry.revision)
        assert re.fullmatch(r"[0-9a-f]{64}", entry.sha256)

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_file_is_a_safetensors_path_inside_the_repo(self, entry: CuratedLora):
        assert entry.file.endswith(".safetensors")
        assert ".." not in Path(entry.file).parts
        assert not entry.file.startswith("/")

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_repo_is_an_owner_and_a_name(self, entry: CuratedLora):
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", entry.repo)
        assert entry.author == entry.repo.split("/")[0]

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_size_is_positive(self, entry: CuratedLora):
        assert entry.size_bytes > 0


class TestPresentation:
    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_texts_are_present_and_short(self, entry: CuratedLora):
        assert entry.display_name.strip()
        assert 0 < len(entry.description) <= 160
        assert entry.description == entry.description.strip()

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_style_or_tool_carries_the_matching_fields(self, entry: CuratedLora):
        assert entry.section in SECTIONS
        if entry.section == "tool":
            assert entry.what_it_does
            assert entry.workflow is None or entry.workflow in WORKFLOWS
        else:
            assert entry.what_it_does is None
            assert entry.workflow is None

    def test_workflow_ids_are_workspace_workflows(self):
        assert set(WORKFLOWS) <= set(CANONICAL_WORKFLOW_VALUES)

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_weight_is_a_sane_recommendation(self, entry: CuratedLora):
        assert 0 < entry.weight <= 2

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_downloads_is_a_count(self, entry: CuratedLora):
        assert entry.downloads >= 0

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_trigger_words_are_comma_separated_without_blanks_or_full_stops(self, entry: CuratedLora):
        if entry.trigger is None:
            return
        assert ".safetensors" not in entry.trigger
        assert "/" not in entry.trigger
        words = entry.trigger.split(", ")
        assert all(word and word == word.strip() and not word.endswith(".") for word in words)
        assert len({word.casefold() for word in words}) == len(words)


class TestPreviews:
    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_two_to_four_previews_on_the_hub(self, entry: CuratedLora):
        urls = entry.previews + entry.preview_videos
        assert 2 <= len(urls) <= 4
        assert len(set(urls)) == len(urls)
        for url in urls:
            parts = urlsplit(url)
            assert parts.scheme == "https"
            assert parts.netloc in PREVIEW_HOSTS

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_videos_only_stand_in_for_missing_images(self, entry: CuratedLora):
        if entry.preview_videos:
            assert len(entry.previews) < 2

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_hub_previews_are_pinned_to_the_revision_of_their_repo(self, entry: CuratedLora):
        for url in entry.previews + entry.preview_videos:
            parts = urlsplit(url)
            if parts.netloc == "huggingface.co":
                assert parts.path.startswith(f"/{entry.repo}/resolve/{entry.revision}/")


class TestLicence:
    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_licence_is_in_the_allowed_set(self, entry: CuratedLora):
        assert entry.licence in LICENCES

    @pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
    def test_non_commercial_flag_follows_the_licence(self, entry: CuratedLora):
        assert entry.non_commercial == (entry.licence in _NON_COMMERCIAL_LICENCES)
