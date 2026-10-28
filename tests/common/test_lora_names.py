"""Tests for the local names downloaded LoRAs are saved under."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.lora_names import MAX_NAME_LENGTH, repo_slug, unique_name


class TestRepoSlug:
    @pytest.mark.parametrize(
        ("repo", "slug"),
        [
            ("renderartist/Technically-Color-Z-Image-Turbo", "technically_color_z_image_turbo"),
            ("owner/My.LoRA_v2", "my_lora_v2"),
            ("owner/--odd--", "odd"),
            ("owner/!!!", "lora"),
        ],
    )
    def test_names_the_repository_not_the_owner_or_file(self, repo, slug):
        assert repo_slug(repo) == slug

    def test_is_cut_to_the_length_limit_without_a_trailing_separator(self):
        slug = repo_slug("owner/" + "a" * 59 + "-" + "b" * 20)

        assert len(slug) <= MAX_NAME_LENGTH
        assert not slug.endswith("_")


class TestUniqueName:
    def test_keeps_a_free_name(self):
        assert unique_name("style", {"other"}) == "style"

    def test_counts_up_from_two(self):
        assert unique_name("style", {"style"}) == "style-2"
        assert unique_name("style", {"style", "style-2"}) == "style-3"

    def test_compares_without_regard_to_case(self):
        assert unique_name("Style", {"style"}) == "Style-2"
