"""Tests for the folder a conversion may write: names that are model aliases are refused."""

from __future__ import annotations

import pytest

from zvisiongenerator.converters.convert_target import checked_target


@pytest.mark.parametrize(("name", "config"), [("ltx-4", {}), ("zit", {}), ("mine", {"model_aliases": {"mine": "owner/repo"}})])
def test_a_built_in_or_user_alias_is_not_a_name_for_a_converted_model(tmp_path, name, config):
    with pytest.raises(ValueError, match="model alias"):
        checked_target(tmp_path, name, force=False, config=config)


def test_any_other_name_is_a_folder_in_the_models_directory(tmp_path):
    assert checked_target(tmp_path, "my-conversion", force=False, config={}) == tmp_path / "my-conversion"
