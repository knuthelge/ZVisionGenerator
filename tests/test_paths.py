"""Tests for zvisiongenerator.utils.paths — get_ziv_data_dir, resolve_model_path, resolve_lora_path."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.paths import get_ziv_data_dir, resolve_lora_path, resolve_model_path


class TestGetZivDataDir:
    def test_default_returns_home_dot_ziv(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ZIV_DATA_DIR", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = get_ziv_data_dir()
        assert result == tmp_path / ".ziv"
        # Should also create subdirs
        assert (result / "models").is_dir()
        assert (result / "loras").is_dir()

    def test_env_var_overrides_default(self, monkeypatch, tmp_path):
        custom_dir = tmp_path / "custom_data"
        monkeypatch.setenv("ZIV_DATA_DIR", str(custom_dir))
        result = get_ziv_data_dir()
        assert result == custom_dir
        assert (result / "models").is_dir()
        assert (result / "loras").is_dir()

    def test_env_var_strips_whitespace(self, monkeypatch, tmp_path):
        custom_dir = tmp_path / "trimmed"
        monkeypatch.setenv("ZIV_DATA_DIR", f"  {custom_dir}  ")
        result = get_ziv_data_dir()
        assert result == custom_dir

    def test_empty_env_var_uses_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", "")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = get_ziv_data_dir()
        assert result == tmp_path / ".ziv"


class TestResolveModelPath:
    def test_absolute_path_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        abs_path = "/some/absolute/model/path"
        assert resolve_model_path(abs_path) == abs_path

    def test_path_with_slash_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        hf_id = "org/model-name"
        assert resolve_model_path(hf_id) == hf_id

    def test_bare_name_resolves_to_models_dir_when_exists(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        model_dir = tmp_path / "models" / "myModel"
        model_dir.mkdir(parents=True)
        result = resolve_model_path("myModel")
        assert result == str(model_dir)

    def test_bare_name_returns_as_is_when_no_local_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        result = resolve_model_path("nonexistent")
        assert result == "nonexistent"


class TestResolveModelAlias:
    """Tests for the model_aliases feature in resolve_model_path()."""

    def test_alias_resolves_to_target(self, monkeypatch, tmp_path):
        """Bare alias name returns the HF repo ID from aliases dict."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        result = resolve_model_path("ltx-4", aliases={"ltx-4": "dgrauet/ltx-2.3-mlx-q4"})
        assert result == "dgrauet/ltx-2.3-mlx-q4"

    def test_alias_resolves_zit(self, monkeypatch, tmp_path):
        """The 'zit' alias resolves to the Z-Image-Turbo HF repo."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        result = resolve_model_path("zit", aliases={"zit": "Tongyi-MAI/Z-Image-Turbo"})
        assert result == "Tongyi-MAI/Z-Image-Turbo"

    def test_local_dir_overrides_alias(self, monkeypatch, tmp_path):
        """When ~/.ziv/models/<alias>/ exists as a local dir, it takes priority over alias."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        local_dir = tmp_path / "models" / "ltx-4"
        local_dir.mkdir(parents=True)
        result = resolve_model_path("ltx-4", aliases={"ltx-4": "dgrauet/ltx-2.3-mlx-q4"})
        assert result == str(local_dir)

    def test_no_alias_match_returns_as_is(self, monkeypatch, tmp_path):
        """Unknown name not in aliases passes through unchanged."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        result = resolve_model_path("nonexistent", aliases={"ltx-4": "dgrauet/ltx-2.3-mlx-q4"})
        assert result == "nonexistent"

    def test_none_aliases_backward_compat(self, monkeypatch, tmp_path):
        """Calling without aliases kwarg works as before (backward compat)."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        result = resolve_model_path("some-model")
        assert result == "some-model"

    def test_empty_aliases_backward_compat(self, monkeypatch, tmp_path):
        """Empty dict aliases works as before (backward compat)."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        result = resolve_model_path("some-model", aliases={})
        assert result == "some-model"

    def test_absolute_path_skips_alias(self, monkeypatch, tmp_path):
        """Absolute path passes through even if it matches an alias key."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        abs_path = "/models/ltx-4"
        result = resolve_model_path(abs_path, aliases={"ltx-4": "dgrauet/ltx-2.3-mlx-q4"})
        assert result == abs_path

    def test_slash_name_skips_alias(self, monkeypatch, tmp_path):
        """HF repo ID format (contains '/') skips alias lookup."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        hf_id = "org/my-model"
        result = resolve_model_path(hf_id, aliases={"org/my-model": "should-not-match"})
        assert result == hf_id

    @pytest.mark.parametrize(
        "alias, target",
        [
            ("ltx-8", "dgrauet/ltx-2.3-mlx-q8"),
            ("ltx-4", "dgrauet/ltx-2.3-mlx-q4"),
            ("zit", "Tongyi-MAI/Z-Image-Turbo"),
            ("klein9b", "black-forest-labs/FLUX.2-klein-9B"),
            ("klein4b", "black-forest-labs/FLUX.2-klein-4B"),
        ],
    )
    def test_all_config_aliases_resolve(self, monkeypatch, tmp_path, alias, target):
        """All 5 configured aliases resolve to their expected targets."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        # Use the actual config aliases dict
        aliases = {
            "ltx-8": "dgrauet/ltx-2.3-mlx-q8",
            "ltx-4": "dgrauet/ltx-2.3-mlx-q4",
            "zit": "Tongyi-MAI/Z-Image-Turbo",
            "klein9b": "black-forest-labs/FLUX.2-klein-9B",
            "klein4b": "black-forest-labs/FLUX.2-klein-4B",
        }
        result = resolve_model_path(alias, aliases=aliases)
        assert result == target


class TestResolveLoraPath:
    def test_absolute_path_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        abs_path = "/some/lora/file.safetensors"
        assert resolve_lora_path(abs_path) == abs_path

    def test_path_with_slash_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        assert resolve_lora_path("some/dir/lora") == "some/dir/lora"

    def test_bare_name_resolves_safetensors(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        lora_file = tmp_path / "loras" / "style.safetensors"
        lora_file.parent.mkdir(parents=True, exist_ok=True)
        lora_file.touch()
        result = resolve_lora_path("style")
        assert result == str(lora_file)

    def test_bare_name_resolves_without_extension(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        lora_file = tmp_path / "loras" / "rawlora"
        lora_file.parent.mkdir(parents=True, exist_ok=True)
        lora_file.touch()
        result = resolve_lora_path("rawlora")
        assert result == str(lora_file)

    def test_bare_name_returns_as_is_when_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "loras").mkdir(parents=True, exist_ok=True)
        result = resolve_lora_path("missing")
        assert result == "missing"
