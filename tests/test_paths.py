"""Tests for zvisiongenerator.utils.paths — get_ziv_data_dir, resolve_model_path, resolve_lora_path."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.paths import (
    display_basename,
    display_stem,
    get_ziv_data_dir,
    is_explicit_local_path,
    is_huggingface_repo_id,
    is_remote_lora_reference,
    parse_huggingface_repo_reference,
    resolve_lora_path,
    resolve_model_path,
)


class TestPathClassification:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("models/foo", True),
            ("models/ltx-mlx", True),
            ("checkpoints/ltx", True),
            ("loras/style.safetensors", True),
            ("owner/repo", False),
            ("owner/repo@revision", False),
            ("owner/repo/subfolder", True),
            ("C:/repo", True),
            (r"C:\repo", True),
            ("C:repo", True),
            ("/abs/model", True),
            ("./models/foo", True),
            ("../models/foo", True),
            ("~/models/foo", True),
            ("//server/share/model", True),
            (r"\\server\share\model", True),
            ("models/foo/", True),
            ("models//foo", True),
            ("owner//repo", True),
            ("models/my model", True),
            ("my model", False),
            ("owner/my model", True),
            ("https://huggingface.co/owner/repo", False),
            ("hf://owner/repo", False),
        ],
    )
    def test_explicit_local_path_table(self, value, expected):
        assert is_explicit_local_path(value) is expected

    @pytest.mark.parametrize(
        ("value", "repo_id", "revision"),
        [
            ("owner/repo", "owner/repo", None),
            ("owner-name/repo_name", "owner-name/repo_name", None),
            ("owner.name/repo.v1", "owner.name/repo.v1", None),
            ("Owner123/Repo-2", "Owner123/Repo-2", None),
            ("owner/repo@main", "owner/repo", "main"),
            ("owner/repo@v1.2.3", "owner/repo", "v1.2.3"),
        ],
    )
    def test_huggingface_repo_reference_accepts_conservative_contract(self, value, repo_id, revision):
        parsed = parse_huggingface_repo_reference(value)
        assert parsed is not None
        assert parsed.repo_id == repo_id
        assert parsed.revision == revision
        assert is_huggingface_repo_id(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "models/foo",
            "checkpoints/ltx",
            "loras/style",
            "owner/repo/subfolder",
            "owner//repo",
            "/owner/repo",
            "./owner/repo",
            "../owner/repo",
            "~/owner/repo",
            "C:/repo",
            r"C:\repo",
            "C:repo",
            r"\\server\share",
            "owner/re po",
            "owner/repo?x=1",
            "owner/repo@",
            "owner/repo@feature/branch",
            "https://huggingface.co/owner/repo",
        ],
    )
    def test_huggingface_repo_reference_rejects_non_contract_values(self, value):
        assert parse_huggingface_repo_reference(value) is None
        assert is_huggingface_repo_id(value) is False

    def test_remote_lora_detection_uses_repo_reference_contract(self):
        assert is_remote_lora_reference("org/lora") is True
        assert is_remote_lora_reference("loras/style.safetensors") is False

    @pytest.mark.parametrize(
        ("value", "basename", "stem"),
        [
            ("/models/style.SAFETENSORS", "style.SAFETENSORS", "style"),
            (r"C:\models\model.fp16.safetensors", "model.fp16.safetensors", "model.fp16"),
            ("C:/models/foo.ckpt", "foo.ckpt", "foo"),
            ("owner/model.v1", "model.v1", "model.v1"),
            ("owner/model.v1.safetensors", "model.v1.safetensors", "model.v1"),
            ("model.safetensors.backup", "model.safetensors.backup", "model.safetensors.backup"),
            ("models/foo/", "foo", "foo"),
        ],
    )
    def test_display_helpers_are_cross_platform(self, value, basename, stem):
        assert display_basename(value) == basename
        assert display_stem(value) == stem


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

    @pytest.mark.parametrize("model_path", ["models/foo", "C:/models/foo", r"C:\models\foo", "owner/repo", "owner/repo@main"])
    def test_cross_platform_or_repo_strings_pass_through(self, monkeypatch, tmp_path, model_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        assert resolve_model_path(model_path) == model_path

    def test_tilde_local_model_expands_inside_resolver(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "ziv"))
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_model_path("~/models/foo") == str(tmp_path / "models" / "foo")

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

    def test_default_aliases_leave_unknown_name_unchanged(self, monkeypatch, tmp_path):
        """Calling without aliases leaves an unknown model name unchanged."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        result = resolve_model_path("some-model")
        assert result == "some-model"

    def test_empty_aliases_leave_unknown_name_unchanged(self, monkeypatch, tmp_path):
        """Empty aliases leave an unknown model name unchanged."""
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


class TestResolveModelAliasPlatformAware:
    def test_per_platform_alias_resolves_for_requested_platform(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)

        aliases = {
            "ltx-2.3": {
                "win32": "dg845/LTX-2.3-Diffusers",
                "linux": "dg845/LTX-2.3-Diffusers",
            }
        }

        result = resolve_model_path("ltx-2.3", aliases=aliases, platform_key="win32")

        assert result == "dg845/LTX-2.3-Diffusers"

    def test_per_platform_message_raises_value_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)

        aliases = {
            "ltx-4": {
                "darwin": "dgrauet/ltx-2.3-mlx-q4",
                "win32": {"message": "Alias 'ltx-4' is macOS-only. On Windows, use 'ltx-2.3' for the CUDA diffusers backend."},
            }
        }

        with pytest.raises(ValueError, match="macOS-only"):
            resolve_model_path("ltx-4", aliases=aliases, platform_key="win32")

    def test_platform_key_none_keeps_dict_alias_unresolved(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)

        aliases = {
            "ltx-2.3": {
                "win32": "dg845/LTX-2.3-Diffusers",
                "linux": "dg845/LTX-2.3-Diffusers",
            }
        }

        result = resolve_model_path("ltx-2.3", aliases=aliases, platform_key=None)

        assert result == "ltx-2.3"

    def test_flat_alias_still_resolves_when_platform_key_is_present(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)

        result = resolve_model_path("zit", aliases={"zit": "Tongyi-MAI/Z-Image-Turbo"}, platform_key="darwin")

        assert result == "Tongyi-MAI/Z-Image-Turbo"

    def test_ideogram4_alias_resolves_for_darwin_platform(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)

        aliases = load_config()["model_aliases"]

        result = resolve_model_path("ideo", aliases=aliases, platform_key="darwin")

        assert result == "ideogram-ai/ideogram-4-fp8"


class TestResolveLoraPath:
    def test_absolute_path_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        abs_path = "/some/lora/file.safetensors"
        assert resolve_lora_path(abs_path) == abs_path

    def test_path_with_slash_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        assert resolve_lora_path("some/dir/lora") == "some/dir/lora"

    @pytest.mark.parametrize("lora_path", ["loras/style.safetensors", "C:/loras/style.safetensors", r"C:\loras\style.safetensors", "org/lora"])
    def test_cross_platform_or_remote_strings_pass_through(self, monkeypatch, tmp_path, lora_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        assert resolve_lora_path(lora_path) == lora_path

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
