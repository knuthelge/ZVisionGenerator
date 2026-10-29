"""Tests for zvisiongenerator.utils.paths — get_ziv_data_dir, resolve_model_path, resolve_lora_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from zvisiongenerator.utils.paths import (
    canonical_model_name,
    collapse_home,
    display_basename,
    display_stem,
    expand_home,
    get_ziv_data_dir,
    is_existing_file,
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
            ("owner/repo", False),
            ("owner/repo@revision", False),
            ("owner/repo/subfolder", True),
            ("C:/repo", True),
            ("C:repo", True),
            ("/abs/model", True),
            ("./models/foo", True),
            ("~/models/foo", True),
            (r"\\server\share\model", True),
            ("models/foo/", True),
            ("owner//repo", True),
            ("my model", False),
            ("owner/my model", True),
            ("https://huggingface.co/owner/repo", False),
        ],
    )
    def test_explicit_local_path_table(self, value, expected):
        assert is_explicit_local_path(value) is expected

    @pytest.mark.parametrize(
        ("value", "repo_id", "revision"),
        [
            ("owner/repo", "owner/repo", None),
            ("owner.name/repo.v1", "owner.name/repo.v1", None),
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
            "owner/repo/subfolder",
            "owner//repo",
            "/owner/repo",
            "C:repo",
            r"\\server\share",
            "owner/re po",
            "owner/repo@",
            "owner/repo@feature/branch",
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

    def test_empty_env_var_uses_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", "")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = get_ziv_data_dir()
        assert result == tmp_path / ".ziv"


class TestResolveModelPath:
    @pytest.mark.parametrize("model_path", ["/some/absolute/model/path", "org/model-name", "C:/models/foo", r"C:\models\foo", "owner/repo@main"])
    def test_paths_and_repo_strings_pass_through(self, monkeypatch, tmp_path, model_path):
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

    def test_local_dir_overrides_alias(self, monkeypatch, tmp_path):
        """When ~/.ziv/models/<alias>/ exists as a local dir, it takes priority over alias."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        local_dir = tmp_path / "models" / "ltx-4"
        local_dir.mkdir(parents=True)
        result = resolve_model_path("ltx-4", aliases={"ltx-4": "dgrauet/ltx-2.3-mlx-q4"})
        assert result == str(local_dir)

    @pytest.mark.parametrize("aliases", [{"ltx-4": "dgrauet/ltx-2.3-mlx-q4"}, {}, None], ids=["other-alias", "empty", "none"])
    def test_an_unknown_name_passes_through_unchanged(self, monkeypatch, tmp_path, aliases):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        assert resolve_model_path("nonexistent", aliases=aliases) == "nonexistent"

    def test_a_path_or_repo_id_skips_the_alias_lookup(self, monkeypatch, tmp_path):
        """An absolute path or an HF repo ID (contains '/') passes through even if it matches an alias key."""
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
        aliases = {"/models/ltx-4": "should-not-match", "org/my-model": "should-not-match"}
        assert resolve_model_path("/models/ltx-4", aliases=aliases) == "/models/ltx-4"
        assert resolve_model_path("org/my-model", aliases=aliases) == "org/my-model"


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

    def test_an_alias_to_a_bare_name_resolves_to_that_models_folder_before_it_exists(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))

        result = resolve_model_path("ideo", aliases={"ideo": "ideo"}, platform_key="darwin")

        assert result == str(tmp_path / "models" / "ideo")
        assert not (tmp_path / "models" / "ideo").exists()

    def test_a_platform_alias_to_a_bare_name_resolves_to_that_models_folder(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))

        result = resolve_model_path("built", aliases={"built": {"linux": "built-copy", "darwin": "owner/repo"}}, platform_key="linux")

        assert result == str(tmp_path / "models" / "built-copy")


class TestResolveLoraPath:
    @pytest.mark.parametrize("lora_path", ["/some/lora/file.safetensors", "some/dir/lora", "C:/loras/style.safetensors", r"C:\loras\style.safetensors", "org/lora"])
    def test_paths_and_remote_strings_pass_through(self, monkeypatch, tmp_path, lora_path):
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


class TestHomeCollapse:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("/home/me/.ziv/loras/style.safetensors", "~/.ziv/loras/style.safetensors"),
            ("/home/me", "~"),
            ("/home/meadow/ref.png", "/home/meadow/ref.png"),
            ("/srv/ref.png", "/srv/ref.png"),
            ("org/repo", "org/repo"),
        ],
    )
    def test_collapses_only_paths_inside_home(self, value, expected):
        assert collapse_home(value, home="/home/me/") == expected

    def test_collapses_windows_home(self):
        assert collapse_home("C:\\Users\\me\\ref.png", home="C:\\Users\\me") == "~\\ref.png"

    def test_expand_reverses_collapse(self, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/me")))

        assert expand_home(collapse_home("/home/me/ref.png")) == "/home/me/ref.png"
        assert expand_home("/srv/ref.png") == "/srv/ref.png"
        assert expand_home("~user/ref.png") == "~user/ref.png"


class TestSplitHuggingfaceRevision:
    def test_a_repo_with_a_revision_is_split(self):
        from zvisiongenerator.utils.paths import split_huggingface_revision

        assert split_huggingface_revision("owner/repo@abc123") == ("owner/repo", "abc123")

    @pytest.mark.parametrize("value", ["owner/repo", "/models/ltx", "./local@dir", "C:\\models\\x", "owner/repo@a@b", "single"])
    def test_anything_else_comes_back_unchanged(self, value):
        from zvisiongenerator.utils.paths import split_huggingface_revision

        assert split_huggingface_revision(value) == (value, None)


class TestCanonicalModelName:
    _PIN = "b" * 40
    _ALIASES = {
        "ltx-4": {"darwin": f"dgrauet/ltx-2.3-mlx-q4@{_PIN}", "win32": "org/diffusers", "linux": "org/diffusers"},
        "ltx-8": {"darwin": "dgrauet/ltx-2.3-mlx-q8", "win32": "org/diffusers", "linux": "org/diffusers"},
        "h3": f"org/h3@{_PIN}",
        "gated": {"darwin": {"message": "not on macOS"}},
        "folder": "ideo",
    }

    def _name(self, name, platform_key="darwin"):
        return canonical_model_name(name, aliases=self._ALIASES, platform_key=platform_key)  # type: ignore[arg-type]

    def test_a_repo_of_one_alias_is_that_alias(self):
        assert self._name("dgrauet/ltx-2.3-mlx-q4") == "ltx-4"
        assert self._name("dgrauet/ltx-2.3-mlx-q8") == "ltx-8"
        assert self._name("org/h3") == "h3"

    def test_the_repo_id_matches_in_any_letter_case(self):
        assert self._name("DGrauet/LTX-2.3-MLX-Q4") == "ltx-4"
        assert self._name(f"dgrauet/LTX-2.3-mlx-q4@{self._PIN}") == "ltx-4"
        assert self._name("ORG/H3") == "h3"

    def test_its_own_revision_is_accepted_and_another_is_not(self):
        assert self._name(f"dgrauet/ltx-2.3-mlx-q4@{self._PIN}") == "ltx-4"
        assert self._name("dgrauet/ltx-2.3-mlx-q4@other") == "dgrauet/ltx-2.3-mlx-q4@other"
        assert self._name(f"dgrauet/ltx-2.3-mlx-q8@{self._PIN}") == f"dgrauet/ltx-2.3-mlx-q8@{self._PIN}"  # that alias has no revision to match

    def test_a_repo_that_two_aliases_resolve_to_stays_a_repo(self):
        assert self._name("org/diffusers", "win32") == "org/diffusers"

    def test_the_platform_picks_the_aliases_repo(self):
        assert self._name("dgrauet/ltx-2.3-mlx-q4", "win32") == "dgrauet/ltx-2.3-mlx-q4"
        assert self._name("org/h3", "win32") == "h3"

    @pytest.mark.parametrize("name", ["ltx-4", "/models/ltx", "./local", "unknown/repo", "single", "ideo"])
    def test_anything_else_comes_back_unchanged(self, name):
        assert self._name(name) == name


class TestIsExistingFile:
    def test_a_file_is_one_and_a_folder_or_missing_path_is_not(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")

        assert is_existing_file(str(tmp_path / "a.png"))
        assert not is_existing_file(str(tmp_path))
        assert not is_existing_file(str(tmp_path / "gone.png"))

    def test_home_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "a.png").write_bytes(b"x")

        assert is_existing_file("~/a.png")

    def test_a_path_the_os_cannot_look_up_is_not_a_file(self):
        assert not is_existing_file("a\x00b.png")
