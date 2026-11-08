"""Tests for the MOSS-Image config entries, their variant capabilities and where they are read."""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests._factories import _make_args, _make_plan, metadata_copy, use_model_metadata
from tests._paths import REPO_ROOT

from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.config import get_variant_key, load_config, resolve_defaults
from zvisiongenerator.utils.model_metadata import model_capabilities
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.model_metadata import alias_map, image_model_family, model_download_size_label
from zvisiongenerator.utils.moss_build import moss_source
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.web import workspace_api as workspace_api_module

_MAKEFILE = REPO_ROOT / "Makefile"

_TURBO = ImageModelInfo(family="moss", is_distilled=True, size=None)
_SFT = ImageModelInfo(family="moss", is_distilled=False, size=None)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


def test_packaged_source_parses_and_matches_the_vendored_revision():
    source = moss_source("linux")
    makefile_revision = re.search(r"^MOSS_REVISION\s*\?=\s*(\w+)", _MAKEFILE.read_text(encoding="utf-8"), re.MULTILINE)

    assert makefile_revision is not None
    assert source.sft_revision == makefile_revision.group(1)


def test_aliases_name_their_folder_on_every_platform_and_are_not_declared_families():
    for alias in ("moss", "moss-sft"):
        assert alias_map({})[alias] == alias
        assert image_model_family(alias) is None
        assert all(model_download_size_label(alias, platform_key) for platform_key in ("darwin", "win32", "linux"))


@pytest.mark.parametrize(("info", "variant"), [(_TURBO, "turbo"), (_SFT, "sft")])
def test_variant_key_follows_is_distilled(info, variant):
    assert get_variant_key(info) == variant


def test_turbo_defaults(config):
    defaults = resolve_defaults(_TURBO, config, {}, "diffusers")

    assert defaults["steps"] == 4
    assert defaults["guidance"] == 1.0
    assert defaults["supports_negative_prompt"] is False
    assert defaults["supports_scheduler"] is False
    assert defaults["upscale_steps"] == 2
    assert defaults["scheduler"] is None


def test_sft_defaults(config):
    defaults = resolve_defaults(_SFT, config, {}, "diffusers")

    assert defaults["steps"] == 50
    assert defaults["guidance"] == 3.0
    assert defaults["supports_negative_prompt"] is True
    assert defaults["supports_scheduler"] is False
    assert defaults["upscale_steps"] == 12


def test_cli_flags_still_win_over_the_variant(config):
    defaults = resolve_defaults(_TURBO, config, {"steps": 8}, "diffusers")

    assert defaults["steps"] == 8


def test_variant_capability_overrides_only_the_flags_it_sets(monkeypatch, tmp_path):
    data = metadata_copy()
    data["image_families"]["fam"] = {"supports_negative_prompt": True, "supports_upscale": False, "variants": {"v": {"supports_negative_prompt": False}}}
    use_model_metadata(monkeypatch, tmp_path, data)

    assert model_capabilities("fam", "v")["supports_negative_prompt"] is False
    assert model_capabilities("fam", "v")["supports_upscale"] is False
    assert model_capabilities("fam")["supports_negative_prompt"] is True
    assert model_capabilities("fam", "missing")["supports_negative_prompt"] is True


def _web_config(app_config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        app_config=app_config,
        image_ratios=("2:3",),
        image_size_options={"2:3": ("m",)},
        quantize_options=(4, 8),
    )


def test_web_bootstrap_defaults_differ_between_turbo_and_sft(config, monkeypatch, tmp_path):
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "diffusers")
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda name, **kwargs: resolve_model_path(name, **{**kwargs, "platform_key": "linux"}))
    web_config = _web_config(config)

    turbo = workspace_api_module._build_image_bootstrap_defaults("moss", web_config)
    sft = workspace_api_module._build_image_bootstrap_defaults("moss-sft", web_config)

    assert (turbo["family"], sft["family"]) == ("moss", "moss")
    assert (turbo["steps"], turbo["guidance"], turbo["supports_negative_prompt"]) == (4, 1.0, False)
    assert (sft["steps"], sft["guidance"], sft["supports_negative_prompt"]) == (50, 3.0, True)
    assert turbo["supports_scheduler"] is False
    assert sft["supports_scheduler"] is False


def _run_with_negative(config: dict, info: ImageModelInfo, model: str = "test-model") -> ImageGenerationRequest:
    stage = MagicMock(return_value=StageOutcome.success)
    from zvisiongenerator.core.workflow import GenerationWorkflow

    workflow = GenerationWorkflow(name="test", stages=[stage])
    prompts = {"set1": [("a cat", "ugly, blurry")]}
    args = _make_args(model=model)
    runner_config = {
        **config,
        "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
        "generation": {**config["generation"], "seed_min": 1, "seed_max": 100},
    }
    with patch("zvisiongenerator.image_runner.build_workflow", return_value=workflow):
        run_batch(MagicMock(), MagicMock(spec=[]), prompts, runner_config, args, model_info=info, plan=_make_plan(prompts, args, config=runner_config))
    return stage.call_args[0][0]


def test_runner_suppresses_a_turbo_negative_and_keeps_an_sft_one(config):
    with pytest.warns(UserWarning):
        turbo_request = _run_with_negative(config, _TURBO)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sft_request = _run_with_negative(config, _SFT)

    assert turbo_request.supports_negative_prompt is False
    assert turbo_request.negative_prompt is None
    assert sft_request.supports_negative_prompt is True
    assert sft_request.negative_prompt == "ugly, blurry"


def test_a_turbo_run_drops_the_negative_and_warns_with_the_presets_hint_and_no_path(config, tmp_path):
    # The CLI replaces args.model with the resolved models-folder path before the run.
    resolved = tmp_path / "models" / "moss"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        request = _run_with_negative(config, _TURBO, str(resolved))

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == 1
    assert request.negative_prompt is None
    hint = model_capabilities("moss", "turbo")["negative_prompt_hint"]
    assert hint and hint in messages[0]
    assert "moss-sft" in hint and "moss-sft" in alias_map(config)
    assert model_capabilities("moss", "sft")["negative_prompt_hint"] is None
    assert "/" not in messages[0] and "\\" not in messages[0] and str(tmp_path) not in messages[0]


def test_a_family_without_variants_warns_without_a_hint(config):
    from zvisiongenerator.image_runner import _negative_prompt_warning

    message = _negative_prompt_warning(_make_args(model="zit"), ImageModelInfo(family="zimage", is_distilled=False, size=None))

    assert "zimage" in message
    assert model_capabilities("zimage")["negative_prompt_hint"] is None


def test_the_runner_adds_whatever_hint_the_preset_sets_and_nothing_else(monkeypatch, tmp_path):
    from zvisiongenerator.image_runner import _negative_prompt_warning

    data = metadata_copy()
    data["image_families"]["flux2_klein"]["supports_negative_prompt"] = True
    data["image_families"]["flux2_klein"]["variants"] = {"distilled": {"supports_negative_prompt": False, "negative_prompt_hint": "HINT"}}
    info = ImageModelInfo(family="flux2_klein", is_distilled=True, size=None)
    args = _make_args(model="flux2-klein-4b")
    use_model_metadata(monkeypatch, tmp_path, data)
    with_hint = _negative_prompt_warning(args, info)
    del data["image_families"]["flux2_klein"]["variants"]["distilled"]["negative_prompt_hint"]
    use_model_metadata(monkeypatch, tmp_path, data)

    assert with_hint.endswith("HINT")
    assert not _negative_prompt_warning(args, info).endswith("HINT")
