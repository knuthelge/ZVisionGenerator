"""Tests for the workspace payload and bootstrap defaults built by the workspace API module."""

from __future__ import annotations


import pytest

from tests._server_helpers import make_resolved_image_defaults, make_web_config, make_workspace_bootstrap_view


from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.utils.model_memory import MemoryBudget


@pytest.fixture(autouse=True)
def _gemma_repo(monkeypatch):
    """Give the minimal test configs the macOS LTX text encoder reference the Models page reads."""
    monkeypatch.setattr(workspace_api_module, "ltx_mlx_text_encoder_repo_id", lambda config: "owner/gemma")


@pytest.mark.parametrize(("family", "alignment"), [("ltx", 8), ("unknown", 1)])
def test_video_bootstrap_defaults_include_frame_rule(monkeypatch, family, alignment):
    """Video bootstrap defaults expose the model's frame rule, and only that, so the UI can snap frame counts."""
    web_config = make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        workspace_api_module,
        "detect_video_model",
        lambda _model: VideoModelInfo(family=family, backend=family, supports_i2v=True, default_fps=24, frame_alignment=alignment, resolution_alignment=32),
    )

    defaults = workspace_api_module._build_video_bootstrap_defaults("ltx-8", web_config)

    assert defaults["frame_rule"]["alignment"] == alignment
    assert "frame_alignment" not in defaults


@pytest.mark.parametrize(("detected", "expected"), [("ltx", "ltx"), ("unknown", None)])
def test_video_bootstrap_defaults_include_detected_family(monkeypatch, detected, expected):
    """Video defaults name the detected family so the Workspace can remember LoRAs per family."""
    web_config = make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        workspace_api_module,
        "detect_video_model",
        lambda _model: VideoModelInfo(family=detected, backend=detected, supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32),
    )

    assert workspace_api_module._build_video_bootstrap_defaults("ltx-8", web_config)["family"] == expected


def test_image_bootstrap_defaults_include_detected_family(monkeypatch):
    """Image defaults name the detected family, and none when detection fails."""
    web_config = make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", lambda *_: make_resolved_image_defaults())
    monkeypatch.setattr(workspace_api_module, "detect_image_model", lambda _model: ImageModelInfo(family="zimage", is_distilled=True, size=None))

    assert workspace_api_module._build_image_bootstrap_defaults("zit", web_config)["family"] == "zimage"

    def _fail(_model):
        raise RuntimeError("no model")

    monkeypatch.setattr(workspace_api_module, "detect_image_model", _fail)

    assert workspace_api_module._build_image_bootstrap_defaults("missing", web_config)["family"] is None


def test_video_bootstrap_defaults_fall_back_to_unaligned_frames_when_detection_fails(monkeypatch):
    """A model that cannot be detected should leave frame counts unconstrained."""
    web_config = make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)

    def _fail(_model):
        raise RuntimeError("no model")

    monkeypatch.setattr(workspace_api_module, "detect_video_model", _fail)

    defaults = workspace_api_module._build_video_bootstrap_defaults("missing", web_config)

    assert defaults["frame_rule"] == {"alignment": 1, "offset": 1, "round": "nearest"}


def test_workspace_bootstrap_defaults_include_ideogram_capability_flags(monkeypatch):
    """Bootstrap defaults should surface the declared-family capability contract."""
    web_config = make_web_config()
    web_config.app_config["model_aliases"] = {
        "ideo": "ideogram-ai/ideogram-4-fp8",
        "zit": "Tongyi-MAI/Z-Image-Turbo",
    }
    web_config.app_config["model_alias_families"] = {"ideo": "ideogram4"}
    web_config.image_model_options = ("ideo", "zit")

    detect_calls: list[str] = []

    def _fake_detect_image_model(value: object) -> ImageModelInfo:
        detect_calls.append(str(value))
        return ImageModelInfo(family="zimage", is_distilled=False, size="xl")

    def _fake_resolve_defaults(model_info, _config, _cli_overrides, _backend_name):
        if model_info.family == "ideogram4":
            return make_resolved_image_defaults(
                supports_negative_prompt=False,
                supports_quantize=False,
                supports_img2img=False,
                supports_upscale=False,
                supports_json_prompt=True,
                supports_first_sigma=True,
                dimension_min=256,
                dimension_max=2048,
                dimension_step=16,
            )
        return make_resolved_image_defaults()

    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: web_config.app_config["model_aliases"].get(model, model))
    monkeypatch.setattr(workspace_api_module, "detect_image_model", _fake_detect_image_model)
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", _fake_resolve_defaults)

    ideogram_defaults = workspace_api_module._build_image_bootstrap_defaults("ideo", web_config)
    zimage_defaults = workspace_api_module._build_image_bootstrap_defaults("zit", web_config)

    assert ideogram_defaults["supports_json_prompt"] is True
    assert ideogram_defaults["dimension_max"] == 2048
    assert zimage_defaults["supports_json_prompt"] is False
    assert zimage_defaults["dimension_max"] is None
    assert detect_calls == ["Tongyi-MAI/Z-Image-Turbo"]


def test_workspace_bootstrap_defaults_include_scheduler_capability(monkeypatch):
    web_config = make_web_config()
    web_config.app_config["model_aliases"] = {"krea2": "unsloth/Krea-2-Turbo", "zit": "Tongyi-MAI/Z-Image-Turbo"}
    web_config.image_model_options = ("krea2", "zit")
    families = {"unsloth/Krea-2-Turbo": "krea2", "Tongyi-MAI/Z-Image-Turbo": "zimage"}

    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: web_config.app_config["model_aliases"].get(model, model))
    monkeypatch.setattr(workspace_api_module, "detect_image_model", lambda value: ImageModelInfo(family=families[str(value)], is_distilled=True, size=None))
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(
        workspace_api_module,
        "resolve_defaults",
        lambda model_info, *_: make_resolved_image_defaults(supports_scheduler=model_info.family != "krea2"),
    )

    assert workspace_api_module._build_image_bootstrap_defaults("krea2", web_config)["supports_scheduler"] is False
    assert workspace_api_module._build_image_bootstrap_defaults("zit", web_config)["supports_scheduler"] is True


def test_workspace_bootstrap_defaults_fall_back_to_supported_ideogram_preset(monkeypatch):
    web_config = make_web_config()
    web_config.app_config["generation"] = {
        "default_ratio": "16:9",
        "default_size": "xl",
    }
    web_config.app_config["sizes"] = {
        "16:9": {
            "m": {"width": 1344, "height": 768},
            "l": {"width": 1888, "height": 1056},
            "xl": {"width": 2112, "height": 1184},
        },
        "2:3": {
            "m": {"width": 832, "height": 1216},
        },
    }
    web_config.image_ratios = ("16:9", "2:3")
    web_config.image_size_options = {
        "16:9": ("m", "l", "xl"),
        "2:3": ("m",),
    }
    web_config.app_config["model_aliases"] = {
        "ideo": "ideogram-ai/ideogram-4-fp8",
    }
    web_config.app_config["model_alias_families"] = {"ideo": "ideogram4"}

    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: web_config.app_config["model_aliases"].get(model, model))
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", lambda *_args, **_kwargs: make_resolved_image_defaults(dimension_min=256, dimension_max=2048, dimension_step=16))

    defaults = workspace_api_module._build_image_bootstrap_defaults("ideo", web_config)

    assert defaults["ratio"] == "16:9"
    assert defaults["size"] == "l"
    assert defaults["width"] == 1888
    assert defaults["height"] == 1056


def test_workspace_bootstrap_defaults_disable_quantize_when_globally_unavailable(monkeypatch):
    web_config = make_web_config()
    web_config.app_config["model_aliases"] = {
        "ideo": "ideogram-ai/ideogram-4-fp8",
        "zit": "Tongyi-MAI/Z-Image-Turbo",
    }
    web_config.app_config["model_alias_families"] = {"ideo": "ideogram4"}
    web_config.image_model_options = ("ideo", "zit")
    web_config.quantize_options = ()

    def _fake_detect_image_model(value: object) -> ImageModelInfo:
        return ImageModelInfo(family="zimage", is_distilled=False, size="xl")

    def _fake_resolve_defaults(model_info, _config, _cli_overrides, _backend_name):
        if model_info.family == "ideogram4":
            return make_resolved_image_defaults(supports_negative_prompt=False, supports_quantize=False)
        return make_resolved_image_defaults(supports_quantize=True)

    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: web_config.app_config["model_aliases"].get(model, model))
    monkeypatch.setattr(workspace_api_module, "detect_image_model", _fake_detect_image_model)
    monkeypatch.setattr(workspace_api_module, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", _fake_resolve_defaults)

    ideogram_defaults = workspace_api_module._build_image_bootstrap_defaults("ideo", web_config)
    zimage_defaults = workspace_api_module._build_image_bootstrap_defaults("zit", web_config)

    assert ideogram_defaults["supports_quantize"] is False
    assert zimage_defaults["supports_quantize"] is False


def test_model_listings_carry_download_and_memory_status(monkeypatch, tmp_path):
    """Workspace and models payloads expose per-model download and memory-fit status from one source."""
    web_config = make_web_config()
    web_config.data_dir = str(tmp_path)
    web_config.image_inventory = (
        ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="owner/zit"),
        ImageInventoryEntry(name="local-image", family="ideogram4", size=None, source="installed", resolved_path=str(tmp_path / "local-image")),
    )
    web_config.video_inventory = (VideoInventoryEntry(name="ltx-8", family="ltx", supports_i2v=True, source="alias", resolved_path="owner/ltx"),)
    web_config.app_config["model_presets"] = {"ideogram4": {"supports_quantize": False}}
    web_config.app_config["video_model_presets"] = {"ltx": {"mlx_text_encoder": "owner/gemma"}}
    calls: list[tuple[str, str, tuple[int, ...]]] = []
    video_names: list[tuple[str | None, bool]] = []

    def _fake_status(resolved_path, *, kind, quantize_options=(), budget=None, find_local_dir=None, model_name=None, app_config=None, family=None):
        calls.append((resolved_path, kind, quantize_options))
        if kind == "video":
            video_names.append((model_name, app_config is web_config.app_config))
        return {"downloaded": resolved_path != "owner/ltx", "memory_fit": None}

    monkeypatch.setattr(workspace_api_module, "describe_model_status", _fake_status)
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: MemoryBudget(10 * 1024**3))
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])
    bootstrap_view = make_workspace_bootstrap_view()
    bootstrap_view["image_model_defaults"]["zit"]["supports_quantize"] = False  # the picker hides quantize for zit

    workspace = workspace_api_module.build_workspace_response(
        web_config,
        [],
        active_job=None,
        prompt_sources=["inline"],
        default_prompt_source="inline",
        prompt_file_contract={},
        workflow_contract={},
        build_bootstrap_view=lambda _cfg: bootstrap_view,
    )
    workspace_calls = list(calls)
    models = workspace_api_module.build_models_response(
        web_config,
        token_source=None,
        image_defaults_for=lambda name, _cfg: bootstrap_view["image_model_defaults"].get(name, {}),
    )

    workspace_images = {entry["id"]: entry for entry in workspace["image_models"]}
    assert workspace_images["zit"]["downloaded"] is True
    assert workspace["video_models"][0]["downloaded"] is False
    assert {entry["name"]: entry["downloaded"] for entry in models["video_models"]} == {"ltx-8": False}
    assert all("memory_fit" in entry for entry in [*workspace["image_models"], *models["image_models"]])
    # Both pages estimate exactly the quantize levels the workspace picker offers (from the bootstrap defaults).
    assert ("owner/zit", "image", ()) in workspace_calls
    assert calls[len(workspace_calls) :] == workspace_calls
    assert ("owner/ltx", "video", ()) in calls
    assert video_names and set(video_names) == {("ltx-8", True)}  # a video model's level comes from its name and the config


def test_workspace_options_carry_the_configured_name_as_their_label(monkeypatch):
    """AC4.3: a named alias is labelled by its name, an unnamed model by its id; the id never changes."""
    web_config = make_web_config()
    web_config.app_config["model_labels"] = {"zit": {"name": "Z-Image Turbo"}, "ltx-8": {"name": "LTX-2.3 8-bit"}}
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)

    workspace = workspace_api_module.build_workspace_response(
        web_config,
        [],
        active_job=None,
        prompt_sources=["inline"],
        default_prompt_source="inline",
        prompt_file_contract={},
        workflow_contract={},
        build_bootstrap_view=lambda _cfg: make_workspace_bootstrap_view(),
    )

    assert {(entry["id"], entry["label"]) for entry in workspace["image_models"]} == {("zit", "Z-Image Turbo"), ("local-image", "local-image")}
    assert [(entry["id"], entry["label"]) for entry in workspace["video_models"]] == [("ltx-8", "LTX-2.3 8-bit")]


def test_workspace_carries_the_first_run_state_and_the_platform_size_labels(monkeypatch):
    """AC5.3: image options carry this platform's download size, and ``first_run`` the skip flag and the recommendation."""
    web_config = make_web_config()
    web_config.image_model_options = ("klein9b", "klein4b", "local-image")
    web_config.app_config["first_run"] = {"large_model": "klein9b", "small_model": "klein4b", "min_memory_gb": 32}
    web_config.app_config["ui"] = {"first_run_skipped": True}
    web_config.app_config["model_labels"] = {
        "klein9b": {"name": "FLUX.2 Klein 9B", "download_size_label": {"darwin": "35 GB", "win32": "36 GB", "linux": "37 GB"}},
        "klein4b": {"name": "FLUX.2 Klein 4B", "download_size_label": {"darwin": "16 GB"}},
    }
    monkeypatch.setattr(workspace_api_module.sys, "platform", "linux")
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: MemoryBudget(gpu_bytes=32607 * 1024**2, system_bytes=64 * 1024**3))

    workspace = workspace_api_module.build_workspace_response(
        web_config,
        [],
        active_job=None,
        prompt_sources=["inline"],
        default_prompt_source="inline",
        prompt_file_contract={},
        workflow_contract={},
        build_bootstrap_view=lambda _cfg: make_workspace_bootstrap_view(),
    )

    assert {entry["id"]: entry["download_size_label"] for entry in workspace["image_models"]} == {"klein9b": "37 GB", "klein4b": None, "local-image": None}
    assert workspace["first_run"] == {"skipped": True, "recommended_model": "klein9b"}


def test_workspace_models_without_inventory_entry_report_unknown_status(monkeypatch):
    """Names with no inventory entry (e.g. stale config) report unknown rather than guessing."""
    web_config = make_web_config()
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)

    workspace = workspace_api_module.build_workspace_response(
        web_config,
        [],
        active_job=None,
        prompt_sources=["inline"],
        default_prompt_source="inline",
        prompt_file_contract={},
        workflow_contract={},
        build_bootstrap_view=lambda _cfg: make_workspace_bootstrap_view(),
    )

    assert {(entry["downloaded"], entry["memory_fit"]) for entry in workspace["image_models"]} == {(None, None)}


def test_one_failing_model_status_does_not_break_the_listing(monkeypatch):
    """A model whose files cannot be inspected reports unknown status instead of failing the whole page."""
    web_config = make_web_config()
    web_config.image_inventory = (
        ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="owner/zit"),
        ImageInventoryEntry(name="local-image", family="zimage", size=None, source="installed", resolved_path="/broken"),
    )

    def _fake_status(resolved_path, **_):
        if resolved_path == "/broken":
            raise AttributeError("corrupt header")
        return {"downloaded": True, "memory_fit": None}

    monkeypatch.setattr(workspace_api_module, "describe_model_status", _fake_status)
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)

    with pytest.warns(UserWarning, match="local-image"):
        workspace = workspace_api_module.build_workspace_response(
            web_config,
            [],
            active_job=None,
            prompt_sources=["inline"],
            default_prompt_source="inline",
            prompt_file_contract={},
            workflow_contract={},
            build_bootstrap_view=lambda _cfg: make_workspace_bootstrap_view(),
        )

    statuses = {entry["id"]: entry["downloaded"] for entry in workspace["image_models"]}
    assert statuses == {"zit": True, "local-image": None}


def test_models_page_skips_quantize_resolution_without_a_memory_budget(monkeypatch):
    """Off macOS there is no budget, so per-model defaults (and their detection lookups) are not resolved."""
    web_config = make_web_config()
    web_config.image_inventory = (ImageInventoryEntry(name="zit", family="zimage", size=None, source="alias", resolved_path="owner/zit"),)
    monkeypatch.setattr(workspace_api_module, "memory_budget", lambda: None)
    monkeypatch.setattr(workspace_api_module, "describe_model_status", lambda resolved_path, **_: {"downloaded": True, "memory_fit": None})
    monkeypatch.setattr(workspace_api_module, "list_loras", lambda _data_dir: [])

    models = workspace_api_module.build_models_response(
        web_config,
        token_source=None,
        image_defaults_for=lambda name, _cfg: pytest.fail("resolved image defaults without a memory budget"),
    )

    assert models["image_models"][0]["downloaded"] is True


def test_image_defaults_carry_the_size_variant(monkeypatch):
    web_config = make_web_config()
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(workspace_api_module, "detect_image_model", lambda _model: ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b"))
    monkeypatch.setattr(workspace_api_module, "resolve_defaults", lambda *_args: {"steps": 4, "guidance": 1.0})

    klein = workspace_api_module._build_image_bootstrap_defaults("zit", web_config)
    monkeypatch.setattr(workspace_api_module, "detect_image_model", lambda _model: (_ for _ in ()).throw(ValueError("no model")))
    failed = workspace_api_module._build_image_bootstrap_defaults("zit", web_config)

    assert (klein["size_variant"], klein["size"]) == ("4b", "m")
    assert (failed["size_variant"], failed["size"]) == (None, "m")
