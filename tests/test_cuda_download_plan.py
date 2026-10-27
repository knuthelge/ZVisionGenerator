"""Tests for the CUDA (diffusers, transformers) download planner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError
from huggingface_hub.utils import filter_repo_objects

from tests._paths import FIXTURES

from zvisiongenerator.backends.cuda.download_plan import (
    DIFFUSERS_IGNORE,
    DIFFUSERS_ROOT_FILES,
    SEPARATE_TRANSFORMER_IGNORE,
    TRANSFORMERS_ALLOW_PATTERNS,
    enhancer_download_plan,
    h3_download_plan,
    image_download_plan,
    video_download_plan,
    video_repo_ignore,
)
from zvisiongenerator.backends.cuda.video.h3 import sources as h3_sources
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils import hf_snapshot
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.h3_preset import h3_preset

_LISTINGS = json.loads((FIXTURES / "diffusers_repo_listings.json").read_text(encoding="utf-8"))
KLEIN = "black-forest-labs/FLUX.2-klein-9B"
ZIMAGE = "Tongyi-MAI/Z-Image-Turbo"
KREA = "unsloth/Krea-2-Turbo"


def _index_reader(repo_id: str, revision: str | None) -> dict[str, Any]:
    return _LISTINGS[repo_id]["model_index"]


def _selected(repo_id: str, extra: tuple[str, ...] = ()) -> set[str]:
    (target,) = image_download_plan(repo_id, family="x", read_index=_index_reader)
    files = [*_LISTINGS[repo_id]["files"], *extra]
    return set(filter_repo_objects(files, allow_patterns=target.allow_patterns, ignore_patterns=target.ignore_patterns))


def test_image_plan_is_one_cache_target_with_component_folders() -> None:
    (target,) = image_download_plan(KLEIN, family="flux2_klein", read_index=_index_reader)

    assert target.repo_id == KLEIN
    assert target.local_dir is None
    assert target.ignore_patterns == DIFFUSERS_IGNORE
    assert target.allow_patterns == ("model_index.json", *DIFFUSERS_ROOT_FILES, "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*")


def test_image_plan_skips_non_component_index_entries() -> None:
    (target,) = image_download_plan(KREA, family="krea2", read_index=_index_reader)

    assert target.allow_patterns is not None
    assert not any(pattern.startswith(("text_encoder_select_layers", "patch_size", "is_distilled", "_")) for pattern in target.allow_patterns)


def test_image_plan_skips_null_library_components() -> None:
    index = {"_class_name": "P", "safety_checker": [None, None], "vae": ["diffusers", "AutoencoderKL"]}
    (target,) = image_download_plan("owner/repo", family="zimage", read_index=lambda repo, rev: index)

    assert target.allow_patterns is not None
    assert "vae/*" in target.allow_patterns
    assert "safety_checker/*" not in target.allow_patterns


@pytest.mark.parametrize("repo_id", [KLEIN, ZIMAGE, KREA])
def test_image_plan_excludes_root_single_file_checkpoints(repo_id: str) -> None:
    selected = _selected(repo_id)

    assert "flux-2-klein-9b.safetensors" not in selected
    assert "turbo.safetensors" not in selected
    assert "model_index.json" in selected
    assert not {name for name in selected if name.endswith((".jpg", ".md", ".pdf"))}


def test_image_plan_includes_krea_transformer_shards() -> None:
    selected = _selected(KREA)

    shards = {name for name in _LISTINGS[KREA]["files"] if name.startswith("transformer/") and name.endswith(".safetensors")}
    assert shards
    assert shards <= selected


def test_image_plan_excludes_variants_and_other_formats() -> None:
    extra = (
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "transformer/diffusion_pytorch_model.bf16-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.fp8.safetensors.index.json",
        "text_encoder/pytorch_model.bin",
        "vae/model.onnx",
    )
    selected = _selected(KLEIN, extra)

    assert not selected & set(extra)
    assert "vae/diffusion_pytorch_model.safetensors" in selected


def test_image_plan_keeps_root_config_files_when_present() -> None:
    selected = _selected(ZIMAGE, ("config.json", "scheduler_config.json"))

    assert {"config.json", "scheduler_config.json"} <= selected


def test_image_plan_passes_revision_to_the_index_reader() -> None:
    seen: list[tuple[str, str | None]] = []

    def reader(repo_id: str, revision: str | None) -> dict[str, Any]:
        seen.append((repo_id, revision))
        return {"vae": ["diffusers", "AutoencoderKL"]}

    (target,) = image_download_plan("owner/repo@abc123", family="zimage", read_index=reader)

    assert seen == [("owner/repo", "abc123")]
    assert (target.repo_id, target.revision) == ("owner/repo", "abc123")


def test_image_plan_unreadable_index_raises_lookup_error() -> None:
    def reader(repo_id: str, revision: str | None) -> dict[str, Any]:
        raise OSError("offline")

    with pytest.raises(LookupError, match="owner/repo"):
        image_download_plan("owner/repo", family="zimage", read_index=reader)


def test_image_plan_hub_refusal_raises_runtime_error_with_the_next_step() -> None:
    def reader(repo_id: str, revision: str | None) -> dict[str, Any]:
        raise GatedRepoError("boom", response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/api/models/owner/repo")))

    with pytest.raises(RuntimeError) as raised:
        image_download_plan("owner/repo", family="zimage", read_index=reader)

    message = str(raised.value)
    assert "\n" not in message
    assert "https://huggingface.co/owner/repo" in message
    assert "hf auth login" in message


def test_image_plan_unreachable_hub_stays_a_lookup_error() -> None:
    def reader(repo_id: str, revision: str | None) -> dict[str, Any]:
        raise httpx.ConnectError("offline")

    with pytest.raises(LookupError, match="owner/repo"):
        image_download_plan("owner/repo", family="zimage", read_index=reader)


def test_image_plan_ideogram4_repo_is_empty_without_reading_the_index() -> None:
    def reader(repo_id: str, revision: str | None) -> dict[str, Any]:
        raise AssertionError("the index must not be read")

    assert image_download_plan("ideogram-ai/ideogram-4-fp8", family="ideogram4", read_index=reader) == ()


def test_image_plan_local_folder_or_non_repo_is_empty(tmp_path: Path) -> None:
    def reader(repo_id: str, revision: str | None) -> dict[str, Any]:
        raise AssertionError("the index must not be read")

    assert image_download_plan(str(tmp_path), family="zimage", read_index=reader) == ()
    assert image_download_plan("not a repo id", family="zimage", read_index=reader) == ()
    assert image_download_plan("/some/missing/path", family="zimage", read_index=reader) == ()


def test_root_files_cover_diffusers_transformers_aux_files() -> None:
    constants = pytest.importorskip("diffusers.utils.constants")

    assert set(DIFFUSERS_ROOT_FILES) >= set(constants.TRANSFORMERS_COMPONENT_AUX_FILES)


def _diffusers_expected_files(repo_id: str, passed_components: tuple[str, ...]) -> set[str]:
    """Return the files ``DiffusionPipeline.download`` needs in the snapshot to skip its own ``snapshot_download`` (diffusers 0.41 logic)."""
    import fnmatch
    import os
    import re

    loading = pytest.importorskip("diffusers.pipelines.pipeline_loading_utils")
    constants = pytest.importorskip("diffusers.utils.constants")

    files = set(_LISTINGS[repo_id]["files"])
    index = _LISTINGS[repo_id]["model_index"]
    _custom, folder_names = loading._get_custom_components_and_folders(repo_id, index, files, None)
    model_folders = {os.path.split(name)[0] for name in loading.filter_model_files(files) if os.path.split(name)[0] in folder_names}
    ignore = loading._get_ignore_patterns(list(passed_components), model_folders, files, True, False, None, False, False, None)
    model_files, _variants = loading.variant_compatible_siblings(files, variant=None, ignore_patterns=ignore)
    allow = list(model_files)
    allow += [f"{name}/*" for name in folder_names if name not in model_folders]
    allow += [f"{name}/config.json" for name in model_folders]
    allow += ["scheduler_config.json", "config.json", "model_index.json", "pipeline.py"]
    allow = [pattern for pattern in allow if not (len(pattern.split("/")) == 2 and pattern.split("/")[0] in passed_components)]
    allow += constants.TRANSFORMERS_COMPONENT_AUX_FILES
    ignore += [f"{pattern}.index.*json" for pattern in ignore]
    ignored = [re.compile(fnmatch.translate(pattern)) for pattern in ignore]
    allowed = [re.compile(fnmatch.translate(pattern)) for pattern in allow]
    return {name for name in files if not any(rx.match(name) for rx in ignored) and any(rx.match(name) for rx in allowed)}


@pytest.mark.parametrize("passed_components", [(), ("text_encoder", "transformer")])
@pytest.mark.parametrize("repo_id", [KLEIN, ZIMAGE, KREA])
def test_image_plan_covers_every_file_diffusers_expects_in_the_snapshot(repo_id: str, passed_components: tuple[str, ...]) -> None:
    """A snapshot filled by the plan makes ``DiffusionPipeline.download`` return without fetching again."""
    expected = _diffusers_expected_files(repo_id, passed_components)

    assert expected
    assert expected <= _selected(repo_id)


# --- video ---

TRANSFORMER = ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-1.1.safetensors")
UPSAMPLER = ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
LTX_REPO = "diffusers/LTX-2.3-Distilled-Diffusers"


def test_video_plan_repo_only_downloads_everything() -> None:
    plan = video_download_plan(LTX_REPO, family="ltx", transformer_source=None, latent_upsampler=UPSAMPLER, upscale=False)

    assert plan == (DownloadTarget(LTX_REPO, ignore_patterns=None),)


def test_video_plan_with_transformer_source_has_two_parts() -> None:
    plan = video_download_plan(LTX_REPO, family="ltx", transformer_source=TRANSFORMER, latent_upsampler=UPSAMPLER, upscale=False)

    assert plan == (
        DownloadTarget(LTX_REPO, ignore_patterns=SEPARATE_TRANSFORMER_IGNORE),
        DownloadTarget(TRANSFORMER[0], allow_patterns=(TRANSFORMER[1],)),
    )


def test_video_plan_upscale_without_transformer_source_has_two_parts() -> None:
    plan = video_download_plan(LTX_REPO, family="ltx", transformer_source=None, latent_upsampler=UPSAMPLER, upscale=True)

    assert plan == (DownloadTarget(LTX_REPO), DownloadTarget(UPSAMPLER[0], allow_patterns=(UPSAMPLER[1],)))


def test_video_plan_upscale_without_configured_upsampler_skips_it() -> None:
    plan = video_download_plan(LTX_REPO, family="ltx", transformer_source=None, latent_upsampler=None, upscale=True)

    assert plan == (DownloadTarget(LTX_REPO),)


@pytest.mark.parametrize(
    ("separate", "extra", "expected"),
    [
        (False, ("a/*",), ("a/*",)),
        (True, ["a/*", "b"], (*SEPARATE_TRANSFORMER_IGNORE, "a/*", "b")),
    ],
)
def test_video_repo_ignore(separate: bool, extra: tuple[str, ...], expected: tuple[str, ...] | None) -> None:
    assert video_repo_ignore(separate_transformer=separate, extra=extra) == expected


def test_2_3_video_plans_are_pinned_without_a_revision() -> None:
    """2.3 guard: the plans of ``ltx-4`` and ``ltx-8`` keep their exact targets, with revision ``None`` and no extra ignore."""
    plan = video_download_plan(LTX_REPO, family="ltx", transformer_source=TRANSFORMER, latent_upsampler=UPSAMPLER, upscale=True, repo_ignore=())

    assert plan == (
        DownloadTarget("diffusers/LTX-2.3-Distilled-Diffusers", None, None, ("transformer/*.safetensors", "transformer/*.safetensors.index.json"), None),
        DownloadTarget("Lightricks/LTX-2.3", None, ("ltx-2.3-22b-distilled-1.1.safetensors",), None, None),
        DownloadTarget("Lightricks/LTX-2.3", None, ("ltx-2.3-spatial-upscaler-x2-1.1.safetensors",), None, None),
    )


LTX25_PIN = "7b40724fc39bbb01d9b4d72638c0d263531c88c2"
LTX25_REPO = "unsloth/LTX-2.5-FP8"
LTX25_IGNORE = ("LTX-2.5-*.safetensors", "samples/*")
LTX25_UPSAMPLER = (f"{LTX25_REPO}@{LTX25_PIN}", "latent_upsampler/diffusion_pytorch_model.safetensors")


@pytest.mark.parametrize("upscale", [False, True])
def test_video_plan_ltx25_is_one_pinned_target_with_the_repo_ignore(upscale: bool) -> None:
    plan = video_download_plan(f"{LTX25_REPO}@{LTX25_PIN}", family="ltx", transformer_source=None, latent_upsampler=LTX25_UPSAMPLER, upscale=upscale, repo_ignore=LTX25_IGNORE)

    assert plan == (DownloadTarget(LTX25_REPO, LTX25_PIN, None, LTX25_IGNORE),)


def test_video_plan_repo_ignore_follows_the_separate_transformer_patterns() -> None:
    plan = video_download_plan(LTX_REPO, family="ltx", transformer_source=TRANSFORMER, repo_ignore=("x/*",))

    assert plan[0].ignore_patterns == (*SEPARATE_TRANSFORMER_IGNORE, "x/*")


def test_video_plan_keeps_the_revision_of_the_reference() -> None:
    (target,) = video_download_plan(f"{LTX_REPO}@abc123", family="ltx")

    assert (target.repo_id, target.revision) == (LTX_REPO, "abc123")


def test_video_plan_splits_the_revision_of_the_transformer_and_upsampler_sources() -> None:
    plan = video_download_plan(
        LTX_REPO,
        family="ltx",
        transformer_source=("owner/tf@rev1", "t.safetensors"),
        latent_upsampler=("owner/up@rev2", "u.safetensors"),
        upscale=True,
    )

    assert plan[1:] == (DownloadTarget("owner/tf", "rev1", ("t.safetensors",)), DownloadTarget("owner/up", "rev2", ("u.safetensors",)))
    assert len({target.key for target in plan}) == 3


def test_video_plan_keeps_an_upsampler_of_the_same_repo_at_another_revision() -> None:
    plan = video_download_plan(f"{LTX25_REPO}@{LTX25_PIN}", family="ltx", latent_upsampler=(f"{LTX25_REPO}@other", "u.safetensors"), upscale=True)

    assert plan == (DownloadTarget(LTX25_REPO, LTX25_PIN), DownloadTarget(LTX25_REPO, "other", ("u.safetensors",)))


def test_video_plan_local_folder_is_empty(tmp_path: Path) -> None:
    assert video_download_plan(str(tmp_path), family="ltx", transformer_source=TRANSFORMER, latent_upsampler=UPSAMPLER, upscale=True) == ()


# --- enhancer ---


H3_REF = "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe"
H3_PRESET = h3_preset(load_config())


def test_h3_plan_is_the_repository_the_text_encoder_and_the_lora() -> None:
    plan = h3_download_plan(H3_REF, H3_PRESET)
    text_encoder, lora = H3_PRESET.text_encoder, H3_PRESET.lora

    assert plan == (
        DownloadTarget(H3_PRESET.repo, H3_PRESET.revision, H3_PRESET.repo_allow, H3_PRESET.repo_ignore),
        DownloadTarget(text_encoder.repo_id, text_encoder.revision, text_encoder.files),
        DownloadTarget(lora.repo_id, lora.revision, lora.files),
    )


def test_h3_plan_takes_the_presets_revision_for_a_reference_without_one() -> None:
    (repo, *_rest) = h3_download_plan("MiniMaxAI/MiniMax-H3", H3_PRESET)

    assert repo.revision == H3_PRESET.revision


def test_h3_plan_keeps_the_revision_of_the_reference() -> None:
    (repo, *_rest) = h3_download_plan("MiniMaxAI/MiniMax-H3@other", H3_PRESET)

    assert repo.revision == "other"


def test_h3_plan_for_a_local_folder_has_no_repository_target(tmp_path: Path) -> None:
    plan = h3_download_plan(str(tmp_path), H3_PRESET)

    assert [target.repo_id for target in plan] == ["unsloth/Qwen3-VL-32B-Instruct-bnb-4bit", "lightx2v/Minimax-h3-Turbo"]


def test_h3_plan_adds_exactly_the_latent_upsampler_with_upscale() -> None:
    base = h3_download_plan(H3_REF, H3_PRESET)
    plan = h3_download_plan(H3_REF, H3_PRESET, upscale=True)

    assert plan[:-1] == base
    upsampler = H3_PRESET.latent_upsampler
    assert plan[-1] == DownloadTarget(upsampler.repo_id, upsampler.revision, upsampler.files)
    assert len(plan) == len(base) + 1


def test_h3_plan_downloads_what_the_backend_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each plan target is the (repo, revision, allow, ignore) the runtime snapshot call uses, or the two diverge."""
    monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda repo_id, revision: [])
    calls: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        folder = tmp_path / kwargs["repo_id"].replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        for name in kwargs.get("allow_patterns") or ():
            if "*" not in name:
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(b"x")
        (folder / "model_index.json").write_text("{}")
        return str(folder)

    for upscale in (False, True):
        calls.clear()
        h3_sources.resolve_files(H3_REF, H3_PRESET, upscale=upscale, snapshot_download=download)

        runtime = [(call["repo_id"], call["revision"], tuple(call["allow_patterns"]), tuple(call.get("ignore_patterns") or ()) or None) for call in calls]
        planned = [(target.repo_id, target.revision, target.allow_patterns, target.ignore_patterns) for target in h3_download_plan(H3_REF, H3_PRESET, upscale=upscale)]
        assert planned == runtime


def test_enhancer_plan_uses_transformers_patterns() -> None:
    assert enhancer_download_plan("coder3101/Qwen3.5-4B-heretic", "dc69b956") == (DownloadTarget("coder3101/Qwen3.5-4B-heretic", "dc69b956", TRANSFORMERS_ALLOW_PATTERNS),)


def test_enhancer_plan_local_folder_is_empty(tmp_path: Path) -> None:
    assert enhancer_download_plan(str(tmp_path), None) == ()


def test_module_imports_nothing_heavy() -> None:
    code = "import sys; import zvisiongenerator.backends.cuda.download_plan; heavy = [m for m in ('torch', 'diffusers', 'transformers', 'mflux', 'mlx') if m in sys.modules]; assert not heavy, heavy"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
