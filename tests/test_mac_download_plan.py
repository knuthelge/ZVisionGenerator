"""Tests for the macOS (mflux, MLX) download planner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from zvisiongenerator.backends.mac import download_plan as mac_plan
from zvisiongenerator.backends.mac.download_plan import MLX_LM_ALLOW_PATTERNS, enhancer_download_plan, h3_mlx_download_plan, image_download_plan, video_download_plan
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils import hf_snapshot
from zvisiongenerator.utils.h3_preset import HubFiles

PIN = "a1" * 20
GEMMA_REF = "mlx-community/gemma-3-12b-it-4bit@" + "c3" * 20
PACK = HubFiles("o/pack", PIN, ("transformer.safetensors", "connector.safetensors", "embedded_config.json"))
PACKS = {f"o/pack@{PIN}": PACK}
BUNDLED_PACK = HubFiles("o/bundled", PIN, ("transformer.safetensors", "text_encoder.safetensors", "text_encoder_config.json"))
BUNDLED_PACKS = {f"o/bundled@{PIN}": BUNDLED_PACK}


@pytest.fixture
def gemma() -> DownloadTarget:
    return DownloadTarget("mlx-community/gemma-3-12b-it-4bit", "c3" * 20, MLX_LM_ALLOW_PATTERNS)


def _plan(model_ref: str, **kwargs):
    return video_download_plan(model_ref, family="ltx", mlx_packs=PACKS, mlx_text_encoder=GEMMA_REF, **kwargs)


# --- video ---


def test_video_plan_pinned_pack_is_its_literal_files_at_its_revision_then_gemma(gemma: DownloadTarget) -> None:
    plan = _plan(f"o/pack@{PIN}")

    assert plan == (DownloadTarget("o/pack", PIN, PACK.files), gemma)
    assert plan[0].ignore_patterns is None


def test_video_plan_other_repo_is_the_whole_repo_then_gemma(gemma: DownloadTarget) -> None:
    plan = _plan("dgrauet/ltx-2.3-mlx-q4")

    assert plan == (DownloadTarget("dgrauet/ltx-2.3-mlx-q4"), gemma)
    assert plan[0].allow_patterns is None


def test_video_plan_keeps_the_revision_of_a_reference_that_is_no_pack(gemma: DownloadTarget) -> None:
    assert _plan("o/p@" + "b2" * 20) == (DownloadTarget("o/p", "b2" * 20), gemma)


def test_video_plan_ignores_diffusers_arguments(gemma: DownloadTarget) -> None:
    plan = _plan(f"o/pack@{PIN}", transformer_source=("a/b", "c"), latent_upsampler=("d/e", "f"), upscale=True, repo_ignore=("LTX-2.5-*.safetensors", "samples/*"))

    assert plan == (DownloadTarget("o/pack", PIN, PACK.files), gemma)


def test_video_plan_bundled_flag_leaves_gemma_out_of_a_model_that_is_no_pack() -> None:
    assert _plan("o/p", text_encoder_bundled=True) == (DownloadTarget("o/p"),)


def test_video_plan_pack_listing_its_own_text_encoder_is_the_pack_alone() -> None:
    plan = video_download_plan(f"o/bundled@{PIN}", family="ltx", mlx_packs=BUNDLED_PACKS, mlx_text_encoder=GEMMA_REF)

    assert plan == (DownloadTarget("o/bundled", PIN, BUNDLED_PACK.files),)


def test_video_plan_pack_files_decide_gemma_not_the_bundled_flag(gemma: DownloadTarget) -> None:
    flagged = _plan(f"o/pack@{PIN}", text_encoder_bundled=True)
    unflagged = video_download_plan("o/bundled", family="ltx", mlx_packs=BUNDLED_PACKS, mlx_text_encoder=GEMMA_REF, text_encoder_bundled=False)

    assert flagged == (DownloadTarget("o/pack", PIN, PACK.files), gemma)
    assert unflagged == (DownloadTarget("o/bundled", PIN, BUNDLED_PACK.files),)


@pytest.mark.parametrize("model_ref", ["o/pack", f"O/PACK@{PIN}"])
def test_video_plan_pack_by_its_repo_id_is_its_pinned_files_whatever_the_case_or_pin(model_ref: str, gemma: DownloadTarget) -> None:
    assert _plan(model_ref) == (DownloadTarget("o/pack", PIN, PACK.files), gemma)


def test_video_plan_pack_at_another_revision_is_the_whole_repo_at_that_revision(gemma: DownloadTarget) -> None:
    plan = _plan("o/pack@" + "b2" * 20)

    assert plan == (DownloadTarget("o/pack", "b2" * 20), gemma)
    assert plan[0].allow_patterns is None


def test_video_plan_local_mlx_layout_needs_only_gemma(tmp_path: Path, gemma: DownloadTarget) -> None:
    (tmp_path / "connector.safetensors").write_bytes(b"")

    assert _plan(str(tmp_path)) == (gemma,)


def test_video_plan_local_bundled_pack_needs_nothing(tmp_path: Path) -> None:
    (tmp_path / "connector.safetensors").write_bytes(b"")
    (tmp_path / "text_encoder.safetensors").write_bytes(b"")
    (tmp_path / "text_encoder_config.json").write_text("{}")

    assert _plan(str(tmp_path)) == ()


def test_video_plan_local_layout_with_bundled_flag_needs_nothing(tmp_path: Path) -> None:
    (tmp_path / "connector.safetensors").write_bytes(b"")

    assert _plan(str(tmp_path), text_encoder_bundled=True) == ()


def test_video_plan_other_local_folder_is_empty(tmp_path: Path) -> None:
    assert _plan(str(tmp_path)) == ()


def test_video_plan_not_a_repo_id_is_empty() -> None:
    assert _plan("/missing/model") == ()


@pytest.mark.parametrize("text_encoder", [None, "", "/local/gemma"])
def test_video_plan_needing_gemma_needs_its_reference(text_encoder: str | None) -> None:
    with pytest.raises(ValueError, match="Gemma 3"):
        video_download_plan(f"o/pack@{PIN}", family="ltx", mlx_packs=PACKS, mlx_text_encoder=text_encoder)


def test_video_plan_without_gemma_needs_no_reference() -> None:
    assert video_download_plan(f"o/bundled@{PIN}", family="ltx", mlx_packs=BUNDLED_PACKS) == (DownloadTarget("o/bundled", PIN, BUNDLED_PACK.files),)


# --- enhancer ---


def test_enhancer_plan_uses_mlx_lm_patterns() -> None:
    assert enhancer_download_plan("McG-221/Qwen3.5-4B-heretic-mlx-4Bit", "7a6b22e4") == (DownloadTarget("McG-221/Qwen3.5-4B-heretic-mlx-4Bit", "7a6b22e4", MLX_LM_ALLOW_PATTERNS),)


def test_enhancer_plan_local_folder_is_empty(tmp_path: Path) -> None:
    assert enhancer_download_plan(str(tmp_path), None) == ()


# --- image (no mflux needed) ---


def test_image_plan_local_folder_or_non_repo_is_empty(tmp_path: Path) -> None:
    assert image_download_plan(str(tmp_path), family="zimage") == ()
    assert image_download_plan("/missing/model", family="zimage") == ()


def test_image_plan_unknown_family_is_empty() -> None:
    assert image_download_plan("owner/repo", family="unknown") == ()


@pytest.mark.parametrize("family", ["flux2_klein", "zimage", "krea2", "ideogram4"])
def test_image_plan_is_one_cache_target_with_the_family_patterns(monkeypatch: pytest.MonkeyPatch, family: str) -> None:
    monkeypatch.setattr(mac_plan, "_mflux_patterns", lambda name: [f"{name}/*.safetensors", "x.json"])

    assert image_download_plan("owner/repo", family=family) == (DownloadTarget("owner/repo", None, (f"{family}/*.safetensors", "x.json")),)


# --- mac-only pins against the installed libraries ---


def test_image_plans_equal_mflux_own_patterns() -> None:
    pytest.importorskip("mflux")
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
    from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    expected = {
        "flux2_klein": Flux2KleinWeightDefinition.get_download_patterns(),
        "zimage": ZImageWeightDefinition.get_download_patterns(),
        "krea2": Krea2WeightDefinition.get_download_patterns(ModelConfig.krea2().model_name),
        "ideogram4": Ideogram4WeightDefinition.get_download_patterns(),
    }
    for family, patterns in expected.items():
        assert image_download_plan("owner/repo", family=family) == (DownloadTarget("owner/repo", None, tuple(patterns)),)


def test_krea2_plan_skips_the_diffusers_transformer_shards() -> None:
    pytest.importorskip("mflux")

    (target,) = image_download_plan("unsloth/Krea-2-Turbo", family="krea2")

    assert target.allow_patterns is not None
    assert "turbo.safetensors" in target.allow_patterns
    assert "transformer/*.safetensors" not in target.allow_patterns


def test_mlx_lm_allow_patterns_equal_mlx_lm_defaults() -> None:
    utils = pytest.importorskip("mlx_lm.utils")

    assert tuple(utils.DEFAULT_ALLOW_PATTERNS) == MLX_LM_ALLOW_PATTERNS


def test_module_imports_nothing_heavy() -> None:
    code = "import sys; import zvisiongenerator.backends.mac.download_plan; heavy = [m for m in ('torch', 'diffusers', 'transformers', 'mflux', 'mlx', 'mlx_lm') if m in sys.modules]; assert not heavy, heavy"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


# --- MiniMax H3 ---


def _shipped_h3_ref() -> str:
    from zvisiongenerator.utils.config import load_config

    return load_config()["model_aliases"]["h3"]


H3_REF = _shipped_h3_ref()


@pytest.fixture(scope="module")
def h3_config() -> dict:
    from zvisiongenerator.utils.config import load_config

    return load_config()


@pytest.fixture
def h3_plan_inputs(h3_config: dict):
    from zvisiongenerator.utils.h3_preset import h3_mlx_preset, h3_turbo_lora

    return h3_mlx_preset(h3_config), h3_turbo_lora(h3_config)[0]


def test_h3_plan_is_the_int8_transformer_the_fl2va_components_and_the_lora(h3_plan_inputs) -> None:
    preset, lora = h3_plan_inputs

    transformer, components, turbo = h3_mlx_download_plan(H3_REF, preset, lora)

    assert (transformer.repo_id, transformer.revision, transformer.allow_patterns) == (preset.transformer.repo_id, preset.transformer.revision, preset.transformer.files)
    assert (components.repo_id, components.revision, components.allow_patterns) == (preset.components.repo_id, preset.components.revision, preset.components.files)
    assert turbo == DownloadTarget(lora.repo_id, lora.revision, lora.files)


def test_h3_plan_lists_every_file_literally(h3_plan_inputs) -> None:
    from zvisiongenerator.downloads.target import is_literal

    plan = h3_mlx_download_plan(H3_REF, *h3_plan_inputs)

    assert all(target.allow_patterns and all(is_literal(name) for name in target.allow_patterns) for target in plan)
    assert all(target.ignore_patterns is None and target.local_dir is None for target in plan)


def test_h3_plan_for_a_local_folder_has_no_component_target(tmp_path: Path, h3_plan_inputs) -> None:
    plan = h3_mlx_download_plan(str(tmp_path), *h3_plan_inputs)

    assert [target.repo_id for target in plan] == ["water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "lightx2v/Minimax-h3-Turbo"]


def test_h3_plan_for_a_local_folder_with_a_transformer_has_no_transformer_target(tmp_path: Path, h3_plan_inputs) -> None:
    (tmp_path / "FL2VA" / "transformer").mkdir(parents=True)
    (tmp_path / "FL2VA" / "transformer" / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    plan = h3_mlx_download_plan(str(tmp_path), *h3_plan_inputs)

    assert [target.repo_id for target in plan] == ["lightx2v/Minimax-h3-Turbo"]


def test_h3_plan_for_a_converted_folder_is_the_components_without_the_lora(tmp_path: Path, h3_plan_inputs) -> None:
    from zvisiongenerator.utils.h3_converted import write_converted_h3

    preset, _lora = h3_plan_inputs
    (tmp_path / "FL2VA" / "transformer").mkdir(parents=True)
    (tmp_path / "FL2VA" / "transformer" / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    write_converted_h3(tmp_path, {"storage": "mlx-int8", "turbo": "merged"})

    plan = h3_mlx_download_plan(str(tmp_path), *h3_plan_inputs)

    assert [target.repo_id for target in plan] == [preset.components.repo_id]


def test_h3_plan_takes_the_presets_components_whatever_the_reference_revision(h3_plan_inputs) -> None:
    preset, _lora = h3_plan_inputs
    (_transformer, components, _lora) = h3_mlx_download_plan("MiniMaxAI/MiniMax-H3@other", *h3_plan_inputs)

    assert components.revision == preset.components.revision


def test_h3_plan_downloads_what_the_backend_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, h3_plan_inputs) -> None:
    """Each plan target is the (repo, revision, allow) the runtime snapshot call uses, or the two diverge."""
    from zvisiongenerator.backends.mac import h3_video

    preset, lora = h3_plan_inputs
    monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda repo_id, revision: [])
    calls: list[dict] = []

    def download(**kwargs) -> str:
        calls.append(kwargs)
        folder = tmp_path / kwargs["repo_id"].replace("/", "--")
        for name in kwargs["allow_patterns"]:
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(b"x")
        return str(folder)

    h3_video.resolve_mlx_files(H3_REF, preset, lora=lora, snapshot_download=download)

    runtime = {(call["repo_id"], call["revision"], tuple(call["allow_patterns"])) for call in calls}
    planned = {(target.repo_id, target.revision, target.allow_patterns) for target in h3_mlx_download_plan(H3_REF, preset, lora)}
    assert len(calls) == len(planned) == 3
    assert planned == runtime


class TestH3PlanPresence:
    """The plan reads as present exactly when the backend resolves its files without the network (SC-10, macOS)."""

    @pytest.fixture
    def cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        import huggingface_hub.constants

        from zvisiongenerator.utils import model_files

        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache_dir))
        monkeypatch.setattr(model_files, "huggingface_cache_dir", lambda: cache_dir)
        return cache_dir

    @staticmethod
    def _snapshot(cache: Path, target: DownloadTarget) -> Path:
        return cache / f"models--{target.repo_id.replace('/', '--')}" / "snapshots" / (target.revision or "main")

    def _download(self, cache: Path, target: DownloadTarget) -> None:
        for name in target.allow_patterns or ():
            path = self._snapshot(cache, target) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    @staticmethod
    def _resolves_offline(preset, lora) -> bool:
        from zvisiongenerator.backends.mac.h3_video import resolve_mlx_files

        def no_network(**_kwargs):
            raise AssertionError("downloaded")

        try:
            resolve_mlx_files(H3_REF, preset, lora=lora, snapshot_download=no_network)
        except AssertionError:
            return False
        return True

    def test_nothing_is_present_before_the_download(self, cache: Path, h3_plan_inputs) -> None:
        from zvisiongenerator.downloads.presence import target_present_offline

        plan = h3_mlx_download_plan(H3_REF, *h3_plan_inputs)

        assert [target_present_offline(target) for target in plan] == [False, False, False]
        assert self._resolves_offline(*h3_plan_inputs) is False

    def test_the_downloaded_plan_is_present_and_resolves_offline(self, cache: Path, h3_plan_inputs) -> None:
        from zvisiongenerator.downloads.presence import target_present_offline

        plan = h3_mlx_download_plan(H3_REF, *h3_plan_inputs)
        for target in plan:
            self._download(cache, target)

        assert [target_present_offline(target) for target in plan] == [True, True, True]
        assert self._resolves_offline(*h3_plan_inputs) is True

    @pytest.mark.parametrize(
        ("index", "missing"), [(0, "model-00008-of-00008.safetensors"), (1, "FL2VA/text_encoder/model-00014-of-00014.safetensors"), (2, "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors")]
    )
    def test_one_missing_file_makes_its_target_absent_and_the_backend_download(self, cache: Path, h3_plan_inputs, index: int, missing: str) -> None:
        from zvisiongenerator.downloads.presence import target_present_offline

        plan = h3_mlx_download_plan(H3_REF, *h3_plan_inputs)
        for target in plan:
            self._download(cache, target)
        (self._snapshot(cache, plan[index]) / missing).unlink()

        assert [target_present_offline(target) for target in plan] == [position != index for position in range(3)]
        assert self._resolves_offline(*h3_plan_inputs) is False
