"""Tests for the macOS (mflux, MLX) download planner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from zvisiongenerator.backends.mac import download_plan as mac_plan
from zvisiongenerator.backends.mac.download_plan import MLX_LM_ALLOW_PATTERNS, enhancer_download_plan, image_download_plan, video_download_plan
from zvisiongenerator.downloads.target import DownloadTarget

GEMMA = "mlx-community/gemma-3-12b-it-4bit"


@pytest.fixture
def gemma(monkeypatch: pytest.MonkeyPatch) -> DownloadTarget:
    monkeypatch.setattr(mac_plan, "ltx_mlx_text_encoder_repo", lambda: GEMMA)
    return DownloadTarget(GEMMA, allow_patterns=MLX_LM_ALLOW_PATTERNS)


# --- video ---


def test_video_plan_repo_id_is_the_whole_repo_then_gemma(gemma: DownloadTarget) -> None:
    plan = video_download_plan("dgrauet/ltx-2.3-mlx-q4", family="ltx")

    assert plan == (DownloadTarget("dgrauet/ltx-2.3-mlx-q4"), gemma)
    assert plan[0].allow_patterns is None
    assert plan[0].ignore_patterns is None


def test_video_plan_ignores_diffusers_arguments(gemma: DownloadTarget) -> None:
    plan = video_download_plan("dgrauet/ltx-2.3-mlx-q4", family="ltx", transformer_source=("a/b", "c"), latent_upsampler=("d/e", "f"), upscale=True)

    assert plan == (DownloadTarget("dgrauet/ltx-2.3-mlx-q4"), gemma)


def test_video_plan_ignores_the_repo_ignore_patterns(gemma: DownloadTarget) -> None:
    plan = video_download_plan("dgrauet/ltx-2.3-mlx-q4", family="ltx", repo_ignore=("LTX-2.5-*.safetensors", "samples/*"))

    assert plan == (DownloadTarget("dgrauet/ltx-2.3-mlx-q4"), gemma)


def test_video_plan_local_mlx_layout_needs_only_gemma(tmp_path: Path, gemma: DownloadTarget) -> None:
    (tmp_path / "connector.safetensors").write_bytes(b"")

    assert video_download_plan(str(tmp_path), family="ltx") == (gemma,)


def test_video_plan_other_local_folder_is_empty(tmp_path: Path, gemma: DownloadTarget) -> None:
    assert video_download_plan(str(tmp_path), family="ltx") == ()


def test_video_plan_not_a_repo_id_is_empty(gemma: DownloadTarget) -> None:
    assert video_download_plan("/missing/model", family="ltx") == ()


# --- enhancer ---


def test_enhancer_plan_uses_mlx_lm_patterns() -> None:
    assert enhancer_download_plan("McG-221/Qwen3.5-4B-heretic-mlx-4Bit", "7a6b22e4") == (DownloadTarget("McG-221/Qwen3.5-4B-heretic-mlx-4Bit", "7a6b22e4", MLX_LM_ALLOW_PATTERNS),)


def test_enhancer_plan_local_folder_is_empty(tmp_path: Path) -> None:
    assert enhancer_download_plan(str(tmp_path), None) == ()


def test_mlx_lm_allow_patterns_are_the_pinned_defaults() -> None:
    assert MLX_LM_ALLOW_PATTERNS == ("*.json", "model*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja")


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
