"""Tests for the Web UI model download / memory-fit status payload."""

from __future__ import annotations

from pathlib import Path

from zvisiongenerator.utils.model_memory import CudaMemoryEstimate, MemoryBudget
from zvisiongenerator.web import model_status
from zvisiongenerator.web.model_status import VIDEO_ESTIMATE_NOTE, describe_model_status

_GIB = 1024**3


def _finder(mapping: dict[str, Path]):
    return lambda reference: mapping.get(reference)


class TestDescribeModelStatus:
    def test_missing_model_is_not_downloaded_and_has_no_estimate(self):
        status = describe_model_status("owner/repo", kind="image", quantize_options=(4, 8), budget=MemoryBudget(10 * _GIB), find_local_dir=_finder({}))

        assert status == {"downloaded": False, "memory_fit": None}

    def test_image_model_reports_every_quantize_level(self, tmp_path, monkeypatch):
        estimates = {None: 16 * _GIB, 8: 13 * _GIB, 4: 5 * _GIB}
        monkeypatch.setattr(model_status, "estimate_image_memory", lambda model_dir, levels: {level: estimates[level] for level in levels})

        status = describe_model_status("owner/repo", kind="image", quantize_options=(4, 8), budget=MemoryBudget(10 * _GIB), find_local_dir=_finder({"owner/repo": tmp_path}))

        assert status["downloaded"] is True
        assert status["memory_fit"] == {
            "kind": "unified",
            "budget_gb": 10.0,
            "by_quantize": {
                "none": {"status": "too_large", "required_gb": 16.0},
                "4": {"status": "fits", "required_gb": 5.0},
                "8": {"status": "tight", "required_gb": 13.0},
            },
        }

    def test_no_budget_means_no_estimate(self, tmp_path):
        status = describe_model_status("owner/repo", kind="image", budget=None, find_local_dir=_finder({"owner/repo": tmp_path}))

        assert status == {"downloaded": True, "memory_fit": None}

    def test_ltx_mlx_video_needs_the_separate_text_encoder(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "ltx"
        model_dir.mkdir()
        (model_dir / "connector.safetensors").write_bytes(b"")
        monkeypatch.setattr(model_status, "estimate_ltx_mlx_memory", lambda model_dir, text_encoder_dir, *, low_memory: (16 if low_memory else 24) * _GIB)
        monkeypatch.setattr(model_status, "ltx_mlx_text_encoder_repo", lambda: "org/gemma")

        without_encoder = describe_model_status("owner/ltx", kind="video", budget=MemoryBudget(10 * _GIB), find_local_dir=_finder({"owner/ltx": model_dir}))
        with_encoder = describe_model_status(
            "owner/ltx",
            kind="video",
            budget=MemoryBudget(10 * _GIB),
            find_local_dir=_finder({"owner/ltx": model_dir, "org/gemma": tmp_path / "gemma"}),
        )

        assert without_encoder == {"downloaded": False, "memory_fit": None}
        assert with_encoder["downloaded"] is True
        assert with_encoder["memory_fit"]["by_quantize"] == {"none": {"status": "too_large", "required_gb": 16.0}}
        assert with_encoder["memory_fit"]["without_low_memory"] == {"status": "too_large", "required_gb": 24.0}
        assert with_encoder["memory_fit"]["note"] == VIDEO_ESTIMATE_NOTE


class TestDescribeCudaModelStatus:
    def test_image_model_reports_gpu_and_system_memory_per_level(self, tmp_path, monkeypatch):
        estimates = {None: CudaMemoryEstimate(4 * _GIB, 34 * _GIB), 8: CudaMemoryEstimate(4 * _GIB, 18 * _GIB), 4: CudaMemoryEstimate(11 * _GIB, 3 * _GIB)}
        monkeypatch.setattr(model_status, "estimate_cuda_image_memory", lambda model_dir, levels: {level: estimates[level] for level in levels})

        status = describe_model_status("owner/repo", kind="image", quantize_options=(4, 8), budget=MemoryBudget(10 * _GIB, 30 * _GIB), find_local_dir=_finder({"owner/repo": tmp_path}))

        assert status["memory_fit"] == {
            "kind": "discrete",
            "budget_gb": 10.0,
            "system_budget_gb": 30.0,
            "by_quantize": {
                "none": {"status": "tight", "required_gb": 4.0, "system_gb": 34.0},
                "4": {"status": "too_large", "required_gb": 11.0, "system_gb": 3.0},
                "8": {"status": "fits", "required_gb": 4.0, "system_gb": 18.0},
            },
        }

    def test_video_is_not_estimated(self, tmp_path):
        status = describe_model_status("owner/ltx", kind="video", budget=MemoryBudget(10 * _GIB, 30 * _GIB), find_local_dir=_finder({"owner/ltx": tmp_path}))

        assert status == {"downloaded": True, "memory_fit": None}
