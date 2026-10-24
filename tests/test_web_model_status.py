"""Tests for the Web UI model download / memory-fit status payload."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from zvisiongenerator.utils.model_memory import CudaMemoryEstimate, LtxCudaMemoryEstimate, MemoryBudget
from zvisiongenerator.web import model_status
from zvisiongenerator.web.model_status import VIDEO_ESTIMATE_NOTE, describe_model_status

_GIB = 1024**3


def _finder(mapping: dict[str, Path]):
    return lambda reference: mapping.get(reference)


def _write_sized(path: Path, gib: float) -> Path:
    """Write a header-only safetensors file whose weights load as *gib* GiB of bfloat16."""
    header = json.dumps({"w": {"dtype": "BF16", "shape": [int(gib * _GIB / 2 / 1024), 1024], "data_offsets": [0, 0]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


def _make_mlx_pack(root: Path, *, bundled: bool) -> Path:
    """Build a fake LTX MLX pack: LTX-2.5 (bundled Gemma 4, conv decoder) or LTX-2.3 (plain decoder)."""
    pack = root / ("ltx25" if bundled else "ltx23")
    _write_sized(pack / "transformer-distilled-1.1.safetensors", 10.5)
    _write_sized(pack / "connector.safetensors", 5.9)
    _write_sized(pack / ("vae_decoder_conv.safetensors" if bundled else "vae_decoder.safetensors"), 0.75)
    if bundled:
        _write_sized(pack / "text_encoder.safetensors", 7.5)
        (pack / "text_encoder_config.json").write_text("{}", encoding="utf-8")
    return pack


class TestDescribeMlxVideoStatus:
    @pytest.fixture(autouse=True)
    def _gemma_repo(self, monkeypatch):
        monkeypatch.setattr(model_status, "ltx_mlx_text_encoder_repo", lambda: "org/gemma")

    def _gemma_dir(self, root: Path) -> Path:
        _write_sized(root / "gemma" / "model-00001-of-00001.safetensors", 7.5)
        return root / "gemma"

    def test_a_bundled_pack_is_downloaded_without_gemma_3_and_estimated(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=True)

        status = describe_model_status("owner/ltx25", kind="video", budget=MemoryBudget(24 * _GIB, unified_bytes=32 * _GIB), find_local_dir=_finder({"owner/ltx25": pack}))

        assert status["downloaded"] is True
        assert status["memory_fit"] is not None
        assert status["memory_fit"]["kind"] == "unified"
        assert set(status["memory_fit"]) == {"kind", "budget_gb", "by_quantize", "without_low_memory", "note"}
        assert set(status["memory_fit"]["by_quantize"]) == {"none"}

    def test_a_bundled_pack_is_estimated_the_same_with_or_without_a_gemma_3_copy(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=True)
        budget = MemoryBudget(24 * _GIB, unified_bytes=32 * _GIB)

        alone = describe_model_status("owner/ltx25", kind="video", budget=budget, find_local_dir=_finder({"owner/ltx25": pack}))
        with_gemma = describe_model_status("owner/ltx25", kind="video", budget=budget, find_local_dir=_finder({"owner/ltx25": pack, "org/gemma": self._gemma_dir(tmp_path)}))

        assert alone == with_gemma

    def test_a_2_3_pack_without_gemma_3_is_not_downloaded(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=False)

        status = describe_model_status("owner/ltx23", kind="video", budget=MemoryBudget(24 * _GIB, unified_bytes=32 * _GIB), find_local_dir=_finder({"owner/ltx23": pack}))

        assert status == {"downloaded": False, "memory_fit": None}

    def test_a_2_3_pack_with_gemma_3_keeps_its_payload(self, tmp_path):
        pack = _make_mlx_pack(tmp_path, bundled=False)
        finder = _finder({"owner/ltx23": pack, "org/gemma": self._gemma_dir(tmp_path)})

        status = describe_model_status("owner/ltx23", kind="video", budget=MemoryBudget(24 * _GIB, unified_bytes=32 * _GIB), find_local_dir=finder)

        assert status == {
            "downloaded": True,
            "memory_fit": {
                "kind": "unified",
                "budget_gb": 24.0,
                "by_quantize": {"none": {"status": "fits", "required_gb": 15.1}},
                "without_low_memory": {"status": "too_large", "required_gb": 36.2},
                "note": VIDEO_ESTIMATE_NOTE,
            },
        }


class TestDescribeLtxStatusGolden:
    """G5: the LTX MLX status dicts, pinned so the family argument cannot move them."""

    @pytest.fixture(autouse=True)
    def _gemma_repo(self, monkeypatch):
        monkeypatch.setattr(model_status, "ltx_mlx_text_encoder_repo", lambda: "org/gemma")

    @pytest.mark.parametrize("family", [pytest.param({}, id="default"), pytest.param({"family": "ltx"}, id="ltx")])
    def test_a_bundled_pack_status_is_pinned(self, tmp_path, family):
        pack = _make_mlx_pack(tmp_path, bundled=True)

        status = describe_model_status("owner/ltx25", kind="video", budget=MemoryBudget(24 * _GIB, unified_bytes=32 * _GIB), find_local_dir=_finder({"owner/ltx25": pack}), **family)

        assert status == {
            "downloaded": True,
            "memory_fit": {
                "kind": "unified",
                "budget_gb": 24.0,
                "by_quantize": {"none": {"status": "fits", "required_gb": 15.1}},
                "without_low_memory": {"status": "too_large", "required_gb": 36.2},
                "note": VIDEO_ESTIMATE_NOTE,
            },
        }

    @pytest.mark.parametrize("family", [pytest.param({}, id="default"), pytest.param({"family": "ltx"}, id="ltx")])
    def test_a_2_3_pack_without_gemma_3_status_is_pinned(self, tmp_path, family):
        pack = _make_mlx_pack(tmp_path, bundled=False)

        status = describe_model_status("owner/ltx23", kind="video", budget=MemoryBudget(24 * _GIB, unified_bytes=32 * _GIB), find_local_dir=_finder({"owner/ltx23": pack}), **family)

        assert status == {"downloaded": False, "memory_fit": None}


class TestH3Status:
    """MiniMax H3 is downloaded when every target of its plan is on disk; on unified memory it gets an estimate."""

    _UNIFIED = MemoryBudget(24 * _GIB, unified_bytes=32 * _GIB)

    @pytest.fixture
    def darwin(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "darwin")

    @staticmethod
    def _status(present, *, budget=_UNIFIED, **extra):
        from zvisiongenerator.utils.config import load_config

        return describe_model_status(
            "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe",
            kind="video",
            budget=budget,
            find_local_dir=lambda reference: None,
            model_name="h3",
            app_config=load_config(),
            family="h3",
            target_present=present,
            **extra,
        )

    def test_macos_checks_the_three_targets_of_the_mac_plan(self, darwin):
        seen = []

        def present(target):
            seen.append((target.repo_id, target.revision))
            return True

        status = self._status(present)

        assert status["downloaded"] is True
        assert [repo for repo, _revision in seen] == ["water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "MiniMaxAI/MiniMax-H3", "lightx2v/Minimax-h3-Turbo"]

    @pytest.mark.parametrize("missing", ["water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8", "MiniMaxAI/MiniMax-H3", "lightx2v/Minimax-h3-Turbo"])
    def test_one_absent_target_is_not_downloaded_and_has_no_estimate(self, darwin, missing):
        status = self._status(lambda target: target.repo_id != missing)

        assert status == {"downloaded": False, "memory_fit": None}

    def test_macos_estimates_the_reference_clip(self, darwin):
        status = self._status(lambda target: True)

        assert status["memory_fit"] == {
            "kind": "unified",
            "budget_gb": 24.0,
            "by_quantize": {"none": {"status": "fits", "required_gb": 15.0}},
            "note": model_status.H3_ESTIMATE_NOTE,
        }

    def test_a_small_budget_is_classified(self, darwin):
        status = self._status(lambda target: True, budget=MemoryBudget(9 * _GIB, unified_bytes=16 * _GIB))

        assert status["memory_fit"]["by_quantize"]["none"] == {"status": "too_large", "required_gb": 15.0}

    def test_a_discrete_gpu_estimates_the_reference_clip(self, monkeypatch):
        import zvisiongenerator.backends as backends

        monkeypatch.setattr(backends.sys, "platform", "linux")
        seen = []

        status = self._status(lambda target: (seen.append(target.repo_id), True)[1], budget=MemoryBudget(10 * _GIB, system_bytes=30 * _GIB))

        assert status["downloaded"] is True
        assert status["memory_fit"] == {
            "kind": "discrete",
            "budget_gb": 10.0,
            "system_budget_gb": 30.0,
            "by_quantize": {"none": {"status": "fits", "required_gb": 7.2, "system_gb": 17.8}},
            "note": "Estimated for a 5 s clip at 768x448. The Workspace checks the size and length you choose.",
        }
        assert seen == ["MiniMaxAI/MiniMax-H3", "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit", "lightx2v/Minimax-h3-Turbo"]

    def test_the_status_agrees_with_the_backends_offline_resolution(self, darwin, tmp_path, monkeypatch):
        """SC-10: the same cache decides ``downloaded`` and whether the backend would download."""
        import huggingface_hub.constants

        from zvisiongenerator.backends.mac.h3_video import resolve_mlx_files
        from zvisiongenerator.downloads.plans import video_plan
        from zvisiongenerator.utils import model_files
        from zvisiongenerator.utils.config import load_config
        from zvisiongenerator.utils.h3_preset import h3_mlx_preset, h3_turbo_lora

        cache = tmp_path / "cache"
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
        monkeypatch.setattr(model_files, "huggingface_cache_dir", lambda: cache)
        config = load_config()
        ref = "MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe"

        def no_network(**_kwargs):
            raise AssertionError("downloaded")

        def resolves() -> bool:
            try:
                resolve_mlx_files(ref, h3_mlx_preset(config), lora=h3_turbo_lora(config)[0], snapshot_download=no_network)
            except AssertionError:
                return False
            return True

        def status() -> bool:
            return describe_model_status(ref, kind="video", budget=self._UNIFIED, find_local_dir=lambda reference: None, model_name="h3", app_config=config, family="h3")["downloaded"]

        assert (status(), resolves()) == (False, False)
        plan = video_plan(ref, family="h3", config=config, model_name="h3", upscale=False)
        for target in plan:
            for name in target.allow_patterns:
                path = cache / f"models--{target.repo_id.replace('/', '--')}" / "snapshots" / target.revision / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
        assert (status(), resolves()) == (True, True)
        (cache / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / "42ed227ee7df40d41602854ae760620d6eb651fe" / "FL2VA" / "audio_vae" / "model.safetensors").unlink()
        assert (status(), resolves()) == (False, False)


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
        calls: list[dict] = []

        def _estimate(model_dir, text_encoder_dir, *, low_memory, **clip):
            calls.append({"low_memory": low_memory, **clip})
            return (16 if low_memory else 24) * _GIB

        monkeypatch.setattr(model_status, "estimate_ltx_mlx_clip_memory", _estimate)
        monkeypatch.setattr(model_status, "ltx_mlx_text_encoder_repo", lambda: "org/gemma")

        without_encoder = describe_model_status("owner/ltx", kind="video", budget=MemoryBudget(10 * _GIB), find_local_dir=_finder({"owner/ltx": model_dir}))
        with_encoder = describe_model_status(
            "owner/ltx",
            kind="video",
            budget=MemoryBudget(10 * _GIB, unified_bytes=32 * _GIB),
            find_local_dir=_finder({"owner/ltx": model_dir, "org/gemma": tmp_path / "gemma"}),
        )

        assert without_encoder == {"downloaded": False, "memory_fit": None}
        assert with_encoder["downloaded"] is True
        assert with_encoder["memory_fit"]["by_quantize"] == {"none": {"status": "too_large", "required_gb": 16.0}}
        assert with_encoder["memory_fit"]["without_low_memory"] == {"status": "too_large", "required_gb": 24.0}
        assert with_encoder["memory_fit"]["note"] == VIDEO_ESTIMATE_NOTE
        # Both estimates are for the reference clip: 768x448, 121 frames (5 s), no upscale; staged is Low memory on.
        reference = {"width": 768, "height": 448, "frames": 121, "unified_bytes": 32 * _GIB}
        assert calls == [{"low_memory": True, **reference}, {"low_memory": False, **reference}]


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

    def test_cuda_video_is_estimated_at_the_models_level(self, tmp_path, monkeypatch):
        system = LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB)
        monkeypatch.setattr(model_status, "ltx_cuda_system_memory", lambda resolved, name, config, finder: system if name == "ltx-4" else None)

        status = describe_model_status("owner/ltx", kind="video", budget=MemoryBudget(10 * _GIB, 30 * _GIB), find_local_dir=_finder({"owner/ltx": tmp_path}), model_name="ltx-4", app_config={})

        fit = status["memory_fit"]
        assert status["downloaded"] is True
        assert (fit["kind"], fit["budget_gb"], fit["system_budget_gb"], fit["note"]) == ("discrete", 10.0, 30.0, VIDEO_ESTIMATE_NOTE)
        assert set(fit["by_quantize"]) == {"none"}
        assert fit["by_quantize"]["none"]["required_gb"] < fit["without_low_memory"]["required_gb"]  # the transformer stays on the GPU
        assert fit["by_quantize"]["none"]["system_gb"] == fit["without_low_memory"]["system_gb"]

    def test_cuda_video_weights_beyond_system_memory_are_too_large(self, tmp_path, monkeypatch):
        system = LtxCudaMemoryEstimate(text_encoder_bytes=12 * _GIB, transformer_bytes=10 * _GIB, resident_bytes=3 * _GIB)
        monkeypatch.setattr(model_status, "ltx_cuda_system_memory", lambda *args: system)

        status = describe_model_status("owner/ltx", kind="video", budget=MemoryBudget(24 * _GIB, 16 * _GIB), find_local_dir=_finder({"owner/ltx": tmp_path}), model_name="ltx-4", app_config={})

        assert status["memory_fit"]["by_quantize"]["none"]["status"] == "too_large"  # the backend refuses the load

    def test_cuda_video_without_its_files_or_level_is_not_estimated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_status, "ltx_cuda_system_memory", lambda *args: None)

        status = describe_model_status("owner/ltx", kind="video", budget=MemoryBudget(10 * _GIB, 30 * _GIB), find_local_dir=_finder({"owner/ltx": tmp_path}), model_name="ltx-4", app_config={})

        assert status == {"downloaded": True, "memory_fit": None}
