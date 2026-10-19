"""Tests for loading SeedVR2 through its stored 8-bit copy (macOS) or its cached FP8 files (CUDA)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from zvisiongenerator.converters.list_assets import list_models
from zvisiongenerator.seedvr2_loader import (
    DOWNLOADING_PHASE,
    LOADING_PHASE,
    SAVING_QUANT_PHASE,
    SEEDVR2_STORED_FORMAT,
    _delete_cached_file,
    build_seedvr2_setup,
    load_seedvr2,
    seedvr2_weights_dir,
    seedvr2_downloaded,
)
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS, SEEDVR2_REPO
from zvisiongenerator.utils.stored_quant import MANIFEST_NAME, write_manifest

(_, TRANSFORMER_7B), (_, VAE) = MLX_WEIGHTS.files["seedvr2-7b"]
(_, TRANSFORMER_3B), _ = MLX_WEIGHTS.files["seedvr2-3b"]


class _Parent:
    """One mock that records the order of calls made on the engine, the phase callback and the file deleter."""

    def __init__(self, *, save_error: BaseException | None = None, delete_error: BaseException | None = None, weights=MLX_WEIGHTS):
        self.manager = MagicMock()
        self.upscaler = self.manager.upscaler
        self.upscaler.weights = weights
        self.on_phase = self.manager.on_phase
        self.delete = self.manager.delete
        self.model = MagicMock(name="loaded")
        self.upscaler.load_model.return_value = self.model
        self.upscaler.save_model.side_effect = self._save
        self._save_error = save_error
        if delete_error is not None:
            self.delete.side_effect = delete_error

    def _save(self, model, path):
        if self._save_error is not None:
            raise self._save_error
        Path(path).mkdir(parents=True)
        (Path(path) / "model.safetensors").write_bytes(b"q")


@pytest.fixture
def models_dir(tmp_path):
    path = tmp_path / "models"
    path.mkdir()
    return path


def _make_copy(models_dir: Path, model: str = "seedvr2-7b", manifest: dict | None = None) -> Path:
    copy = seedvr2_weights_dir(model, models_dir, MLX_WEIGHTS)
    copy.mkdir(parents=True)
    write_manifest(copy, {"bits": 8, "backend_format": SEEDVR2_STORED_FORMAT, "model": model} if manifest is None else manifest)
    return copy


def _load(parent: _Parent, models_dir: Path, model: str = "seedvr2-7b"):
    return load_seedvr2(parent.upscaler, model, models_dir=models_dir, on_phase=parent.on_phase, delete_cached_file=parent.delete)


class TestCopyLocation:
    def test_copy_dir_is_hidden_and_named_by_model_and_bits(self, models_dir):
        assert seedvr2_weights_dir("seedvr2-3b", models_dir, MLX_WEIGHTS) == models_dir / ".seedvr2" / "seedvr2-3b@q8"

    def test_downloaded_needs_a_matching_manifest(self, models_dir):
        assert not seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)
        copy = seedvr2_weights_dir("seedvr2-7b", models_dir, MLX_WEIGHTS)
        copy.mkdir(parents=True)
        assert not seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)  # weights without a manifest are incomplete

        write_manifest(copy, {"bits": 8, "backend_format": SEEDVR2_STORED_FORMAT, "model": "seedvr2-7b"})

        assert seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)
        assert not seedvr2_downloaded("seedvr2-3b", models_dir, MLX_WEIGHTS)

    @pytest.mark.parametrize(
        "manifest",
        [
            {"bits": 4, "backend_format": SEEDVR2_STORED_FORMAT, "model": "seedvr2-7b"},
            {"bits": 8, "backend_format": "other-format", "model": "seedvr2-7b"},
            {"bits": 8, "backend_format": SEEDVR2_STORED_FORMAT, "model": "seedvr2-3b"},
        ],
    )
    def test_a_copy_made_another_way_is_not_current(self, models_dir, manifest):
        _make_copy(models_dir, manifest=manifest)

        assert not seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)

    def test_listings_ignore_the_copy_folder(self, tmp_path, models_dir):
        _make_copy(models_dir)

        assert list_models(tmp_path) == []

    def test_listings_skip_the_copy_folder_without_probing_it(self, tmp_path, models_dir):
        _make_copy(models_dir)

        with patch("zvisiongenerator.converters.list_assets.detect_image_model") as detect:
            detect.return_value = ImageModelInfo(family="flux2_klein", is_distilled=True, size="9b")
            assert list_models(tmp_path) == []

        detect.assert_not_called()


class TestFirstLoad:
    def test_calls_run_in_order_and_the_model_is_returned(self, models_dir):
        parent = _Parent()

        loaded = _load(parent, models_dir)

        assert loaded is parent.model
        copy = seedvr2_weights_dir("seedvr2-7b", models_dir, MLX_WEIGHTS)
        assert [c[0] for c in parent.manager.mock_calls] == [
            "on_phase",
            "upscaler.load_model",
            "on_phase",
            "upscaler.save_model",
            "delete",
            "delete",
        ]
        assert parent.on_phase.call_args_list == [call(DOWNLOADING_PHASE), call(SAVING_QUANT_PHASE)]
        parent.upscaler.load_model.assert_called_once_with("seedvr2-7b", quantize=8, model_path=SEEDVR2_REPO)
        saved_to = parent.upscaler.save_model.call_args.args[1]
        assert parent.upscaler.save_model.call_args.args[0] is parent.model
        assert saved_to.parent == copy.parent and saved_to != copy  # written to a hidden partial, then promoted
        assert (copy / "model.safetensors").is_file()
        assert seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)

    def test_the_manifest_records_the_model_bits_and_format(self, models_dir):
        _load(_Parent(), models_dir)

        manifest = json.loads((seedvr2_weights_dir("seedvr2-7b", models_dir, MLX_WEIGHTS) / MANIFEST_NAME).read_text())
        assert manifest == {"bits": 8, "backend_format": SEEDVR2_STORED_FORMAT, "model": "seedvr2-7b"}

    @pytest.mark.parametrize(("model", "transformer"), [("seedvr2-7b", TRANSFORMER_7B), ("seedvr2-3b", TRANSFORMER_3B)])
    def test_deletes_exactly_the_tiers_two_files(self, models_dir, model, transformer):
        parent = _Parent()

        _load(parent, models_dir, model)

        assert parent.delete.call_args_list == [call(SEEDVR2_REPO, transformer), call(SEEDVR2_REPO, VAE)]

    def test_the_7b_tier_never_touches_the_3b_file(self, models_dir):
        parent = _Parent()

        _load(parent, models_dir, "seedvr2-7b")

        assert call(SEEDVR2_REPO, TRANSFORMER_3B) not in parent.delete.call_args_list

    def test_a_failed_save_warns_deletes_nothing_and_leaves_no_folder(self, models_dir):
        parent = _Parent(save_error=OSError("disk full"))

        with pytest.warns(UserWarning, match="disk full"):
            loaded = _load(parent, models_dir)

        assert loaded is parent.model
        parent.delete.assert_not_called()
        assert not seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)
        assert list((models_dir / ".seedvr2").iterdir()) == []

    def test_a_failed_save_warning_names_the_caller_of_the_loader(self, models_dir):
        parent = _Parent(save_error=OSError("disk full"))

        with pytest.warns(UserWarning, match="disk full") as warned:
            _load(parent, models_dir)

        assert Path(warned[0].filename) == Path(__file__)

    def test_a_failed_delete_warning_names_the_caller_of_the_loader(self, models_dir):
        parent = _Parent(delete_error=OSError("busy"))

        with pytest.warns(UserWarning, match="busy") as warned:
            _load(parent, models_dir)

        assert {Path(item.filename) for item in warned} == {Path(__file__)}

    def test_a_failed_delete_warns_and_still_returns_the_model(self, models_dir):
        parent = _Parent(delete_error=OSError("busy"))

        with pytest.warns(UserWarning, match="busy") as warned:
            loaded = _load(parent, models_dir)

        assert loaded is parent.model
        assert seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)
        assert len(warned) == 2  # each file is tried even after the first fails
        assert parent.delete.call_count == 2

    def test_a_failed_load_propagates_and_stores_nothing(self, models_dir):
        parent = _Parent()
        parent.upscaler.load_model.side_effect = RuntimeError("offline")

        with pytest.raises(RuntimeError, match="offline"):
            _load(parent, models_dir)

        parent.upscaler.save_model.assert_not_called()
        parent.delete.assert_not_called()
        assert not (models_dir / ".seedvr2").exists()

    def test_files_are_deleted_only_once_the_copy_is_current(self, models_dir):
        parent = _Parent()
        current_at_delete: list[bool] = []
        parent.delete.side_effect = lambda *_: current_at_delete.append(seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS))

        _load(parent, models_dir)

        assert current_at_delete == [True, True]

    def test_an_interrupted_save_discards_the_partial_and_deletes_nothing(self, models_dir):
        parent = _Parent(save_error=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            _load(parent, models_dir)

        parent.delete.assert_not_called()
        assert list((models_dir / ".seedvr2").iterdir()) == []

    def test_a_stale_partial_is_swept(self, models_dir):
        copy = seedvr2_weights_dir("seedvr2-7b", models_dir, MLX_WEIGHTS)
        stale = copy.parent / f".{copy.name}.deadbeef.partial"
        stale.mkdir(parents=True)
        (stale / "model.safetensors").write_bytes(b"x")
        old = time.time() - 2 * 60 * 60
        os.utime(stale / "model.safetensors", (old, old))
        os.utime(stale, (old, old))

        _load(_Parent(), models_dir)

        assert not stale.exists()
        assert seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)

    def test_a_missing_on_phase_is_allowed(self, models_dir):
        parent = _Parent()

        load_seedvr2(parent.upscaler, "seedvr2-7b", models_dir=models_dir, on_phase=None, delete_cached_file=parent.delete)

        assert seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)


class TestLaterLoads:
    def test_the_copy_is_loaded_without_quantizing_or_downloading(self, models_dir):
        copy = _make_copy(models_dir)
        parent = _Parent()

        loaded = _load(parent, models_dir)

        assert loaded is parent.model
        assert parent.on_phase.call_args_list == [call(LOADING_PHASE)]
        parent.upscaler.load_model.assert_called_once_with("seedvr2-7b", quantize=None, model_path=str(copy))
        parent.upscaler.save_model.assert_not_called()
        parent.delete.assert_not_called()

    def test_a_second_load_after_the_first_uses_the_copy(self, models_dir):
        first, second = _Parent(), _Parent()

        _load(first, models_dir)
        _load(second, models_dir)

        assert second.on_phase.call_args_list == [call(LOADING_PHASE)]
        assert second.upscaler.load_model.call_args.kwargs["quantize"] is None

    def test_a_stale_copy_is_downloaded_again(self, models_dir):
        _make_copy(models_dir, manifest={"bits": 8, "backend_format": "older-format", "model": "seedvr2-7b"})
        parent = _Parent()

        _load(parent, models_dir)

        assert parent.on_phase.call_args_list == [call(DOWNLOADING_PHASE), call(SAVING_QUANT_PHASE)]
        assert seedvr2_downloaded("seedvr2-7b", models_dir, MLX_WEIGHTS)


class TestDownloadedFp8Files:
    """An engine that keeps no stored copy is handed its weights folder and downloads into it: no quantizing or deleting."""

    def _put(self, models_dir: Path, model: str, names: list[str]) -> Path:
        folder = seedvr2_weights_dir(model, models_dir, FP8_WEIGHTS)
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            (folder / name).write_bytes(b"w")
        return folder

    def test_the_folder_is_named_by_model_and_precision(self, models_dir):
        assert seedvr2_weights_dir("seedvr2-7b", models_dir, FP8_WEIGHTS) == models_dir / ".seedvr2" / "seedvr2-7b@fp8"

    def test_downloaded_needs_both_of_the_tiers_files_in_its_folder(self, models_dir):
        (_, transformer), (_, vae) = FP8_WEIGHTS.files["seedvr2-7b"]
        self._put(models_dir, "seedvr2-7b", [vae])

        assert not seedvr2_downloaded("seedvr2-7b", models_dir, FP8_WEIGHTS)

        self._put(models_dir, "seedvr2-7b", [transformer])

        assert seedvr2_downloaded("seedvr2-7b", models_dir, FP8_WEIGHTS)
        assert not seedvr2_downloaded("seedvr2-3b", models_dir, FP8_WEIGHTS)

    def test_a_stored_copy_does_not_count(self, models_dir):
        _make_copy(models_dir)

        assert not seedvr2_downloaded("seedvr2-7b", models_dir, FP8_WEIGHTS)

    def test_first_use_reports_downloading_and_hands_over_the_new_folder(self, models_dir):
        parent = _Parent(weights=FP8_WEIGHTS)

        loaded = _load(parent, models_dir)

        folder = seedvr2_weights_dir("seedvr2-7b", models_dir, FP8_WEIGHTS)
        assert loaded is parent.model
        assert parent.on_phase.call_args_list == [call(DOWNLOADING_PHASE)]
        parent.upscaler.load_model.assert_called_once_with("seedvr2-7b", quantize=None, model_path=str(folder))
        assert folder.is_dir()
        parent.upscaler.save_model.assert_not_called()
        parent.delete.assert_not_called()

    def test_downloaded_files_report_loading(self, models_dir):
        folder = self._put(models_dir, "seedvr2-3b", [name for _repo, name in FP8_WEIGHTS.files["seedvr2-3b"]])
        parent = _Parent(weights=FP8_WEIGHTS)

        _load(parent, models_dir, "seedvr2-3b")

        assert parent.on_phase.call_args_list == [call(LOADING_PHASE)]
        parent.upscaler.load_model.assert_called_once_with("seedvr2-3b", quantize=None, model_path=str(folder))


class TestDeleteCachedFile:
    def _layout(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create an HF cache layout with a snapshot link pointing at a blob; return ``(link, blob)``."""
        blob = tmp_path / "blobs" / "abc123"
        blob.parent.mkdir()
        blob.write_bytes(b"weights")
        link = tmp_path / "snapshots" / "rev" / TRANSFORMER_7B
        link.parent.mkdir(parents=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
        return link, blob

    def test_removes_the_link_and_its_blob(self, tmp_path):
        link, blob = self._layout(tmp_path)

        with patch("huggingface_hub.try_to_load_from_cache", return_value=str(link)) as lookup:
            _delete_cached_file(SEEDVR2_REPO, TRANSFORMER_7B)

        lookup.assert_called_once_with(SEEDVR2_REPO, TRANSFORMER_7B)
        assert not link.is_symlink()
        assert not blob.exists()

    @pytest.mark.parametrize("cached", [None, object()])
    def test_a_file_that_is_not_cached_is_left_alone(self, tmp_path, cached):
        link, blob = self._layout(tmp_path)

        with patch("huggingface_hub.try_to_load_from_cache", return_value=cached):
            _delete_cached_file(SEEDVR2_REPO, TRANSFORMER_7B)

        assert link.is_symlink()
        assert blob.exists()

    def test_a_file_already_gone_is_not_an_error(self, tmp_path):
        link, blob = self._layout(tmp_path)
        blob.unlink()
        link.unlink()

        with patch("huggingface_hub.try_to_load_from_cache", return_value=str(link)):
            _delete_cached_file(SEEDVR2_REPO, TRANSFORMER_7B)


class TestBuildSetup:
    def _config(self) -> dict:
        return {
            "seedvr2": {
                "model": "auto",
                "softness": 0.75,
                "reference_megapixels": 16,
                "tiers": [
                    {"model": "seedvr2-7b", "weights_gb": 9.26, "activation_floor_gb": 10.79, "activation_base_gb": 1.88, "activation_gb_per_mp": 0.83},
                    {"model": "seedvr2-3b", "weights_gb": 4.72, "activation_floor_gb": 10.79, "activation_base_gb": 1.10, "activation_gb_per_mp": 0.89},
                ],
            }
        }

    def test_picks_the_tier_and_reports_whether_its_copy_exists(self, models_dir):
        setup = build_seedvr2_setup(self._config(), weights=MLX_WEIGHTS, system_bytes=None, budget_bytes=40_000_000_000, models_dir=models_dir)

        assert setup.tier.model == "seedvr2-7b"
        assert setup.budget_bytes == 40_000_000_000
        assert setup.downloaded is False

        _make_copy(models_dir, "seedvr2-7b")

        assert build_seedvr2_setup(self._config(), weights=MLX_WEIGHTS, system_bytes=None, budget_bytes=40_000_000_000, models_dir=models_dir).downloaded is True

    def test_downloaded_follows_the_picked_tier(self, models_dir):
        _make_copy(models_dir, "seedvr2-7b")

        setup = build_seedvr2_setup(self._config(), weights=MLX_WEIGHTS, system_bytes=None, budget_bytes=10_000_000_000, models_dir=models_dir)

        assert setup.tier.model == "seedvr2-3b"
        assert setup.downloaded is False

    def test_fp8_weights_pick_from_the_discrete_tiers_by_system_memory(self, models_dir):
        config = self._config()
        config["seedvr2"]["discrete_tiers"] = [
            {"model": "seedvr2-7b", "weights_gb": 0.4, "activation_floor_gb": 5, "activation_base_gb": 0.6, "activation_gb_per_mp": 0.55, "system_gb": 22},
            {"model": "seedvr2-3b", "weights_gb": 0.3, "activation_floor_gb": 5, "activation_base_gb": 0.6, "activation_gb_per_mp": 0.6, "system_gb": 12.5},
        ]

        roomy = build_seedvr2_setup(config, weights=FP8_WEIGHTS, budget_bytes=10_000_000_000, system_bytes=32_000_000_000, models_dir=models_dir)
        small = build_seedvr2_setup(config, weights=FP8_WEIGHTS, budget_bytes=10_000_000_000, system_bytes=16_000_000_000, models_dir=models_dir)

        assert roomy.tier.model == "seedvr2-7b"
        assert small.tier.model == "seedvr2-3b"
        assert roomy.weights is FP8_WEIGHTS

    def test_an_invalid_block_raises(self, models_dir):
        with pytest.raises(ValueError, match="seedvr2"):
            build_seedvr2_setup({}, weights=MLX_WEIGHTS, system_bytes=None, budget_bytes=None, models_dir=models_dir)
