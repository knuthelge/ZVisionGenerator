"""Tests for the SeedVR2 upscaler backend, its platform flags, and the pure settings/tier helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zvisiongenerator import backends
from zvisiongenerator.core.seedvr2 import SeedVR2Upscaler
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.seedvr2 import (
    FP8_WEIGHTS,
    MLX_WEIGHTS,
    SeedVR2Settings,
    SeedVR2Tier,
    estimate_seedvr2_bytes,
    pick_seedvr2_tier,
    seedvr2_download_bytes,
    seedvr2_over_budget,
    seedvr2_row_name,
    seedvr2_settings,
    seedvr2_size_label,
    seedvr2_tier_label,
)

TIER_7B = SeedVR2Tier("seedvr2-7b", weights_gb=9.26, activation_floor_gb=10.79, activation_base_gb=1.88, activation_gb_per_mp=0.83)
TIER_3B = SeedVR2Tier("seedvr2-3b", weights_gb=4.72, activation_floor_gb=10.79, activation_base_gb=1.10, activation_gb_per_mp=0.89)
STREAMED_7B = SeedVR2Tier("seedvr2-7b", weights_gb=0.4, activation_floor_gb=5.0, activation_base_gb=0.6, activation_gb_per_mp=0.55, system_gb=22)
STREAMED_3B = SeedVR2Tier("seedvr2-3b", weights_gb=0.3, activation_floor_gb=5.0, activation_base_gb=0.6, activation_gb_per_mp=0.6, system_gb=12.5)


def _settings(model: str = "auto", reference_megapixels: float = 16) -> SeedVR2Settings:
    return SeedVR2Settings(model=model, softness=0.75, video_softness=0.0, reference_megapixels=reference_megapixels, tiers=(TIER_7B, TIER_3B))


def _config(**overrides):
    block = {
        "model": "auto",
        "softness": 0.75,
        "video_softness": 0.0,
        "reference_megapixels": 16,
        "tiers": [
            {"model": "seedvr2-7b", "weights_gb": 9.26, "activation_floor_gb": 10.79, "activation_base_gb": 1.88, "activation_gb_per_mp": 0.83},
            {"model": "seedvr2-3b", "weights_gb": 4.72, "activation_floor_gb": 10.79, "activation_base_gb": 1.10, "activation_gb_per_mp": 0.89},
        ],
    }
    block.update(overrides)
    return {"seedvr2": block}


# ── Backend ─────────────────────────────────────────────────────────────────


@pytest.fixture
def seedvr2_mac(monkeypatch):
    pytest.importorskip("mflux")
    pytest.importorskip("mlx.core")
    import mflux.models.seedvr2 as mflux_seedvr2
    from mflux.models.common.config.model_config import ModelConfig

    import zvisiongenerator.backends.mac.seedvr2 as module

    monkeypatch.setattr(ModelConfig, "precision", "other")
    seedvr2_class = MagicMock(name="SeedVR2")
    monkeypatch.setattr(mflux_seedvr2, "SeedVR2", seedvr2_class)
    monkeypatch.setattr(module.memory_mac, "materialize_weights", MagicMock(name="materialize_weights"))
    monkeypatch.setattr(module.memory_mac, "apply_buffer_cache_policy", MagicMock(name="apply_buffer_cache_policy"))
    return module, seedvr2_class, ModelConfig


@pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")
class TestMfluxSeedVR2Upscaler:
    def test_implements_the_protocol(self, seedvr2_mac):
        module, _cls, _config = seedvr2_mac

        assert isinstance(module.MfluxSeedVR2Upscaler(), SeedVR2Upscaler)

    def test_load_model_builds_the_variant_with_its_config(self, seedvr2_mac):
        module, seedvr2_class, model_config = seedvr2_mac

        loaded = module.MfluxSeedVR2Upscaler().load_model("seedvr2-7b", quantize=8, model_path="/models/seedvr2")

        assert loaded is seedvr2_class.return_value
        assert seedvr2_class.call_args.kwargs == {"quantize": 8, "model_path": "/models/seedvr2", "model_config": model_config.seedvr2_7b()}

    def test_load_model_uses_the_3b_config_for_the_3b_model(self, seedvr2_mac):
        module, seedvr2_class, model_config = seedvr2_mac

        module.MfluxSeedVR2Upscaler().load_model("seedvr2-3b", quantize=None, model_path="/models/seedvr2")

        assert seedvr2_class.call_args.kwargs["model_config"] == model_config.seedvr2_3b()

    def test_load_model_keeps_weights_resident_then_applies_the_cache_policy(self, seedvr2_mac):
        module, seedvr2_class, _model_config = seedvr2_mac

        module.MfluxSeedVR2Upscaler().load_model("seedvr2-3b", quantize=8, model_path="/m")

        module.memory_mac.materialize_weights.assert_called_once_with(seedvr2_class.return_value)
        module.memory_mac.apply_buffer_cache_policy.assert_called_once_with()

    def test_load_model_sets_bfloat16_precision(self, seedvr2_mac):
        import mlx.core as mx

        module, _cls, model_config = seedvr2_mac

        module.MfluxSeedVR2Upscaler().load_model("seedvr2-3b", quantize=8, model_path="/m")

        assert model_config.precision == mx.bfloat16

    def test_unknown_model_raises(self, seedvr2_mac):
        module, seedvr2_class, _model_config = seedvr2_mac

        with pytest.raises(ValueError, match="seedvr2-9b"):
            module.MfluxSeedVR2Upscaler().load_model("seedvr2-9b", quantize=8, model_path="/m")

        seedvr2_class.assert_not_called()

    def test_save_model_writes_to_the_path(self, seedvr2_mac):
        module, _cls, _config = seedvr2_mac
        model = MagicMock()

        module.MfluxSeedVR2Upscaler().save_model(model, Path("/models/.seedvr2/copy"))

        model.save_model.assert_called_once_with("/models/.seedvr2/copy")

    def test_upscale_image_sizes_by_the_shorter_side_and_returns_the_image(self, seedvr2_mac):
        import mlx.core as mx

        module, _cls, model_config = seedvr2_mac
        model = MagicMock()

        image = module.MfluxSeedVR2Upscaler().upscale_image(model, "/in.png", width=2048, height=1024, seed=7, softness=0.75)

        model.generate_image.assert_called_once_with(seed=7, image_path="/in.png", resolution=1024, softness=0.75)
        assert image is model.generate_image.return_value.image
        assert model_config.precision == mx.bfloat16

    def test_upscale_image_uses_the_width_when_it_is_the_shorter_side(self, seedvr2_mac):
        module, _cls, _config = seedvr2_mac
        model = MagicMock()

        module.MfluxSeedVR2Upscaler().upscale_image(model, "/in.png", width=1000, height=1500, seed=1, softness=0.0)

        assert model.generate_image.call_args.kwargs["resolution"] == 1000


# ── Platform flags ──────────────────────────────────────────────────────────


class TestPlatformFlags:
    @pytest.mark.parametrize(("platform", "supported"), [("darwin", True), ("win32", True), ("linux", True), ("freebsd14", False)])
    def test_supports_seedvr2_on_every_backend_platform(self, monkeypatch, platform, supported):
        monkeypatch.setattr(backends.sys, "platform", platform)

        assert backends.supports_seedvr2() is supported

    def test_upscaler_is_none_where_unsupported(self, monkeypatch):
        monkeypatch.setattr(backends.sys, "platform", "freebsd14")

        assert backends.get_seedvr2_upscaler() is None

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_cuda_platforms_get_the_fp8_engine(self, monkeypatch, platform):
        monkeypatch.setattr(backends.sys, "platform", platform)
        monkeypatch.setattr(backends, "_SEEDVR2_UPSCALERS", {})

        upscaler = backends.get_seedvr2_upscaler()

        assert isinstance(upscaler, SeedVR2Upscaler)
        assert upscaler.weights is FP8_WEIGHTS

    def test_upscaler_is_created_once_on_macos(self, monkeypatch):
        factory = MagicMock(return_value=MagicMock(name="upscaler"))
        monkeypatch.setattr(backends.sys, "platform", "darwin")
        monkeypatch.setattr(backends, "_SEEDVR2_UPSCALERS_MAP", {"darwin": factory})
        monkeypatch.setattr(backends, "_SEEDVR2_UPSCALERS", {})

        first = backends.get_seedvr2_upscaler()
        second = backends.get_seedvr2_upscaler()

        assert first is second is factory.return_value
        factory.assert_called_once_with()


# ── Memory estimate and tier choice ─────────────────────────────────────────


class TestEstimate:
    def test_activations_stay_on_the_floor_for_small_outputs(self):
        assert estimate_seedvr2_bytes(TIER_7B, 4) == int((9.26 + 10.79) * 1e9)

    def test_activations_follow_the_line_for_large_outputs(self):
        assert estimate_seedvr2_bytes(TIER_7B, 16) == int((9.26 + 1.88 + 0.83 * 16) * 1e9)
        assert estimate_seedvr2_bytes(TIER_3B, 16) == int((4.72 + 1.10 + 0.89 * 16) * 1e9)


class TestOverBudget:
    def test_over_only_when_the_estimate_is_strictly_larger(self):
        assert seedvr2_over_budget(101, 100) is True
        assert seedvr2_over_budget(100, 100) is False
        assert seedvr2_over_budget(99, 100) is False

    def test_never_over_an_unknown_budget(self):
        assert seedvr2_over_budget(10**12, None) is False


class TestPickTier:
    def test_budget_equal_to_the_7b_estimate_picks_7b(self):
        budget = estimate_seedvr2_bytes(TIER_7B, 16)

        assert pick_seedvr2_tier(_settings(), budget) is TIER_7B

    def test_one_byte_less_picks_3b(self):
        budget = estimate_seedvr2_bytes(TIER_7B, 16) - 1

        assert pick_seedvr2_tier(_settings(), budget) is TIER_3B

    def test_packaged_config_picks_7b_on_a_32gb_mac_and_3b_below_the_7b_estimate(self):
        settings = seedvr2_settings(load_config())

        assert pick_seedvr2_tier(settings, int(26.80e9)).model == "seedvr2-7b"
        assert pick_seedvr2_tier(settings, int(24.41e9)).model == "seedvr2-3b"

    def test_unknown_budget_picks_the_last_row(self):
        assert pick_seedvr2_tier(_settings(), None) is TIER_3B

    def test_nothing_within_budget_picks_the_last_candidate(self):
        assert pick_seedvr2_tier(_settings(), 1) is TIER_3B

    def test_configured_model_is_used_even_with_a_large_budget(self):
        assert pick_seedvr2_tier(_settings("seedvr2-3b"), 10**12) is TIER_3B

    def test_configured_7b_is_used_even_when_over_budget(self):
        assert pick_seedvr2_tier(_settings("seedvr2-7b"), 1) is TIER_7B


class TestPickStreamedTier:
    def _settings(self, model: str = "auto") -> SeedVR2Settings:
        return SeedVR2Settings(model=model, softness=0.75, video_softness=0.0, reference_megapixels=16, tiers=(STREAMED_7B, STREAMED_3B))

    def test_system_memory_equal_to_the_7b_need_picks_7b(self):
        assert pick_seedvr2_tier(self._settings(), 1, int(22e9)) is STREAMED_7B

    def test_less_system_memory_picks_3b(self):
        assert pick_seedvr2_tier(self._settings(), 10**12, int(22e9) - 1) is STREAMED_3B

    def test_the_gpu_budget_does_not_decide(self):
        assert pick_seedvr2_tier(self._settings(), 1, 10**12) is STREAMED_7B

    def test_unknown_system_memory_picks_the_last_row(self):
        assert pick_seedvr2_tier(self._settings(), 10**12, None) is STREAMED_3B

    def test_configured_model_wins(self):
        assert pick_seedvr2_tier(self._settings("seedvr2-7b"), 1, 1) is STREAMED_7B

    def test_packaged_config_picks_7b_with_32gb_and_3b_with_16gb(self):
        settings = seedvr2_settings(load_config(), FP8_WEIGHTS.tiers_key)

        assert pick_seedvr2_tier(settings, int(10.7e9), int(33.4e9)).model == "seedvr2-7b"
        assert pick_seedvr2_tier(settings, int(10.7e9), int(16.7e9)).model == "seedvr2-3b"


class TestLabels:
    def test_tier_label_names_the_engines_precision(self):
        assert seedvr2_tier_label("seedvr2-7b", MLX_WEIGHTS) == "SeedVR2 7B q8"
        assert seedvr2_tier_label("seedvr2-3b", MLX_WEIGHTS) == "SeedVR2 3B q8"
        assert seedvr2_tier_label("seedvr2-7b", FP8_WEIGHTS) == "SeedVR2 7B fp8"

    def test_the_fp8_7b_is_the_mixed_file_from_the_companion_repo(self):
        assert FP8_WEIGHTS.files["seedvr2-7b"][0] == ("AInVFX/SeedVR2_comfyUI", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors")
        assert {repo for _model, files in FP8_WEIGHTS.files.items() for repo, name in files if name.startswith("ema_vae")} == {"numz/SeedVR2_comfyUI"}

    def test_row_name(self):
        assert seedvr2_row_name("seedvr2-7b", MLX_WEIGHTS) == "seedvr2-7b@q8"
        assert seedvr2_row_name("seedvr2-3b", FP8_WEIGHTS) == "seedvr2-3b@fp8"

    def test_download_bytes_cover_the_transformer_and_the_vae(self):
        assert seedvr2_download_bytes("seedvr2-7b", MLX_WEIGHTS) == 16_479_334_424 + 501_324_814
        assert seedvr2_download_bytes("seedvr2-3b", MLX_WEIGHTS) == 6_783_018_808 + 501_324_814
        assert seedvr2_download_bytes("seedvr2-7b", FP8_WEIGHTS) == 8_466_296_338 + 501_324_814
        assert seedvr2_download_bytes("seedvr2-3b", FP8_WEIGHTS) == 3_391_544_696 + 501_324_814

    def test_size_label(self):
        assert seedvr2_size_label(16_980_659_238) == "17.0 GB"
        assert seedvr2_size_label(7_284_343_622) == "7.3 GB"


# ── Settings validation ─────────────────────────────────────────────────────


class TestSettings:
    def test_packaged_config_loads(self):
        settings = seedvr2_settings(load_config())

        assert settings.model == "auto"
        assert settings.softness == 0.75
        assert settings.video_softness == 0.0
        assert settings.reference_megapixels == 16
        assert [tier.model for tier in settings.tiers] == ["seedvr2-7b", "seedvr2-3b"]

    def test_packaged_config_loads_the_discrete_tiers_with_system_memory(self):
        settings = seedvr2_settings(load_config(), FP8_WEIGHTS.tiers_key)

        assert [tier.model for tier in settings.tiers] == ["seedvr2-7b", "seedvr2-3b"]
        assert all(tier.system_gb is not None for tier in settings.tiers)

    def test_discrete_tiers_parse_with_system_memory(self):
        config = _config(discrete_tiers=[{"model": "seedvr2-7b", "weights_gb": 0.4, "activation_floor_gb": 5.0, "activation_base_gb": 0.6, "activation_gb_per_mp": 0.55, "system_gb": 22}])

        assert seedvr2_settings(config, "discrete_tiers").tiers == (STREAMED_7B,)

    @pytest.mark.parametrize("system_gb", [0, -1, "22", True])
    def test_system_memory_must_be_a_positive_number(self, system_gb):
        config = _config(discrete_tiers=[{"model": "seedvr2-7b", "weights_gb": 0.4, "activation_floor_gb": 5, "activation_base_gb": 0.6, "activation_gb_per_mp": 0.55, "system_gb": system_gb}])

        with pytest.raises(ValueError, match=r"seedvr2\.discrete_tiers\[0\]\.system_gb"):
            seedvr2_settings(config, "discrete_tiers")

    def test_a_missing_discrete_list_names_its_key(self):
        with pytest.raises(ValueError, match=r"seedvr2\.discrete_tiers"):
            seedvr2_settings(_config(), "discrete_tiers")

    def test_a_valid_block_parses_into_tiers(self):
        settings = seedvr2_settings(_config())

        assert settings.tiers == (TIER_7B, TIER_3B)

    def test_a_pinned_model_is_accepted(self):
        assert seedvr2_settings(_config(model="seedvr2-3b")).model == "seedvr2-3b"

    @pytest.mark.parametrize("softness", [0, 1, 0.5])
    def test_softness_bounds_are_inclusive(self, softness):
        assert seedvr2_settings(_config(softness=softness)).softness == softness

    @pytest.mark.parametrize("video_softness", [0, 1, 0.5])
    def test_video_softness_bounds_are_inclusive(self, video_softness):
        settings = seedvr2_settings(_config(video_softness=video_softness))

        assert settings.video_softness == video_softness
        assert settings.softness == 0.75

    @pytest.mark.parametrize(
        ("overrides", "key"),
        [
            ({"tiers": None}, "seedvr2.tiers"),
            ({"tiers": []}, "seedvr2.tiers"),
            ({"tiers": [{"model": "seedvr2-9b", "weights_gb": 1, "activation_floor_gb": 1, "activation_base_gb": 1, "activation_gb_per_mp": 1}]}, "seedvr2.tiers[0].model"),
            ({"tiers": [{"model": ["seedvr2-3b"], "weights_gb": 1, "activation_floor_gb": 1, "activation_base_gb": 1, "activation_gb_per_mp": 1}]}, "seedvr2.tiers[0].model"),
            ({"tiers": [{"model": 3, "weights_gb": 1, "activation_floor_gb": 1, "activation_base_gb": 1, "activation_gb_per_mp": 1}]}, "seedvr2.tiers[0].model"),
            ({"tiers": [{"model": "seedvr2-3b", "activation_floor_gb": 1, "activation_base_gb": 1, "activation_gb_per_mp": 1}]}, "seedvr2.tiers[0].weights_gb"),
            ({"tiers": [{"model": "seedvr2-3b", "weights_gb": 0, "activation_floor_gb": 1, "activation_base_gb": 1, "activation_gb_per_mp": 1}]}, "seedvr2.tiers[0].weights_gb"),
            ({"tiers": [{"model": "seedvr2-3b", "weights_gb": 1, "activation_floor_gb": 1, "activation_base_gb": 1, "activation_gb_per_mp": 1, "quantize": 4}]}, "seedvr2.tiers[0]"),
            ({"model": "seedvr2-9b"}, "seedvr2.model"),
            ({"model": ["seedvr2-7b"]}, "seedvr2.model"),
            ({"model": 7}, "seedvr2.model"),
            ({"softness": 1.5}, "seedvr2.softness"),
            ({"softness": -0.1}, "seedvr2.softness"),
            ({"video_softness": 1.5}, "seedvr2.video_softness"),
            ({"video_softness": -0.1}, "seedvr2.video_softness"),
            ({"video_softness": "0"}, "seedvr2.video_softness"),
            ({"video_softness": True}, "seedvr2.video_softness"),
            ({"reference_megapixels": 0}, "seedvr2.reference_megapixels"),
        ],
    )
    def test_invalid_blocks_name_the_key(self, overrides, key):
        with pytest.raises(ValueError, match=key.replace("[", r"\[").replace("]", r"\]")):
            seedvr2_settings(_config(**overrides))

    @pytest.mark.parametrize("key", ["softness", "video_softness", "reference_megapixels"])
    def test_required_keys_must_be_present(self, key):
        config = _config()
        del config["seedvr2"][key]

        with pytest.raises(ValueError, match=f"seedvr2.{key}"):
            seedvr2_settings(config)

    def test_missing_block_raises(self):
        with pytest.raises(ValueError, match="seedvr2"):
            seedvr2_settings({})


def test_model_not_among_the_tiers_is_rejected():
    config = _config(tiers=[_config()["seedvr2"]["tiers"][1]], model="seedvr2-7b")

    with pytest.raises(ValueError, match="seedvr2.model"):
        seedvr2_settings(config)
