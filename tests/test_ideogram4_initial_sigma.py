"""Contract tests for the always-on Ideogram 4 first-step sigma shim."""

from __future__ import annotations

import sys

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="mflux/MLX backend is macOS-only")

_TIMESTEP_KWARGS = {
    "num_steps": 20,
    "height": 1024,
    "width": 1024,
    "mu": 0.0,
    "std": 1.75,
}
_DEFAULT_INITIAL_SIGMA = 1.004
_NATURAL_LAST_TIMESTEP = 0.00012339458044152707


def _import_backend_module():
    pytest.importorskip("mflux")
    pytest.importorskip("mlx.core")

    import zvisiongenerator.backends.image_mac as image_mac_module

    return image_mac_module


@pytest.fixture()
def image_mac_module(monkeypatch):
    image_mac = _import_backend_module()
    monkeypatch.setattr(image_mac, "IDEOGRAM4_INITIAL_SIGMA", _DEFAULT_INITIAL_SIGMA)
    monkeypatch.setattr(image_mac, "_initial_sigma_override", image_mac._INITIAL_SIGMA_UNSET)
    return image_mac


def test_make_timesteps_is_wrapped_on_image_mac_import(image_mac_module):
    scheduler_class = image_mac_module.Ideogram4Scheduler

    assert getattr(scheduler_class.make_timesteps, "_ziv_initial_sigma_wrapped", False) is True


def test_make_timesteps_applies_default_sigma_override_to_last_timestep(image_mac_module):
    scheduler_class = image_mac_module.Ideogram4Scheduler

    t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

    assert image_mac_module.IDEOGRAM4_INITIAL_SIGMA == _DEFAULT_INITIAL_SIGMA
    assert t_values[-1] == pytest.approx(1.0 - _DEFAULT_INITIAL_SIGMA)


def test_make_timesteps_only_changes_last_timestep_and_leaves_sigmas_untouched(image_mac_module, monkeypatch):
    scheduler_class = image_mac_module.Ideogram4Scheduler
    shimmed_t_values, shimmed_s_values = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

    monkeypatch.setattr(image_mac_module, "IDEOGRAM4_INITIAL_SIGMA", None)
    natural_t_values, natural_s_values = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

    assert np.allclose(shimmed_t_values[:-1], natural_t_values[:-1])
    assert np.allclose(shimmed_s_values, natural_s_values)
    assert shimmed_t_values[-1] == pytest.approx(1.0 - _DEFAULT_INITIAL_SIGMA)
    assert natural_t_values[-1] == pytest.approx(_NATURAL_LAST_TIMESTEP)


def test_make_timesteps_reads_sigma_dynamically_and_none_disables_override(image_mac_module, monkeypatch):
    scheduler_class = image_mac_module.Ideogram4Scheduler

    monkeypatch.setattr(image_mac_module, "IDEOGRAM4_INITIAL_SIGMA", None)
    natural_t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

    monkeypatch.setattr(image_mac_module, "IDEOGRAM4_INITIAL_SIGMA", 1.01)
    overridden_t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

    assert natural_t_values[-1] == pytest.approx(_NATURAL_LAST_TIMESTEP)
    assert overridden_t_values[-1] == pytest.approx(1.0 - 1.01)


def test_use_initial_sigma_sets_and_restores_previous_value(image_mac_module):
    scheduler_class = image_mac_module.Ideogram4Scheduler

    with image_mac_module._use_initial_sigma(1.006):
        overridden_t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

        assert image_mac_module._effective_initial_sigma() == 1.006
        assert overridden_t_values[-1] == pytest.approx(1.0 - 1.006)

        with image_mac_module._use_initial_sigma(1.002):
            nested_t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

            assert image_mac_module._effective_initial_sigma() == 1.002
            assert nested_t_values[-1] == pytest.approx(1.0 - 1.002)

        restored_t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

        assert image_mac_module._initial_sigma_override == 1.006
        assert restored_t_values[-1] == pytest.approx(1.0 - 1.006)

    default_t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

    assert image_mac_module._initial_sigma_override is image_mac_module._INITIAL_SIGMA_UNSET
    assert default_t_values[-1] == pytest.approx(1.0 - _DEFAULT_INITIAL_SIGMA)


def test_use_initial_sigma_none_disables_override_for_one_run(image_mac_module):
    scheduler_class = image_mac_module.Ideogram4Scheduler

    with image_mac_module._use_initial_sigma(None):
        natural_t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

        assert image_mac_module._effective_initial_sigma() is None
        assert natural_t_values[-1] == pytest.approx(_NATURAL_LAST_TIMESTEP)

    restored_t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

    assert image_mac_module._initial_sigma_override is image_mac_module._INITIAL_SIGMA_UNSET
    assert restored_t_values[-1] == pytest.approx(1.0 - _DEFAULT_INITIAL_SIGMA)


def test_effective_initial_sigma_distinguishes_sentinel_from_none(image_mac_module, monkeypatch):
    monkeypatch.setattr(image_mac_module, "_initial_sigma_override", image_mac_module._INITIAL_SIGMA_UNSET)
    assert image_mac_module._effective_initial_sigma() == _DEFAULT_INITIAL_SIGMA

    monkeypatch.setattr(image_mac_module, "_initial_sigma_override", 1.006)
    assert image_mac_module._effective_initial_sigma() == 1.006

    monkeypatch.setattr(image_mac_module, "_initial_sigma_override", None)
    assert image_mac_module._effective_initial_sigma() is None


def test_install_ideogram4_initial_sigma_is_idempotent(image_mac_module):
    scheduler_class = image_mac_module.Ideogram4Scheduler
    wrapped_make_timesteps = scheduler_class.make_timesteps

    image_mac_module._install_ideogram4_initial_sigma()
    t_values, _ = scheduler_class.make_timesteps(**_TIMESTEP_KWARGS)

    assert scheduler_class.make_timesteps is wrapped_make_timesteps
    assert getattr(scheduler_class.make_timesteps, "_ziv_initial_sigma_wrapped", False) is True
    assert t_values[-1] == pytest.approx(1.0 - _DEFAULT_INITIAL_SIGMA)


def test_shim_target_is_the_ideogram4_scheduler_symbol(image_mac_module):
    from mflux.models.ideogram4.model.ideogram4_scheduler import scheduler as ideogram4_scheduler_module

    assert image_mac_module.Ideogram4Scheduler is ideogram4_scheduler_module.Ideogram4Scheduler
    assert getattr(ideogram4_scheduler_module.Ideogram4Scheduler.make_timesteps, "_ziv_initial_sigma_wrapped", False) is True
