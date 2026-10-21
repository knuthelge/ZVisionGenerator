"""Tests for the first-run dialog's server side: the recommendation, the payload and the skip route."""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.web import config_contract as config_contract_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.first_run import first_run_payload, recommended_first_run_model

_CONFIG = {"first_run": {"large_model": "klein9b", "small_model": "klein4b", "min_memory_gb": 32}}
_BOTH = ("klein9b", "klein4b", "zit")
_MIB = 1024**2


class TestRecommendedFirstRunModel:
    """AC5.1: the large model needs 32 GB of unified memory or VRAM (within 5%); everything else gets the small one."""

    @pytest.mark.parametrize(
        ("budget", "expected"),
        [
            (MemoryBudget(gpu_bytes=int(26.8 * GIB), unified_bytes=32 * GIB), "klein9b"),
            (MemoryBudget(gpu_bytes=int(10.7 * GIB), unified_bytes=16 * GIB), "klein4b"),
            (MemoryBudget(gpu_bytes=32607 * _MIB, system_bytes=64 * GIB), "klein9b"),
            (MemoryBudget(gpu_bytes=24 * GIB, system_bytes=64 * GIB), "klein4b"),
            (MemoryBudget(gpu_bytes=int(30.5 * GIB), system_bytes=64 * GIB), "klein9b"),
            (MemoryBudget(gpu_bytes=int(30.3 * GIB), system_bytes=64 * GIB), "klein4b"),
            (None, "klein4b"),
            (MemoryBudget(gpu_bytes=64 * GIB), "klein4b"),
        ],
    )
    def test_picks_by_memory(self, budget, expected):
        assert recommended_first_run_model(_CONFIG, budget, _BOTH) == expected

    def test_a_large_model_that_is_not_listed_falls_back_to_the_small_one(self):
        budget = MemoryBudget(gpu_bytes=24 * GIB, unified_bytes=64 * GIB)

        assert recommended_first_run_model(_CONFIG, budget, ("klein4b", "zit")) == "klein4b"

    def test_no_recommendation_when_neither_model_is_listed(self):
        budget = MemoryBudget(gpu_bytes=24 * GIB, unified_bytes=64 * GIB)

        assert recommended_first_run_model(_CONFIG, budget, ("zit", "krea2")) is None
        assert recommended_first_run_model(_CONFIG, budget, ()) is None

    def test_no_recommendation_without_a_first_run_section(self):
        assert recommended_first_run_model({}, MemoryBudget(gpu_bytes=GIB, unified_bytes=64 * GIB), _BOTH) is None
        assert recommended_first_run_model({"first_run": None}, None, _BOTH) is None


class TestFirstRunPayload:
    """AC5.3: the payload says whether the dialog was skipped and which model it preselects."""

    @pytest.mark.parametrize(
        ("ui", "skipped"),
        [({"first_run_skipped": True}, True), ({"first_run_skipped": False}, False), ({}, False), (None, False), ({"first_run_skipped": "yes"}, False)],
    )
    def test_skipped_only_when_the_flag_is_true(self, ui, skipped):
        payload = first_run_payload({**_CONFIG, "ui": ui}, None, _BOTH)

        assert payload == {"skipped": skipped, "recommended_model": "klein4b"}

    def test_the_recommendation_follows_the_budget(self):
        budget = MemoryBudget(gpu_bytes=int(26.8 * GIB), unified_bytes=32 * GIB)

        assert first_run_payload(_CONFIG, budget, _BOTH)["recommended_model"] == "klein9b"


class TestPackagedFirstRunConfig:
    """AC5.2: the packaged config carries per-platform sizes for the image aliases and valid first-run models."""

    @pytest.fixture
    def config(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils import config as config_module

        monkeypatch.setattr(config_module, "get_ziv_data_dir", lambda: tmp_path / ".ziv")
        return load_config()

    @pytest.mark.parametrize("alias", ["zit", "klein9b", "klein4b", "krea2", "ideo"])
    def test_an_image_alias_has_a_size_label_for_every_platform(self, config, alias):
        sizes = config["model_labels"][alias]["download_size_label"]

        assert set(sizes) == {"darwin", "win32", "linux"}
        assert all(isinstance(label, str) and label.endswith(" GB") for label in sizes.values())

    def test_the_first_run_models_are_aliases(self, config):
        section = config["first_run"]

        assert section["large_model"] in config["model_aliases"]
        assert section["small_model"] in config["model_aliases"]
        assert section["min_memory_gb"] > 0

    def test_the_dialog_starts_unskipped(self, config):
        assert config["ui"]["first_run_skipped"] is False


class TestFirstRunSkipRoute:
    """AC5.4: skipping is stored in the user config next to the user's other keys."""

    @pytest.fixture
    def data_dir(self, monkeypatch, tmp_path):
        directory = tmp_path / "data"
        directory.mkdir()
        monkeypatch.setattr(config_contract_module, "get_ziv_data_dir", lambda: directory)
        return directory

    def test_sets_the_flag_and_keeps_other_keys(self, data_dir):
        (data_dir / "config.yaml").write_text(yaml.safe_dump({"ui": {"default_models": {"image": "zit"}}, "generation": {"default_size": "l"}}), encoding="utf-8")

        with TestClient(web_server.app) as client:
            response = client.post("/api/first-run/skip")

        assert response.status_code == 200
        assert response.json() == {"skipped": True}
        stored = yaml.safe_load((data_dir / "config.yaml").read_text(encoding="utf-8"))
        assert stored == {"ui": {"default_models": {"image": "zit"}, "first_run_skipped": True}, "generation": {"default_size": "l"}}

    def test_creates_the_user_config_when_there_is_none(self, data_dir):
        with TestClient(web_server.app) as client:
            assert client.post("/api/first-run/skip").status_code == 200

        assert yaml.safe_load((data_dir / "config.yaml").read_text(encoding="utf-8")) == {"ui": {"first_run_skipped": True}}

    def test_an_unreadable_user_config_is_a_422(self, data_dir):
        (data_dir / "config.yaml").write_text("- not\n- a mapping\n", encoding="utf-8")

        with TestClient(web_server.app) as client:
            response = client.post("/api/first-run/skip")

        assert response.status_code == 422

    def test_the_flag_is_not_a_writable_config_field(self):
        keys = {field.key for field in config_contract_module.WRITABLE_CONFIG_FIELDS}

        assert "ui.first_run_skipped" not in keys
