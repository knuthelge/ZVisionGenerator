"""Tests for MOSS-Image's build settings: source config, model folders, build records and shared stored copies."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.utils.built_model import BUILD_RECORD_NAME, read_build_record
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.moss_build import (
    TURBO_DIR_NAME,
    TURBO_LORA_NAME,
    MossSource,
    build_record,
    built_moss_base,
    is_built,
    is_moss_dir,
    is_turbo_dir,
    moss_base_dir,
    moss_source,
    shared_quant_source,
    turbo_lora_file,
    write_build_record,
)
from zvisiongenerator.utils.stored_quant import build_manifest, write_manifest

_SFT = "owner/sft@" + "a" * 40
_CONFIG = {"moss_image": {"sft": _SFT, "turbo_lora": {"repo": "owner/lora", "revision": "b" * 40, "filename": "lora.safetensors"}}}


def _source(sft: str = _SFT, lora_revision: str = "b" * 40) -> MossSource:
    return MossSource(sft, "owner/lora", lora_revision, "lora.safetensors")


@pytest.fixture
def models(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    return tmp_path / "models"


def _weights(folder: Path, size: int = 8) -> None:
    (folder / "transformer").mkdir(parents=True, exist_ok=True)
    (folder / "transformer" / "model.safetensors").write_bytes(b"x" * size)


def _built(models: Path, name: str, source: MossSource) -> Path:
    folder = models / name
    _weights(folder)
    write_build_record(folder, source, turbo=name == TURBO_DIR_NAME)
    return folder


def _stored_copy(models: Path, base: Path, bits: int = 4, fmt: str = "nf4") -> Path:
    copy = models / f"{base.name}@q{bits}"
    copy.mkdir()
    write_manifest(copy, build_manifest(base, bits, fmt))
    return copy


class TestMossSource:
    def test_reads_the_block_and_splits_the_sft_pin(self):
        source = moss_source(_CONFIG)

        assert source == _source()
        assert source.sft_repo == "owner/sft"
        assert source.sft_revision == "a" * 40

    @pytest.mark.parametrize(
        ("block", "key"),
        [
            (None, "moss_image"),
            ({"sft": "", "turbo_lora": _CONFIG["moss_image"]["turbo_lora"]}, "moss_image.sft"),
            ({"sft": "owner/sft", "turbo_lora": _CONFIG["moss_image"]["turbo_lora"]}, "moss_image.sft"),
            ({"sft": _SFT}, "moss_image.turbo_lora"),
            ({"sft": _SFT, "turbo_lora": {"repo": "o/l", "revision": "", "filename": "f"}}, "moss_image.turbo_lora"),
            ({"sft": _SFT, "turbo_lora": {"repo": "o/l", "revision": "r"}}, "moss_image.turbo_lora"),
        ],
    )
    def test_a_bad_block_names_the_key(self, block, key):
        config = {} if block is None else {"moss_image": block}

        with pytest.raises(ValueError, match=key):
            moss_source(config)


class TestMossDir:
    def test_only_the_two_folders_in_the_models_directory_count(self, models, tmp_path):
        assert is_moss_dir(models / "moss")
        assert is_moss_dir(models / "moss-sft")
        assert not is_moss_dir(tmp_path / "elsewhere" / "moss")
        assert not is_moss_dir(models / "zit")
        assert not is_moss_dir(models / "moss@q4")

    def test_only_the_moss_folder_is_turbo(self, models):
        assert is_turbo_dir(models / "moss")
        assert not is_turbo_dir(models / "moss-sft")
        assert not is_turbo_dir(models / "zit")


class TestDetection:
    def test_unbuilt_turbo_folder_is_distilled(self, models):
        info = detect_image_model(str(models / "moss"))

        assert (info.family, info.is_distilled, info.size) == ("moss", True, None)

    def test_unbuilt_sft_folder_is_not_distilled(self, models):
        info = detect_image_model(str(models / "moss-sft"))

        assert (info.family, info.is_distilled) == ("moss", False)

    @pytest.mark.parametrize(("index", "distilled"), [({"is_distilled": True}, True), ({"is_distilled": False}, False), ({}, False)])
    def test_built_folder_reads_the_index_flag(self, tmp_path, index, distilled):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "MOSSImagePipeline", **index}))

        info = detect_image_model(str(tmp_path))

        assert (info.family, info.is_distilled) == ("moss", distilled)

    def test_stored_copy_of_a_turbo_index_is_distilled(self, models):
        copy = models / "moss@q4"
        copy.mkdir(parents=True)
        (copy / "model_index.json").write_text(json.dumps({"_class_name": "MOSSImagePipeline", "is_distilled": True}))

        assert detect_image_model(str(copy)).is_distilled is True

    def test_a_pinned_revision_is_refused(self, tmp_path):
        index = tmp_path / "model_index.json"
        index.write_text(json.dumps({"_class_name": "MOSSImagePipeline"}))

        with patch("huggingface_hub.hf_hub_download", MagicMock(return_value=str(index))), pytest.raises(ValueError, match="pinned revision"):
            detect_image_model("OpenMOSS-Team/MOSS-Image-SFT@" + "a" * 40)


class TestBuildRecord:
    def test_only_turbo_records_the_lora(self):
        assert build_record(_source(), turbo=False) == {"version": 1, "sft": _SFT}
        assert build_record(_source(), turbo=True) == {
            "version": 1,
            "sft": _SFT,
            "turbo_lora": {"repo": "owner/lora", "revision": "b" * 40, "filename": "lora.safetensors"},
        }

    def test_a_written_record_is_built_for_the_same_source(self, models):
        turbo, sft = _built(models, "moss", _source()), _built(models, "moss-sft", _source())

        assert is_built(turbo, _source())
        assert is_built(sft, _source())
        assert (turbo / BUILD_RECORD_NAME).is_file()

    def test_the_turbo_flag_decides_what_the_record_holds_whatever_the_folder_is_called(self, models):
        partial = models / ".moss.1.partial"
        partial.mkdir(parents=True)

        write_build_record(partial, _source(), turbo=True)
        assert read_build_record(partial) == build_record(_source(), turbo=True)

        write_build_record(partial, _source(), turbo=False)
        assert read_build_record(partial) == build_record(_source(), turbo=False)

    def test_a_folder_without_a_record_is_not_built(self, models):
        folder = models / "moss"
        _weights(folder)

        assert not is_built(folder, _source())

    def test_an_sft_pin_change_unbuilds_both(self, models):
        turbo, sft = _built(models, "moss", _source()), _built(models, "moss-sft", _source())
        changed = _source(sft="owner/sft@" + "c" * 40)

        assert not is_built(turbo, changed)
        assert not is_built(sft, changed)

    def test_a_lora_pin_change_unbuilds_turbo_only(self, models):
        turbo, sft = _built(models, "moss", _source()), _built(models, "moss-sft", _source())
        changed = _source(lora_revision="d" * 40)

        assert not is_built(turbo, changed)
        assert is_built(sft, changed)


class TestTurboLora:
    def test_the_base_folder_of_a_stored_copy_finds_the_file(self, models):
        base = _built(models, "moss", _source())
        (base / TURBO_LORA_NAME).write_bytes(b"lora")
        copy = _stored_copy(models, base)

        assert turbo_lora_file(base) == base / TURBO_LORA_NAME
        assert turbo_lora_file(moss_base_dir(copy)) == base / TURBO_LORA_NAME

    def test_missing_file_is_none(self, models):
        base = _built(models, "moss", _source())
        sft = _built(models, "moss-sft", _source())

        assert turbo_lora_file(base) is None
        assert turbo_lora_file(moss_base_dir(models / "moss@q8")) is None
        assert turbo_lora_file(sft) is None


class TestMossBaseDir:
    def test_a_stored_copy_maps_to_its_base_and_any_other_path_to_itself(self, models, tmp_path):
        assert moss_base_dir(models / "moss@q4") == models / "moss"
        assert moss_base_dir(models / "moss-sft") == models / "moss-sft"
        assert moss_base_dir(tmp_path / "elsewhere" / "zit@q8") == tmp_path / "elsewhere" / "zit"


class TestBuiltMossBase:
    def test_a_base_folder_and_its_stored_copies_resolve_to_the_base(self, models):
        for name in ("moss", "moss-sft"):
            assert built_moss_base(models / name) == models / name
            assert built_moss_base(models / f"{name}@q4") == models / name
            assert built_moss_base(models / f"{name}@q8") == models / name

    def test_a_foreign_folder_or_repo_id_is_none(self, models, tmp_path):
        assert built_moss_base(tmp_path / "elsewhere" / "moss") is None
        assert built_moss_base(tmp_path / "elsewhere" / "moss@q4") is None
        assert built_moss_base(models / "zit@q4") is None
        assert built_moss_base("OpenMOSS-Team/MOSS-Image-SFT") is None


class TestSharedQuantSource:
    @pytest.fixture
    def both(self, models) -> tuple[Path, Path]:
        return _built(models, "moss", _source()), _built(models, "moss-sft", _source())

    def test_returns_the_other_aliass_copy_in_both_directions(self, models, both):
        turbo, sft = both
        sft_copy = _stored_copy(models, sft)

        assert shared_quant_source(turbo, 4, "nf4") == sft_copy
        assert shared_quant_source(sft, 4, "nf4") is None

        turbo_copy = _stored_copy(models, turbo)

        assert shared_quant_source(sft, 4, "nf4") == turbo_copy

    def test_none_when_the_copy_is_missing(self, both):
        turbo, sft = both

        assert shared_quant_source(turbo, 4, "nf4") is None
        assert shared_quant_source(sft, 4, "nf4") is None

    def test_none_when_the_copy_is_stale(self, models, both):
        turbo, sft = both
        _stored_copy(models, sft)
        _weights(sft, size=16)

        assert shared_quant_source(turbo, 4, "nf4") is None

    def test_none_for_another_level_or_format(self, models, both):
        turbo, sft = both
        _stored_copy(models, sft, bits=4)

        assert shared_quant_source(turbo, 8, "nf4") is None
        assert shared_quant_source(turbo, 4, "fp8") is None

    def test_none_when_the_sft_pins_differ(self, models):
        turbo = _built(models, "moss", _source(sft="owner/sft@" + "c" * 40))
        sft = _built(models, "moss-sft", _source())
        _stored_copy(models, sft)

        assert shared_quant_source(turbo, 4, "nf4") is None

    def test_a_lora_pin_change_keeps_the_sft_copy_shareable(self, models):
        turbo = _built(models, "moss", _source(lora_revision="d" * 40))
        sft = _built(models, "moss-sft", _source())
        copy = _stored_copy(models, sft)

        assert shared_quant_source(turbo, 4, "nf4") == copy

    def test_none_when_a_folder_is_unbuilt_or_foreign(self, models, tmp_path, both):
        turbo, sft = both
        _stored_copy(models, sft)
        (turbo / BUILD_RECORD_NAME).unlink()

        assert shared_quant_source(turbo, 4, "nf4") is None
        assert shared_quant_source(tmp_path / "elsewhere" / "moss", 4, "nf4") is None
