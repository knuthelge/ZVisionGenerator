"""Tests for stored-quant naming, manifests and folder handling."""

from __future__ import annotations

import json
import os

import pytest

from zvisiongenerator.utils import stored_quant as sq


def _make_source(root, name="atlas"):
    """Create a minimal diffusers-layout model folder with weights and detection files."""
    source = root / name
    (source / "transformer").mkdir(parents=True)
    (source / "scheduler").mkdir()
    (source / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "Flux2Transformer"], "scheduler": ["diffusers", "Scheduler"]}))
    (source / "transformer" / "config.json").write_text(json.dumps({"num_single_layers": 24}))
    (source / "transformer" / "weights.safetensors").write_bytes(b"x" * 64)
    (source / "scheduler" / "scheduler_config.json").write_text("{}")
    return source


class TestNames:
    def test_name_round_trip(self):
        assert sq.stored_quant_name("atlas", 8) == "atlas@q8"
        assert sq.parse_stored_quant_name("atlas@q8") == ("atlas", 8)
        assert sq.parse_stored_quant_name("my@model@q4") == ("my@model", 4)

    @pytest.mark.parametrize("name", ["atlas", "atlas@q6", "atlas@q8x", "@q8", "atlas-q8"])
    def test_other_names_are_not_stored_quants(self, name):
        assert sq.parse_stored_quant_name(name) is None

    def test_stored_quant_dir_is_a_sibling(self, tmp_path):
        assert sq.stored_quant_dir(tmp_path / "models" / "atlas", 4) == tmp_path / "models" / "atlas@q4"


class TestManifest:
    def test_current_after_writing_manifest(self, tmp_path):
        source = _make_source(tmp_path)
        stored = tmp_path / "atlas@q8"
        stored.mkdir()
        sq.write_manifest(stored, sq.build_manifest(source, 8, "mflux-1.0"))

        assert sq.is_current(stored, source, 8, "mflux-1.0")

    def test_missing_manifest_is_not_current(self, tmp_path):
        source = _make_source(tmp_path)
        stored = tmp_path / "atlas@q8"
        stored.mkdir()

        assert not sq.is_current(stored, source, 8, "mflux-1.0")

    def test_corrupt_manifest_is_not_current(self, tmp_path):
        source = _make_source(tmp_path)
        stored = tmp_path / "atlas@q8"
        stored.mkdir()
        (stored / sq.MANIFEST_NAME).write_text("{not json")

        assert not sq.is_current(stored, source, 8, "mflux-1.0")

    @pytest.mark.parametrize(("bits", "backend_format"), [(4, "mflux-1.0"), (8, "mflux-2.0")])
    def test_other_bits_or_format_is_stale(self, tmp_path, bits, backend_format):
        source = _make_source(tmp_path)
        stored = tmp_path / "atlas@q8"
        stored.mkdir()
        sq.write_manifest(stored, sq.build_manifest(source, 8, "mflux-1.0"))

        assert not sq.is_current(stored, source, bits, backend_format)

    def test_changed_source_weights_are_stale(self, tmp_path):
        source = _make_source(tmp_path)
        stored = tmp_path / "atlas@q8"
        stored.mkdir()
        sq.write_manifest(stored, sq.build_manifest(source, 8, "mflux-1.0"))

        weights = source / "transformer" / "weights.safetensors"
        weights.write_bytes(b"y" * 128)
        os.utime(weights, ns=(1, 1))

        assert not sq.is_current(stored, source, 8, "mflux-1.0")

    def test_moved_source_folder_stays_current(self, tmp_path):
        source = _make_source(tmp_path / "old")
        stored = tmp_path / "atlas@q8"
        stored.mkdir()
        sq.write_manifest(stored, sq.build_manifest(source, 8, "mflux-1.0"))
        moved = source.rename(tmp_path / "atlas")

        assert sq.is_current(stored, moved, 8, "mflux-1.0")


class TestFolders:
    def test_copy_detection_files(self, tmp_path):
        source = _make_source(tmp_path)
        stored = tmp_path / "out"
        stored.mkdir()

        sq.copy_detection_files(source, stored)

        assert (stored / "model_index.json").read_text() == (source / "model_index.json").read_text()
        assert (stored / "transformer" / "config.json").is_file()
        assert (stored / "scheduler" / "scheduler_config.json").is_file()
        assert not (stored / "transformer" / "weights.safetensors").exists()

    def test_partial_dir_is_hidden_and_unique(self, tmp_path):
        target = tmp_path / "atlas@q8"
        first, second = sq.partial_dir(target), sq.partial_dir(target)

        assert first.parent == tmp_path
        assert first.name.startswith(".atlas@q8.") and first.name.endswith(".partial")
        assert first != second

    def test_promote_partial_replaces_existing_target(self, tmp_path):
        target = tmp_path / "atlas@q8"
        target.mkdir()
        (target / "old.txt").write_text("old")
        partial = sq.partial_dir(target)
        partial.mkdir()
        (partial / "new.txt").write_text("new")

        sq.promote_partial(partial, target)

        assert (target / "new.txt").is_file()
        assert not (target / "old.txt").exists()
        assert not partial.exists()

    def test_discard_partial_tolerates_missing_folder(self, tmp_path):
        partial = tmp_path / ".x.partial"
        sq.discard_partial(partial)
        partial.mkdir()
        sq.discard_partial(partial)

        assert not partial.exists()
