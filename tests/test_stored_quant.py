"""Tests for stored-quant naming, manifests and folder handling."""

from __future__ import annotations

import json
import os
import time

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

    def test_manifest_matches_needs_every_expected_key_equal(self, tmp_path):
        sq.write_manifest(tmp_path, {"bits": 8, "model": "atlas", "extra": 1})

        assert sq.manifest_matches(tmp_path, {"bits": 8, "model": "atlas"})
        assert not sq.manifest_matches(tmp_path, {"bits": 8, "model": "other"})
        assert not sq.manifest_matches(tmp_path, {"bits": 8, "missing": 1})

    def test_manifest_matches_is_false_without_a_manifest(self, tmp_path):
        assert not sq.manifest_matches(tmp_path, {})

    def test_stored_quant_bits_reads_the_manifest(self, tmp_path):
        source = _make_source(tmp_path)
        stored = tmp_path / "atlas@q4"
        stored.mkdir()
        sq.write_manifest(stored, sq.build_manifest(source, 4, "fmt"))

        assert sq.stored_quant_bits(stored) == 4

    @pytest.mark.parametrize("manifest", [None, "not json", '{"bits": 3}', "[]"])
    def test_stored_quant_bits_is_none_without_a_valid_manifest(self, tmp_path, manifest):
        if manifest is not None:
            (tmp_path / sq.MANIFEST_NAME).write_text(manifest)
        assert sq.stored_quant_bits(tmp_path) is None

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

    def test_copy_detection_files_keeps_component_configs_the_backend_wrote(self, tmp_path):
        source = _make_source(tmp_path)
        stored = tmp_path / "out"
        (stored / "transformer").mkdir(parents=True)
        (stored / "transformer" / "config.json").write_text('{"quantization_config": {}}')
        (stored / "model_index.json").write_text('{"saved": true}')

        sq.copy_detection_files(source, stored)

        assert (stored / "transformer" / "config.json").read_text() == '{"quantization_config": {}}'
        assert (stored / "model_index.json").read_text() == (source / "model_index.json").read_text()

    def test_copy_detection_files_never_writes_through_a_hard_link_to_the_source(self, tmp_path):
        source = _make_source(tmp_path)
        (source / "model_index.json").write_text('{"source": true}')
        stored = tmp_path / "out"
        stored.mkdir()
        os.link(source / "model_index.json", stored / "model_index.json")

        sq.copy_detection_files(source, stored)

        assert (source / "model_index.json").read_text() == '{"source": true}'
        assert (stored / "model_index.json").stat().st_ino != (source / "model_index.json").stat().st_ino

    def test_flush_to_disk_syncs_written_files_and_skips_links_to_the_source(self, tmp_path, monkeypatch):
        source = _make_source(tmp_path)
        stored = tmp_path / "out"
        (stored / "transformer").mkdir(parents=True)
        (stored / "transformer" / "w.safetensors").write_bytes(b"q")
        os.link(source / "model_index.json", stored / "model_index.json")
        synced: list[int] = []
        monkeypatch.setattr(sq.os, "fsync", synced.append)

        sq.flush_to_disk(stored)

        assert len(synced) == 1  # the written weights; the hard link to the source's model_index.json is skipped

    def test_a_file_that_cannot_be_synced_never_fails_the_flush(self, tmp_path, monkeypatch):
        stored = tmp_path / "out"
        stored.mkdir()
        (stored / "w.safetensors").write_bytes(b"q")

        def refuse(_fd):
            raise OSError("EBADF")

        monkeypatch.setattr(sq.os, "fsync", refuse)

        sq.flush_to_disk(stored)  # no exception: the copy is kept

    def test_flush_to_disk_syncs_read_only_files(self, tmp_path, monkeypatch):
        stored = tmp_path / "out"
        stored.mkdir()
        weights = stored / "w.safetensors"
        weights.write_bytes(b"q")
        weights.chmod(0o444)
        synced: list[int] = []
        monkeypatch.setattr(sq.os, "fsync", synced.append)

        sq.flush_to_disk(stored)

        assert len(synced) == 1

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


class TestEnsureStoredCopy:
    """The write-once routine behind the video backend's prepared copies."""

    _MANIFEST = {"layout": 1, "bits": 8, "format": "fp8-e4m3"}

    def _write_x(self, folder):
        (folder / "w").write_text("x")

    def test_a_missing_copy_is_written_then_marked_complete(self, tmp_path):
        target = tmp_path / "cache" / "model@q8"

        result = sq.ensure_stored_copy(target, self._MANIFEST, self._write_x)

        assert result == target
        assert (target / "w").read_text() == "x"
        assert json.loads((target / sq.MANIFEST_NAME).read_text()) == self._MANIFEST

    def test_an_extra_manifest_key_that_differs_makes_the_copy_stale(self, tmp_path):
        target = tmp_path / "model@q8"
        sq.ensure_stored_copy(target, {**self._MANIFEST, "lora_scale": 0.5}, lambda folder: (folder / "w").write_text("old"))
        writes: list[object] = []

        sq.ensure_stored_copy(target, {**self._MANIFEST, "lora_scale": 0.75}, lambda folder: (writes.append(folder), (folder / "w").write_text("new")))

        assert len(writes) == 1
        assert (target / "w").read_text() == "new"
        assert json.loads((target / sq.MANIFEST_NAME).read_text())["lora_scale"] == 0.75

    def test_a_current_copy_is_returned_without_writing(self, tmp_path):
        target = tmp_path / "model@q8"
        sq.ensure_stored_copy(target, self._MANIFEST, self._write_x)
        writes: list[object] = []

        sq.ensure_stored_copy(target, self._MANIFEST, writes.append)

        assert writes == []

    def test_reuse_never_picks_the_target_itself(self, tmp_path):
        target = tmp_path / "model@q8"
        sq.ensure_stored_copy(target, {**self._MANIFEST, "n": 1}, lambda folder: (folder / "w").write_text("old"))
        writes: list[object] = []

        # The target is stale for the new manifest and also listed to reuse: it is written anew, not linked from itself.
        sq.ensure_stored_copy(target, {**self._MANIFEST, "n": 2}, lambda folder: (writes.append(folder), (folder / "w").write_text("new")), reuse=[target])

        assert len(writes) == 1
        assert (target / "w").read_text() == "new"

    def test_a_current_copy_in_reuse_is_hard_linked_with_its_own_manifest(self, tmp_path):
        first = tmp_path / "a" / "text_encoder"
        sq.ensure_stored_copy(first, self._MANIFEST, self._write_x)
        writes: list[object] = []

        second = sq.ensure_stored_copy(tmp_path / "b" / "text_encoder", self._MANIFEST, writes.append, reuse=[first])

        assert writes == []
        assert (second / "w").stat().st_ino == (first / "w").stat().st_ino
        assert (second / sq.MANIFEST_NAME).stat().st_ino != (first / sq.MANIFEST_NAME).stat().st_ino

    def test_the_reuse_branch_also_sweeps_a_stale_partial_of_the_target(self, tmp_path):
        first = tmp_path / "a" / "text_encoder"
        sq.ensure_stored_copy(first, self._MANIFEST, self._write_x)
        target = tmp_path / "b" / "text_encoder"
        stale = target.parent / ".text_encoder.deadbeef.partial"
        stale.mkdir(parents=True)
        (stale / "half").write_text("half")
        old = time.time() - 2 * 60 * 60
        for path in (stale / "half", stale):
            os.utime(path, (old, old))

        sq.ensure_stored_copy(target, self._MANIFEST, lambda folder: pytest.fail("reuse must not write"), reuse=[first])

        assert not stale.exists()
        assert (target / "w").exists()

    def test_flush_syncs_the_files_before_the_manifest_exists(self, monkeypatch, tmp_path):
        seen: list[tuple[bool, bool]] = []
        monkeypatch.setattr(sq, "flush_to_disk", lambda folder: seen.append(((folder / "w").exists(), (folder / sq.MANIFEST_NAME).exists())))

        sq.ensure_stored_copy(tmp_path / "model@q8", self._MANIFEST, self._write_x, flush=True)

        assert seen == [(True, False)]

    def test_flush_is_off_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sq, "flush_to_disk", lambda folder: pytest.fail("flush_to_disk must not run"))

        sq.ensure_stored_copy(tmp_path / "model@q8", self._MANIFEST, self._write_x)

    @pytest.mark.parametrize("error", [RuntimeError("disk full"), KeyboardInterrupt()])
    def test_a_failed_or_interrupted_write_leaves_neither_target_nor_partial(self, tmp_path, error):
        target = tmp_path / "model@q8"

        def _fail(folder):
            (folder / "w").write_text("half")
            raise error

        with pytest.raises(type(error)):
            sq.ensure_stored_copy(target, self._MANIFEST, _fail)

        assert not target.exists()
        assert list(tmp_path.glob(".model@q8.*.partial")) == []


class TestFileIdentities:
    def test_a_symlink_records_the_resolved_name_and_size(self, tmp_path):
        real = tmp_path / "real-name.safetensors"
        real.write_bytes(b"1234567")
        link = tmp_path / "link.safetensors"
        link.symlink_to(real)
        plain = tmp_path / "plain.safetensors"
        plain.write_bytes(b"01")

        assert sq.file_identities([link, plain]) == [{"name": "real-name.safetensors", "bytes": 7}, {"name": "plain.safetensors", "bytes": 2}]
