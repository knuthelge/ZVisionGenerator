"""Tests for the LoRA sources store, the trigger order of trust and the listing of installed LoRAs' details."""

from __future__ import annotations

import dataclasses
import json
from contextlib import closing

import pytest

from tests.web._helpers import safetensors_bytes
from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.lora_family import LORA_FAMILY_VERSION
from zvisiongenerator.utils.curated_loras import load_curated_loras
from zvisiongenerator.web import lora_details, lora_index
from zvisiongenerator.web.lora_details import describe_installed, installed_details, resolve_trigger
from zvisiongenerator.web.lora_sources import LoraSourceRow, LoraSources, LoraSourcesError


@pytest.fixture
def sources() -> LoraSources:
    return LoraSources(app_db.app_db_path)


def _row(name: str = "style", size: int = 10, **fields) -> LoraSourceRow:
    return LoraSourceRow(name=name, size_bytes=size, repo="o/r", revision="a" * 40, file="f.safetensors", sha256="b" * 64, curated_id=None, **fields)


class TestStore:
    def test_a_recorded_row_reads_back_by_name(self, sources):
        sources.record(_row(auto_trigger="t3chnic4lly", trigger_source="card"))

        found = sources.rows(["style", "missing"])

        assert list(found) == ["style"]
        assert (found["style"].repo, found["style"].file, found["style"].sha256, found["style"].auto_trigger, found["style"].trigger_source) == (
            "o/r",
            "f.safetensors",
            "b" * 64,
            "t3chnic4lly",
            "card",
        )
        assert found["style"].downloaded_at > 0

    def test_recording_a_name_again_keeps_the_trigger_the_user_typed(self, sources):
        sources.record(_row())
        sources.set_user_trigger("style", 10, "mine")

        sources.record(_row(size=20, auto_trigger="card trigger", trigger_source="card"))

        row = sources.rows(["style"])["style"]
        assert (row.size_bytes, row.auto_trigger, row.user_trigger) == (20, "card trigger", "mine")

    def test_a_typed_trigger_for_a_file_with_no_source_makes_a_row_that_holds_only_it(self, sources):
        sources.set_user_trigger("hand_copied", 7, "ohwx")

        row = sources.rows(["hand_copied"])["hand_copied"]
        assert (row.repo, row.user_trigger, row.size_bytes) == (None, "ohwx", 7)

    def test_clearing_the_typed_trigger_removes_a_row_that_held_nothing_else_and_keeps_a_source_row(self, sources):
        sources.set_user_trigger("hand_copied", 7, "ohwx")
        sources.record(_row())
        sources.set_user_trigger("style", 10, "mine")

        sources.set_user_trigger("hand_copied", 7, None)
        sources.set_user_trigger("style", 10, None)

        assert set(sources.rows(["hand_copied", "style"])) == {"style"}
        assert sources.rows(["style"])["style"].user_trigger is None

    def test_a_hugging_face_download_keeps_its_facts_and_preview_image_until_forgotten(self, sources):
        sources.record(_row(author="owner", downloads=1234, licence="mit"), ("image/webp", b"IMG"))

        row = sources.rows(["style"])["style"]
        assert (row.author, row.downloads, row.licence, row.has_image) == ("owner", 1234, "mit", True)
        assert sources.image("style") == ("image/webp", b"IMG")

        sources.forget("style")
        assert sources.image("style") is None

    def test_recording_again_without_an_image_drops_the_old_one(self, sources):
        sources.record(_row(), ("image/webp", b"IMG"))

        sources.record(_row())

        assert (sources.rows(["style"])["style"].has_image, sources.image("style")) == (False, None)

    def test_an_image_added_later_is_kept_only_for_the_file_it_was_fetched_for(self, sources):
        sources.record(_row())
        size = sources.rows(["style"])["style"].size_bytes

        assert sources.set_image("style", size + 1, ("image/webp", b"OLD")) is False
        assert sources.set_image("gone", size, ("image/webp", b"OLD")) is False
        assert sources.image("style") is None
        assert sources.set_image("style", size, ("image/webp", b"IMG")) is True

        assert (sources.rows(["style"])["style"].has_image, sources.image("style")) == (True, ("image/webp", b"IMG"))

    def test_a_database_written_by_a_newer_ziv_is_left_alone_and_reported(self, sources):
        with closing(app_db.connect()) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_versions (component TEXT PRIMARY KEY, version INTEGER NOT NULL)")
            conn.execute("INSERT OR REPLACE INTO schema_versions VALUES ('lora_sources', 99)")

        with pytest.raises(LoraSourcesError, match="newer"):
            sources.rows(["style"])

    def test_a_family_rules_version_bump_leaves_the_sources_intact(self, sources, tmp_path, monkeypatch):
        sources.record(_row(auto_trigger="keep me", trigger_source="card"))
        loras = tmp_path / "loras"
        loras.mkdir()
        (loras / "style.safetensors").write_bytes(safetensors_bytes(metadata={"ss_base_model_version": "zimage"}))
        assert lora_index.lookup(loras, ["style"])["style"].family == "zimage"
        bumped = LORA_FAMILY_VERSION + 1
        monkeypatch.setattr(lora_index, "LORA_FAMILY_VERSION", bumped)
        monkeypatch.setattr(lora_index, "_TABLE", f"{lora_index._TABLE_PREFIX}{bumped}")
        monkeypatch.setattr(lora_index, "_COMPONENT", app_db.Component("loras", bumped, lora_index._migrate))

        assert lora_index.lookup(loras, ["style"])["style"].family == "zimage"

        row = sources.rows(["style"])["style"]
        assert (row.repo, row.auto_trigger) == ("o/r", "keep me")
        with closing(app_db.connect()) as conn:
            tables = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert f"loras_entries_v{bumped}" in tables
        assert f"loras_entries_v{LORA_FAMILY_VERSION}" not in tables
        assert "lora_sources_v1" in tables


def _curated(**fields):
    base = dataclasses.replace(load_curated_loras()[0], name="curated_one", id="curated-one", size_bytes=10, trigger="curated trigger")
    return dataclasses.replace(base, **fields)


class TestTriggerOrderOfTrust:
    ROW = LoraSourceRow(name="x", size_bytes=1, repo="o/r", auto_trigger="card trigger", trigger_source="card", user_trigger="typed")

    def test_what_the_user_typed_comes_first(self):
        assert resolve_trigger(curated=_curated(), row=self.ROW, header="header caption") == ("typed", "user")
        assert resolve_trigger(curated=None, row=self.ROW, header=None) == ("typed", "user")

    def test_then_the_curated_entrys_trigger(self):
        row = dataclasses.replace(self.ROW, user_trigger=None)

        assert resolve_trigger(curated=_curated(), row=row, header="header caption") == ("curated trigger", "curated")

    def test_then_the_model_cards_instance_prompt(self):
        row = dataclasses.replace(self.ROW, user_trigger=None)

        assert resolve_trigger(curated=None, row=row, header="header caption") == ("card trigger", "card")
        assert resolve_trigger(curated=_curated(trigger=None), row=row, header="header caption") == ("card trigger", "card")

    def test_then_a_header_caption(self):
        row = dataclasses.replace(self.ROW, auto_trigger=None, trigger_source=None, user_trigger=None)

        assert resolve_trigger(curated=None, row=row, header="header caption") == ("header caption", "header")

    def test_otherwise_there_is_none(self):
        assert resolve_trigger(curated=None, row=None, header=None) == (None, None)


class TestInstalledDetails:
    @pytest.fixture
    def loras(self, tmp_path, monkeypatch):
        folder = tmp_path / "loras"
        folder.mkdir()
        entry = _curated()
        monkeypatch.setattr(lora_details, "curated_by_name", lambda name: entry if name == entry.name else None)
        monkeypatch.setattr(lora_details, "curated_by_id", lambda curated_id: entry if curated_id == entry.id else None)
        return folder

    def test_a_file_is_a_curated_entry_when_its_name_and_size_match(self, loras):
        (loras / "curated_one.safetensors").write_bytes(b"x" * 10)
        (loras / "other.safetensors").write_bytes(b"x" * 10)

        details = describe_installed(loras, ["curated_one", "other"], {})

        assert (details["curated_one"]["curated_id"], details["curated_one"]["trigger"], details["curated_one"]["trigger_source"]) == ("curated-one", "curated trigger", "curated")
        assert details["other"]["curated_id"] is None

    def test_a_file_with_a_curated_name_but_another_size_is_not_the_curated_entry(self, loras):
        (loras / "curated_one.safetensors").write_bytes(b"x" * 11)

        assert describe_installed(loras, ["curated_one"], {})["curated_one"]["curated_id"] is None

    def test_a_downloaded_file_reports_its_source_and_a_stale_row_is_ignored(self, loras):
        (loras / "style.safetensors").write_bytes(b"x" * 10)
        row = _row("style", 10, auto_trigger="card trigger", trigger_source="card")

        fresh = describe_installed(loras, ["style"], {"style": row})["style"]
        stale = describe_installed(loras, ["style"], {"style": dataclasses.replace(row, size_bytes=99)})["style"]

        assert fresh["source"] == {"repo": "o/r", "revision": "a" * 40, "file": "f.safetensors", "sha256": "b" * 64, "curated_id": None}
        assert (fresh["trigger"], fresh["trigger_source"]) == ("card trigger", "card")
        assert (stale["source"], stale["trigger"]) == (None, None)

    def test_a_single_caption_in_the_header_is_the_trigger_but_the_output_name_and_file_name_are_not(self, loras):
        frequencies = json.dumps({"10_x": {"t3chnic4lly": 30}})
        (loras / "style_v1.safetensors").write_bytes(safetensors_bytes(metadata={"ss_tag_frequency": frequencies, "ss_output_name": "should_not_be_used"}))
        (loras / "named.safetensors").write_bytes(safetensors_bytes(metadata={"ss_output_name": "should_not_be_used"}))

        details = describe_installed(loras, ["style_v1", "named"], {})

        assert (details["style_v1"]["trigger"], details["style_v1"]["trigger_source"]) == ("t3chnic4lly", "header")
        assert (details["named"]["trigger"], details["named"]["trigger_source"]) == (None, None)

    def test_a_typed_trigger_fills_in_when_nothing_better_names_one(self, loras):
        (loras / "style.safetensors").write_bytes(safetensors_bytes())
        row = LoraSourceRow(name="style", size_bytes=(loras / "style.safetensors").stat().st_size, user_trigger="ohwx")

        details = describe_installed(loras, ["style"], {"style": row})["style"]

        assert (details["trigger"], details["trigger_source"], details["source"]) == ("ohwx", "user", None)

    def test_a_caption_the_family_lookup_already_read_costs_no_second_header_read(self, loras, monkeypatch):
        path = loras / "style.safetensors"
        path.write_bytes(safetensors_bytes(metadata={"ss_tag_frequency": json.dumps({"a": {"one": 1}})}))
        lora_index.lookup(loras, ["style"])
        reads = []
        real = lora_index.read_header_and_metadata
        monkeypatch.setattr(lora_index, "read_header_and_metadata", lambda p: reads.append(p) or real(p))

        assert describe_installed(loras, ["style"], {})["style"]["trigger"] == "one"
        assert reads == []

    def test_describing_one_file_keeps_the_cached_rows_of_the_others(self, loras, monkeypatch):
        (loras / "style.safetensors").write_bytes(safetensors_bytes())
        (loras / "other.safetensors").write_bytes(safetensors_bytes())
        lora_index.lookup(loras, ["style", "other"])
        reads = []
        real = lora_index.read_header_and_metadata
        monkeypatch.setattr(lora_index, "read_header_and_metadata", lambda p: reads.append(p) or real(p))

        describe_installed(loras, ["style"], {})
        lora_index.lookup(loras, ["style", "other"])

        assert reads == []

    def test_a_hugging_face_download_offers_its_kept_facts_and_a_preview_address(self, loras):
        (loras / "style.safetensors").write_bytes(b"x" * 10)
        row = _row("style", 10, author="owner", downloads=1234, licence="cc-by-4.0", has_image=True, downloaded_at=99.5)

        described = describe_installed(loras, ["style"], {"style": row})["style"]

        assert described["hub"] == {
            "name": "r",
            "author": "owner",
            "downloads": 1234,
            "licence": "cc-by-4.0",
            "licence_label": "CC BY 4.0",
            "model_card_url": "https://huggingface.co/o/r",
            "thumbnail_url": "/api/discover/installed-thumbnails/style?v=99",
        }
        assert (described["size_bytes"], described["size_label"], described["title"]) == (10, "1 KB", "r")

    def test_a_download_without_an_image_has_no_preview_address_and_a_curated_file_has_no_hub_facts(self, loras):
        (loras / "style.safetensors").write_bytes(b"x" * 10)
        (loras / "curated_one.safetensors").write_bytes(b"x" * 10)

        plain = describe_installed(loras, ["style"], {"style": _row("style", 10, author="owner")})["style"]
        curated = describe_installed(loras, ["curated_one"], {"curated_one": _row("curated_one", 10, author="owner")})["curated_one"]

        assert plain["hub"]["thumbnail_url"] is None
        assert curated["hub"] is None

    def test_a_curated_file_is_titled_by_its_curated_display_name(self, loras):
        (loras / "curated_one.safetensors").write_bytes(b"x" * 10)

        assert describe_installed(loras, ["curated_one"], {})["curated_one"]["title"] == load_curated_loras()[0].display_name

    def test_a_missing_file_has_no_details(self, loras):
        assert describe_installed(loras, ["gone"], {})["gone"] == {
            "curated_id": None,
            "title": None,
            "trigger": None,
            "trigger_source": None,
            "source": None,
            "hub": None,
            "size_bytes": None,
            "size_label": None,
        }

    def test_installed_details_read_rows_from_the_store_and_survive_its_failure(self, loras, sources, monkeypatch):
        (loras / "style.safetensors").write_bytes(b"x" * 10)
        sources.record(_row("style", 10, auto_trigger="stored", trigger_source="card"))

        assert installed_details(loras, ["style"], sources=sources)["style"]["trigger"] == "stored"

        def broken(_names):
            raise LoraSourcesError("disk")

        monkeypatch.setattr(sources, "rows", broken)
        with pytest.warns(UserWarning, match="sources are unavailable"):
            assert installed_details(loras, ["style"], sources=sources)["style"]["trigger"] is None
