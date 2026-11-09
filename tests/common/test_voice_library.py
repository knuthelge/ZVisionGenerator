"""Tests for the saved-voice library: the files, the record rules and the recording reader."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from tests._speech_fakes import RATE, tone
from zvisiongenerator.utils import voice_library as library
from zvisiongenerator.utils.slug import MAX_SLUG_LENGTH, is_slug
from zvisiongenerator.utils.audio_voice import CONSENT_REQUIRED, consent_for_saving
from zvisiongenerator.utils.voice_library import (
    delete_voice,
    find_voice,
    list_voices,
    read_recording,
    rename_voice,
    save_voice,
    save_voice_file,
    voice_by_slug,
    voice_names,
    voice_slug_for_path,
    voices_dir,
)

_LIMITS = {"min_seconds": 3.0, "max_seconds": 30.0}


def _save(name="Anna", seconds=4.0, **overrides):
    arguments = {"text": "Hello there.", "consent": "given", **overrides}
    return save_voice(name, tone(seconds), RATE, **arguments)


def test_save_list_find_rename_delete_round_trip():
    saved = _save("Anna Lund")

    slug = saved.slug
    assert slug.startswith("anna-lund-")
    assert (saved.name, saved.origin, saved.consent, saved.seconds, saved.description) == ("Anna Lund", "recording", "given", 4.0, None)
    assert saved.path == voices_dir() / f"{slug}.wav" and saved.path.is_file()
    assert saved.voice_ref().path == str(saved.path) and saved.voice_ref().text == "Hello there."
    assert list_voices() == [saved]
    assert find_voice("Anna Lund") == find_voice("anna lund") == find_voice(slug) == saved
    assert voice_by_slug(slug) == saved

    renamed = rename_voice(slug, "  Anna L  ")
    assert (renamed.slug, renamed.name, renamed.path) == (slug, "Anna L", saved.path)
    assert sorted(path.name for path in voices_dir().iterdir()) == [f"{slug}.wav", f"{slug}.yaml"]
    assert find_voice("Anna Lund") is None and find_voice("Anna L") == renamed

    delete_voice(slug)
    assert list_voices() == [] and list(voices_dir().iterdir()) == []
    with pytest.raises(FileNotFoundError):
        voice_by_slug(slug)
    with pytest.raises(FileNotFoundError):
        delete_voice(slug)


def test_the_stored_recording_is_a_tagless_mono_wav_at_the_given_rate():
    import wave

    saved = _save()
    with wave.open(str(saved.path), "rb") as reader:
        assert (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes()) == (1, 2, RATE, 4 * RATE)
    assert b"LIST" not in saved.path.read_bytes()[:64]


def test_list_is_sorted_by_name_ignoring_case():
    for name in ("charlie", "Bravo", "alpha"):
        _save(name)

    assert [voice.name for voice in list_voices()] == ["alpha", "Bravo", "charlie"]


def test_a_taken_name_is_refused_on_save_and_on_rename_ignoring_case():
    _save("Anna")
    ben = _save("Ben")

    with pytest.raises(FileExistsError):
        _save("ANNA")
    with pytest.raises(FileExistsError):
        rename_voice(ben.slug, "anna")
    assert rename_voice(ben.slug, "BEN").name == "BEN"


def test_every_slug_is_the_slugified_name_with_a_short_random_hex_suffix():
    first, second, fallback = _save("Anna!"), _save("Anna?"), _save("花子")

    assert re.fullmatch(r"anna-[0-9a-f]{4}", first.slug) and re.fullmatch(r"anna-[0-9a-f]{4}", second.slug)
    assert re.fullmatch(r"voice-[0-9a-f]{4}", fallback.slug)
    assert first.slug != second.slug


def test_a_drawn_suffix_that_clashes_with_a_record_or_a_recording_is_drawn_again(monkeypatch):
    draws = iter(["aaaa", "bbbb", "cccc", "dddd"])
    monkeypatch.setattr(library, "_random_suffix", lambda: next(draws))
    assert _save("Anna").slug == "anna-aaaa"
    (voices_dir() / "anna-bbbb.wav").write_bytes(b"orphan")
    (voices_dir() / "anna-cccc.yaml").write_text("name: x", encoding="utf-8")

    assert _save("Anna!").slug == "anna-dddd"


@pytest.mark.parametrize("name", ["x" * 60, "a" * 58 + " b"])
def test_a_slug_stays_within_the_slug_length_with_its_suffix(name):
    slug = _save(name).slug

    assert is_slug(slug) and len(slug) <= MAX_SLUG_LENGTH


def test_the_slug_stays_when_the_voice_is_renamed_and_a_new_voice_takes_a_fresh_one():
    first = _save("Anna")
    rename_voice(first.slug, "Zed")
    second = _save("Anna")

    assert second.slug != first.slug
    assert voice_by_slug(first.slug).name == "Zed"


@pytest.mark.parametrize("name", ["", "   ", "x" * 61])
def test_an_invalid_name_is_refused(name):
    with pytest.raises(ValueError, match="name"):
        _save(name)
    saved = _save("Anna")
    with pytest.raises(ValueError, match="name"):
        rename_voice(saved.slug, name)


def test_a_name_of_60_characters_is_allowed():
    assert _save("x" * 60).name == "x" * 60


def test_a_wav_is_saved_as_a_copy_and_the_source_stays(tmp_path):
    source = _save("Source").path
    copy = save_voice_file("Copy", source, 4.0004, text="  Hello.  ", description=" A calm voice. ")

    assert copy.path != source and copy.path.read_bytes() == source.read_bytes()
    assert (copy.seconds, copy.text, copy.origin, copy.consent, copy.description) == (4.0, "Hello.", "design", "not_needed", "A calm voice.")
    assert voice_by_slug(copy.slug) == copy
    assert source.is_file()


def test_a_failed_record_write_leaves_no_files(monkeypatch):
    def _fail(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(library, "write_text_atomic", _fail)

    with pytest.raises(OSError, match="disk full"):
        _save()
    assert list(voices_dir().iterdir()) == []


def test_a_record_missing_a_field_is_not_listed_and_not_found():
    slug = _save("Anna").slug
    record = yaml.safe_load((voices_dir() / f"{slug}.yaml").read_text(encoding="utf-8"))
    del record["name"]
    (voices_dir() / f"{slug}.yaml").write_text(yaml.safe_dump(record), encoding="utf-8")

    assert list_voices() == []
    assert find_voice("Anna") is None and find_voice(slug) is None
    assert voice_names() == {}
    with pytest.raises(FileNotFoundError):
        voice_by_slug(slug)
    with pytest.raises(FileNotFoundError):
        rename_voice(slug, "Other")
    with pytest.raises(FileNotFoundError):
        delete_voice(slug)


def test_unreadable_records_and_records_without_a_recording_are_skipped():
    good = _save("Good")
    orphan = _save("Orphan")
    orphan.path.unlink()
    (voices_dir() / "broken.yaml").write_text("name: [", encoding="utf-8")
    (voices_dir() / "list.yaml").write_text("- a\n- b\n", encoding="utf-8")
    (voices_dir() / "Bad_Slug.yaml").write_text((voices_dir() / f"{good.slug}.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    assert [voice.slug for voice in list_voices()] == [good.slug]


@pytest.mark.parametrize("slug", ["../anna", "Anna", "a/b", "", "anna.wav"])
def test_a_slug_that_is_not_a_slug_is_not_found(slug):
    _save("Anna")

    with pytest.raises(FileNotFoundError):
        voice_by_slug(slug)


def test_a_slug_cannot_reach_a_voice_outside_the_library():
    saved = _save("Anna")
    outside = voices_dir().parent / "evil"
    outside.with_suffix(".wav").write_bytes(saved.path.read_bytes())
    outside.with_suffix(".yaml").write_text(saved.path.with_suffix(".yaml").read_text(encoding="utf-8"), encoding="utf-8")

    for action in (voice_by_slug, delete_voice, lambda slug: rename_voice(slug, "Other")):
        with pytest.raises(FileNotFoundError):
            action("../evil")
    assert outside.with_suffix(".wav").is_file() and outside.with_suffix(".yaml").is_file()


def test_find_prefers_a_name_over_a_slug():
    slug_owner = _save("Anna?")
    name_owner = _save(slug_owner.slug)

    assert find_voice(slug_owner.slug) == name_owner
    assert find_voice("Anna?") == slug_owner


def test_voice_slug_for_path_names_library_wavs_only_whether_or_not_the_voice_exists(tmp_path):
    saved = _save("Anna")
    outside = tmp_path / saved.path.name
    outside.write_bytes(saved.path.read_bytes())

    assert voice_slug_for_path(saved.path) == saved.slug
    assert voice_slug_for_path(str(saved.path)) == saved.slug
    assert voice_slug_for_path(voices_dir() / ".." / "voices" / saved.path.name) == saved.slug
    assert voice_slug_for_path(voices_dir() / "gone.wav") == "gone"
    assert voice_slug_for_path(outside) is None
    assert voice_slug_for_path(saved.path.with_suffix(".yaml")) is None
    assert voice_slug_for_path(voices_dir() / ".upload-1a2b.wav") is None


def test_voice_names_maps_every_valid_slug_to_its_name():
    anna, bea = _save("Anna"), _save("Bea Lund")
    (voices_dir() / "broken.yaml").write_text("not: a voice", encoding="utf-8")

    assert voice_names() == {anna.slug: "Anna", bea.slug: "Bea Lund"}


def test_consent_for_saving_covers_its_three_outcomes(monkeypatch, tmp_path):
    needs = MagicMock(return_value=True)
    monkeypatch.setattr("zvisiongenerator.utils.audio_voice.voice_needs_consent", needs)

    assert consent_for_saving(tmp_path / "a.wav", True) == "given"
    with pytest.raises(ValueError, match=CONSENT_REQUIRED):
        consent_for_saving(tmp_path / "a.wav", False)
    needs.return_value = False
    assert consent_for_saving(tmp_path / "a.wav", False) == "not_needed"


def test_read_recording_probes_first_and_mixes_to_mono_48k(monkeypatch):
    stereo = np.stack([np.full(48000 * 4, 0.2, dtype=np.float32), np.full(48000 * 4, 0.4, dtype=np.float32)])
    probe, read = MagicMock(return_value=4.0), MagicMock(return_value=stereo)
    monkeypatch.setattr(library, "probe_track_seconds", probe)
    monkeypatch.setattr(library, "read_track", read)

    samples, rate = read_recording("take.flac", **_LIMITS)

    probe.assert_called_once_with("take.flac")
    track, read_rate, read_seconds = read.call_args.args
    assert (track.path, track.start, read_rate, read_seconds) == ("take.flac", 0.0, 48000, 4.0)
    assert rate == 48000 and samples.dtype == np.float32 and samples.shape == (48000 * 4,)
    assert samples == pytest.approx(0.3)


@pytest.mark.parametrize("seconds", [45.0, 2.0])
def test_read_recording_refuses_a_probed_length_outside_the_limits_before_decoding(monkeypatch, seconds):
    read = MagicMock()
    monkeypatch.setattr(library, "probe_track_seconds", MagicMock(return_value=seconds))
    monkeypatch.setattr(library, "read_track", read)

    with pytest.raises(ValueError):
        read_recording("take.flac", **_LIMITS)
    read.assert_not_called()


def test_read_recording_passes_on_an_unreadable_file(monkeypatch):
    read = MagicMock()
    monkeypatch.setattr(library, "probe_track_seconds", MagicMock(side_effect=ValueError("Can't read the audio track")))
    monkeypatch.setattr(library, "read_track", read)

    with pytest.raises(ValueError, match="Can't read"):
        read_recording("take.flac", **_LIMITS)
    read.assert_not_called()
