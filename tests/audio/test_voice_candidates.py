"""Tests for the candidate store of the voice design runs: the files, the batch record, stable ids, saving marks and clearing."""

from __future__ import annotations

import numpy as np
import pytest

from tests._fake_audio import read_fake_audio
from tests._speech_fakes import RATE, tone
from zvisiongenerator.utils.voice_candidates import (
    DesignBatch,
    DesignCandidate,
    add_batch,
    candidate_file,
    clear_candidates,
    find_candidate,
    mark_saved,
    read_batches,
    write_candidate,
)


def _batch(*candidates: DesignCandidate, batch_id: str = "b1", description: str = "A calm, low voice.", text: str = "Hello there.") -> DesignBatch:
    return DesignBatch(batch_id, "Qwen3-TTS VoiceDesign", description, text, "2026-11-08T10:00:00+00:00", candidates)


def test_a_candidate_round_trips_as_a_tagless_mono_wav_at_its_rate_under_a_new_id(tmp_path):
    samples = tone(1.5)

    first = write_candidate(tmp_path / "dir", samples, RATE)
    second = write_candidate(tmp_path / "dir", samples, RATE)
    add_batch(tmp_path / "dir", _batch(first))
    stored = read_fake_audio(candidate_file(tmp_path / "dir", first.id))

    assert first.id != second.id
    assert first.seconds == 1.5 and first.saved_as is None
    assert stored.rate == RATE
    assert stored.samples.shape == (1, samples.shape[0])
    assert np.allclose(stored.samples[0], samples, atol=1e-3)
    assert candidate_file(tmp_path / "dir", first.id) == tmp_path / "dir" / f"{first.id}.wav"
    assert sorted(path.name for path in (tmp_path / "dir").iterdir()) == sorted([f"{first.id}.wav", f"{second.id}.wav", "candidates.yaml"])


def test_batches_add_up_oldest_first_and_round_trip(tmp_path):
    one, two, three = (write_candidate(tmp_path, tone(2.0), RATE) for _ in range(3))

    add_batch(tmp_path, _batch(one, two, batch_id="b1", description="first"))
    add_batch(tmp_path, _batch(three, batch_id="b2", description="second", text="Another line."))

    batches = read_batches(tmp_path)
    assert [batch.batch_id for batch in batches] == ["b1", "b2"]
    assert batches[0].candidates == (one, two)
    assert (batches[1].description, batches[1].text, batches[1].created_at) == ("second", "Another line.", "2026-11-08T10:00:00+00:00")
    assert batches[1].candidates == (three,)


def test_a_batch_without_candidates_is_not_added(tmp_path):
    add_batch(tmp_path, _batch())

    assert read_batches(tmp_path) == ()
    assert not tmp_path.joinpath("candidates.yaml").exists()


def test_an_id_finds_its_candidate_and_batch_whatever_is_added_later(tmp_path):
    one = write_candidate(tmp_path, tone(2.0), RATE)
    add_batch(tmp_path, _batch(one, batch_id="b1"))
    two = write_candidate(tmp_path, tone(2.0), RATE)
    add_batch(tmp_path, _batch(two, batch_id="b2"))

    batch, candidate = find_candidate(tmp_path, one.id)

    assert (batch.batch_id, candidate) == ("b1", one)
    assert find_candidate(tmp_path, two.id)[0].batch_id == "b2"


def test_an_unknown_id_is_file_not_found(tmp_path):
    one = write_candidate(tmp_path, tone(2.0), RATE)
    add_batch(tmp_path, _batch(one))

    for reader in (find_candidate, candidate_file):
        with pytest.raises(FileNotFoundError):
            reader(tmp_path, "nope")


def test_a_wav_that_no_batch_lists_is_not_served(tmp_path):
    unlisted = write_candidate(tmp_path, tone(2.0), RATE)

    with pytest.raises(FileNotFoundError):
        candidate_file(tmp_path, unlisted.id)


def test_marking_a_candidate_saved_keeps_the_others_and_the_rest_of_the_record(tmp_path):
    one, two = (write_candidate(tmp_path, tone(2.0), RATE) for _ in range(2))
    add_batch(tmp_path, _batch(one, two))

    mark_saved(tmp_path, two.id, "morgan", "2026-11-08T11:00:00+00:00")

    (batch,) = read_batches(tmp_path)
    assert [(item.id, item.saved_as, item.saved_at) for item in batch.candidates] == [(one.id, None, None), (two.id, "morgan", "2026-11-08T11:00:00+00:00")]
    assert (batch.description, batch.text) == ("A calm, low voice.", "Hello there.")


def test_marking_a_gone_candidate_changes_nothing(tmp_path):
    mark_saved(tmp_path / "nowhere", "nope", "morgan", "2026-11-08T11:00:00+00:00")
    one = write_candidate(tmp_path, tone(2.0), RATE)
    add_batch(tmp_path, _batch(one))

    mark_saved(tmp_path, "nope", "morgan", "2026-11-08T11:00:00+00:00")

    assert not (tmp_path / "nowhere").exists()
    assert read_batches(tmp_path)[0].candidates == (one,)


def test_a_missing_record_reads_as_no_batches(tmp_path):
    assert read_batches(tmp_path) == ()
    assert read_batches(tmp_path / "nowhere") == ()


@pytest.mark.parametrize("text", ["{not: [yaml", "just text", "- a list", "batches: 3", "batches: [{batch_id: b}]"])
def test_an_unreadable_record_reads_as_no_batches(tmp_path, text):
    (tmp_path / "candidates.yaml").write_text(text, encoding="utf-8")

    assert read_batches(tmp_path) == ()


def test_clearing_removes_every_wav_and_the_record_only(tmp_path):
    one, two = (write_candidate(tmp_path, tone(1.0), RATE) for _ in range(2))
    add_batch(tmp_path, _batch(one))
    add_batch(tmp_path, _batch(two, batch_id="b2"))
    (tmp_path / "keep.txt").write_text("kept", encoding="utf-8")

    clear_candidates(tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == ["keep.txt"]
    assert read_batches(tmp_path) == ()
    with pytest.raises(FileNotFoundError):
        find_candidate(tmp_path, one.id)


def test_clearing_a_missing_folder_does_nothing(tmp_path):
    clear_candidates(tmp_path / "nowhere")

    assert not (tmp_path / "nowhere").exists()
