"""Tests that a Qwen3-TTS model counts as downloaded from a real fake Hub cache.

Nothing about presence is mocked: the Models-page row, the runner's pre-load refusal and the backend's snapshot lookup all
read the same cache folder, laid out as a finished download leaves it.
"""

from __future__ import annotations

from pathlib import Path

import huggingface_hub.constants
import pytest

from tests._audio_presets import QWEN_BASE_REPO, QWEN_CUSTOM_REPO, audio_config, qwen3_tts_preset
from tests.audio._helpers import preset

from zvisiongenerator.audio_runner import _require_downloaded
from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.web.audio_inventory import audio_inventory_rows

QWEN = preset("qwen3_tts")
_BUDGET = MemoryBudget(10 * GIB, system_bytes=64 * GIB)
_LITERAL = ("config.json", "generation_config.json", "merges.txt", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json", "vocab.json")
_TOKENIZER = ("speech_tokenizer/config.json", "speech_tokenizer/configuration.json", "speech_tokenizer/model.safetensors", "speech_tokenizer/preprocessor_config.json")


@pytest.fixture
def hub(monkeypatch, tmp_path) -> Path:
    """An empty Hub cache."""
    folder = tmp_path / "hub"
    folder.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(folder))
    return folder


def _lay_out(hub: Path, repo: str, *, skip: tuple[str, ...] = ()) -> Path:
    """Write the Qwen layout of *repo* (``owner/name@commit``) as a finished download leaves it, minus the files in *skip*."""
    name, commit = repo.split("@")
    snapshot = hub / f"models--{name.replace('/', '--')}" / "snapshots" / commit
    for relative in (*_LITERAL, *_TOKENIZER):
        if relative not in skip:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
    return snapshot


def _row(hub: Path) -> dict:
    return next(row for row in audio_inventory_rows(audio_config(qwen3_tts=qwen3_tts_preset()), _BUDGET) if row["name"] == "qwen3_tts")


class TestSpeechTokenizerInTheFileList:
    def test_the_preset_lists_the_speech_tokenizer_files(self):
        assert set(_TOKENIZER) <= set(QWEN.files[0].patterns) and set(_TOKENIZER) <= set(QWEN.voice_files[0].patterns)

    def test_customvoice_on_disk_is_downloaded_and_a_built_in_voice_take_is_not_refused(self, hub):
        _lay_out(hub, QWEN_CUSTOM_REPO)

        _require_downloaded(QWEN)

        row = _row(hub)
        assert (row["downloaded"], row["download_needed"]) == (True, False)
        assert row["voice_download"] == {"downloaded": False}

    def test_the_backend_finds_the_snapshot_by_its_literal_files(self, hub):
        snapshot = _lay_out(hub, QWEN_CUSTOM_REPO)

        assert snapshot_dir(QWEN.files[0]) == snapshot

    def test_nothing_on_disk_is_not_downloaded_and_the_take_is_refused(self, hub):
        with pytest.raises(FileNotFoundError, match="Qwen3-TTS 1.7B is not downloaded"):
            _require_downloaded(QWEN)

        assert _row(hub)["downloaded"] is False

    def test_missing_weights_are_not_downloaded(self, hub):
        _lay_out(hub, QWEN_CUSTOM_REPO, skip=("model.safetensors",))

        with pytest.raises(FileNotFoundError, match="is not downloaded"):
            _require_downloaded(QWEN)
        assert _row(hub)["downloaded"] is False

    def test_a_missing_speech_tokenizer_is_not_downloaded(self, hub):
        """The tokenizer files have to count: the codec the model decodes with lives in ``speech_tokenizer/``."""
        _lay_out(hub, QWEN_CUSTOM_REPO, skip=_TOKENIZER)

        with pytest.raises(FileNotFoundError, match="is not downloaded"):
            _require_downloaded(QWEN)
        assert _row(hub)["downloaded"] is False

    @pytest.mark.parametrize("name", ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json"])
    def test_a_missing_literal_small_file_is_not_downloaded(self, hub, name):
        _lay_out(hub, QWEN_CUSTOM_REPO, skip=(name,))

        with pytest.raises(FileNotFoundError, match="is not downloaded"):
            _require_downloaded(QWEN)
        assert _row(hub)["downloaded"] is False


class TestBothRepositories:
    def test_a_recording_needs_the_cloning_checkpoint_too(self, hub):
        _lay_out(hub, QWEN_CUSTOM_REPO)

        _require_downloaded(QWEN)
        with pytest.raises(FileNotFoundError, match="is not downloaded"):
            _require_downloaded(QWEN, voice=True)

        _lay_out(hub, QWEN_BASE_REPO)

        _require_downloaded(QWEN, voice=True)

    def test_the_row_reports_the_two_repositories_apart(self, hub):
        _lay_out(hub, QWEN_BASE_REPO)
        base_only = _row(hub)

        assert (base_only["downloaded"], base_only["voice_download"]) == (False, {"downloaded": True})

        _lay_out(hub, QWEN_CUSTOM_REPO)
        both = _row(hub)

        assert (both["downloaded"], both["voice_download"]) == (True, {"downloaded": True})

    def test_the_cloning_checkpoint_with_a_missing_speech_tokenizer_is_not_downloaded(self, hub):
        _lay_out(hub, QWEN_CUSTOM_REPO)
        _lay_out(hub, QWEN_BASE_REPO, skip=_TOKENIZER)

        assert _row(hub)["voice_download"] == {"downloaded": False}
        with pytest.raises(FileNotFoundError):
            _require_downloaded(QWEN, voice=True)
