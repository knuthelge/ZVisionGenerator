"""Tests for the Higgs TTS 3 MLX backend: its frame loop on mlx-audio's sampler, and the engine it gives the shared take logic.

The model is a tiny fake with real MLX arrays and mlx-audio's real sampler step; nothing is loaded from disk.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_audio")

from mlx_audio.tts.models.higgs_audio_v3.config import HiggsAudioV3Config  # noqa: E402

from tests._audio_presets import audio_config  # noqa: E402
from zvisiongenerator.backends.audio import higgs3_mlx  # noqa: E402
from zvisiongenerator.backends.audio.higgs3_mlx import Higgs3MlxBackend, _generate_rows  # noqa: E402
from zvisiongenerator.backends.mac import memory as mac_memory  # noqa: E402
from zvisiongenerator.core.audio_backend import AudioCancelled  # noqa: E402
from zvisiongenerator.core.audio_types import AudioGenerationRequest  # noqa: E402
from zvisiongenerator.utils.audio_preset import parse_audio_presets  # noqa: E402

CONFIG = HiggsAudioV3Config(audio_num_codebooks=2)
EOC = CONFIG.audio_eoc_token_id
MAC_PRESET = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "higgs3").on_platform("mps")


class FakeModel:
    """A stand-in for mlx-audio's model whose parts end after a scripted number of head calls.

    ``takes`` lists, per part, how many head calls the part runs before it ends by itself (more than the two-frame
    ramp); 50 when the list is used up.
    """

    config = CONFIG
    layers: list = []

    def __init__(self, takes: list[int] | None = None) -> None:
        self.takes = list(takes or [])
        self.prompts: list[tuple[str, list]] = []
        self.encoded: list[np.ndarray] = []
        self.decoded: list[int] = []
        self.decoded_max = -1
        self.calls = self.length = 0

    def _build_prompt_embeddings(self, text, references):
        self.prompts.append((text, references))
        self.calls, self.length = 0, self.takes.pop(0) if self.takes else 50
        return mx.zeros((1, 3, 4)), 3

    def backbone(self, ids, cache=None, input_embeddings=None):
        return mx.zeros((1, input_embeddings.shape[1], 4))

    def _audio_logits(self, hidden):
        self.calls += 1
        logits = np.zeros((1, 2, CONFIG.audio_codebook_size), dtype=np.float32)
        logits[0, 0, EOC if self.calls == self.length else 3] = 100.0
        logits[0, 1, 5] = 100.0
        return mx.array(logits)

    def _embed_audio_codes(self, codes):
        return mx.zeros((1, 4))

    def _decode_audio(self, rows):
        self.decoded.append(len(rows))
        self.decoded_max = max(self.decoded_max, int(mx.max(mx.stack(rows))))
        return mx.full((len(rows) * 960,), 0.25)

    def encode_reference_audio(self, wave):
        self.encoded.append(wave)
        return mx.zeros((4, 2), dtype=mx.int32)


def _rows(model, *, max_new_tokens=100, on_frame=lambda: None, reference=None, seed=1):
    return _generate_rows(model, "Hello there.", reference, seed=seed, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.95, on_frame=on_frame)


class TestFrameLoop:
    def test_a_part_that_ends_by_itself_calls_back_on_every_frame_and_is_not_cut(self):
        frames = []

        rows, cut = _rows(FakeModel([6]), on_frame=lambda: frames.append(1))

        assert (len(rows), len(frames), cut) == (6, 6, False)

    def test_a_part_that_reaches_the_cap_is_cut(self):
        frames = []

        rows, cut = _rows(FakeModel([500]), max_new_tokens=20, on_frame=lambda: frames.append(1))

        assert (len(rows), len(frames), cut) == (20, 20, True)

    def test_an_exception_from_the_frame_callback_stops_the_part(self):
        model = FakeModel([500])

        def stop():
            if model.calls == 4:
                raise AudioCancelled

        with pytest.raises(AudioCancelled):
            _rows(model, on_frame=stop)

        assert model.calls == 4

    def test_a_reference_reaches_the_prompt_with_its_transcript(self):
        model = FakeModel([6])
        codes = mx.zeros((4, 2), dtype=mx.int32)

        _rows(model, reference=(codes, "What it says."))

        ((_, references),) = model.prompts
        assert (references[0].codes is codes, references[0].text) == (True, "What it says.")


@pytest.fixture
def backend(monkeypatch):
    """A backend loaded with a :class:`FakeModel`, with the cache and loader replaced by recorders."""
    model = FakeModel()
    loads: list[str] = []
    limits: list[int] = []
    monkeypatch.setattr(higgs3_mlx, "snapshot_dir", lambda files: f"/hf/{files.repo_id}")
    monkeypatch.setattr("mlx_audio.tts.utils.load", lambda path: loads.append(path) or model)
    monkeypatch.setattr(mac_memory, "load_resident", lambda build: build())

    @contextmanager
    def cache_limit(limit):
        limits.append(limit)
        yield

    monkeypatch.setattr(mac_memory, "mlx_cache_limit", cache_limit)
    backend = Higgs3MlxBackend()
    backend.load(MAC_PRESET, "mps")
    return SimpleNamespace(backend=backend, model=model, loads=loads, limits=limits)


def _request(text: str = "Hello there.") -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="higgs3", kind="speech", prompt=text, seed=7)


class TestBackend:
    def test_the_mac_preset_loads_only_the_model_repository(self, backend):
        assert backend.loads == ["/hf/bosonai/higgs-tts-3-4b"]

    def test_a_take_runs_with_the_mlx_buffer_cache_off(self, backend):
        backend.backend.generate(_request(), lambda *args: None, lambda: False)

        assert backend.limits == [0]

    def test_a_part_reports_its_decode_after_its_frames(self, backend):
        events = []

        backend.backend.generate(_request(), lambda phase, step, *_: events.append(phase), lambda: False)

        assert events[-3:] == ["Speech tokens", "Decode", "Decode"]
        assert backend.model.decoded == [50]

    def test_later_parts_follow_the_first_parts_own_voice(self, backend):
        text = ("a" * 299 + ". ") * 2

        backend.backend.generate(_request(text), lambda *args: None, lambda: False)

        (first, _), (_, references) = backend.model.prompts
        assert len(backend.model.encoded) == 1 and references[0].text == first

    def test_no_stream_marker_reaches_the_codec(self, backend):
        backend.backend.read_part("Hello there.", 1, None, 100, lambda: None, lambda done: None)

        assert 0 <= backend.model.decoded_max < CONFIG.audio_boc_token_id

    def test_a_part_too_short_for_sound_is_not_decoded(self, backend):
        wave, cut = backend.backend.read_part("Hi.", 1, None, 1, lambda: None, lambda done: pytest.fail("decoded"))

        assert (wave.size, cut) == (0, True)

    def test_close_drops_the_model(self, backend):
        backend.backend.close()

        assert backend.backend._model is None
