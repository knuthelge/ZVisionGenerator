"""Fakes for the Higgs TTS 3 backend: a scripted model, a fake codec and tokenizer, and the patches that make ``load()`` use them.

Import them with ``from tests._higgs3_fakes import ...``. Nothing here loads a real model.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch
from torch import nn

from tests._audio_presets import audio_config

from zvisiongenerator.backends.audio import higgs3 as higgs3_module
from zvisiongenerator.backends.audio import higgs3_model
from zvisiongenerator.backends.audio.higgs3_config import Higgs3Config
from zvisiongenerator.utils import hf_snapshot
from zvisiongenerator.utils.audio_preset import parse_audio_presets

NUM_CODEBOOKS = 2
CODEBOOK_SIZE = 1026
"""The codec's 1024 values plus the two stream markers (begin 1024, end 1025)."""
EOC = CODEBOOK_SIZE - 1
OFFICIAL_REPO = "bosonai/higgs-tts-3-4b"
CODEC_REPO = "bosonai/higgs-audio-v2-tokenizer"

PRESET = next(preset for preset in parse_audio_presets(audio_config()) if preset.id == "higgs3")

CONFIG = Higgs3Config(
    hidden_size=4,
    intermediate_size=8,
    num_layers=1,
    num_heads=1,
    num_kv_heads=1,
    head_dim=4,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    text_vocab_size=64,
    max_positions=128,
    num_codebooks=NUM_CODEBOOKS,
    codebook_size=CODEBOOK_SIZE,
)


def codes_of(encode_number: int) -> torch.Tensor:
    """Return the ``[5, N]`` codes the fake codec returns from its *encode_number*-th encode (counting from 0)."""
    return (torch.arange(5 * NUM_CODEBOOKS).reshape(NUM_CODEBOOKS, 5) + 100 * encode_number).T


class FakeTokenizer:
    """Encodes a string to its character codes and knows the five structural tokens."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(char) for char in text]

    def get_added_vocab(self) -> dict[str, int]:
        return {"<|tts|>": 500, "<|ref_text|>": 501, "<|ref_audio|>": 502, "<|text|>": 503, "<|audio|>": 504}


class FakeCodec(nn.Module):
    """A codec that records what it encodes and decodes, and decodes to the next wave of *waves*."""

    def __init__(self, events: list, waves: list[torch.Tensor]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.events, self.waves = events, waves
        self.encoded: list[torch.Tensor] = []
        self.decoded: list[torch.Tensor] = []

    def to(self, device, *args, **kwargs):
        self.events.append(("codec.to", device))
        return self

    def encode(self, batch):
        codes = codes_of(len(self.encoded)).T.unsqueeze(0)  # [1, N, 5]
        self.encoded.append(batch)
        return SimpleNamespace(audio_codes=codes)

    def decode(self, batch):
        self.decoded.append(batch)
        return SimpleNamespace(audio_values=self.waves.pop(0).reshape(1, 1, -1))


class _Head:
    """The audio head of :class:`FakeModel`: it ends the current take on its last scripted call."""

    num_codebooks = NUM_CODEBOOKS
    codebook_size = CODEBOOK_SIZE

    def __init__(self, owner: FakeModel) -> None:
        self.owner = owner

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        owner = self.owner
        owner.head_calls += 1
        logits = torch.zeros(1, NUM_CODEBOOKS, CODEBOOK_SIZE)
        code = EOC if owner.head_calls == owner.take_calls else 0
        logits[0, 0, code] = 100.0  # the sampler then picks it whatever the temperature
        logits[0, 1:, 0] = 100.0
        return logits


class FakeModel:
    """A stand-in for ``Higgs3Model`` whose takes end after a scripted number of head calls.

    ``takes`` lists, per part in order, how many times the head runs before the part ends by itself (at least
    ``NUM_CODEBOOKS + 1``, the length of the start-up ramp plus the end frame). A part runs 50 calls when the list is
    used up. A part that would run longer than ``max_new_tokens`` is cut by the loop.
    """

    def __init__(self, takes: list[int] | None = None) -> None:
        self.takes = list(takes) if takes is not None else []
        self.audio_head = _Head(self)
        self.head_calls = 0
        self.take_calls = 0

    def embed_prompt(self, prompt_ids: list[int], reference_rows: torch.Tensor | None) -> torch.Tensor:
        self.head_calls = 0
        self.take_calls = self.takes.pop(0) if self.takes else 50
        assert self.take_calls > NUM_CODEBOOKS, "a take needs more head calls than the ramp has frames"
        return torch.zeros(1, len(prompt_ids), 4)

    def prefill(self, embeds: torch.Tensor) -> tuple[torch.Tensor, str]:
        return torch.zeros(1, 4), "cache"

    def step(self, codes: torch.Tensor, cache: str) -> torch.Tensor:
        return torch.zeros(1, 4)


def install_fakes(monkeypatch: Any, tmp_path: Path, model: Any, codec: nn.Module) -> SimpleNamespace:
    """Make ``Higgs3Backend.load`` read fake cached repositories and return *model* and *codec*.

    Both repositories get a snapshot folder holding every file the fixture preset lists, and a Hugging Face lookup for
    any other repository finds nothing. Reading the config, loading the tokenizer, the model and the codec are mocks.

    Returns:
        A namespace with the mocks (``read_config``, ``load_model``, ``codec_load``), the folders (``official``,
        ``codec_dir``) and ``snapshots``, the mutable map from repo id to the snapshot folders the cache lookup returns.
    """
    folders = {files.repo_id: tmp_path / "hub" / files.repo_id.replace("/", "--") for files in PRESET.files}
    for files in PRESET.files:
        for name in files.patterns:
            (folders[files.repo_id] / name).parent.mkdir(parents=True, exist_ok=True)
            (folders[files.repo_id] / name).write_text("x")
    snapshots = {repo_id: [folder] for repo_id, folder in folders.items()}
    monkeypatch.setattr(hf_snapshot, "cached_snapshot_dirs", lambda repo_id, revision: snapshots.get(repo_id, []))
    read_config = MagicMock(return_value=CONFIG)
    load_model = MagicMock(return_value=model)
    codec_load = MagicMock(return_value=codec)
    tokenizer_load = MagicMock(side_effect=lambda folder: FakeTokenizer())
    monkeypatch.setattr(higgs3_module, "read_higgs3_config", read_config)
    monkeypatch.setattr(higgs3_module, "load_tokenizer", tokenizer_load)
    monkeypatch.setattr(higgs3_model, "load_higgs3_model", load_model)
    import transformers

    monkeypatch.setattr(transformers, "HiggsAudioV2TokenizerModel", SimpleNamespace(from_pretrained=codec_load), raising=False)
    return SimpleNamespace(
        read_config=read_config,
        load_model=load_model,
        codec_load=codec_load,
        tokenizer_load=tokenizer_load,
        official=folders[OFFICIAL_REPO],
        codec_dir=folders[CODEC_REPO],
        snapshots=snapshots,
    )
