"""Tests for the Higgs TTS 3 tokenizer loader, the special ids and the prompt layouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


from zvisiongenerator.backends.audio.higgs3_prompt import AUDIO_SLOT, SpecialIds, build_prompt_ids, load_tokenizer, special_ids

TTS, REF_TEXT, REF_AUDIO, TEXT, AUDIO = 500, 501, 502, 503, 504
IDS = SpecialIds(tts=TTS, ref_text=REF_TEXT, ref_audio=REF_AUDIO, text=TEXT, audio=AUDIO)
SPECIALS = ["<|tts|>", "<|ref_text|>", "<|ref_audio|>", "<|text|>", "<|audio|>"]


class FakeTokenizer:
    """Encodes a string to its character codes, like a byte tokenizer, and holds an added vocabulary."""

    def __init__(self, added: dict[str, int] | None = None) -> None:
        self._added = added if added is not None else dict(zip(SPECIALS, (TTS, REF_TEXT, REF_AUDIO, TEXT, AUDIO), strict=True))
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return [ord(char) for char in text]

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)


class TestBuildPromptIds:
    def test_zero_shot_layout(self) -> None:
        assert build_prompt_ids(FakeTokenizer(), IDS, "hi", reference_rows=0, reference_text=None) == [TTS, TEXT, 104, 105, AUDIO]  # type: ignore[arg-type]

    def test_voice_with_transcript_layout(self) -> None:
        prompt = build_prompt_ids(FakeTokenizer(), IDS, "hi", reference_rows=3, reference_text="ab")  # type: ignore[arg-type]

        assert prompt == [TTS, REF_TEXT, 97, 98, REF_AUDIO, AUDIO_SLOT, AUDIO_SLOT, AUDIO_SLOT, TEXT, 104, 105, AUDIO]

    def test_voice_without_transcript_layout(self) -> None:
        prompt = build_prompt_ids(FakeTokenizer(), IDS, "hi", reference_rows=2, reference_text=None)  # type: ignore[arg-type]

        assert prompt == [TTS, REF_AUDIO, AUDIO_SLOT, AUDIO_SLOT, TEXT, 104, 105, AUDIO]

    def test_an_empty_transcript_counts_as_none(self) -> None:
        with_empty = build_prompt_ids(FakeTokenizer(), IDS, "hi", reference_rows=2, reference_text="")  # type: ignore[arg-type]

        assert REF_TEXT not in with_empty
        assert with_empty == build_prompt_ids(FakeTokenizer(), IDS, "hi", reference_rows=2, reference_text=None)  # type: ignore[arg-type]

    def test_a_transcript_without_a_reference_is_left_out(self) -> None:
        prompt = build_prompt_ids(FakeTokenizer(), IDS, "hi", reference_rows=0, reference_text="ab")  # type: ignore[arg-type]

        assert prompt == [TTS, TEXT, 104, 105, AUDIO]

    def test_text_is_encoded_without_special_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        build_prompt_ids(tokenizer, IDS, "hi", reference_rows=1, reference_text="ab")  # type: ignore[arg-type]

        assert tokenizer.calls == [("ab", False), ("hi", False)]


class TestSpecialIds:
    def test_the_ids_come_from_the_added_vocabulary(self) -> None:
        assert special_ids(FakeTokenizer()) == IDS  # type: ignore[arg-type]


def _write_tiny_tokenizer(folder: Path) -> None:
    """Write a byte-level BPE tokenizer with the five structural tokens as added special tokens, and a config.json that names remote code."""
    from tokenizers import AddedToken, Tokenizer, decoders, models
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in SPECIALS])
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", "backend": "tokenizers", "add_prefix_space": False, "extra_special_tokens": SPECIALS, "eos_token": None}),
        encoding="utf-8",
    )
    (folder / "config.json").write_text(json.dumps({"model_type": "no_such_model", "auto_map": {"AutoTokenizer": ["no_such_module.Tokenizer", None]}}), encoding="utf-8")


class TestLoadTokenizer:
    def test_it_loads_beside_a_config_that_names_remote_code(self, tmp_path: Path) -> None:
        _write_tiny_tokenizer(tmp_path)

        tokenizer = load_tokenizer(tmp_path)
        ids = special_ids(tokenizer)

        assert type(tokenizer).__name__ == "Qwen2Tokenizer"
        assert len({ids.tts, ids.ref_text, ids.ref_audio, ids.text, ids.audio}) == 5
        assert not any(name.startswith("no_such") for name in sys.modules)

    def test_a_structural_token_in_the_text_is_one_id(self, tmp_path: Path) -> None:
        _write_tiny_tokenizer(tmp_path)
        tokenizer = load_tokenizer(tmp_path)
        ids = special_ids(tokenizer)

        prompt = build_prompt_ids(tokenizer, ids, "a<|tts|>", reference_rows=0, reference_text=None)

        assert prompt[0] == ids.tts
        assert prompt[-2] == ids.tts
        assert prompt[-1] == ids.audio
