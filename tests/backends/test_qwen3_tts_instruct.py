"""Tests for Qwen3-TTS's Delivery instruction: every setting's value renders, neutral asks for nothing, and a Vibe is one sentence."""

from __future__ import annotations

import pytest

from zvisiongenerator.backends.audio.qwen3_tts_instruct import delivery_instruction
from zvisiongenerator.core.audio_types import SpeechDelivery
from zvisiongenerator.utils.audio_settings import EMOTIONS, LEVELS, MANNERS, PACES, VIBES


class TestNothingToSay:
    def test_no_delivery_is_no_instruction(self):
        assert delivery_instruction(None) is None

    def test_the_default_delivery_is_no_instruction(self):
        assert delivery_instruction(SpeechDelivery()) is None


class TestEachValue:
    @pytest.mark.parametrize("emotion", EMOTIONS[1:])
    def test_every_emotion_renders_after_the_verb(self, emotion):
        sentence = delivery_instruction(SpeechDelivery(emotion=emotion))

        assert sentence is not None and sentence.startswith("Speak ") and sentence.endswith(".")

    def test_the_emotions_phrases_are_all_different(self):
        sentences = {delivery_instruction(SpeechDelivery(emotion=emotion)) for emotion in EMOTIONS[1:]}

        assert len(sentences) == len(EMOTIONS) - 1

    def test_the_blind_tested_wording_is_kept(self):
        assert delivery_instruction(SpeechDelivery(emotion="sad")) == "Speak sadly, in a low and sorrowful voice."
        assert delivery_instruction(SpeechDelivery(emotion="angry")) == "Speak angrily, in a sharp and furious voice."
        assert delivery_instruction(SpeechDelivery(emotion="enthusiastic")) == "Speak with excitement, bright and full of energy."
        assert delivery_instruction(SpeechDelivery(manner="whisper")) == "Whisper softly."
        assert delivery_instruction(SpeechDelivery(pace="slow")) == "Speak slowly."

    @pytest.mark.parametrize(("manner", "sentence"), [("natural", None), ("whisper", "Whisper softly."), ("shout", "Shout loudly.")])
    def test_a_manner_alone(self, manner, sentence):
        assert delivery_instruction(SpeechDelivery(manner=manner)) == sentence

    @pytest.mark.parametrize(
        ("pace", "sentence"),
        [("very-slow", "Speak very slowly."), ("slow", "Speak slowly."), ("normal", None), ("fast", "Speak quickly."), ("very-fast", "Speak very quickly.")],
    )
    def test_a_pace_alone(self, pace, sentence):
        assert delivery_instruction(SpeechDelivery(pace=pace)) == sentence

    @pytest.mark.parametrize(("level", "sentence"), [("low", "Speak in a low-pitched voice."), ("normal", None), ("high", "Speak in a high-pitched voice.")])
    def test_a_pitch_alone(self, level, sentence):
        assert delivery_instruction(SpeechDelivery(pitch=level)) == sentence

    @pytest.mark.parametrize(("level", "sentence"), [("low", "Speak with little expression."), ("normal", None), ("high", "Speak with lots of expression.")])
    def test_an_expressiveness_alone(self, level, sentence):
        assert delivery_instruction(SpeechDelivery(expressiveness=level)) == sentence

    def test_every_value_of_the_vocabulary_is_covered(self):
        for manner in MANNERS:
            delivery_instruction(SpeechDelivery(manner=manner))
        for pace in PACES:
            delivery_instruction(SpeechDelivery(pace=pace))
        for level in LEVELS:
            delivery_instruction(SpeechDelivery(pitch=level, expressiveness=level))


class TestOneSentence:
    def test_the_settings_follow_in_the_order_emotion_pace_pitch_expressiveness(self):
        delivery = SpeechDelivery(emotion="sad", pace="slow", pitch="low", expressiveness="high")

        assert delivery_instruction(delivery) == "Speak sadly, in a low and sorrowful voice, slowly, in a low-pitched voice, with lots of expression."

    def test_a_manner_leads_and_the_other_settings_follow(self):
        assert delivery_instruction(SpeechDelivery(emotion="afraid", manner="whisper")) == "Whisper fearfully, in a trembling and frightened voice."
        assert delivery_instruction(SpeechDelivery(manner="shout", pace="fast")) == "Shout quickly."

    @pytest.mark.parametrize("vibe", list(VIBES))
    def test_every_vibe_is_one_sentence(self, vibe):
        sentence = delivery_instruction(VIBES[vibe])

        assert sentence is not None
        assert sentence[0].isupper() and sentence.endswith(".")
        assert sentence.count(".") == 1  # one sentence: a single full stop, at the end
        assert sentence.split(" ", 1)[0] in {"Speak", "Whisper", "Shout"}

    @pytest.mark.parametrize("emotion", EMOTIONS[1:])
    @pytest.mark.parametrize("manner", MANNERS)
    def test_any_combination_is_one_sentence(self, emotion, manner):
        sentence = delivery_instruction(SpeechDelivery(emotion=emotion, manner=manner, pace="very-fast", pitch="high", expressiveness="low"))

        assert sentence.count(".") == 1 and sentence.endswith(".")
