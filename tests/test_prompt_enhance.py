"""Tests for zvisiongenerator.utils.prompt_enhance — matrix, length math, messages, hygiene, retry."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.prompt_enhance import (
    EnhanceContext,
    EnhanceSettings,
    build_messages,
    clean_output,
    enhance_context,
    enhance_prompt,
    format_enhance_spec,
    is_unusable,
    matrix_contract,
    parse_enhance_entry,
    parse_enhance_spec,
    plan_length,
    resolve_enhance_ceiling,
    settings_from_mapping,
    validate_settings,
)


class _FakeEnhancer:
    """Yield scripted outputs, one per generate() call, in small deltas."""

    repo = "fake/repo"
    revision = None

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        self.calls.append({"messages": messages, "seed": seed, "max_tokens": max_tokens, "temperature": temperature})
        text = self.outputs.pop(0)
        for index in range(0, len(text), 4):
            if cancelled is not None and cancelled():
                return
            yield text[index : index + 4]

    def close(self) -> None:
        pass


# ── Matrix and settings ─────────────────────────────────────────────────────


class TestMatrix:
    def test_contract_lists_axes_and_defaults(self):
        contract = matrix_contract()
        assert [axis["key"] for axis in contract["axes"]] == ["style", "mood", "details", "motion", "sound", "length"]
        assert contract["defaults"] == {
            "style": "keep",
            "mood": "keep",
            "details": ["lighting", "composition"],
            "length": "same",
            "motion": ["action"],
            "sound": [],
            "note": "",
        }
        motion, sound, length = contract["axes"][3], contract["axes"][4], contract["axes"][5]
        assert (motion["video_only"], motion["audio_only"]) == (True, False)
        assert (sound["video_only"], sound["audio_only"], sound["multi"]) == (True, True, True)
        assert [option["slug"] for option in sound["options"]] == ["ambient", "effects", "dialogue", "music"]
        assert [option["slug"] for option in length["options"]] == ["shorter", "same", "longer", "extra"]

    def test_sound_alone_is_not_no_op_for_video(self):
        validate_settings(EnhanceSettings(details=(), motion=(), sound=("ambient",)), mode="video")

    def test_sound_does_not_count_for_image(self):
        with pytest.raises(ValueError, match="Nothing to enhance"):
            validate_settings(EnhanceSettings(details=(), sound=("ambient",)), mode="image")

    @pytest.mark.parametrize("mode", ["image", "video"])
    def test_note_alone_is_not_no_op(self, mode):
        validate_settings(EnhanceSettings(details=(), motion=(), note="at night"), mode=mode)

    def test_no_op_combination_rejected(self):
        with pytest.raises(ValueError, match="Nothing to enhance"):
            validate_settings(EnhanceSettings(style="keep", details=(), length="same", motion=()), mode="image")

    def test_mood_alone_is_not_no_op(self):
        validate_settings(EnhanceSettings(style="keep", mood="eerie", details=(), length="same", motion=()), mode="image")

    def test_motion_alone_is_not_no_op_for_video(self):
        validate_settings(EnhanceSettings(style="keep", details=(), length="same", motion=("action",)), mode="video")

    def test_motion_does_not_count_for_image(self):
        with pytest.raises(ValueError, match="Nothing to enhance"):
            validate_settings(EnhanceSettings(style="keep", details=(), length="same", motion=("action",)), mode="image")

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown enhancement mode"):
            validate_settings(EnhanceSettings(), mode="audio")


class TestSpec:
    def test_empty_spec_is_defaults(self):
        assert parse_enhance_spec(None, mode="image") == EnhanceSettings()
        assert parse_enhance_spec("  ", mode="image") == EnhanceSettings()

    def test_full_spec(self):
        settings = parse_enhance_spec("style=cinematic,mood=dramatic,details=lighting+camera,length=longer,motion=action+camera-move", mode="video")
        assert settings == EnhanceSettings(style="cinematic", mood="dramatic", details=("lighting", "camera"), length="longer", motion=("action", "camera-move"))

    def test_partial_spec_keeps_defaults(self):
        settings = parse_enhance_spec("length=extra", mode="image")
        assert settings.length == "extra"
        assert settings.details == ("lighting", "composition")

    def test_empty_multi_value(self):
        assert parse_enhance_spec("details=,style=anime", mode="image").details == ()

    @pytest.mark.parametrize(
        "settings",
        [
            EnhanceSettings(),
            EnhanceSettings(style="3d", mood="nostalgic", details=(), length="shorter", motion=("pacing",)),
            EnhanceSettings(sound=("ambient", "dialogue")),
        ],
    )
    def test_round_trip(self, settings):
        assert parse_enhance_spec(format_enhance_spec(settings), mode="video") == settings

    def test_image_spec_omits_video_axes(self):
        spec = format_enhance_spec(EnhanceSettings(sound=("music",)), mode="image")
        assert "motion" not in spec and "sound" not in spec

    def test_sound_spec(self):
        assert parse_enhance_spec("sound=effects+music", mode="video").sound == ("effects", "music")

    def test_sound_rejected_for_image_spec(self):
        with pytest.raises(ValueError, match="only applies to video"):
            parse_enhance_spec("sound=music", mode="image")

    def test_note_not_in_spec(self):
        with pytest.raises(ValueError, match="--enhance-note"):
            parse_enhance_spec("note=night", mode="image")

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("style=watercolor", "keep, photo, candid, street, film, bw, portrait, product, cinematic, illustration, anime, comic, 3d, painterly"),
            ("mood=angry", "keep, serene, joyful, romantic, melancholic, mysterious, eerie, dramatic, epic, whimsical, nostalgic"),
            ("details=lighting+smell", "lighting, composition, camera, materials, color, environment, subject"),
            ("length=huge", "shorter, same, longer, extra"),
            ("colour=red", "Valid keys: style, mood, details, motion, sound, length, note"),
            ("cinematic", "Use key=value"),
        ],
    )
    def test_errors_list_valid_values(self, spec, message):
        with pytest.raises(ValueError, match=message):
            parse_enhance_spec(spec, mode="image")

    def test_motion_rejected_for_image_spec(self):
        with pytest.raises(ValueError, match="only applies to video"):
            parse_enhance_spec("motion=action", mode="image")

    def test_duplicates_collapse(self):
        assert parse_enhance_spec("details=lighting+lighting", mode="image").details == ("lighting",)


class TestEntry:
    @pytest.mark.parametrize("value", [None, False])
    def test_off(self, value):
        assert parse_enhance_entry(value, mode="image", where="set 'a' entry 1") is None

    def test_true_is_defaults(self):
        assert parse_enhance_entry(True, mode="image", where="x") == EnhanceSettings()

    def test_mapping(self):
        settings = parse_enhance_entry({"style": "anime", "details": ["subject"], "length": "longer"}, mode="image", where="x")
        assert settings == EnhanceSettings(style="anime", details=("subject",), length="longer")

    def test_motion_in_image_yaml_warns_and_is_ignored(self):
        with pytest.warns(UserWarning, match="only applies to video"):
            settings = parse_enhance_entry({"style": "photo", "motion": ["pacing"]}, mode="image", where="x")
        assert settings.motion == EnhanceSettings().motion

    def test_invalid_names_location(self):
        with pytest.raises(ValueError, match="set 'woman' entry 2"):
            parse_enhance_entry({"style": "nope"}, mode="image", where="set 'woman' entry 2")

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="expected true, false, or a mapping"):
            parse_enhance_entry("yes", mode="image", where="x")

    def test_mapping_from_json(self):
        assert settings_from_mapping({"details": "lighting+color"}, mode="image").details == ("lighting", "color")

    def test_note_is_one_trimmed_line(self):
        assert parse_enhance_entry({"note": "  at night,\n in the rain "}, mode="image", where="x").note == "at night, in the rain"

    @pytest.mark.parametrize("value", [42, "x" * 301])
    def test_invalid_note(self, value):
        with pytest.raises(ValueError, match="Direction"):
            settings_from_mapping({"note": value}, mode="image")

    def test_sound_in_image_yaml_warns_and_is_ignored(self):
        with pytest.warns(UserWarning, match="only applies to video"):
            settings = parse_enhance_entry({"style": "photo", "sound": ["music"]}, mode="image", where="x")
        assert settings.sound == ()


class TestContext:
    @pytest.mark.parametrize(
        ("kwargs", "orientation"),
        [
            ({"width": 704, "height": 448}, "landscape"),
            ({"width": 448, "height": 704}, "portrait"),
            ({"width": 512, "height": 512}, "square"),
            ({"ratio": "9:16"}, "portrait"),
            ({"ratio": "1:1"}, "square"),
            ({"width": 1024, "height": 576, "ratio": "9:16"}, "landscape"),  # explicit size wins over ratio
            ({"ratio": "wide"}, None),
            ({}, None),
        ],
    )
    def test_orientation(self, kwargs, orientation):
        assert enhance_context(mode="image", **kwargs).orientation == orientation

    def test_video_duration_and_audio(self):
        context = enhance_context(mode="video", frames=97, fps=24, audio=True, reference_image=True)
        assert context == EnhanceContext(audio=True, seconds=97 / 24, orientation=None, reference_image=True)

    def test_image_has_no_audio_or_duration(self):
        context = enhance_context(mode="image", frames=97, fps=24, audio=True)
        assert (context.audio, context.seconds) == (False, None)

    def test_duration_needs_fps(self):
        assert enhance_context(mode="video", frames=97).seconds is None


# ── Length math ─────────────────────────────────────────────────────────────


class TestLength:
    @pytest.mark.parametrize(
        ("in_words", "length", "target", "clamped"),
        [
            (6, "shorter", 6, True),  # at or below the 12-word floor: keeps about the same length
            (12, "shorter", 12, True),
            (15, "shorter", 12, False),  # round(7.5) lifted to the floor of 12, still shorter than 15
            (29, "shorter", 14, False),  # round(14.5) is 14 (round half to even)
            (46, "shorter", 23, False),
            (1, "shorter", 1, True),
            (15, "same", 15, False),
            (400, "same", 300, False),
            (6, "longer", 40, False),
            (46, "longer", 92, False),
            (150, "longer", 300, False),
            (400, "longer", 300, True),
            (6, "extra", 80, False),
            (46, "extra", 138, False),
            (150, "extra", 300, False),
            (300, "extra", 300, True),
        ],
    )
    def test_targets(self, in_words, length, target, clamped):
        plan = plan_length(in_words, length, ceiling=300)
        assert (plan.target, plan.clamped) == (target, clamped)

    @pytest.mark.parametrize("in_words", [1, 2, 4, 6, 15, 46, 150, 400])
    @pytest.mark.parametrize("length", ["shorter", "same", "longer", "extra"])
    def test_range_invariants(self, in_words, length):
        plan = plan_length(in_words, length, ceiling=300)
        assert 1 <= plan.lo <= plan.target <= plan.hi
        assert plan.target <= 300

    def test_flux1_ceiling(self):
        plan = plan_length(100, "extra", ceiling=180)
        assert plan.target == 180

    def test_resolve_ceiling_per_family(self):
        config = {"model_presets": {"flux1": {"enhance_max_words": 180}, "zimage": {}}, "prompt_enhancer": {"length": {"max_words": 250}}}
        assert resolve_enhance_ceiling(config, family="flux1", mode="image") == 180
        assert resolve_enhance_ceiling(config, family="zimage", mode="image") == 250
        assert resolve_enhance_ceiling({}, family=None, mode="video") == 300

    def test_unknown_length(self):
        with pytest.raises(ValueError, match="Unknown length"):
            plan_length(10, "huge", ceiling=300)


# ── Messages ────────────────────────────────────────────────────────────────


def _system(settings: EnhanceSettings, *, mode: str = "image", in_words: int = 10, context: EnhanceContext | None = None) -> str:
    plan = plan_length(in_words, settings.length, ceiling=300)
    return build_messages("p", settings, mode=mode, plan=plan, context=context)[0]["content"]


def _rule(system: str, prefix: str) -> str | None:
    """Return the rule line that starts with *prefix*, or None."""
    return next((line[2:] for line in system.splitlines() if line.startswith(f"- {prefix}")), None)


_AUDIO = EnhanceContext(audio=True)


class TestMessages:
    def test_user_message_is_prompt(self):
        plan = plan_length(3, "same", ceiling=300)
        messages = build_messages("a red fox", EnhanceSettings(), mode="image", plan=plan)
        assert messages[1] == {"role": "user", "content": "a red fox"}

    def test_details_line(self):
        assert "Add detail on: lighting, composition and framing." in _system(EnhanceSettings())

    def test_shorter_keeps_details(self):
        assert "Keep these aspects while cutting: lighting" in _system(EnhanceSettings(length="shorter"), in_words=40)
        assert "Length: at most" in _system(EnhanceSettings(length="shorter"), in_words=40)

    def test_mood_line(self):
        assert "Mood: keep the user's existing mood" in _system(EnhanceSettings())
        assert "Mood: eerie and unsettling." in _system(EnhanceSettings(mood="eerie"))

    def test_empty_details_omitted(self):
        system = _system(EnhanceSettings(style="anime", details=()))
        assert "Add detail" not in system and "Keep these aspects" not in system

    @pytest.mark.parametrize("settings", [EnhanceSettings(), EnhanceSettings(style="anime", details=(), length="shorter")])
    def test_fidelity_rule_always_present(self, settings):
        assert "EVERY DETAIL IS IMPORTANT" in _system(settings, in_words=40)

    def test_shorter_never_cuts_named_elements(self):
        assert "never an element the user named" in _system(EnhanceSettings(length="shorter"), in_words=40)

    def test_video_mode_lines(self):
        system = _system(EnhanceSettings(motion=("action", "camera-move")), mode="video")
        assert "text-to-video" in system
        assert "visible motion and camera movement" in system
        assert "Motion: describe the action as a clear sequence of events, camera movement." in system

    def test_image_mode_has_no_motion(self):
        assert "Motion:" not in _system(EnhanceSettings(motion=("action",)))

    @pytest.mark.parametrize(("length", "in_words", "expected"), [("same", 10, True), ("longer", 400, True), ("longer", 10, False), ("shorter", 40, False)])
    def test_no_echo_rule_only_when_length_stays_the_same(self, length, in_words, expected):
        assert ("don't return the input unchanged" in _system(EnhanceSettings(length=length), in_words=in_words)) is expected

    def test_clamped_uses_same_wording(self):
        system = _system(EnhanceSettings(length="longer"), in_words=400)
        assert "Do not exceed" in system

    def test_video_with_audio_allows_and_keeps_sounds(self):
        system = _system(EnhanceSettings(), mode="video", context=_AUDIO)
        assert "seen and heard" in system and "no sounds" not in system
        assert "Keep every sound and every line of quoted speech" in system

    @pytest.mark.parametrize(("mode", "context"), [("video", EnhanceContext(audio=False)), ("image", _AUDIO), ("video", None)])
    def test_no_audio_forbids_sounds(self, mode, context):
        assert "no sounds" in _system(EnhanceSettings(), mode=mode, context=context)

    def test_sound_rule_lists_choices(self):
        rule = _rule(_system(EnhanceSettings(sound=("ambient", "dialogue")), mode="video", context=_AUDIO), "Sound:")
        assert rule is not None
        assert "ambient soundscape" in rule and "write short lines" in rule and "double quotes" in rule

    @pytest.mark.parametrize(("mode", "context"), [("video", EnhanceContext(audio=False)), ("image", _AUDIO)])
    def test_sound_rule_needs_video_with_audio(self, mode, context):
        assert _rule(_system(EnhanceSettings(sound=("music",)), mode=mode, context=context), "Sound:") is None

    def test_no_sound_choice_no_sound_rule(self):
        assert _rule(_system(EnhanceSettings(), mode="video", context=_AUDIO), "Sound:") is None

    def test_direction_rule(self):
        assert _rule(_system(EnhanceSettings(note="at night, in the rain")), "Direction from the user") == "Direction from the user (follow it): at night, in the rain"
        assert _rule(_system(EnhanceSettings()), "Direction") is None

    @pytest.mark.parametrize(("seconds", "beats"), [(2.0, "one continuous action"), (5.0, "one continuous action"), (8.0, "two or three beats"), (20.0, "short sequence")])
    def test_duration_rule(self, seconds, beats):
        rule = _rule(_system(EnhanceSettings(), mode="video", context=EnhanceContext(seconds=seconds)), "Duration:")
        assert rule is not None and f"about {round(seconds)} seconds" in rule and beats in rule

    def test_duration_only_for_video(self):
        assert _rule(_system(EnhanceSettings(), context=EnhanceContext(seconds=4.0)), "Duration:") is None

    def test_frame_rule(self):
        assert _rule(_system(EnhanceSettings(), context=EnhanceContext(orientation="portrait")), "Frame:") == "Frame: portrait; compose for that shape."
        assert _rule(_system(EnhanceSettings()), "Frame:") is None

    def test_reference_image_rules(self):
        video = _system(EnhanceSettings(), mode="video", context=EnhanceContext(reference_image=True))
        image = _system(EnhanceSettings(), context=EnhanceContext(reference_image=True))
        assert _rule(video, "The first frame is supplied as an image") is not None
        assert _rule(image, "A reference image supplies the subject and layout") is not None
        assert "reference image" not in _system(EnhanceSettings()) and "first frame" not in _system(EnhanceSettings(), mode="video")


# ── Hygiene ─────────────────────────────────────────────────────────────────


class TestHygiene:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ("<think>plan</think>A fox.", "A fox."),
            ("<think>unfinished", ""),
            ("Prompt: A fox.", "A fox."),
            ("Rewritten prompt:  A fox.", "A fox."),
            ('"A fox."', "A fox."),
            ("“A fox.”", "A fox."),
            ("A fox.<|im_end|>", "A fox."),
        ],
    )
    def test_clean(self, raw, clean):
        assert clean_output(raw) == clean

    @pytest.mark.parametrize(
        ("output", "unusable"),
        [
            ("", True),
            ("   ", True),
            ("a  RED fox", True),
            ("I can't help with that.", True),
            ("I’m sorry, but no.", True),
            ("As an AI, I cannot.", True),
            ("A fox in an inappropriate hat.", False),
            ("A red fox in snow, golden light.", False),
        ],
    )
    def test_unusable(self, output, unusable):
        assert is_unusable(output, "a red fox") is unusable


# ── enhance_prompt ──────────────────────────────────────────────────────────


class TestEnhancePrompt:
    def test_success_streams_and_returns(self):
        enhancer = _FakeEnhancer(["A grey fox in deep snow, soft light."])
        seen: list[str] = []
        result = enhance_prompt(enhancer, "a grey fox", EnhanceSettings(), mode="image", seed=7, ceiling=300, on_text=seen.append)
        assert result.prompt == "A grey fox in deep snow, soft light."
        assert result.clamped is False
        assert seen[0] == "" and seen[-1] == result.prompt
        assert len(seen) >= 3  # first delta is shown immediately; later ones are throttled
        assert enhancer.calls[0]["seed"] == 7
        assert enhancer.calls[0]["temperature"] == pytest.approx(0.7)

    @pytest.mark.parametrize("bad", ["", "a red fox", "I cannot do that."])
    def test_retry_once_with_next_seed(self, bad):
        enhancer = _FakeEnhancer([bad, "A red fox, golden light."])
        result = enhance_prompt(enhancer, "a red fox", EnhanceSettings(), mode="image", seed=5, ceiling=300)
        assert result.prompt == "A red fox, golden light."
        assert [call["seed"] for call in enhancer.calls] == [5, 6]

    def test_two_failures_raise(self):
        enhancer = _FakeEnhancer(["", "a red fox"])
        with pytest.raises(RuntimeError, match="no usable rewrite"):
            enhance_prompt(enhancer, "a red fox", EnhanceSettings(), mode="image", seed=1, ceiling=300)
        assert len(enhancer.calls) == 2

    def test_cancel_raises(self):
        enhancer = _FakeEnhancer(["A long rewrite that will be cancelled."])
        with pytest.raises(RuntimeError, match="cancelled"):
            enhance_prompt(enhancer, "a red fox", EnhanceSettings(), mode="image", seed=1, ceiling=300, cancelled=lambda: True)

    def test_empty_prompt(self):
        with pytest.raises(ValueError, match="Enter a prompt"):
            enhance_prompt(_FakeEnhancer([]), "  ", EnhanceSettings(), mode="image", seed=1, ceiling=300)

    def test_clamped_reported(self):
        enhancer = _FakeEnhancer(["word " * 300])
        result = enhance_prompt(enhancer, "word " * 400, EnhanceSettings(length="longer"), mode="image", seed=1, ceiling=300)
        assert result.clamped is True


class TestReviewFixes:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ('"Welcome" reads the neon sign above a bar called "Moe\'s"', '"Welcome" reads the neon sign above a bar called "Moe\'s"'),
            ("'Moe's bar at night'", "'Moe's bar at night'"),
            ('"A fox in snow."', "A fox in snow."),
        ],
    )
    def test_quotes_only_strip_a_wrapping_pair(self, raw, clean):
        assert clean_output(raw) == clean

    def test_enhance_by_set_for_mode_drops_motion_only_entries_for_images(self):
        from zvisiongenerator.utils.prompt_enhance import enhance_by_set_for_mode

        motion_only = EnhanceSettings(style="keep", details=(), length="same", motion=("pacing",))
        anime = EnhanceSettings(style="anime")
        with pytest.warns(UserWarning, match="no change in images"):
            assert enhance_by_set_for_mode({"a": [motion_only, anime, None]}, mode="image") == {"a": [None, anime, None]}
        assert enhance_by_set_for_mode({"a": [motion_only]}, mode="video") == {"a": [motion_only]}
        assert enhance_by_set_for_mode(None, mode="image") is None

    def test_hi_never_exceeds_ceiling(self):
        plan = plan_length(100, "extra", ceiling=180)
        assert plan.target == 180 and plan.hi == 180 and plan.lo < plan.hi

    def test_text_updates_are_throttled(self, monkeypatch):
        import zvisiongenerator.utils.prompt_enhance as module

        clock = iter([0.0, 0.05, 0.06, 0.2, 0.21] + [1.0] * 50)
        monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
        seen: list[str] = []

        class _Chunks:
            repo = "f"
            revision = None

            def generate(self, messages, **kwargs):
                yield from ["A ", "red ", "fox ", "in ", "snow."]

            def close(self):
                pass

        enhance_prompt(_Chunks(), "a fox", EnhanceSettings(), mode="image", seed=1, ceiling=300, on_text=seen.append)
        assert seen == ["", "A", "A red fox in", "A red fox in snow."]

    def test_valid_user_config_is_used(self):
        from zvisiongenerator.utils.prompt_enhance import enhance_options

        options = enhance_options({"prompt_enhancer": {"temperature": 0, "length": {"max_words": 250, "percent": {"longer": 150}}}})
        assert options["temperature"] == 0
        assert options["length"]["max_words"] == 250 and options["length"]["percent"]["longer"] == 150
