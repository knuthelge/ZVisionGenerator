"""Tests for zvisiongenerator.utils.prompt_enhance — matrix, length math, messages, hygiene, retry."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.prompt_enhance import (
    DEFAULT_LENGTH_CONFIG,
    EnhanceContext,
    EnhanceSettings,
    build_messages,
    clean_output,
    enhance_by_set_for_mode,
    enhance_context,
    enhance_prompt,
    format_enhance_spec,
    is_noop,
    is_unusable,
    matrix_contract,
    parse_enhance_entry,
    parse_enhance_spec,
    plan_length,
    resolve_enhance_ceiling,
    resolve_parts,
    settings_from_mapping,
    system_message,
    validate_settings,
)
from zvisiongenerator.utils.prompt_enhance_video import (
    CONTINUE_NATURALLY,
    SCENE_HEAVY_WARNING,
    EnhancePairResult,
    PairLengthPlan,
    build_pair_messages,
    enhance_video_prompt,
    parse_scene_action,
    plan_pair_length,
    repeats_source,
    split_scene_action,
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
            "parts": None,
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
            ("colour=red", "Valid keys: style, mood, details, motion, sound, length, note, parts"),
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
            (15, "same", 25, False),  # a short prompt grows to the Same floor of 25 words
            (30, "same", 30, False),
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
    def test_system_message_lists_rules_between_intro_and_closing(self):
        assert system_message("Intro.", ["one", "two"], "Closing.") == "Intro.\nRules:\n- one\n- two\nClosing."

    def test_user_message_is_prompt(self):
        plan = plan_length(3, "same", ceiling=300)
        messages = build_messages("a red fox", EnhanceSettings(), mode="image", plan=plan)
        assert messages[1] == {"role": "user", "content": "Prompt: a red fox"}
        assert messages[0]["content"].splitlines()[2].startswith("- The user gives a Prompt.")

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
        assert _rule(_system(EnhanceSettings(), context=EnhanceContext(orientation="portrait")), "Compose for") == "Compose for a portrait frame; never mention the frame or its shape."
        assert _rule(_system(EnhanceSettings()), "Compose for") is None

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


# ── Video prompts as scene and action ───────────────────────────────────────

_KEEP_ALL = EnhanceSettings(style="keep", mood="keep", details=(), length="same", motion=(), sound=())


def _pair_system(settings: EnhanceSettings | None = None, *, scene: str = "", prompt: str = "a knight rides in", context: EnhanceContext | None = None) -> str:
    settings = settings or EnhanceSettings()
    plan = plan_pair_length(len(scene.split()), len(prompt.split()), settings.length, ceiling=300)
    return build_pair_messages(scene, prompt, settings, plan=plan, context=context)[0]["content"]


class TestPairLength:
    @pytest.mark.parametrize(("scene_words", "prompt_words", "scene", "action"), [(10, 20, 10, 20), (0, 30, 10, 20), (30, 0, 10, 20), (30, 90, 30, 90)])
    def test_split_goldens(self, scene_words, prompt_words, scene, action):
        plan = plan_pair_length(scene_words, prompt_words, "same", ceiling=300)
        assert isinstance(plan, PairLengthPlan)
        assert plan.total.target == scene_words + prompt_words
        assert (plan.scene.target, plan.action.target) == (scene, action)
        assert plan.scene_heavy is False

    def test_ranges_are_fifteen_percent_with_a_minimum_of_three(self):
        plan = plan_pair_length(100, 100, "same", ceiling=300)
        assert (plan.scene.lo, plan.scene.hi) == (85, 115)
        small = plan_pair_length(0, 30, "same", ceiling=300)
        assert (small.scene.lo, small.scene.hi) == (7, 13)

    def test_scene_heavy_input_keeps_an_action_floor_and_the_ceiling(self):
        plan = plan_pair_length(280, 5, "same", ceiling=300)
        assert plan.scene_heavy is True
        assert plan.action.target >= 12
        assert plan.scene.hi + plan.action.hi <= 300
        assert plan.scene.lo <= plan.scene.target <= plan.scene.hi

    def test_a_short_input_still_gets_the_action_floor(self):
        no_same_floor = {**DEFAULT_LENGTH_CONFIG, "floor_words": {**DEFAULT_LENGTH_CONFIG["floor_words"], "same": 1}}
        plan = plan_pair_length(1, 2, "same", ceiling=300, length_cfg=no_same_floor)
        assert plan.action.target == 12 and plan.scene.target == 1

    def test_upper_bounds_never_exceed_the_ceiling_after_a_longer_request(self):
        plan = plan_pair_length(60, 60, "extra", ceiling=150)
        assert plan.total.target == 150
        assert plan.scene.hi + plan.action.hi <= 150

    def test_clamped_total_is_carried(self):
        assert plan_pair_length(100, 100, "longer", ceiling=300).total.clamped is False
        clamped = plan_pair_length(200, 200, "longer", ceiling=100)
        assert clamped.total.clamped is True and clamped.scene.clamped is True


class TestPairMessages:
    def test_user_message_carries_both_inputs(self):
        plan = plan_pair_length(3, 4, "same", ceiling=300)
        messages = build_pair_messages("A dark alley", "a cat jumps", EnhanceSettings(), plan=plan)
        assert messages[1] == {"role": "user", "content": "Scene: A dark alley\nPrompt: a cat jumps"}

    def test_blank_scene_is_none(self):
        plan = plan_pair_length(0, 4, "same", ceiling=300)
        assert build_pair_messages("  ", "a cat jumps", EnhanceSettings(), plan=plan)[1]["content"].startswith("Scene: (none)\n")

    def test_format_line_names_both_markers(self):
        system = _pair_system()
        assert "`SCENE:` then `ACTION:`" in system
        assert "text-to-video" in system

    def test_option_mapping_style_mood_details_go_to_the_scene(self):
        system = _pair_system(EnhanceSettings(style="anime", mood="eerie", details=("lighting",)))
        assert _rule(system, "For the SCENE: Style: anime") is not None
        assert _rule(system, "For the SCENE: Mood: eerie") is not None
        assert _rule(system, "For the SCENE: add detail on: lighting") is not None
        assert _rule(system, "For the ACTION: Style") is None

    def test_option_mapping_motion_and_sound_go_to_the_action(self):
        system = _pair_system(EnhanceSettings(motion=("camera-move",), sound=("dialogue",)), context=_AUDIO)
        assert _rule(system, "For the ACTION: describe camera movement") is not None
        assert _rule(system, "For the ACTION: describe speech") is not None
        assert _rule(system, "For the SCENE: describe") is None

    def test_sound_needs_an_audio_track(self):
        assert "quoted lines" not in _pair_system(EnhanceSettings(sound=("dialogue",)))
        assert "quoted lines" in _pair_system(EnhanceSettings(sound=("dialogue",)), context=_AUDIO)

    def test_direction_applies_to_both_parts(self):
        system = _pair_system(EnhanceSettings(note="make it rain"))
        assert _rule(system, "Direction from the user (follow it in both parts): make it rain") is not None

    def test_part_length_lines(self):
        plan = plan_pair_length(10, 20, "same", ceiling=300)
        system = build_pair_messages("a b", "c d", EnhanceSettings(), plan=plan)[0]["content"]
        assert f"SCENE length: {plan.scene.lo}-{plan.scene.hi} words." in system
        assert f"ACTION length: {plan.action.lo}-{plan.action.hi} words." in system

    def test_text_to_video_splits_and_expands(self):
        rule = _rule(_pair_system(), "Sort the user's text")
        assert rule is not None and "everything in the Prompt" in rule

    def test_image_to_video_keeps_the_scene_short(self):
        system = _pair_system(context=EnhanceContext(reference_image=True))
        assert "Keep the SCENE short" in system
        assert "Sort the user's text" not in system

    def test_scene_rule_forbids_events_and_speech(self):
        assert "No events, no camera moves, no quoted speech." in _pair_system()

    def test_audio_adds_dialogue_to_the_action_role(self):
        assert "sound and quoted dialogue" in _pair_system(context=_AUDIO)
        assert "sound and quoted dialogue" not in _pair_system()

    def test_blank_prompt_asks_for_an_action(self):
        assert "The Prompt is (none)" in _pair_system(scene="A castle at dusk", prompt="")


class TestParseSceneAction:
    @pytest.mark.parametrize(
        "text",
        [
            "SCENE: A castle at dusk.\nACTION: A knight rides in.",
            "**Scene:** A castle at dusk.\n**Action:** A knight rides in.",
            "**Scene**: A castle at dusk.\n**Action**: A knight rides in.",
            "## Scene:\nA castle at dusk.\n\n## Action:\nA knight rides in.",
            "SCENE: A castle at dusk. ACTION: A knight rides in.",
            "SCENE: A castle at dusk.\nPROMPT: A knight rides in.",
            "Here you go:\nscene: A castle at dusk.\naction: A knight rides in.",
            "<think>plan the shot</think>\nSCENE: A castle at dusk.\nACTION: A knight rides in.",
            "<think>SCENE: nope\nACTION: nope</think>SCENE: A castle at dusk.\nACTION: A knight rides in.",
        ],
    )
    def test_marker_styles(self, text):
        assert parse_scene_action(text) == ("A castle at dusk.", "A knight rides in.")

    def test_the_word_scene_inside_the_action_is_not_a_marker(self):
        text = 'SCENE: A stage.\nACTION: The scene goes dark and the actor says "Bye."'
        assert parse_scene_action(text) == ("A stage.", 'The scene goes dark and the actor says "Bye."')

    def test_markers_after_the_action_marker_are_action_text(self):
        assert parse_scene_action("SCENE: A.\nACTION: B. Scene: C. Action: D.") == ("A.", "B. Scene: C. Action: D.")

    def test_a_marker_inside_a_word_does_not_count(self):
        assert parse_scene_action("SCENE: A crowd.\nREACTION: they cheer.") is None

    @pytest.mark.parametrize(
        "text",
        ["SCENE: A castle.", "ACTION: A knight rides in.", "A knight rides in.", "SCENE: A castle.\nACTION:", "SCENE:\nACTION: A knight rides in.", "<think>SCENE: A. ACTION: B."],
    )
    def test_missing_marker_or_part_is_none(self, text):
        assert parse_scene_action(text) is None

    def test_parts_are_cleaned(self):
        assert parse_scene_action('SCENE: "A castle."\nACTION: <|im_end|>A knight rides in.') == ("A castle.", "A knight rides in.")


class TestSplitSceneAction:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", ("", "")),
            ("SCENE: A castle.\nACTION: A kni", ("A castle.", "A kni")),
            ("SCENE: A castle.\nACT", ("A castle.", "")),
            ("SCENE: A castle.\nACTION", ("A castle.", "")),
            ("SCENE: A castle.\n**Ac", ("A castle.", "")),
            ("SCENE: A castle. P", ("A castle.", "")),
            ("SCENE: A castle.\nPROMPT: A knight", ("A castle.", "A knight")),
            ("SCENE: A cas", ("A cas", "")),
            ("SCENE:", ("", "")),
            ("**Scene:** A castle.\n**Action:** A knight", ("A castle.", "A knight")),
            ("**Scene:*", ("", "")),
        ],
    )
    def test_partials(self, raw, expected):
        assert split_scene_action(raw) == expected

    def test_text_before_the_scene_marker_shows_as_action(self):
        assert split_scene_action("A knight rides") == ("", "A knight rides")
        assert split_scene_action("A knight. SC") == ("", "A knight.")
        assert split_scene_action("SCEN") == ("", "")

    def test_a_word_that_cannot_become_a_marker_is_not_held_back(self):
        assert split_scene_action("SCENE: A castle. Another") == ("A castle. Another", "")
        assert split_scene_action("SCENE: A castle. Scenic") == ("A castle. Scenic", "")

    def test_nothing_is_held_back_inside_the_action(self):
        assert split_scene_action("SCENE: A.\nACTION: The scene ends. S") == ("A.", "The scene ends. S")

    def test_think_blocks_are_hidden(self):
        assert split_scene_action("<think>SCENE: x ACTION: y") == ("", "")
        assert split_scene_action("<think>x</think>SCENE: A.\nACTION: B") == ("A.", "B")

    def test_action_only_returns_the_raw_text(self):
        assert split_scene_action("SCENE: A.\nACTION: B", action_only=True) == ("", "SCENE: A.\nACTION: B")

    def test_no_prefix_of_a_stream_leaks_a_marker(self):
        full = "SCENE: A castle at dusk.\nACTION: A knight rides in."
        seen = [split_scene_action(full[:end]) for end in range(len(full) + 1)]
        assert seen[-1] == ("A castle at dusk.", "A knight rides in.")
        assert all("ACT" not in scene and "ACT" not in action for scene, action in seen)


_USER_SCENE = "A castle at dusk."


def _pair_run(outputs: list[str], *, scene: str = "", prompt: str = "a knight rides in", settings: EnhanceSettings | None = None, **kwargs):
    enhancer = _FakeEnhancer(outputs)
    result = enhance_video_prompt(enhancer, scene, prompt, settings or EnhanceSettings(), seed=10, ceiling=300, **kwargs)
    return result, enhancer


class TestEnhanceVideoPrompt:
    def test_success(self):
        result, enhancer = _pair_run(["SCENE: A castle at dusk, soft light.\nACTION: A knight rides in through the gate."])
        assert result == EnhancePairResult(scene="A castle at dusk, soft light.", prompt="A knight rides in through the gate.", clamped=False, warning=None)
        assert len(enhancer.calls) == 1 and enhancer.calls[0]["seed"] == 10

    def test_streams_raw_text_and_ends_with_the_cleaned_output(self):
        seen: list[str] = []
        text = "SCENE: A castle at dusk.\nACTION: A knight rides in."
        _pair_run([text], on_text=seen.append)
        assert seen[0] == "" and seen[-1] == text

    def test_an_echoed_scene_is_valid(self):
        result, enhancer = _pair_run([f"SCENE: {_USER_SCENE}\nACTION: A knight rides in at speed."], scene=_USER_SCENE)
        assert result.scene == _USER_SCENE and len(enhancer.calls) == 1

    def test_an_echoed_action_is_rejected_and_retried(self):
        result, enhancer = _pair_run(["SCENE: A castle.\nACTION: A Knight  rides in", "SCENE: A castle.\nACTION: A knight rides in on a grey horse."])
        assert result.prompt == "A knight rides in on a grey horse."
        assert [call["seed"] for call in enhancer.calls] == [10, 11]

    @pytest.mark.parametrize("bad", ["", "A knight rides in on a horse.", "SCENE: A castle.", "SCENE: A castle.\nACTION: I cannot do that."])
    def test_unusable_output_is_retried_once(self, bad):
        result, enhancer = _pair_run([bad, "SCENE: A castle.\nACTION: A knight gallops in."])
        assert result.prompt == "A knight gallops in."
        assert [call["seed"] for call in enhancer.calls] == [10, 11]

    def test_a_new_quote_in_the_scene_is_rejected(self):
        result, _ = _pair_run(['SCENE: A man says "hi".\nACTION: A knight gallops in.', "SCENE: A castle.\nACTION: A knight gallops in."], scene=_USER_SCENE)
        assert result.scene == "A castle."

    def test_a_quote_already_in_the_users_scene_is_kept(self):
        scene = 'A man says "Hi there" at a door.'
        result, enhancer = _pair_run(['SCENE: A man says "hi  there" at an old door.\nACTION: He steps inside.'], scene=scene)
        assert result.scene == 'A man says "hi  there" at an old door.' and len(enhancer.calls) == 1

    def test_quotes_in_the_action_are_always_fine(self):
        result, _ = _pair_run(['SCENE: A castle.\nACTION: The knight shouts "Charge!" and rides in.'])
        assert '"Charge!"' in result.prompt

    def test_fallback_uses_the_action_segment_and_leaves_the_scene_unrewritten(self):
        """The fallback result's scene is "" (not rewritten), so callers keep the user's scene and never copy it into an enhanced field."""
        result, enhancer = _pair_run(["no markers at all", "ACTION: A knight rides in on a horse."], scene=_USER_SCENE)
        assert (result.scene, result.prompt) == ("", "A knight rides in on a horse.")
        assert result.warning is not None and "scene is left as written" in result.warning
        assert len(enhancer.calls) == 2

    def test_fallback_without_markers_uses_the_whole_text(self):
        result, _ = _pair_run(["", "A knight gallops across the drawbridge."])
        assert (result.scene, result.prompt) == ("", "A knight gallops across the drawbridge.")

    @pytest.mark.parametrize(
        "second",
        [
            "SCENE: A castle at dusk with a lone tower.\n\nA knight rides in on a horse.",
            "SCENE: A castle at dusk with a lone tower.\nA knight rides in on a horse.",
            'SCENE: A castle at dusk with a lone tower, a man says "lone tower".\nACTION: A knight rides in on a horse.',
        ],
    )
    def test_a_fallback_composes_without_duplicated_scene_text(self, second):
        result, _ = _pair_run(["no markers", second], scene=_USER_SCENE)
        # The fallback leaves the scene unrewritten (""), so the caller composes the user's own scene with the action.
        assert result.scene == ""
        composed = f"{_USER_SCENE} {result.prompt}"
        assert composed == f"{_USER_SCENE} A knight rides in on a horse."
        assert composed.count("castle") == 1 and "lone tower" not in composed

    @pytest.mark.parametrize("second", ["SCENE: A castle at dusk with a lone tower.", "a knight rides in", "I cannot do that."], ids=["scene segment only", "echoed action", "refusal"])
    def test_an_unusable_fallback_raises(self, second):
        with pytest.raises(RuntimeError, match="no usable rewrite"):
            _pair_run(["", second], scene=_USER_SCENE)

    def test_cancel_raises(self):
        with pytest.raises(RuntimeError, match="cancelled"):
            _pair_run(["SCENE: A.\nACTION: B."], cancelled=lambda: True)

    def test_both_blank_raises(self):
        with pytest.raises(ValueError, match="scene or a prompt"):
            enhance_video_prompt(_FakeEnhancer([]), " ", None, EnhanceSettings(), seed=1, ceiling=300)

    def test_scene_only_asks_for_an_action(self):
        result, enhancer = _pair_run(["SCENE: A castle at dusk.\nACTION: Mist drifts past the towers as a bell tolls."], scene=_USER_SCENE, prompt="")
        assert result.prompt.startswith("Mist drifts")
        assert enhancer.calls[0]["messages"][1]["content"] == f"Scene: {_USER_SCENE}\nPrompt: (none)"

    def test_none_scene_is_blank(self):
        enhancer = _FakeEnhancer(["SCENE: A castle.\nACTION: A knight gallops in."])
        result = enhance_video_prompt(enhancer, None, "a knight rides in", EnhanceSettings(), seed=1, ceiling=300)
        assert result.scene == "A castle."

    def test_heavy_scene_warns_and_reports(self):
        with pytest.warns(UserWarning, match="most of the word budget"):
            result, _ = _pair_run(["SCENE: A castle.\nACTION: A knight gallops in."], scene="word " * 250, prompt="a knight rides in")
        assert result.warning == SCENE_HEAVY_WARNING

    def test_clamped_is_reported(self):
        enhancer = _FakeEnhancer(["SCENE: A castle.\nACTION: A knight gallops in."])
        result = enhance_video_prompt(enhancer, "word " * 60, "word " * 60, EnhanceSettings(length="longer"), seed=1, ceiling=100)
        assert result.clamped is True

    def test_all_keep_settings_are_allowed_when_the_prompt_is_split(self):
        result, _ = _pair_run(["SCENE: A castle.\nACTION: A knight gallops in."], settings=_KEEP_ALL)
        assert result.prompt == "A knight gallops in."

    def test_all_keep_settings_are_a_noop_with_a_scene(self):
        with pytest.raises(ValueError, match="Nothing to enhance"):
            _pair_run([], scene=_USER_SCENE, settings=_KEEP_ALL)

    def test_image_to_video_context_reaches_the_prompt(self):
        _, enhancer = _pair_run(["SCENE: A castle.\nACTION: A knight gallops in."], context=EnhanceContext(reference_image=True))
        assert "Keep the SCENE short" in enhancer.calls[0]["messages"][0]["content"]


class TestNoOpSplit:
    def test_all_keep_video_with_a_split_is_a_change(self):
        assert is_noop(_KEEP_ALL, mode="video") is True
        assert is_noop(_KEEP_ALL, mode="video", split=True) is False

    def test_images_ignore_split(self):
        assert is_noop(_KEEP_ALL, mode="image", split=True) is True

    def test_validate_settings_accepts_a_split_video(self):
        validate_settings(_KEEP_ALL, mode="video", split=True)
        with pytest.raises(ValueError, match="Nothing to enhance"):
            validate_settings(_KEEP_ALL, mode="video")
        with pytest.raises(ValueError, match="Nothing to enhance"):
            validate_settings(_KEEP_ALL, mode="image", split=True)

    def test_enhance_by_set_keeps_entries_without_a_scene(self):
        with pytest.warns(UserWarning, match="no change in videos"):
            result = enhance_by_set_for_mode({"a": [_KEEP_ALL, _KEEP_ALL, None]}, mode="video", scene_by_set={"a": [None, "A castle.", None]})
        assert result == {"a": [_KEEP_ALL, None, None]}

    def test_enhance_by_set_uses_the_job_scene_for_entries_without_one(self):
        with pytest.warns(UserWarning, match="no change in videos"):
            result = enhance_by_set_for_mode({"a": [_KEEP_ALL, _KEEP_ALL]}, mode="video", scene_by_set={"a": ["", "Own scene."]}, job_scene="Job scene.")
        assert result == {"a": [None, None]}

    def test_blank_job_scene_still_splits(self):
        assert enhance_by_set_for_mode({"a": [_KEEP_ALL]}, mode="video", job_scene="  ") == {"a": [_KEEP_ALL]}

    def test_images_drop_all_keep_entries_whatever_the_scene(self):
        with pytest.warns(UserWarning, match="no change in images"):
            assert enhance_by_set_for_mode({"a": [_KEEP_ALL]}, mode="image", scene_by_set={"a": [None]}) == {"a": [None]}


# ── Extending a clip ────────────────────────────────────────────────────────

_EXT_SCENE = "A woman in a red raincoat stands on a rainy street at night."
_EXT_PREVIOUS = 'She looks at the camera and says "I told you so". Then she sighs.'
_EXT_LEGACY = f"{_EXT_SCENE} {_EXT_PREVIOUS}"
_EXT_PAIR = f"SCENE: {_EXT_SCENE}\nACTION: She opens a black umbrella."


def _extend_context(**overrides) -> EnhanceContext:
    values = {"mode": "video", "width": 512, "height": 768, "frames": 73, "fps": 24.0, "new_frames": 48, "extends": True, "previous": _EXT_PREVIOUS}
    values.update(overrides)
    return enhance_context(**values)


def _extend_run(
    outputs: list[str],
    *,
    scene: str = _EXT_SCENE,
    prompt: str = "She grabs an umbrella.",
    context: EnhanceContext | None = None,
    settings: EnhanceSettings | None = None,
    **kwargs,
):
    enhancer = _FakeEnhancer(outputs)
    context = context or _extend_context()
    result = enhance_video_prompt(enhancer, scene, prompt, settings or EnhanceSettings(), seed=10, ceiling=300, context=context, **kwargs)
    return result, enhancer


def _sent_system(enhancer: _FakeEnhancer, call: int = 0) -> str:
    return enhancer.calls[call]["messages"][0]["content"]


def _sent_user(enhancer: _FakeEnhancer, call: int = 0) -> str:
    return enhancer.calls[call]["messages"][1]["content"]


class TestExtendContext:
    def test_defaults_do_not_extend(self):
        context = enhance_context(mode="video", frames=48, fps=24.0)
        assert (context.extends, context.previous, context.extract_from, context.prompt_is_original, context.new_seconds) == (False, None, None, False, None)

    def test_the_window_and_the_new_part_are_in_seconds_at_the_source_fps(self):
        context = _extend_context(frames=25 + 48, new_frames=48, fps=24.0)
        assert context.seconds == pytest.approx(73 / 24) and context.new_seconds == pytest.approx(2.0)
        assert context.extends and context.previous == _EXT_PREVIOUS

    def test_extension_fields_are_dropped_for_images(self):
        context = enhance_context(mode="image", extends=True, previous="x", extract_from="y", prompt_is_original=True, new_frames=8, fps=24.0)
        assert not context.extends and context.previous is None and context.extract_from is None
        assert not context.prompt_is_original and context.new_seconds is None

    def test_the_legacy_fields_pass_through(self):
        context = _extend_context(extract_from=_EXT_LEGACY, prompt_is_original=True)
        assert context.extract_from == _EXT_LEGACY and context.prompt_is_original


class TestRepeatsSource:
    def test_a_repeated_quoted_line_is_a_repeat(self):
        assert repeats_source('She smiles and says "I told you so!"', previous=_EXT_PREVIOUS)

    def test_the_comparison_ignores_case_punctuation_and_curly_quotes(self):
        assert repeats_source("She says “i TOLD you... so”.", previous=_EXT_PREVIOUS)

    def test_a_new_quote_is_not_a_repeat(self):
        assert not repeats_source('She says "Goodbye now".', previous=_EXT_PREVIOUS)

    def test_a_longer_line_containing_the_old_one_is_a_repeat(self):
        assert repeats_source('She says "I told you so, again and again".', previous=_EXT_PREVIOUS)

    def test_a_short_line_must_match_exactly(self):
        assert not repeats_source('She says "Yes, I will".', previous='He says "Yes".')
        assert repeats_source('She says "Yes".', previous='He says "Yes".')

    def test_an_action_almost_equal_to_the_previous_prompt_is_too_close(self):
        previous = "She walks to the door and opens it. She steps outside into the rain."
        action = "She walks to the door and opens it. She steps outside into the rain!"
        assert repeats_source(action, previous=previous)

    def test_closeness_compares_without_the_previous_quotes(self):
        previous = 'She looks at the camera. She says "I told you so". Then she sighs.'
        assert repeats_source("She looks at the camera. Then she sighs.", previous=previous)

    def test_a_continuation_that_reuses_the_earlier_vocabulary_is_not_a_repeat(self):
        previous = "She opens a black umbrella and walks away down the street in the rain."
        action = "She walks back down the street in the rain, opens the black umbrella and looks at the sky."
        assert not repeats_source(action, previous=previous)

    def test_a_short_common_quoted_word_is_only_a_repeat_on_an_exact_match(self):
        assert repeats_source('He says "Yes."', previous='She says "Yes."')
        assert not repeats_source('He says "Yes, I know."', previous='She says "Yes."')
        assert not repeats_source('He says "No."', previous='She says "Yes."')

    def test_a_different_action_is_fine(self):
        assert not repeats_source("She opens a black umbrella and walks away down the street.", previous=_EXT_PREVIOUS)

    def test_blank_previous_never_repeats(self):
        assert not repeats_source("Anything.", previous="")


class TestExtendWithScene:
    def test_the_scene_is_not_rewritten_and_the_output_is_the_action(self):
        """The result's scene is "" (kept as written by the caller), where it once echoed the user's scene back."""
        result, enhancer = _extend_run(["She opens a black umbrella and walks away."])
        assert result == EnhancePairResult(scene="", prompt="She opens a black umbrella and walks away.", clamped=False, warning=None)
        assert len(enhancer.calls) == 1 and enhancer.calls[0]["seed"] == 10

    def test_the_scene_and_the_previous_prompt_are_in_the_system_message_only(self):
        _, enhancer = _extend_run(["She opens a black umbrella and walks away."])
        system = _sent_system(enhancer)
        assert f"The scene (kept as is, do not restate): {_EXT_SCENE}" in system
        assert "Never repeat their clothing, appearance or the setting" in system
        assert f"What just happened (do not repeat its actions or quoted lines): {_EXT_PREVIOUS}" in system
        assert _sent_user(enhancer) == "She grabs an umbrella."

    def test_an_empty_prompt_continues_naturally(self):
        result, enhancer = _extend_run(["She walks away slowly."], prompt="")
        assert _sent_user(enhancer) == CONTINUE_NATURALLY and result.prompt == "She walks away slowly."

    def test_the_prompt_is_stripped_in_the_user_message(self):
        _, enhancer = _extend_run(["She walks away slowly."], prompt="  She leaves.\n")
        assert _sent_user(enhancer) == "She leaves."

    def test_none_inputs_are_allowed(self):
        enhancer = _FakeEnhancer(["She walks away slowly."])
        result = enhance_video_prompt(enhancer, _EXT_SCENE, None, EnhanceSettings(), seed=1, ceiling=300, context=_extend_context())
        assert result.prompt == "She walks away slowly." and _sent_user(enhancer) == CONTINUE_NATURALLY

    @pytest.mark.parametrize(("new_frames", "fps", "fitted"), [(48, 24.0, "2 seconds of a 3 second window"), (80, 16.0, "5 seconds of a 7 second window")])
    def test_the_window_and_the_new_part_are_in_the_rules_in_seconds_at_the_source_fps(self, new_frames, fps, fitted):
        _, enhancer = _extend_run(["She walks away slowly."], context=_extend_context(frames=25 + new_frames, new_frames=new_frames, fps=fps))
        assert f"fitted to {fitted}" in _sent_system(enhancer)

    def test_the_rules_start_from_where_the_clip_ends(self):
        _, enhancer = _extend_run(["She walks away slowly."])
        system = _sent_system(enhancer)
        assert "final positions and poses" in system and "describe speakers as before" in system and "Compose for a portrait frame" in system

    def test_no_reference_image_rule_and_no_style_rules_for_the_kept_scene(self):
        context = _extend_context(reference_image=True)
        _, enhancer = _extend_run(["She walks away slowly."], context=context, settings=EnhanceSettings(style="anime", mood="eerie"))
        system = _sent_system(enhancer)
        assert "first frame is supplied" not in system and "Style:" not in system and "Mood:" not in system and "Add detail on" not in system

    def test_motion_sound_and_direction_apply(self):
        settings = EnhanceSettings(motion=("camera-move",), sound=("dialogue",), note="Keep it quiet.")
        _, enhancer = _extend_run(["She walks away slowly."], context=_extend_context(audio=True), settings=settings)
        system = _sent_system(enhancer)
        assert "Motion: describe camera movement." in system and "Sound: describe speech" in system
        assert "Direction from the user (follow it): Keep it quiet." in system

    def test_sound_is_left_out_without_audio(self):
        _, enhancer = _extend_run(["She walks away slowly."], settings=EnhanceSettings(sound=("dialogue",)))
        assert "Sound:" not in _sent_system(enhancer)

    def test_no_previous_prompt_leaves_the_line_out(self):
        _, enhancer = _extend_run(["She walks away slowly."], context=_extend_context(previous=None))
        assert "What just happened" not in _sent_system(enhancer)

    def test_an_action_too_close_to_the_previous_prompt_is_rejected_and_retried(self):
        context = _extend_context(previous='She looks at the camera. She says "I told you so". Then she sighs.')
        result, enhancer = _extend_run(["She looks at the camera. Then she sighs.", "She opens a black umbrella."], context=context)
        assert len(enhancer.calls) == 2
        assert result.prompt == "She opens a black umbrella."

    @pytest.mark.parametrize(
        "bad", ["", "I cannot help with that.", "She  grabs an Umbrella.", 'She smiles and says "I told you so".'], ids=["empty", "refusal", "echo of the prompt", "repeated quote"]
    )
    def test_unusable_output_an_echo_of_the_prompt_and_a_repeated_quote_are_retried(self, bad):
        result, enhancer = _extend_run([bad, "She opens a black umbrella and steps away."])
        assert result.prompt == "She opens a black umbrella and steps away."
        assert [call["seed"] for call in enhancer.calls] == [10, 11]

    def test_two_repeating_outputs_raise(self):
        with pytest.raises(RuntimeError, match="no usable rewrite"):
            _extend_run(['She says "I told you so".', 'She says "I told you so".'])

    def test_a_leading_action_marker_is_stripped(self):
        result, _ = _extend_run(["ACTION: She walks away slowly."])
        assert result.prompt == "She walks away slowly."

    def test_a_scene_and_action_reply_keeps_only_the_action(self):
        result, enhancer = _extend_run([f"SCENE: {_EXT_SCENE} Neon signs glow.\nACTION: She walks away slowly."])
        assert result.prompt == "She walks away slowly." and "Neon" not in result.prompt and result.scene == ""
        assert len(enhancer.calls) == 1

    def test_a_scene_without_an_action_is_retried(self):
        result, enhancer = _extend_run([f"SCENE: {_EXT_SCENE} Neon signs glow.", "She walks away slowly."])
        assert result.prompt == "She walks away slowly." and len(enhancer.calls) == 2

    def test_two_scene_only_replies_raise(self):
        with pytest.raises(RuntimeError, match="no usable rewrite"):
            _extend_run([f"SCENE: {_EXT_SCENE}", f"SCENE: {_EXT_SCENE}"])

    def test_streaming_ends_with_the_action_text(self):
        seen: list[str] = []
        _extend_run(["She walks away slowly."], on_text=seen.append)
        assert seen[0] == "" and seen[-1] == "She walks away slowly."
        assert split_scene_action(seen[-1], action_only=True) == ("", "She walks away slowly.")

    def test_cancellation_raises(self):
        with pytest.raises(RuntimeError, match="cancelled"):
            _extend_run(["She walks away slowly."], cancelled=lambda: True)

    def test_default_settings_that_change_nothing_elsewhere_still_write_an_action(self):
        result, _ = _extend_run(["She walks away slowly."], settings=EnhanceSettings(details=(), motion=()))
        assert result.prompt == "She walks away slowly."

    @pytest.mark.parametrize(("length", "words"), [("same", "34-46"), ("longer", "68-92")])
    def test_the_action_gets_two_thirds_of_the_total_scaled_by_the_length_setting(self, length, words):
        _, enhancer = _extend_run(["She walks away slowly."], scene=" ".join(["scene"] * 30), prompt=" ".join(["act"] * 30), settings=EnhanceSettings(length=length))
        assert f"Length: {words} words." in _sent_system(enhancer)

    def test_a_long_scene_leaves_a_floor_for_the_action_and_warns(self):
        enhancer = _FakeEnhancer(["She walks away slowly."])
        with pytest.warns(UserWarning, match="scene uses most"):
            result = enhance_video_prompt(enhancer, " ".join(["scene"] * 295), "", EnhanceSettings(), seed=1, ceiling=300, context=_extend_context())
        assert result.warning == SCENE_HEAVY_WARNING
        assert "Length: 9-15 words." in _sent_system(enhancer) or "Length: 9-12 words." in _sent_system(enhancer)


class TestExtendWithoutScene:
    def test_a_legacy_source_extracts_the_scene_with_the_old_prompt_in_the_system_message(self):
        context = _extend_context(previous=_EXT_LEGACY, extract_from=_EXT_LEGACY)
        result, enhancer = _extend_run([_EXT_PAIR], scene="", context=context)
        assert (result.scene, result.prompt) == (_EXT_SCENE, "She opens a black umbrella.")
        system = _sent_system(enhancer)
        assert f"Extract the scene (who, where, look, style, camera) from this earlier prompt: {_EXT_LEGACY}" in system
        assert "Write the action for what happens next" in system
        assert "SCENE: who and where" in system and "final positions and poses" in system

    @pytest.mark.parametrize(
        ("prompt", "is_original", "expected"),
        [(_EXT_LEGACY, True, CONTINUE_NATURALLY), ("She opens an umbrella.", False, "She opens an umbrella."), ("", False, CONTINUE_NATURALLY)],
        ids=["unchanged", "edited", "blank"],
    )
    def test_a_legacy_prompt_is_sent_only_when_it_was_edited(self, prompt, is_original, expected):
        context = _extend_context(previous=_EXT_LEGACY, extract_from=_EXT_LEGACY, prompt_is_original=is_original)
        _, enhancer = _extend_run([_EXT_PAIR], scene="", prompt=prompt, context=context)
        assert _sent_user(enhancer) == expected

    def test_the_previous_prompt_is_listed_separately_when_it_differs_from_the_extract_source(self):
        context = _extend_context(previous="She sighs.", extract_from=_EXT_LEGACY)
        _, enhancer = _extend_run([_EXT_PAIR], scene="", context=context)
        assert "What just happened (do not repeat its actions or quoted lines): She sighs." in _sent_system(enhancer)

    @pytest.mark.parametrize("action", ['She says "I told you so".', _EXT_LEGACY], ids=["quoted line", "close to the old prompt"])
    def test_the_repeat_guard_retries_even_though_the_user_prompt_was_not_sent(self, action):
        context = _extend_context(previous=_EXT_LEGACY, extract_from=_EXT_LEGACY, prompt_is_original=True)
        result, enhancer = _extend_run([f"SCENE: {_EXT_SCENE}\nACTION: {action}", _EXT_PAIR], scene="", prompt=_EXT_LEGACY, context=context)
        assert result.prompt == "She opens a black umbrella." and len(enhancer.calls) == 2

    def test_a_quote_in_the_extracted_scene_is_rejected(self):
        context = _extend_context(previous=_EXT_LEGACY, extract_from=_EXT_LEGACY)
        result, _ = _extend_run([f'SCENE: {_EXT_SCENE} She says "I told you so".\nACTION: She walks off.', f"SCENE: {_EXT_SCENE}\nACTION: She walks off."], scene="", context=context)
        assert result.scene == _EXT_SCENE

    def test_the_fallback_keeps_an_empty_scene_and_warns(self):
        context = _extend_context(previous=_EXT_LEGACY, extract_from=_EXT_LEGACY)
        result, enhancer = _extend_run(["no markers", "She opens a black umbrella."], scene="", context=context)
        assert (result.scene, result.prompt) == ("", "She opens a black umbrella.")
        assert result.warning is not None and len(enhancer.calls) == 2

    def test_a_repeating_fallback_raises(self):
        context = _extend_context(previous=_EXT_LEGACY, extract_from=_EXT_LEGACY)
        with pytest.raises(RuntimeError, match="no usable rewrite"):
            _extend_run(['no markers: She says "I told you so".', 'She says "I told you so".'], scene="", context=context)

    def test_the_window_rule_uses_the_new_part_and_the_window(self):
        context = _extend_context(frames=25 + 48, new_frames=48, fps=24.0, extract_from=_EXT_LEGACY)
        _, enhancer = _extend_run([_EXT_PAIR], scene="", context=context)
        assert "fitted to 2 seconds of a 3 second window" in _sent_system(enhancer) and "Duration:" not in _sent_system(enhancer)

    def test_no_source_prompt_at_all_still_writes_a_pair_from_blank_inputs(self):
        context = _extend_context(previous=None)
        result, enhancer = _extend_run(["SCENE: A street at night.\nACTION: A woman walks away."], scene="", prompt="", context=context)
        assert (result.scene, result.prompt) == ("A street at night.", "A woman walks away.")
        assert _sent_user(enhancer) == CONTINUE_NATURALLY and "Extract the scene" not in _sent_system(enhancer)


class TestExtendBlankInputs:
    def test_blank_inputs_are_refused_for_a_context_that_does_not_extend(self):
        with pytest.raises(ValueError, match="Enter a scene or a prompt"):
            _extend_run(["x"], scene="  ", prompt="\n", context=_extend_context(extends=False))

    def test_a_non_extending_run_is_unchanged_by_extension_fields_it_ignores(self):
        context = EnhanceContext(previous='x says "I told you so".')
        result, _ = _pair_run(['SCENE: A castle.\nACTION: He says "I told you so".'], context=context)
        assert result.prompt == 'He says "I told you so".'


# ── The parts setting: rewrite both or only the action ──────────────────────

_KEPT_SCENE = "A knight in dull armor stands in a castle courtyard at dusk."
_ACTION_ONLY = EnhanceSettings(parts="action")


def _action_run(outputs: list[str], *, scene: str = _KEPT_SCENE, prompt: str = "he draws his sword", context: EnhanceContext | None = None, settings: EnhanceSettings = _ACTION_ONLY):
    enhancer = _FakeEnhancer(outputs)
    result = enhance_video_prompt(enhancer, scene, prompt, settings, seed=10, ceiling=300, context=context or enhance_context(mode="video", frames=121, fps=24.0))
    return result, enhancer


class TestPartsGrammar:
    def test_default_is_auto(self):
        assert EnhanceSettings().parts is None
        assert parse_enhance_spec("", mode="video").parts is None

    @pytest.mark.parametrize("value", ["both", "action"])
    def test_spec_round_trip(self, value):
        settings = parse_enhance_spec(f"style=anime,parts={value}", mode="video")
        assert settings.parts == value
        assert parse_enhance_spec(format_enhance_spec(settings), mode="video") == settings

    def test_format_omits_the_default_and_writes_a_set_value_last(self):
        assert "parts" not in format_enhance_spec(EnhanceSettings())
        assert format_enhance_spec(EnhanceSettings(parts="action")).endswith(",parts=action")

    def test_image_format_omits_parts(self):
        assert "parts" not in format_enhance_spec(EnhanceSettings(parts="action"), mode="image")

    def test_mapping_round_trip_and_null_means_auto(self):
        settings = settings_from_mapping({"parts": "action", "length": "longer"}, mode="video")
        assert settings.parts == "action"
        assert settings_from_mapping({"parts": None}, mode="video").parts is None

    def test_invalid_value_lists_the_choices(self):
        with pytest.raises(ValueError, match="both, action"):
            parse_enhance_spec("parts=scene", mode="video")
        with pytest.raises(ValueError, match="both, action"):
            settings_from_mapping({"parts": ""}, mode="video")
        with pytest.raises(ValueError, match="both, action"):
            settings_from_mapping({"parts": 3}, mode="video")

    def test_images_reject_or_ignore_parts_like_motion(self):
        with pytest.raises(ValueError, match="only applies to video"):
            parse_enhance_spec("parts=action", mode="image")
        with pytest.warns(UserWarning, match="ignored for images"):
            assert settings_from_mapping({"parts": "action"}, mode="image", motion_in_image="warn").parts is None

    def test_contract_describes_the_choice(self):
        parts = matrix_contract()["parts"]
        assert (parts["key"], parts["video_only"], parts["default"]) == ("parts", True, None)
        assert [(option["slug"], option["label"]) for option in parts["options"]] == [("both", "Scene and action"), ("action", "Action only")]

    def test_prompt_file_entry_accepts_parts(self):
        assert parse_enhance_entry({"parts": "action"}, mode="video", where="entry 0").parts == "action"
        with pytest.raises(ValueError, match="Invalid 'enhance' in entry 0"):
            parse_enhance_entry({"parts": "scene"}, mode="video", where="entry 0")


class TestResolveParts:
    @pytest.mark.parametrize(
        ("parts", "scene", "extends", "reference_image", "expected"),
        [
            (None, "A castle.", False, False, "both"),
            (None, "A castle.", True, False, "action"),
            (None, "A castle.", False, True, "action"),
            (None, "", True, False, "both"),
            (None, "", False, True, "both"),
            (None, "  ", False, False, "both"),
            ("both", "A castle.", True, False, "both"),
            ("both", "A castle.", False, True, "both"),
            ("action", "A castle.", False, False, "action"),
            ("action", "A castle.", True, False, "action"),
            ("action", "", False, False, "both"),
            ("action", None, True, False, "both"),
        ],
    )
    def test_rule(self, parts, scene, extends, reference_image, expected):
        assert resolve_parts(EnhanceSettings(parts=parts), scene=scene, extends=extends, reference_image=reference_image) == expected


class TestPartsNoop:
    def test_action_is_a_change_even_with_keep_all_settings(self):
        keep = EnhanceSettings(details=(), motion=(), parts="action")
        assert is_noop(keep, mode="video") is False
        validate_settings(keep, mode="video")

    def test_both_and_auto_keep_the_existing_rule(self):
        for parts in (None, "both"):
            assert is_noop(EnhanceSettings(details=(), motion=(), parts=parts), mode="video") is True

    def test_images_ignore_parts(self):
        assert is_noop(EnhanceSettings(details=(), motion=(), parts="action"), mode="image") is True

    def test_a_keep_all_action_entry_with_its_own_scene_is_kept(self):
        entry = EnhanceSettings(details=(), motion=(), parts="action")
        plain = EnhanceSettings(details=(), motion=())
        with pytest.warns(UserWarning, match="no change in videos"):
            result = enhance_by_set_for_mode({"a": [entry, plain]}, mode="video", scene_by_set={"a": ["A castle.", "A castle."]})
        assert result == {"a": [entry, None]}


class TestActionOnlyWithoutExtending:
    def test_the_scene_is_not_in_the_result(self):
        result, enhancer = _action_run(["The knight draws his sword and raises it."])
        assert result == EnhancePairResult(scene="", prompt="The knight draws his sword and raises it.", clamped=False, warning=None)
        assert len(enhancer.calls) == 1

    def test_the_prompt_says_to_keep_the_scene(self):
        _, enhancer = _action_run(["The knight draws his sword and raises it."])
        system, user = _sent_system(enhancer), _sent_user(enhancer)
        assert f"The scene (kept exactly as written, do not restate or change it): {_KEPT_SCENE}" in system
        assert "Never repeat their clothing, appearance or the setting" in system
        assert "SCENE:" not in system and _KEPT_SCENE not in user and user == "he draws his sword"

    def test_style_mood_and_details_belong_to_the_scene_and_are_left_out(self):
        settings = EnhanceSettings(style="anime", mood="eerie", details=("lighting",), motion=("camera-move",), note="Keep it quiet.", parts="action")
        _, enhancer = _action_run(["The knight draws his sword."], settings=settings)
        system = _sent_system(enhancer)
        assert "Style:" not in system and "Mood:" not in system and "Add detail on" not in system
        assert "Motion: describe camera movement." in system and "Direction from the user (follow it): Keep it quiet." in system

    def test_a_blank_prompt_asks_for_an_action_that_suits_the_scene(self):
        result, enhancer = _action_run(["The knight walks across the courtyard."], prompt="")
        assert "write an action that suits the scene" in _sent_user(enhancer) and result.scene == ""

    def test_a_reference_image_gets_the_describe_change_rule(self):
        context = enhance_context(mode="video", frames=121, fps=24.0, reference_image=True)
        _, enhancer = _action_run(["The knight draws his sword."], context=context)
        assert "The first frame is supplied as an image. Describe the motion, camera movement and changes over time" in _sent_system(enhancer)

    def test_no_reference_image_rule_without_an_image(self):
        _, enhancer = _action_run(["The knight draws his sword."])
        assert "first frame is supplied" not in _sent_system(enhancer)

    def test_duration_and_frame_rules_apply(self):
        context = enhance_context(mode="video", width=512, height=768, frames=121, fps=24.0)
        _, enhancer = _action_run(["The knight draws his sword."], context=context)
        system = _sent_system(enhancer)
        assert "Compose for a portrait frame" in system and "Duration: about 5 seconds" in system

    def test_sound_needs_audio(self):
        settings = EnhanceSettings(sound=("effects",), parts="action")
        _, silent = _action_run(["The knight draws his sword."], settings=settings)
        _, loud = _action_run(["The knight draws his sword."], settings=settings, context=enhance_context(mode="video", frames=121, fps=24.0, audio=True))
        assert "Sound:" not in _sent_system(silent) and "Sound: describe sound effects" in _sent_system(loud)

    def test_an_echo_of_the_prompt_is_retried(self):
        result, enhancer = _action_run(["he draws his sword", "The knight draws his sword and raises it."])
        assert result.prompt == "The knight draws his sword and raises it." and [call["seed"] for call in enhancer.calls] == [10, 11]

    def test_a_scene_and_action_reply_keeps_only_the_action(self):
        result, _ = _action_run([f"SCENE: {_KEPT_SCENE} Torches burn.\nACTION: The knight draws his sword."])
        assert (result.scene, result.prompt) == ("", "The knight draws his sword.")

    def test_two_unusable_outputs_raise(self):
        with pytest.raises(RuntimeError, match="no usable rewrite"):
            _action_run(["", "I cannot help with that."])

    def test_a_scene_that_fills_the_budget_warns_and_still_runs(self):
        long_scene = " ".join(["word"] * 260)
        with pytest.warns(UserWarning, match="scene uses most of the word budget"):
            result, _ = _action_run(["The knight draws his sword."], scene=long_scene)
        assert result.warning == SCENE_HEAVY_WARNING

    def test_both_asks_for_a_scene_and_action_pair_even_with_a_scene(self):
        result, enhancer = _action_run(["SCENE: A castle courtyard.\nACTION: A knight draws his sword."], settings=EnhanceSettings(parts="both"))
        assert (result.scene, result.prompt) == ("A castle courtyard.", "A knight draws his sword.")
        assert "SCENE:" in _sent_system(enhancer)

    def test_a_blank_scene_falls_back_to_both_without_error(self):
        result, enhancer = _action_run(["SCENE: A castle courtyard.\nACTION: A knight draws his sword."], scene="")
        assert (result.scene, result.prompt) == ("A castle courtyard.", "A knight draws his sword.")
        assert "Prompt: he draws his sword" in _sent_user(enhancer)


class TestActionOnlyWhenExtending:
    def test_explicit_action_matches_the_auto_default_for_a_scene(self):
        explicit, _ = _extend_run(["She opens a black umbrella."], settings=_ACTION_ONLY)
        auto, _ = _extend_run(["She opens a black umbrella."])
        assert explicit == auto and explicit.scene == ""

    def test_a_blank_scene_falls_back_to_a_pair(self):
        result, _ = _extend_run([_EXT_PAIR], scene="", settings=_ACTION_ONLY)
        assert result.scene == _EXT_SCENE

    def test_both_with_a_scene_rewrites_the_scene_and_sends_it_to_the_model(self):
        result, enhancer = _extend_run([_EXT_PAIR], settings=EnhanceSettings(parts="both"))
        assert (result.scene, result.prompt) == (_EXT_SCENE, "She opens a black umbrella.")
        assert _sent_user(enhancer) == f"Scene: {_EXT_SCENE}\nWhat happens next: She grabs an umbrella."
        system = _sent_system(enhancer)
        assert "SCENE:" in system and "rewrite it, keeping its facts" in system and "What just happened" in system

    def test_both_with_a_scene_still_rejects_a_repeat_of_the_previous_prompt(self):
        repeat = f'SCENE: {_EXT_SCENE}\nACTION: She says "I told you so".'
        result, enhancer = _extend_run([repeat, _EXT_PAIR], settings=EnhanceSettings(parts="both"))
        assert result.prompt == "She opens a black umbrella." and len(enhancer.calls) == 2
