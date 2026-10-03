"""Tests for zvisiongenerator.utils.prompt_enhance — matrix, length math, protection, hygiene, retry."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.prompt_compose import expand_random_choices
from zvisiongenerator.utils.prompt_enhance import (
    EnhanceSettings,
    build_messages,
    clean_output,
    enhance_prompt,
    format_enhance_spec,
    is_unusable,
    matrix_contract,
    parse_enhance_entry,
    parse_enhance_spec,
    plan_length,
    protect_groups,
    resolve_enhance_ceiling,
    restore_groups,
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
        assert [axis["key"] for axis in contract["axes"]] == ["style", "details", "length", "motion"]
        assert contract["defaults"] == {"style": "keep", "details": ["lighting", "composition"], "length": "same", "motion": ["action"]}
        motion = contract["axes"][3]
        assert motion["video_only"] is True
        assert [option["slug"] for option in contract["axes"][2]["options"]] == ["shorter", "same", "longer", "extra"]

    def test_no_op_combination_rejected(self):
        with pytest.raises(ValueError, match="Nothing to enhance"):
            validate_settings(EnhanceSettings(style="keep", details=(), length="same", motion=()), mode="image")

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
        settings = parse_enhance_spec("style=cinematic,details=lighting+camera,length=longer,motion=action+camera-move", mode="video")
        assert settings == EnhanceSettings(style="cinematic", details=("lighting", "camera"), length="longer", motion=("action", "camera-move"))

    def test_partial_spec_keeps_defaults(self):
        settings = parse_enhance_spec("length=extra", mode="image")
        assert settings.length == "extra"
        assert settings.details == ("lighting", "composition")

    def test_empty_multi_value(self):
        assert parse_enhance_spec("details=,style=anime", mode="image").details == ()

    @pytest.mark.parametrize("settings", [EnhanceSettings(), EnhanceSettings(style="3d", details=(), length="shorter", motion=("pacing",))])
    def test_round_trip(self, settings):
        assert parse_enhance_spec(format_enhance_spec(settings), mode="video") == settings

    def test_image_spec_omits_motion(self):
        assert "motion" not in format_enhance_spec(EnhanceSettings(), mode="image")

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("style=watercolor", "keep, photo, cinematic, illustration, anime, 3d, painterly"),
            ("details=lighting+smell", "lighting, composition, camera, materials, color, environment, subject"),
            ("length=huge", "shorter, same, longer, extra"),
            ("colour=red", "Valid keys: style, details, length, motion"),
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


def _system(settings: EnhanceSettings, *, mode: str = "image", in_words: int = 10, protected: bool = False) -> str:
    plan = plan_length(in_words, settings.length, ceiling=300)
    return build_messages("p", settings, mode=mode, plan=plan, protected=protected)[0]["content"]


class TestMessages:
    def test_user_message_is_prompt(self):
        plan = plan_length(3, "same", ceiling=300)
        messages = build_messages("a red fox", EnhanceSettings(), mode="image", plan=plan, protected=False)
        assert messages[1] == {"role": "user", "content": "a red fox"}

    def test_details_line(self):
        assert "Add detail on: lighting, composition and framing." in _system(EnhanceSettings())

    def test_shorter_keeps_details(self):
        assert "Keep these aspects while cutting: lighting" in _system(EnhanceSettings(length="shorter"), in_words=40)
        assert "Length: at most" in _system(EnhanceSettings(length="shorter"), in_words=40)

    def test_empty_details_omitted(self):
        system = _system(EnhanceSettings(style="anime", details=()))
        assert "Add detail" not in system and "Keep these aspects" not in system

    def test_placeholder_rule_only_when_protected(self):
        assert "Placeholders like [[1]]" not in _system(EnhanceSettings())
        assert "Placeholders like [[1]]" in _system(EnhanceSettings(), protected=True)

    def test_video_mode_lines(self):
        system = _system(EnhanceSettings(motion=("action", "camera-move")), mode="video")
        assert "text-to-video" in system
        assert "visible motion and camera movement" in system
        assert "Motion: describe the action as a clear sequence of events, camera movement." in system

    def test_image_mode_has_no_motion(self):
        assert "Motion:" not in _system(EnhanceSettings(motion=("action",)))

    def test_clamped_uses_same_wording(self):
        system = _system(EnhanceSettings(length="longer"), in_words=400)
        assert "Do not exceed" in system


# ── Protection ──────────────────────────────────────────────────────────────


def _groups_once(text: str, groups: tuple[str, ...]) -> bool:
    stripped = text
    for group in groups:
        if stripped.count(group) != 1:
            return False
        stripped = stripped.replace(group, "")
    return "{" not in stripped and "}" not in stripped


class TestProtect:
    @pytest.mark.parametrize(
        ("prompt", "protected", "groups"),
        [
            ("a fox", "a fox", ()),
            ("a {red|blue} fox", "a [[1]] fox", ("{red|blue}",)),
            ("{a|b} and {c|d}", "[[1]] and [[2]]", ("{a|b}", "{c|d}")),
            ("a {red|{dark|light} blue} roof", "a [[1]] roof", ("{red|{dark|light} blue}",)),
            ("a {single} word", "a [[1]] word", ("{single}",)),
            ("unbalanced { brace", "unbalanced { brace", ()),
        ],
    )
    def test_protect(self, prompt, protected, groups):
        assert protect_groups(prompt) == (protected, groups)

    def test_protect_matches_expand_grammar(self):
        prompt = "x {a|{b|c}} y {z}"
        _protected, groups = protect_groups(prompt)
        for group in groups:
            assert "{" not in expand_random_choices(group)


class TestRestore:
    GROUPS = ("{red|blue}", "{day|night}")

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("A [[1]] roof at [[2]].", "A {red|blue} roof at {day|night}."),
            ("A [[1]] roof.", "A {red|blue} roof, {day|night}"),
            ("A roof at [[2]].", "A roof at {day|night}, {red|blue}"),
            ("A roof.", "A roof, {red|blue}, {day|night}"),
            ("A [[1]] roof, [[1]] tiles at [[2]].", "A {red|blue} roof, tiles at {day|night}."),
            ("A [[1]] roof at [[2]] [[9]].", "A {red|blue} roof at {day|night}."),
            ("A [[1]] roof at [[2]], {sunny|rainy}.", "A {red|blue} roof at {day|night}, sunny|rainy."),
            ("", "{red|blue}, {day|night}"),
        ],
    )
    def test_restore(self, output, expected):
        restored = restore_groups(output, self.GROUPS)
        assert restored == expected
        assert _groups_once(restored, self.GROUPS)

    def test_nested_group_restored_verbatim(self):
        groups = ("{red|{dark|light} blue}",)
        assert restore_groups("a [[1]] roof", groups) == "a {red|{dark|light} blue} roof"

    def test_no_groups_still_strips_invented_braces(self):
        assert restore_groups("a {x|y} fox", ()) == "a x|y fox"


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
        enhancer = _FakeEnhancer(["A [[1]] fox in deep snow, soft light."])
        seen: list[str] = []
        result = enhance_prompt(enhancer, "a {red|grey} fox", EnhanceSettings(), mode="image", seed=7, ceiling=300, protect=True, on_text=seen.append)
        assert result.prompt == "A {red|grey} fox in deep snow, soft light."
        assert result.clamped is False
        assert seen[0] == "" and seen[-1] == result.prompt
        assert len(seen) >= 3  # first delta is shown immediately; later ones are throttled
        assert enhancer.calls[0]["seed"] == 7
        assert enhancer.calls[0]["temperature"] == pytest.approx(0.7)

    def test_auto_mode_does_not_protect(self):
        enhancer = _FakeEnhancer(["A red fox, golden light."])
        enhance_prompt(enhancer, "a red fox", EnhanceSettings(), mode="image", seed=1, ceiling=300, protect=False)
        system = enhancer.calls[0]["messages"][0]["content"]
        assert "Placeholders" not in system

    @pytest.mark.parametrize("bad", ["", "a red fox", "I cannot do that."])
    def test_retry_once_with_next_seed(self, bad):
        enhancer = _FakeEnhancer([bad, "A red fox, golden light."])
        result = enhance_prompt(enhancer, "a red fox", EnhanceSettings(), mode="image", seed=5, ceiling=300, protect=True)
        assert result.prompt == "A red fox, golden light."
        assert [call["seed"] for call in enhancer.calls] == [5, 6]

    def test_two_failures_raise(self):
        enhancer = _FakeEnhancer(["", "a red fox"])
        with pytest.raises(RuntimeError, match="no usable rewrite"):
            enhance_prompt(enhancer, "a red fox", EnhanceSettings(), mode="image", seed=1, ceiling=300, protect=True)
        assert len(enhancer.calls) == 2

    def test_cancel_raises(self):
        enhancer = _FakeEnhancer(["A long rewrite that will be cancelled."])
        with pytest.raises(RuntimeError, match="cancelled"):
            enhance_prompt(enhancer, "a red fox", EnhanceSettings(), mode="image", seed=1, ceiling=300, protect=True, cancelled=lambda: True)

    def test_empty_prompt(self):
        with pytest.raises(ValueError, match="Enter a prompt"):
            enhance_prompt(_FakeEnhancer([]), "  ", EnhanceSettings(), mode="image", seed=1, ceiling=300, protect=True)

    def test_clamped_reported(self):
        enhancer = _FakeEnhancer(["word " * 300])
        result = enhance_prompt(enhancer, "word " * 400, EnhanceSettings(length="longer"), mode="image", seed=1, ceiling=300, protect=False)
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

        enhance_prompt(_Chunks(), "a fox", EnhanceSettings(), mode="image", seed=1, ceiling=300, protect=False, on_text=seen.append)
        assert seen == ["", "A", "A red fox in", "A red fox in snow."]

    def test_valid_user_config_is_used(self):
        from zvisiongenerator.utils.prompt_enhance import enhance_options

        options = enhance_options({"prompt_enhancer": {"temperature": 0, "length": {"max_words": 250, "percent": {"longer": 150}}}})
        assert options["temperature"] == 0
        assert options["length"]["max_words"] == 250 and options["length"]["percent"]["longer"] == 150

    def test_groups_count_at_average_option_length(self):
        from zvisiongenerator.utils.prompt_enhance import prompt_word_count

        prompt = "{a majestic red fox in fresh snow at golden hour|a snowy owl on a frosted branch at dawn}, photorealistic"
        protected, groups = protect_groups(prompt)
        assert prompt_word_count(protected, groups) == 11  # options of 10 and 9 words → 10, plus "photorealistic" (comma is not a word)
        assert prompt_word_count(*protect_groups("a {red|{dark|light} blue} roof")) == 4
        assert prompt_word_count("plain words here", ()) == 3

    def test_enhance_button_plans_length_from_real_prompt_size(self):
        enhancer = _FakeEnhancer(["A [[1]] scene, photorealistic, soft golden light."])
        prompt = "{a majestic red fox in fresh snow at golden hour|a snowy owl on a frosted branch at dawn}, photorealistic"
        enhance_prompt(enhancer, prompt, EnhanceSettings(), mode="image", seed=1, ceiling=300, protect=True)
        assert "Length: 8-14 words" in enhancer.calls[0]["messages"][0]["content"]  # was "1-5 words" before the fix

    def test_placeholders_with_spaces_are_restored(self):
        assert restore_groups("a [[ 1 ]] car under a [[2]] sky", ("{red|blue}", "{grey|clear}")) == "a {red|blue} car under a {grey|clear} sky"
