"""Tests for the prompt-enhancer bench: cases, checks, the runner with a fake model, and the compare report."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import yaml

from bench.enhance import __main__ as bench_cli
from bench.enhance.cases import case_seeds, load_cases, parse_case, select_cases
from bench.enhance.checks import (
    CHECK_NAMES,
    audio_format,
    audio_shape_name,
    banned_clear,
    case_score,
    caption_format,
    count_kept,
    forbid_clear,
    keep_share,
    length_check,
    length_plan,
    marks_kept,
    lyrics_format,
    quotes_kept,
    scope_kept,
    tags_kept,
    fields_filled,
    fill_tags,
    fit_ratio,
    marks_known,
    only_empty,
    values_valid,
    script_format,
    script_words,
    words_kept,
)
from bench.enhance.report import Run, check_rates, compare_runs, load_run
from bench.enhance.sheet import find_image, prompts_file_text, render_command, sheet_rows
from bench.enhance.runner import META_FILE, RESULTS_FILE, RecordingEnhancer, rescore_run, run_bench, run_case
from zvisiongenerator.utils.audio_settings import reading_characters
from zvisiongenerator.utils.audio_writers import NOTHING_TO_ADD_MESSAGE, SECTIONS_CHANGED_MESSAGE
from zvisiongenerator.utils.prompt_enhance import DEFAULT_LENGTH_CONFIG, LengthPlan
from zvisiongenerator.utils.prompt_enhance_audio import AUDIO_AXES, caption_rules
from zvisiongenerator.utils.speech_text import strip_marks

_OPTIONS = {"temperature": 0.7, "max_new_tokens": 700, "idle_release_seconds": 0.0, "length": DEFAULT_LENGTH_CONFIG}
_CONFIG: dict = {}


class _FakeEnhancer:
    """Yields the next scripted reply per call, word by word."""

    def __init__(self, replies: list[str]) -> None:
        self.repo = "fake/model"
        self.revision = None
        self._replies = list(replies)
        self.closed = False

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled: Callable[[], bool] | None = None) -> Iterator[str]:
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        for word in reply.split(" "):
            yield word + " "

    def close(self) -> None:
        self.closed = True


def _audio_case(mode: str, family: str, **overrides):
    entry = {"id": "a1", "tags": ["audio"], "mode": mode, "family": family, "prompt": "a song", "settings": {}}
    entry.update(overrides)
    return parse_case(entry)


_SA3_CASE = {"mode": "audio-sound", "family": "stable_audio_3"}


def _case(**overrides):
    entry = {"id": "c1", "tags": ["t"], "prompt": "a fox in snow", "settings": {}}
    entry.update(overrides)
    return parse_case(entry)


class TestCases:
    def test_shipped_cases_load(self):
        cases = load_cases()
        assert len(cases) >= 50
        assert {case.mode for case in cases} == {"image", "video", "caption", "audio-sound", "audio-style", "audio-lyrics", "audio-script", "audio-fill"}

    @pytest.mark.parametrize("mode", ["audio-sound", "audio-style", "audio-lyrics", "audio-script", "audio-fill"])
    def test_shipped_audio_modes_have_at_least_four_cases(self, mode):
        cases = [case for case in load_cases() if case.mode == mode]
        assert len(cases) >= 4
        assert all(case.audio_context is not None and case.family for case in cases)

    def test_shipped_audio_cases_cover_both_families_and_the_starter_set(self):
        cases = [case for case in load_cases() if case.is_audio]
        assert {case.family for case in cases} == {"yue2", "stable_audio_3", "qwen3_tts"}
        assert {"song", "instrumental", "music", "sound-type", "names"} <= {tag for case in cases for tag in case.tags}
        assert {case.audio_context.sound_type for case in cases if case.task == "sound"} == {"auto", "ambience", "effect", "one-shot"}

    def test_seeds_are_stable_and_at_least_eight_digits(self):
        seeds = case_seeds("short-default", 3)
        assert seeds == case_seeds("short-default", 3)
        assert all(seed >= 10_000_000 for seed in seeds)
        assert seeds[1] == seeds[0] + 1
        assert case_seeds("other", 1) != seeds[:1]

    @pytest.mark.parametrize(
        "entry",
        [
            {"id": "x", "prompt": "a", "mode": "audio"},
            {"id": "x", "prompt": "a", "mode": "audio-style"},
            {"id": "x", "prompt": "a", "mode": "audio-style", "family": "yue2", "settings": {"space": "forest"}},
            {"id": "x", "prompt": "a", "mode": "audio-sound", "family": "stable_audio_3", "settings": {"genre": "jazz"}},
            {"id": "x", "prompt": "a", "mode": "audio-sound", "family": "stable_audio_3", "context": {"ratio": "1:1"}},
            {"id": "x", "prompt": "a", "mode": "audio-sound", "family": "stable_audio_3", "context": {"sound_type": "loop"}},
            {"id": "x", "prompt": "a", "mode": "audio-sound", "family": "stable_audio_3", "context": {"seconds": 0}},
            {"id": "x", "prompt": "[Verse]\nx", "mode": "audio-lyrics"},
            {"id": "x", "prompt": "[Verse]\nx", "mode": "audio-lyrics", "family": "yue2", "settings": {"improve": ["louder"]}},
            {"id": "x", "prompt": "[Verse]\nx", "mode": "audio-lyrics", "family": "yue2", "settings": {"action": "write", "section": 0}},
            {"id": "x", "prompt": "[Verse]\n[Chorus]", "mode": "audio-lyrics", "family": "yue2"},
            {"id": "x", "prompt": "[Verse]\nx", "mode": "audio-lyrics", "family": "yue2", "context": {"tempo": 90}},
            {"id": "x", "prompt": "[Verse]\nx", "mode": "audio-lyrics", "family": "yue2", "context": {"style": 3}},
            {"id": "x", "prompt": "Hello there.", "mode": "audio-script"},
            {"id": "x", "prompt": "Hello there.", "mode": "audio-script", "family": "qwen3_tts"},
            {"id": "x", "prompt": "Hello there.", "mode": "audio-script", "family": "qwen3_tts", "settings": {"actions": ["fit"]}},
            {"id": "x", "prompt": "Hello there.", "mode": "audio-script", "family": "qwen3_tts", "settings": {"actions": ["delivery"], "seconds": 20}},
            {"id": "x", "prompt": "[sad] ...", "mode": "audio-script", "family": "qwen3_tts", "settings": {"actions": ["delivery"]}},
            {"id": "x", "prompt": "Hello there.", "mode": "audio-script", "family": "qwen3_tts", "settings": {"actions": ["delivery"]}, "context": {"style": "folk"}},
            {"id": "x", "prompt": "a", "mode": "audio-style", "family": "yue2", "context": {"style": "folk"}},
            {"id": "x", "prompt": "", "mode": "audio-style", "family": "yue2"},
            {"id": "x", "prompt": "a", "mode": "audio-style", "family": "yue2", "context": {"describe": "a door"}},
            {"id": "x", "prompt": "a", "settings": {"style": "nope"}},
            {"id": "x", "prompt": "a", "scene": "b"},
            {"id": "x", "prompt": "", "settings": {}},
            {"id": "x", "prompt": "a", "colour": "red"},
            {"id": "x", "prompt": "a", "context": {"speed": 2}},
            {"id": "x", "prompt": "a", "settings": {"theme": "western"}},
        ],
    )
    def test_rejects_malformed_case(self, entry):
        with pytest.raises(ValueError, match="x"):
            parse_case(entry)

    def test_audio_case_parses_settings_context_and_task(self):
        case = _audio_case("audio-sound", "stable_audio_3", prompt="rain", settings={"space": "forest"}, context={"sound_type": "ambience", "seconds": 12})
        assert (case.task, case.media, case.is_audio) == ("sound", "audio", True)
        assert case.settings.space == "forest"
        assert (case.audio_context.sound_type, case.audio_context.seconds) == ("ambience", 12)

    def test_a_lyrics_case_parses_the_lyrics_settings_and_the_style_context(self):
        case = _audio_case("audio-lyrics", "yue2", prompt="[Verse]\nrain\n\n[Bridge]\n", settings={"action": "write", "section": 1}, context={"style": "folk"})

        assert (case.task, case.media, case.is_audio) == ("lyrics", "audio", True)
        assert (case.settings.action, case.settings.section, case.audio_context.style) == ("write", 1, "folk")

    def test_the_shipped_lyrics_cases_cover_improve_a_section_and_write(self):
        cases = [case for case in load_cases() if case.mode == "audio-lyrics"]

        assert {"improve", "section", "write", "rhymes", "simpler", "fit", "language", "names", "empty", "repeat"} <= {tag for case in cases for tag in case.tags}
        assert {case.settings.action for case in cases} == {"improve", "write"}
        assert any(case.settings.section is None for case in cases) and any(case.settings.section is not None for case in cases)

    def test_a_script_case_parses_the_script_settings_and_takes_no_context(self):
        case = _audio_case("audio-script", "qwen3_tts", prompt="Hello there.", settings={"actions": ["fit"], "seconds": 20, "pace": "slow"})

        assert (case.task, case.media, case.is_audio) == ("script", "audio", True)
        assert (case.settings.actions, case.settings.seconds, case.settings.pace) == (("fit",), 20, "slow")

    def test_the_shipped_script_cases_cover_delivery_and_fit(self):
        cases = [case for case in load_cases() if case.mode == "audio-script"]

        assert {"delivery", "fit", "marks", "names", "quotes", "pace", "lengthen"} <= {tag for case in cases for tag in case.tags}
        assert {action for case in cases for action in case.settings.actions} == {"delivery", "fit"}
        assert any(case.settings.pace != "normal" for case in cases) and {case.family for case in cases} == {"qwen3_tts"}
        assert any("fit" in case.settings.actions and 1 < reading_characters(case.settings.seconds, case.settings.pace) / len(strip_marks(case.prompt)) for case in cases)  # a Fit that lengthens
        assert any("fit" in case.settings.actions and reading_characters(case.settings.seconds, case.settings.pace) / len(strip_marks(case.prompt)) < 0.8 for case in cases)  # and one that shortens

    def test_rejects_duplicate_ids(self, tmp_path):
        path = tmp_path / "cases.yaml"
        path.write_text("cases:\n  - {id: a, prompt: x}\n  - {id: a, prompt: y}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate"):
            load_cases(path)

    def test_select_by_tag_and_id(self):
        cases = (_case(id="a", tags=["x"]), _case(id="b", tags=["y"]))
        assert [case.id for case in select_cases(cases, tags={"y"})] == ["b"]
        assert [case.id for case in select_cases(cases, ids={"a"})] == ["a"]
        assert len(select_cases(cases)) == 2


class TestChecks:
    def test_length_check(self):
        plan = LengthPlan(length="same", target=20, lo=17, hi=23, clamped=False)
        assert length_check(20, plan) == (True, 1.0)
        assert length_check(30, plan) == (False, 1.5)
        assert length_check(5, None) == (None, None)

    def test_quotes_kept(self):
        source = 'a diner with a sign that says "OPEN 24 HOURS"'
        assert quotes_kept(source, 'A diner glows; its sign reads "OPEN  24 HOURS".') is True
        assert quotes_kept(source, 'A diner with a sign reading "Open 24 Hours".') is False
        assert quotes_kept("a fox in snow", "anything") is None

    def test_keep_share_allows_case_and_plurals(self):
        assert keep_share("Two Foxes rest in deep snow.", ("fox", "snow", "owl")) == pytest.approx(2 / 3, abs=0.001)
        assert keep_share("text", ()) is None
        assert keep_share("Two friends at a café table.", ("cafe",)) == 1.0

    def test_keep_matches_stems_alternatives_and_inserted_words(self):
        assert keep_share("A walk along the riverbank.", ("river",)) == 1.0
        assert keep_share("An elderly man reads.", ("old man|elderly man", "read")) == 1.0
        assert keep_share("A green desk lamp glows.", ("green lamp",)) == 1.0
        assert keep_share("A green field and a quiet room with a lamp.", ("green lamp",)) == 0.0

    def test_count_kept_as_word_or_digit(self):
        assert count_kept("Three cats sleep.", "three cats") is True
        assert count_kept("Three fluffy grey cats sleep.", "three cats") is True
        assert count_kept("3 cats sleep.", "three cats") is True
        assert count_kept("Two cats sleep.", "three cats") is False
        assert count_kept("x", None) is None

    def test_forbid_matches_whole_words(self):
        assert forbid_clear("A woman plays chess.", ("man",)) is True
        assert forbid_clear("A man plays chess.", ("man",)) is False
        assert forbid_clear("x", ()) is None

    def test_caption_format(self):
        elements = [{"type": "obj", "desc": f"thing {index}"} for index in range(3)]
        caption = {"high_level_description": "A fox.", "compositional_deconstruction": {"background": "snow", "elements": elements}}
        assert caption_format(json.dumps(caption)) is True
        caption["compositional_deconstruction"]["elements"] = [{"type": "text", "text": "LISBOA", "desc": "big letters"}]
        assert caption_format(json.dumps(caption)) is False
        assert caption_format("not json") is False

    def test_case_score(self):
        checks = {name: None for name in CHECK_NAMES} | {"usable": True, "first_try": False, "keep": 0.5}
        assert case_score(checks) == 0.5
        assert case_score({"usable": False}) == 0.0

    def test_length_plan_per_mode(self):
        assert length_plan(_case(), ceiling=300, length_cfg=None)[1] == "prompt"
        assert length_plan(_case(mode="caption"), ceiling=300, length_cfg=None) == (None, "caption")
        action = _case(mode="video", scene="A cat on a counter.", prompt="it jumps", context={"reference_image": True})
        assert length_plan(action, ceiling=300, length_cfg=None)[1] == "action"
        pair = _case(mode="video", prompt="a cat jumps off a counter")
        assert length_plan(pair, ceiling=300, length_cfg=None)[1] == "pair"


class TestAudioChecks:
    def test_shapes_follow_the_caption_rules(self):
        assert audio_shape_name(caption_rules("yue2")) == "descriptors"
        assert audio_shape_name(caption_rules("stable_audio_3")) == "sentence"
        assert audio_shape_name(caption_rules("unknown")) == "prose"
        assert audio_shape_name(caption_rules(None)) == "prose"

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("Jazz, piano, tenor saxophone, upright bass, warm", True),
            ("Jazz, piano", False),
            ("Jazz, piano, upright bass, warm.", False),
            ("Jazz, piano. Upright bass, warm", False),
            ("A warm jazz trio plays a slow song with piano and bass, mellow and relaxed throughout", False),
            ("Jazz, piano, [sad], warm", False),
            ("Jazz, piano, **bass**", False),
            ("Jazz, piano,\nbass, warm", False),
            ("", False),
        ],
    )
    def test_descriptor_format(self, output, expected):
        assert audio_format(_audio_case("audio-style", "yue2"), output) is expected

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("A slow jazz trio with soft piano and a warm upright bass.", True),
            ("A slow jazz trio with soft piano and a warm upright bass", True),
            ("A slow jazz trio plays. The bass is warm.", False),
            ("A slow jazz trio plays at 2.5 times the speed of a ballad.", True),
            ("A slow jazz trio [with piano].", False),
            ("- A slow jazz trio with piano.", False),
            ("A *slow* jazz trio with piano.", False),
            ("A slow jazz trio.\nWith piano.", False),
        ],
    )
    def test_sentence_format(self, output, expected):
        assert audio_format(_audio_case("audio-style", "stable_audio_3"), output) is expected

    def test_unknown_family_gets_two_sentences(self):
        case = _audio_case("audio-style", "other")
        assert audio_format(case, "A slow song. It is calm.") is True
        assert audio_format(case, "A slow song. It is calm. It ends.") is False

    def test_marks_in_the_source_are_not_brackets_to_ban(self):
        case = _audio_case("audio-sound", "stable_audio_3", prompt="a door [slam] in a hall")
        assert audio_format(case, "A door [slam] in a large hall.") is True
        assert audio_format(case, "A door [bang] in a large hall.") is False

    @pytest.mark.parametrize(
        ("kwargs", "output", "expected"),
        [
            (_SA3_CASE, "A bright pop tune with a female singer.", False),
            (_SA3_CASE, "Soft vocals float over a warm pad.", False),
            (_SA3_CASE, "A warm pad with a gentle pulse.", True),
            (_SA3_CASE, "A warm pad played entirely on strings without vocals or singing.", True),
            (_SA3_CASE, "A warm pad, no vocals, no singer.", True),
            (_SA3_CASE, "A warm pad that carries the melody without any vocals.", True),
            (_SA3_CASE, "A warm pad without vocals but with a female singer.", False),
            (_SA3_CASE, "A warm pad. No vocals. Soft vocals enter later.", False),
            ({"mode": "audio-style", "family": "yue2", "context": {"instrumental": True}}, "Lo-fi, soft guitar, instrumental, no vocals", True),
            (_SA3_CASE, "A warm pad at 120 BPM.", False),
            (_SA3_CASE, "A warm pad. TrackType: music.", False),
            (_SA3_CASE, "A warm pad with a length of 30 seconds.", False),
            (_SA3_CASE, "A mid tempo warm pad.", False),
            ({"mode": "audio-style", "family": "yue2"}, "Pop, female lead vocal, synth, bright", True),
            ({"mode": "audio-style", "family": "yue2"}, "Pop, synth, 120 BPM", False),
            ({"mode": "audio-style", "family": "yue2", "context": {"instrumental": True}}, "Pop, female lead vocal, synth", False),
            ({"mode": "audio-style", "family": "yue2", "context": {"instrumental": True}}, "Pop, synth, bright, no drums", True),
        ],
    )
    def test_banned_words_per_family(self, kwargs, output, expected):
        assert banned_clear(_audio_case(**kwargs), output) is expected

    def test_a_banned_word_from_the_users_own_text_is_allowed(self):
        case = _audio_case("audio-sound", "stable_audio_3", prompt="a choir singer rehearsing, the length of a hall")
        assert banned_clear(case, "A singer rehearses in a hall.") is True

    def test_marks_kept(self):
        assert marks_kept("[Verse 1] la [sad] da", "[sad]   then [Verse 1] la") is True
        assert marks_kept("[Verse 1] la [sad] da", "[Verse 1] la") is False
        assert marks_kept("no marks", "anything") is None

    def test_length_plan_is_the_rules_range(self):
        plan, measures = length_plan(_audio_case("audio-sound", "stable_audio_3"), ceiling=0, length_cfg=None)
        assert (measures, plan.hi) == ("audio", 45)
        plan, _ = length_plan(_audio_case("audio-style", "yue2"), ceiling=0, length_cfg=None)
        assert plan.hi < 45
        assert length_check(46, length_plan(_audio_case("audio-sound", "stable_audio_3"), ceiling=0, length_cfg=None)[0])[0] is False

    def test_every_axis_of_the_contract_is_accepted_by_a_case(self):
        for item in AUDIO_AXES:
            task = "sound" if item.kind == "sound" else "style"
            slug = item.axis.options[-1].slug
            value = [slug] if item.axis.multi else slug
            assert _audio_case(f"audio-{task}", "yue2", settings={item.axis.key: value}).settings


class TestRunner:
    def test_recording_enhancer_records_each_call(self):
        ticks = iter(range(100))
        recorder = RecordingEnhancer(_FakeEnhancer(["one two three"]), clock=lambda: float(next(ticks)))
        text = "".join(recorder.generate([{"role": "user", "content": "x"}], seed=1, max_tokens=10, temperature=0.5))
        assert text.split() == ["one", "two", "three"]
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["words"] == 3
        assert recorder.calls[0]["first_text_seconds"] is not None

    def test_recording_enhancer_leaves_a_default_call_alone(self):
        recorder = RecordingEnhancer(_FakeEnhancer(["one two"]))  # the fake's generate() takes no extra parameters, so none may be passed
        list(recorder.generate([], seed=1, max_tokens=100, temperature=0.5))
        assert [call["raw"] for call in recorder.calls] == ["one two "]

    def test_run_case_image_scores_output(self):
        recorder = RecordingEnhancer(_FakeEnhancer(["A red fox curls up in deep fresh snow under soft morning light, framed low and wide."]))
        record = run_case(recorder, _case(keep=["fox", "snow"]), 12345678, config=_CONFIG, options=_OPTIONS)
        assert record["error"] is None
        assert record["checks"]["usable"] is True
        assert record["checks"]["first_try"] is True
        assert record["checks"]["keep"] == 1.0
        assert record["system_prompt"]
        assert record["plan"]["measures"] == "prompt"

    def test_run_case_records_retry_and_failure(self):
        recorder = RecordingEnhancer(_FakeEnhancer(["a fox in snow"]))
        record = run_case(recorder, _case(), 12345678, config=_CONFIG, options=_OPTIONS)
        assert record["error"]
        assert record["checks"]["usable"] is False
        assert len(record["attempts"]) == 2

    def test_run_case_video_pair(self):
        reply = "SCENE: A busy kitchen with steel counters, warm light.\nACTION: The chef flips a pancake high and laughs as it lands."
        case = _case(mode="video", prompt="a chef flips a pancake and laughs", context={"audio": True, "frames": 97, "fps": 24}, keep=["chef", "kitchen"])
        record = run_case(RecordingEnhancer(_FakeEnhancer([reply])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        assert record["scene"].startswith("A busy kitchen")
        assert record["checks"]["format"] is True
        assert record["checks"]["keep"] == 1.0

    def test_run_case_audio_sound_scores_shape_and_banned_words(self):
        reply = "A heavy wooden door slams shut in a large stone hall and the echo fades slowly."
        case = _audio_case("audio-sound", "stable_audio_3", prompt="a heavy wooden door slamming shut", context={"sound_type": "effect", "seconds": 5}, keep=["wooden door", "slam"])
        record = run_case(RecordingEnhancer(_FakeEnhancer([reply])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        assert record["error"] is None
        checks = record["checks"]
        assert (checks["format"], checks["length_in_range"], checks["banned"], checks["keep"]) == (True, True, True, 1.0)
        assert checks["marks_kept"] is None
        assert record["plan"]["measures"] == "audio"
        assert "sound effect" in record["system_prompt"]

    def test_run_case_audio_style_flags_a_sentence_and_vocals(self):
        reply = "A warm jazz band plays a slow song with a female vocalist and soft piano."
        case = _audio_case("audio-style", "yue2", prompt="slow jazz", context={"instrumental": True})
        checks = run_case(RecordingEnhancer(_FakeEnhancer([reply])), case, 12345678, config=_CONFIG, options=_OPTIONS)["checks"]
        assert checks["format"] is False
        assert checks["banned"] is False

    def test_run_case_audio_keeps_quoted_names(self):
        case = _audio_case("audio-sound", "stable_audio_3", prompt='the bell of the "Santa Maria" ringing', keep=["bell"])
        kept = run_case(RecordingEnhancer(_FakeEnhancer(['The bell of the "Santa Maria" rings out over a calm harbour.'])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        assert kept["checks"]["quotes_kept"] is True
        lost = run_case(RecordingEnhancer(_FakeEnhancer(["The bell of a ship rings out over a calm harbour."])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        assert lost["checks"]["quotes_kept"] is False

    def test_run_bench_writes_results_and_meta(self, tmp_path):
        fake = _FakeEnhancer(["A red fox curls up in deep fresh snow under soft morning light."])
        out = tmp_path / "base"
        path = run_bench(
            (_case(),), factory=lambda repo, revision: fake, repo="fake/model", revision=None, model_choice="fast", config=_CONFIG, options=_OPTIONS, seeds=2, out_dir=out, echo=lambda _line: None
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert {json.loads(line)["seed"] for line in lines} == set(case_seeds("c1", 2))
        assert json.loads((out / META_FILE).read_text(encoding="utf-8"))["repo"] == "fake/model"
        assert fake.closed
        with pytest.raises(FileExistsError):
            run_bench((_case(),), factory=lambda repo, revision: fake, repo="r", revision=None, model_choice="fast", config=_CONFIG, options=_OPTIONS, seeds=1, out_dir=out, echo=lambda _line: None)


def _write_run(run_dir: Path, reply: str) -> None:
    run_bench(
        (_case(keep=["fox"]),),
        factory=lambda repo, revision: _FakeEnhancer([reply]),
        repo="fake/model",
        revision=None,
        model_choice="fast",
        config=_CONFIG,
        options=_OPTIONS,
        seeds=1,
        out_dir=run_dir,
        echo=lambda _line: None,
    )


class TestRescore:
    def test_rescore_uses_current_expectations(self, tmp_path):
        _write_run(tmp_path / "run", "A red fox curls up in deep fresh snow.")
        assert load_run(tmp_path / "run").records[0]["checks"]["keep"] == 1.0
        count = rescore_run(tmp_path / "run", (_case(keep=["owl"]),), config=_CONFIG, options=_OPTIONS)
        assert count == 1
        record = load_run(tmp_path / "run").records[0]
        assert record["checks"]["keep"] == 0.0
        assert record["output"] == "A red fox curls up in deep fresh snow."

    def test_rescore_recomputes_audio_shape(self, tmp_path):
        case = _audio_case("audio-style", "yue2", prompt="slow jazz")
        run_bench(
            (case,),
            factory=lambda repo, revision: _FakeEnhancer(["Jazz, piano, upright bass, warm"]),
            repo="fake/model",
            revision=None,
            model_choice="fast",
            config=_CONFIG,
            options=_OPTIONS,
            seeds=1,
            out_dir=tmp_path / "run",
            echo=lambda _line: None,
        )
        assert load_run(tmp_path / "run").records[0]["checks"]["format"] is True
        assert rescore_run(tmp_path / "run", (_audio_case("audio-style", "stable_audio_3", prompt="slow jazz"),), config=_CONFIG, options=_OPTIONS) == 1
        record = load_run(tmp_path / "run").records[0]
        assert record["checks"]["format"] is True  # one clause, no sentence break: still one sentence
        assert record["plan"]["measures"] == "audio"

    def test_rescore_keeps_results_of_removed_cases(self, tmp_path):
        _write_run(tmp_path / "run", "A red fox curls up in deep fresh snow.")
        assert rescore_run(tmp_path / "run", (), config=_CONFIG, options=_OPTIONS) == 0
        assert len(load_run(tmp_path / "run").records) == 1


class TestCompare:
    def test_compare_reports_checks_and_changed_cases(self, tmp_path):
        _write_run(tmp_path / "base", "A red fox curls up in deep fresh snow.")
        _write_run(tmp_path / "new", "A quiet white field under falling snow at dawn.")
        base, new = load_run(tmp_path / "base"), load_run(tmp_path / "new")
        assert check_rates(base.records)["keep"] == (1.0, 1)
        assert check_rates(new.records)["keep"] == (0.0, 1)
        assert "c1" in compare_runs(base, new)

    def test_cli_compare_writes_report(self, tmp_path):
        _write_run(tmp_path / "base", "A red fox curls up in deep fresh snow.")
        _write_run(tmp_path / "new", "A red fox sleeps in deep fresh snow.")
        assert bench_cli.main(["compare", "base", "new", "--runs-dir", str(tmp_path)]) == 0
        assert (tmp_path / "compare-base-new.md").exists()

    def test_load_run_requires_results(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=RESULTS_FILE):
            load_run(tmp_path)


class TestSheet:
    def _run(self, tmp_path, label: str, reply: str) -> Run:
        _write_run(tmp_path / label, reply)
        return load_run(tmp_path / label)

    def test_rows_hold_the_input_and_each_runs_first_rewrite(self, tmp_path):
        base, new = self._run(tmp_path, "base", "A red fox in snow."), self._run(tmp_path, "new", "A red fox curls up in deep snow.")
        rows = sheet_rows((base, new), ("c1", "missing"))
        assert [row.case for row in rows] == ["c1"]
        assert [(cell.column, cell.prompt) for cell in rows[0].cells] == [("Your prompt", "a fox in snow"), ("base", "A red fox in snow."), ("new", "A red fox curls up in deep snow.")]
        assert len({cell.name for cell in rows[0].cells}) == 3

    def test_prompts_file_has_one_set_per_cell(self, tmp_path):
        rows = sheet_rows((self._run(tmp_path, "base", "A red fox in snow."),), ("c1",))
        data = yaml.safe_load(prompts_file_text(rows))
        assert data == {cell.name: [{"prompt": cell.prompt}] for cell in rows[0].cells}

    def test_find_image_by_set_name(self, tmp_path):
        (tmp_path / "s002_2026-10-19_12-00-00.png").write_bytes(b"")
        assert find_image(tmp_path, "s002").name.startswith("s002_")
        assert find_image(tmp_path, "s001") is None

    def test_render_command_fixes_model_seed_and_turns_enhancement_off(self, tmp_path):
        command = render_command(tmp_path / "p.yaml", tmp_path, model="zit", seed=73914826, ratio="2:3", size="s")
        assert command[command.index("-m") + 1] == "zit" and command[command.index("--seed") + 1] == "73914826"
        assert "--no-enhance" in command and "-q" not in command

    def test_render_command_passes_the_quantize_level(self, tmp_path):
        command = render_command(tmp_path / "p.yaml", tmp_path, model="krea2", seed=1, ratio="2:3", size="s", quantize=8)
        assert command[command.index("-q") + 1] == "8"


_SONG = "[Verse]\nThe porch light flickers\nI keep your coat\n\n[Chorus]\nCome back home\n\n[Bridge]\n\n[Chorus]\nCome back home\n"


def _lyrics_case(**overrides):
    return _audio_case("audio-lyrics", "yue2", prompt=_SONG, **overrides)


class TestLyricsChecks:
    def test_the_tags_must_stay_in_order_each_on_its_own_line(self):
        case = _lyrics_case()

        assert tags_kept(case, _SONG.replace("flickers", "glows")) is True
        assert tags_kept(case, _SONG.replace("[Bridge]\n\n", "")) is False
        assert tags_kept(case, _SONG.replace("[Verse]\n", "[Verse] ")) is False
        assert tags_kept(case, _SONG.replace("[Bridge]", "[Refrain]")) is False
        assert tags_kept(case, "[Chorus]\nCome back home\n\n[Verse]\nx\n\n[Bridge]\n\n[Chorus]\nx") is False

    def test_a_whole_song_improvement_keeps_empty_sections_empty_and_repeats_repeated(self):
        case = _lyrics_case()
        assert scope_kept(case, _SONG.replace("Come back home", "Come home")) is True  # both choruses changed alike
        assert scope_kept(case, _SONG.replace("[Bridge]\n", "[Bridge]\nan invented bridge\n")) is False
        assert scope_kept(case, _SONG.replace("[Verse]\nThe porch light flickers\nI keep your coat\n", "[Verse]\n")) is False
        assert scope_kept(case, _SONG.replace("[Chorus]\nCome back home\n\n[Bridge]", "[Chorus]\nCome home\n\n[Bridge]")) is False  # one chorus no longer a repeat
        assert scope_kept(case, _SONG.replace("[Bridge]\n\n", "")) is False

    def test_a_section_improvement_changes_only_its_section(self):
        case = _lyrics_case(settings={"section": 0, "improve": ["tighten"]})

        assert scope_kept(case, _SONG.replace("I keep your coat", "Your coat")) is True
        assert scope_kept(case, _SONG.replace("I keep your coat", "Your coat").replace("Come back home", "Come home")) is False
        assert scope_kept(case, _SONG.replace("[Verse]\nThe porch light flickers\nI keep your coat\n", "[Verse]\n")) is False

    def test_a_write_fills_the_empty_section_and_no_other(self):
        case = _lyrics_case(settings={"action": "write", "section": 2})

        assert scope_kept(case, _SONG.replace("[Bridge]\n", "[Bridge]\nthe night is long\n")) is True
        assert scope_kept(case, _SONG) is False
        assert scope_kept(case, _SONG.replace("[Bridge]\n", "[Bridge]\nthe night is long\n").replace("flickers", "glows")) is False

    @pytest.mark.parametrize(
        ("output", "expected"), [("[Verse]\nplain words", True), ("", False), ("**[Verse]**\nwords", False), ("[Verse]\n- a list\n- of lines", False), ("```\n[Verse]\nx\n```", False)]
    )
    def test_lyrics_are_plain_text(self, output, expected):
        assert lyrics_format(output) is expected

    def test_the_new_checks_are_reported_and_only_for_lyrics(self):
        assert {"tags_kept", "scope_kept"} <= set(CHECK_NAMES)
        sound = run_case(
            RecordingEnhancer(_FakeEnhancer(["A heavy door slams shut in a large hall."])),
            _audio_case("audio-sound", "stable_audio_3", prompt="a heavy door slamming"),
            12345678,
            config=_CONFIG,
            options=_OPTIONS,
        )

        assert sound["checks"]["tags_kept"] is None and sound["checks"]["scope_kept"] is None


class TestLyricsRun:
    def test_a_run_scores_the_structure_of_the_lyrics(self):
        reply = "[Verse]\nPorch light flickers\nI keep your coat\n\n[Chorus]\nCome home\n\n[Bridge]\n\n[Chorus]\nCome home\n"
        case = _lyrics_case(settings={"improve": ["tighten"]}, context={"style": "folk"}, keep=["coat", "porch"])

        record = run_case(RecordingEnhancer(_FakeEnhancer([reply])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["error"] is None and record["output"] == reply
        checks = record["checks"]
        assert (checks["usable"], checks["first_try"], checks["format"], checks["tags_kept"], checks["scope_kept"], checks["marks_kept"]) == (True, True, True, True, True, True)
        assert checks["banned"] is None and checks["length_in_range"] is None and checks["keep"] == 1.0 and record["plan"] is None
        assert "Tighten" in record["system_prompt"] and "folk" in record["system_prompt"] and record["user_message"].startswith("Lyrics:")

    def test_a_write_run_fills_only_the_target(self):
        case = _lyrics_case(settings={"action": "write", "section": 2})
        reply = "the night is long and I am still here"

        record = run_case(RecordingEnhancer(_FakeEnhancer([reply])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["output"] == _SONG.replace("[Bridge]\n", "[Bridge]\nthe night is long and I am still here\n")
        assert (record["checks"]["tags_kept"], record["checks"]["scope_kept"]) == (True, True)

    def test_changed_sections_are_a_failed_run_after_the_retry(self):
        record = run_case(RecordingEnhancer(_FakeEnhancer(["[Verse]\nonly this"])), _lyrics_case(), 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["error"] == SECTIONS_CHANGED_MESSAGE and record["checks"]["usable"] is False and len(record["attempts"]) == 2

    def test_a_changed_quote_is_scored_on_double_quotes_only(self):
        case = _audio_case("audio-lyrics", "yue2", prompt="[Verse]\nI'm goin' down, 'cause you're gone\nShe said \"hold on, Marie\" twice")

        kept = run_case(RecordingEnhancer(_FakeEnhancer(['[Verse]\nI\'m going down, you are gone\nShe said "hold on, Marie" two times'])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        lost = run_case(RecordingEnhancer(_FakeEnhancer(["[Verse]\nI'm going down, you are gone\nShe said wait, Marie, two times"])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert kept["checks"]["quotes_kept"] is True and lost["checks"]["quotes_kept"] is False and lost["warnings"]

    def test_rescoring_a_stored_lyrics_run_recomputes_the_structure_checks(self, tmp_path):
        reply = "[Verse]\nPorch light flickers\nI keep your coat\n\n[Chorus]\nCome home\n\n[Bridge]\n\n[Chorus]\nCome home\n"
        case = _lyrics_case()
        record = run_case(RecordingEnhancer(_FakeEnhancer([reply])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / RESULTS_FILE).write_text(json.dumps(record) + "\n", encoding="utf-8")

        assert rescore_run(run_dir, (_lyrics_case(settings={"action": "write", "section": 2}),), config=_CONFIG, options=_OPTIONS) == 1

        rescored = json.loads((run_dir / RESULTS_FILE).read_text(encoding="utf-8"))
        assert rescored["checks"]["scope_kept"] is False and rescored["checks"]["tags_kept"] is True


_SCRIPT = "I never told anyone. Not even you. The door was open."
_MARKED = "[whisper] I never told anyone. …\nNot even you. [as set] The door was open."


def _script_case(**overrides):
    return _audio_case("audio-script", "qwen3_tts", **{"prompt": _SCRIPT, "settings": {"actions": ["delivery"]}, **overrides})


class TestScriptChecks:
    def test_the_words_of_a_script_are_what_is_left_without_marks_and_pauses(self):
        assert script_words(_MARKED) == script_words(_SCRIPT) == _SCRIPT
        assert script_words("[Sad] Hi… [sadd] there") == "Hi [sadd] there"

    def test_add_delivery_must_leave_every_word_as_it_was(self):
        case = _script_case()

        assert words_kept(case, _MARKED) is True
        assert words_kept(case, "[sad] I never told anybody. Not even you. The door was open.") is False
        assert words_kept(case, "I never told anyone, not even you. The door was open.") is False
        assert words_kept(case, "I never told anyone. Not even you.") is False

    def test_a_fit_has_no_words_check(self):
        assert words_kept(_script_case(settings={"actions": ["fit"], "seconds": 5}), "Anything.") is None

    def test_a_bracketed_word_that_is_not_a_mark_is_not_known(self):
        assert marks_known(_MARKED) is True
        assert marks_known("[sad] Hello [sighs] there") is False
        assert marks_known("Press [1] now") is True

    def test_a_fit_is_scored_by_the_share_of_its_time_it_reads_for(self):
        case = _script_case(settings={"actions": ["fit"], "seconds": 20})

        assert fit_ratio(case, "x" * 280) == 1.0
        assert fit_ratio(case, "[sad] " + "x" * 252 + " [as set]") == round((252 + 2) / 280, 3)  # the marks are not counted, the spaces around them are
        assert fit_ratio(case, "x" * 322) == 1.15 and fit_ratio(case, "x" * 140) == 0.5
        assert fit_ratio(_script_case(), "x") is None

    def test_a_fit_counts_at_the_pace(self):
        case = _script_case(settings={"actions": ["fit"], "seconds": 20, "pace": "slow"})

        assert fit_ratio(case, "x" * round(20 * 14 * 0.81)) == 1.0

    @pytest.mark.parametrize(("output", "expected"), [("[sad] plain words…", True), ("", False), ("**bold**", False), ("- a list\n- of lines", False)])
    def test_a_script_is_plain_text(self, output, expected):
        assert script_format(output) is expected


class TestScriptRun:
    def test_add_delivery_is_scored_on_its_words_and_marks(self):
        record = run_case(RecordingEnhancer(_FakeEnhancer([_MARKED])), _script_case(keep=["door"]), 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["error"] is None and record["output"] == _MARKED
        checks = record["checks"]
        assert (checks["usable"], checks["first_try"], checks["format"], checks["words_kept"], checks["marks_known"], checks["marks_kept"]) == (True, True, True, True, True, None)
        assert checks["fit_in_range"] is None and checks["length_ratio"] is None and checks["banned"] is None and checks["length_in_range"] is None and checks["keep"] == 1.0
        assert record["plan"] is None and "Never change the words" in record["system_prompt"] and record["user_message"].startswith("Script:")

    def test_a_mark_the_script_had_survives_an_add_delivery_whatever_the_model_writes(self):
        case = _audio_case("audio-script", "qwen3_tts", prompt="[sad] I never told anyone.", settings={"actions": ["delivery"]})

        kept = run_case(RecordingEnhancer(_FakeEnhancer(["[sad] I never told anyone. …"])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        swapped = run_case(RecordingEnhancer(_FakeEnhancer(["[whisper] I never told anyone. …"])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert kept["checks"]["marks_kept"] is True and swapped["checks"]["marks_kept"] is True
        assert swapped["output"] == "[sad] I never told anyone. …"

    def test_a_fit_is_scored_on_how_close_it_lands_and_not_on_its_words(self):
        case = _script_case(prompt="The harbour was quiet that morning. " * 20, settings={"actions": ["fit"], "seconds": 20})

        near = run_case(RecordingEnhancer(_FakeEnhancer(["y" * 300])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        far = run_case(RecordingEnhancer(_FakeEnhancer(["y" * 400, "y" * 400])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert (near["checks"]["fit_in_range"], near["checks"]["length_ratio"], near["checks"]["words_kept"]) == (True, 1.071, None)
        assert (far["checks"]["fit_in_range"], far["checks"]["length_ratio"]) == (False, 1.429) and far["warnings"]

    def test_an_answer_that_adds_nothing_is_a_failed_run_after_the_retry(self):
        record = run_case(RecordingEnhancer(_FakeEnhancer([_SCRIPT, _SCRIPT])), _script_case(), 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["error"] == NOTHING_TO_ADD_MESSAGE and record["checks"]["usable"] is False and len(record["attempts"]) == 2

    def test_words_the_model_changed_come_back_as_the_scripts_own(self):
        record = run_case(RecordingEnhancer(_FakeEnhancer(["[sad] Nobody was told."])), _script_case(), 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["checks"]["words_kept"] is True and record["output"] == f"[sad] {_SCRIPT}"

    def test_the_new_checks_are_reported_and_only_for_scripts(self):
        assert {"words_kept", "marks_known", "fit_in_range"} <= set(CHECK_NAMES)
        sound = run_case(
            RecordingEnhancer(_FakeEnhancer(["A heavy door slams shut in a large hall."])),
            _audio_case("audio-sound", "stable_audio_3", prompt="a heavy door slamming"),
            12345678,
            config=_CONFIG,
            options=_OPTIONS,
        )

        assert (sound["checks"]["words_kept"], sound["checks"]["marks_known"], sound["checks"]["fit_in_range"]) == (None, None, None)

    def test_a_changed_quote_in_a_script_is_scored(self):
        case = _audio_case("audio-script", "qwen3_tts", prompt='She said "Meet me at Pier 9." Then she left.', settings={"actions": ["delivery"]})

        kept = run_case(RecordingEnhancer(_FakeEnhancer(['She said "Meet me at Pier 9." … Then she left.'])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert kept["checks"]["quotes_kept"] is True


_FILL_ANSWER = "STYLE: folk, acoustic guitar, wistful vocal\nTEMPO: 84\nKEY: A minor\nLYRICS:\n[Verse]\nI pack the boat at dawn\n\n[Chorus]\nGoodbye to the coast"


def _fill_case(**overrides):
    return _audio_case("audio-fill", "yue2", **{"prompt": "a sad folk song about leaving the coast", "settings": {"fields": ["style", "tempo", "key", "lyrics"]}, **overrides})


class TestFillCases:
    def test_a_fill_case_parses_the_fields_the_kind_and_what_is_already_set(self):
        case = _fill_case(settings={"fields": ["tempo", "lyrics"]}, context={"style": "folk", "tempo": 90})

        assert (case.task, case.media, case.is_audio) == ("fill", "audio", True)
        assert (case.settings.kind, case.settings.fields, case.audio_context.style) == ("music", ("tempo", "lyrics"), "folk")
        assert _audio_case("audio-fill", "qwen3_tts", prompt="a secret", settings={"fields": ["vibe", "text"]}).settings.kind == "speech"

    @pytest.mark.parametrize(
        "entry",
        [
            {"prompt": "a song", "settings": {}},
            {"prompt": "a song", "settings": {"fields": ["mood"]}},
            {"prompt": "  ", "settings": {"fields": ["style"]}},
            {"prompt": "a song", "settings": {"fields": ["style", "text"]}},
            {"prompt": "a song", "settings": {"fields": ["style"]}, "context": {"describe": "x"}},
        ],
    )
    def test_a_malformed_fill_case_fails_to_load(self, entry):
        with pytest.raises(ValueError, match="a1"):
            _audio_case("audio-fill", "yue2", **entry)

    def test_the_shipped_fill_cases_cover_both_kinds_all_three_families_and_the_partial_fills(self):
        cases = [case for case in load_cases() if case.mode == "audio-fill"]

        assert {case.settings.kind for case in cases} == {"music", "speech"} and {case.family for case in cases} == {"yue2", "stable_audio_3", "qwen3_tts"}
        assert {"names", "quotes", "instrumental", "partial"} <= {tag for case in cases for tag in case.tags}
        assert all(case.audio_context.style or case.audio_context.lyrics or case.audio_context.text for case in cases if "partial" in case.tags and case.id != "qwen3-fill-keeps-vibe-text-empty")


class TestFillChecks:
    def test_the_checks_are_reported_and_only_for_fills(self):
        assert {"fields_filled", "only_empty", "values_valid", "fill_tags"} <= set(CHECK_NAMES)
        sound = run_case(RecordingEnhancer(_FakeEnhancer(["A heavy door slams shut."])), _audio_case("audio-sound", "stable_audio_3", prompt="a door"), 12345678, config=_CONFIG, options=_OPTIONS)

        assert (sound["checks"]["fields_filled"], sound["checks"]["only_empty"], sound["checks"]["values_valid"], sound["checks"]["fill_tags"]) == (None, None, None, None)

    def test_every_asked_field_filled_scores_one(self):
        record = run_case(RecordingEnhancer(_FakeEnhancer([_FILL_ANSWER])), _fill_case(keep=["coast"]), 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["error"] is None and record["plan"] is None
        checks = record["checks"]
        assert (checks["usable"], checks["first_try"], checks["format"], checks["fields_filled"], checks["only_empty"], checks["values_valid"], checks["fill_tags"]) == (
            True,
            True,
            True,
            1.0,
            True,
            1.0,
            True,
        )
        assert (checks["banned"], checks["keep"], checks["length_in_range"], checks["marks_kept"]) == (True, 1.0, None, None)

    def test_a_field_left_empty_after_the_retry_lowers_the_share(self):
        answer = "STYLE: folk\nTEMPO: fast\nKEY: A minor"
        case = _fill_case(settings={"fields": ["style", "tempo", "key"]})

        record = run_case(RecordingEnhancer(_FakeEnhancer([answer, answer])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["checks"]["fields_filled"] == pytest.approx(0.667, abs=0.001) and record["checks"]["first_try"] is False
        assert record["checks"]["values_valid"] == 0.5 and record["warnings"]

    def test_only_the_first_answer_decides_what_was_valid(self):
        case = _fill_case(settings={"fields": ["tempo", "key"]})

        record = run_case(RecordingEnhancer(_FakeEnhancer(["TEMPO: 900\nKEY: H", "TEMPO: 90\nKEY: A minor"])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert record["checks"]["fields_filled"] == 1.0 and record["checks"]["values_valid"] == 0.0 and record["checks"]["first_try"] is False

    def test_a_field_the_user_already_set_must_not_be_written_again(self):
        case = _fill_case(settings={"fields": ["tempo", "key"]}, context={"style": "folk"})

        assert only_empty(case, "TEMPO: 90\nKEY: Am") is True
        assert only_empty(case, "STYLE: folk\nTEMPO: 90\nKEY: Am") is False

    def test_an_instrumental_is_filled_by_lyrics_none_and_words_are_not_filling_it(self):
        case = _fill_case(tags=["audio", "instrumental"], settings={"fields": ["style", "lyrics"]})

        assert fields_filled(case, "STYLE: piano\nLYRICS: none") == 1.0
        assert fields_filled(case, "STYLE: piano\nLYRICS:\n[Verse]\nla la") == 0.5
        assert fill_tags(case, "LYRICS: none") is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("LYRICS:\n[Verse]\nrain\n\n[Chorus]\nhome", True),
            ("LYRICS:\n[Verse] rain on my window\n[Chorus]\nhome", False),
            ("LYRICS:\nrain\n[Verse]\nhome", False),
            ("LYRICS:\njust words", False),
            ("STYLE: folk", False),
        ],
    )
    def test_lyrics_tags_are_alone_on_their_line_and_first(self, raw, expected):
        assert fill_tags(_fill_case(), raw) is expected

    def test_values_valid_is_not_measured_when_none_is_asked(self):
        assert values_valid(_fill_case(settings={"fields": ["style", "lyrics"]}), "STYLE: folk") is None
        assert values_valid(_audio_case("audio-fill", "qwen3_tts", prompt="a secret", settings={"fields": ["vibe", "text"]}), "VIBE: storyteller\nTEXT: hi") == 1.0

    def test_a_style_for_a_model_that_cannot_sing_is_banned_from_naming_a_voice_or_a_tempo(self):
        case = _audio_case("audio-fill", "stable_audio_3", prompt="calm piano", settings={"fields": ["style", "tempo", "key"]})

        good = run_case(RecordingEnhancer(_FakeEnhancer(["STYLE: Calm solo piano with soft pads.\nTEMPO: 70\nKEY: C"])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        bad = run_case(RecordingEnhancer(_FakeEnhancer(["STYLE: Calm piano at 70 BPM with a warm vocal.\nTEMPO: 70\nKEY: C"])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert good["checks"]["banned"] is True and bad["checks"]["banned"] is False

    def test_a_quoted_name_in_the_idea_is_scored_in_the_fields(self):
        case = _fill_case(prompt='a rock anthem whose chorus is "We Never Sleep"', keep=["We Never Sleep"])
        answer = "STYLE: rock\nTEMPO: 130\nKEY: E minor\nLYRICS:\n[Chorus]\nWe Never Sleep"

        kept = run_case(RecordingEnhancer(_FakeEnhancer([answer])), case, 12345678, config=_CONFIG, options=_OPTIONS)

        assert kept["checks"]["quotes_kept"] is True and kept["checks"]["keep"] == 1.0

    def test_rescoring_a_stored_fill_run_recomputes_the_fill_checks(self, tmp_path):
        case = _fill_case()
        record = run_case(RecordingEnhancer(_FakeEnhancer([_FILL_ANSWER])), case, 12345678, config=_CONFIG, options=_OPTIONS)
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / RESULTS_FILE).write_text(json.dumps({**record, "checks": {}}) + "\n", encoding="utf-8")

        assert rescore_run(run_dir, (case,), config=_CONFIG, options=_OPTIONS) == 1
        rescored = json.loads((run_dir / RESULTS_FILE).read_text(encoding="utf-8"))
        assert rescored["checks"] == record["checks"]
