"""Tests for the audio enhancer: option axes, caption rules per model family, the chat messages, the output cleaner and the run."""

from __future__ import annotations

import warnings

import pytest

from zvisiongenerator.core.audio_types import AudioGenerationRequest
from zvisiongenerator.utils import prompt_enhance_audio as audio_module
from zvisiongenerator.utils.prompt_enhance import NO_USABLE_MESSAGE, QUOTES_CHANGED_WARNING
from zvisiongenerator.utils.prompt_enhance_audio import (
    AUDIO_AXES,
    LYRICS_CONTEXT_CHARS,
    AudioEnhanceContext,
    AudioEnhanceSettings,
    CaptionRules,
    audio_enhance_context,
    audio_matrix_contract,
    audio_settings_from_mapping,
    audio_settings_from_spec,
    audio_settings_summary,
    audio_settings_to_mapping,
    build_audio_messages,
    caption_rules,
    caption_task,
    clean_audio_output,
    enhance_audio_caption,
    without_phrase,
)


class _FakeEnhancer:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.seeds: list[int] = []
        self.max_tokens: list[int] = []
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        self.seeds.append(seed)
        self.max_tokens.append(max_tokens)
        self.messages.append(messages)
        yield self.replies[len(self.seeds) - 1]


def _style(text: str = "sad folk", family: str | None = "yue2", **settings) -> str:
    """Return the system message of a style rewrite."""
    return _messages(text, "style", family, **settings)[0]["content"]


def _messages(text: str, task: str, family: str | None, *, context: AudioEnhanceContext | None = None, **settings) -> list[dict[str, str]]:
    return build_audio_messages(text, audio_settings_from_mapping(settings, task=task), task=task, rules=caption_rules(family), context=context or AudioEnhanceContext())


def _rules(system: str) -> list[str]:
    return [line.removeprefix("- ") for line in system.splitlines() if line.startswith("- ")]


class TestCommandLineSettings:
    def test_a_spec_sets_the_axes_with_plus_between_several_and_the_note_comes_apart(self):
        settings = audio_settings_from_spec("genre=folk, instruments=piano+violin,without=drums", task="style", note="  slow   and sparse ")

        assert (settings.genre, settings.instruments, settings.without, settings.note) == ("folk", ("piano", "violin"), ("drums",), "slow and sparse")

    def test_no_spec_is_the_defaults_and_a_note_alone_is_a_direction(self):
        assert audio_settings_from_spec(None, task="sound") == AudioEnhanceSettings()
        assert audio_settings_from_spec("", task="sound", note="at night") == AudioEnhanceSettings(note="at night")

    @pytest.mark.parametrize(
        ("spec", "task", "message"),
        [
            ("genre", "style", "Use key=value"),
            ("note=slow", "style", "--enhance-note"),
            ("genre=nope", "style", "Unknown genre"),
            ("genre=folk", "sound", "only applies to music"),
            ("instruments=piano,without=piano", "style", "both list: piano"),
        ],
    )
    def test_a_wrong_spec_is_refused_in_words_the_user_sees(self, spec, task, message):
        with pytest.raises(ValueError, match=message):
            audio_settings_from_spec(spec, task=task)

    def test_music_and_sound_are_rewritten_and_speech_is_not(self):
        assert (caption_task("music"), caption_task("sound"), caption_task("speech")) == ("style", "sound", None)


class TestContextOfARequest:
    def test_a_sound_is_told_its_type_and_its_length(self):
        request = AudioGenerationRequest(preset="sa3", kind="sound", prompt="rain", seconds=5.0, sound_type="effect")

        assert audio_enhance_context(request, task="sound", sings=False) == AudioEnhanceContext(sound_type="effect", seconds=5.0)

    def test_a_sung_song_is_told_the_start_of_its_lyrics_and_its_tempo_and_key(self):
        request = AudioGenerationRequest(preset="yue2", kind="music", prompt="folk", lyrics="la " * LYRICS_CONTEXT_CHARS, tempo=96, key="Am")

        context = audio_enhance_context(request, task="style", sings=True)

        assert (context.instrumental, context.tempo, context.key) == (False, 96, "Am") and len(context.lyrics) == LYRICS_CONTEXT_CHARS

    def test_an_instrumental_is_told_so_and_not_its_lyrics(self):
        request = AudioGenerationRequest(preset="yue2", kind="music", prompt="folk", lyrics="[verse]\nwords")

        assert audio_enhance_context(request, task="style", sings=False) == AudioEnhanceContext(instrumental=True)


class TestAxes:
    def test_the_keys_are_unique_and_the_ones_the_contract_names(self):
        keys = [item.axis.key for item in AUDIO_AXES]

        assert keys == ["genre", "music_mood", "instruments", "without", "singer", "production", "space", "distance", "over_time", "movement"]
        assert len(set(keys)) == len(keys)

    def test_the_axes_of_music_and_sound_belong_to_their_kind(self):
        kinds = {item.axis.key: item.kind for item in AUDIO_AXES}

        assert {key for key, kind in kinds.items() if kind == "music"} == {"genre", "music_mood", "instruments", "without", "singer", "production"}
        assert {key for key, kind in kinds.items() if kind == "sound"} == {"space", "distance", "over_time", "movement"}

    def test_genre_is_one_choice_in_eight_groups_and_instruments_take_many(self):
        by_key = {item.axis.key: item.axis for item in AUDIO_AXES}

        assert not by_key["genre"].multi and by_key["genre"].default == ("keep",)
        assert len({option.group for option in by_key["genre"].options if option.group}) == 8
        assert by_key["instruments"].multi and by_key["without"].multi and by_key["instruments"].default == ()
        assert [option.slug for option in by_key["singer"].options] == ["keep", "female", "male", "duet", "choir"]
        assert [option.slug for option in by_key["production"].options][1:] == ["polished", "lo-fi", "live", "vintage", "cinematic"]
        assert [option.slug for option in by_key["space"].options][1:] == ["small-room", "large-hall", "street", "open-country", "forest", "underwater"]
        assert [option.slug for option in by_key["distance"].options][1:] == ["close", "medium", "far"]
        assert [option.slug for option in by_key["over_time"].options][1:] == ["steady", "builds", "hit-then-tail"]
        assert [option.slug for option in by_key["movement"].options][1:] == ["still", "passing", "approaching", "receding"]

    def test_the_contract_serves_every_axis_with_its_kind_and_the_defaults(self):
        contract = audio_matrix_contract()

        assert [axis["key"] for axis in contract["axes"]] == [item.axis.key for item in AUDIO_AXES]
        singer = next(axis for axis in contract["axes"] if axis["key"] == "singer")
        assert (singer["kind"], singer["vocal"], singer["multi"], singer["default"]) == ("music", True, False, ["keep"])
        assert {"slug", "label", "description", "group"} == set(singer["options"][1])
        assert all(axis["vocal"] is False for axis in contract["axes"] if axis["key"] != "singer")
        assert contract["defaults"] == {
            "genre": "keep",
            "music_mood": "keep",
            "instruments": [],
            "without": [],
            "singer": "keep",
            "production": "keep",
            "space": "keep",
            "distance": "keep",
            "over_time": "keep",
            "movement": "keep",
            "note": "",
        }


class TestSettings:
    def test_settings_start_at_the_defaults_and_parse_every_kind_of_value(self):
        settings = audio_settings_from_mapping({"genre": "folk", "instruments": "piano+violin", "without": ["drums"], "note": "  slow   and sparse "}, task="style")

        assert (settings.genre, settings.instruments, settings.without, settings.note) == ("folk", ("piano", "violin"), ("drums",), "slow and sparse")
        assert audio_settings_from_mapping({}, task="sound") == AudioEnhanceSettings()

    @pytest.mark.parametrize(
        ("data", "task", "message"),
        [
            ({"genre": "nope"}, "style", "Unknown genre"),
            ({"colour": "red"}, "style", "Unknown enhancement setting"),
            ({"genre": "folk"}, "sound", "only applies to music"),
            ({"space": "forest"}, "style", "only applies to sound"),
            ({"instruments": ["piano"], "without": ["piano"]}, "style", "both list: piano"),
            ({"note": "x" * 400}, "style", "too long"),
            ("folk", "style", "must be a mapping"),
        ],
    )
    def test_a_wrong_setting_is_refused_in_words_the_user_sees(self, data, task, message):
        with pytest.raises(ValueError, match=message):
            audio_settings_from_mapping(data, task=task)

    def test_an_unknown_task_is_refused(self):
        with pytest.raises(ValueError, match="Unknown audio task"):
            audio_settings_from_mapping({}, task="lyrics")

    def test_the_mapping_lists_every_axis_or_only_the_changed_ones(self):
        settings = audio_settings_from_mapping({"genre": "folk", "without": ["guitar"]}, task="style")

        assert audio_settings_to_mapping(settings, changed_only=True) == {"genre": "folk", "without": ["guitar"]}
        assert len(audio_settings_to_mapping(settings)) == 11

    def test_the_summary_names_the_picked_options_in_axis_order_then_the_direction(self):
        assert audio_settings_summary({"music_mood": "melancholic", "genre": "folk", "note": "sparse"}) == ["Folk", "Melancholic", '"sparse"']
        assert audio_settings_summary({"genre": "keep", "unknown": "x", "instruments": ["piano", "nope"]}) == ["Piano"]

    def test_without_is_written_as_no_instrument(self):
        assert without_phrase(("guitar", "drums")) == "no guitar, no drums"


class TestEveryAxisReachesTheModel:
    @pytest.mark.parametrize(
        ("task", "key", "slug"),
        [
            (item.kind if item.kind == "sound" else "style", item.axis.key, option.slug)
            for item in AUDIO_AXES
            for option in item.axis.options
            if option.slug != "keep" and item.axis.key not in ("instruments", "without")
        ],
    )
    def test_a_chosen_option_puts_its_instruction_in_the_rules(self, task, key, slug):
        item = next(axis for axis in AUDIO_AXES if axis.axis.key == key)

        system = _messages("text", task, "yue2", **{key: slug})[0]["content"]

        assert item.axis.option(slug).instruction in _rules(system)

    def test_instruments_are_featured_and_without_is_written_as_no_instrument(self):
        rules = _rules(_style(instruments=["piano", "violin"], without=["guitar", "drums"]))

        assert "Instruments to feature: piano, violin." in rules
        assert 'Leave these out and write them as: "no guitar, no drums".' in rules

    def test_the_direction_is_a_rule(self):
        assert "Direction from the user (follow it): keep it sparse" in _rules(_style(note="keep it sparse"))

    def test_defaults_add_no_axis_rule_and_the_other_kinds_axes_are_ignored(self):
        plain = _rules(_style())
        with_sound_axis = _rules(build_audio_messages("sad folk", AudioEnhanceSettings(space="forest"), task="style", rules=caption_rules("yue2"), context=AudioEnhanceContext())[0]["content"])

        assert not any(rule.startswith(("Genre:", "Mood:", "Vocal:", "Production:", "Instruments", "Leave these out")) for rule in plain)
        assert with_sound_axis == plain

    def test_the_text_is_sent_as_a_labelled_prompt_to_rewrite(self):
        messages = _messages("sad folk song", "style", "yue2")

        assert messages[1] == {"role": "user", "content": "Prompt: sad folk song"}


class TestCaptionRules:
    def test_yue2_writes_comma_descriptors_without_tempo_or_key(self):
        rules = _rules(_style(family="yue2", genre="jazz"))

        assert any("comma-separated descriptors" in rule for rule in rules)
        assert any("no full sentences" in rule for rule in rules)
        assert any("Do not write a tempo, BPM or key" in rule for rule in rules)
        assert not any("45 words" in rule or "TrackType" in rule for rule in rules)

    def test_stable_audio_writes_one_sentence_of_at_most_45_words_without_vocals_or_the_shell(self):
        for task in ("sound", "style"):
            rules = _rules(_messages("a text", task, "stable_audio_3")[0]["content"])

            assert any("one sentence of at most 45 words" in rule for rule in rules)
            assert any("cannot sing" in rule and "never describe a voice or singing" in rule for rule in rules)
            assert any('"TrackType", "Length"' in rule and "tempo" in rule for rule in rules)
            assert not any("comma-separated" in rule for rule in rules)

    @pytest.mark.parametrize("family", ["something_new", None])
    def test_an_unknown_family_gets_plain_prose(self, family):
        rules = _rules(_style(family=family))

        assert any("plain sentences of at most 60 words" in rule for rule in rules)
        assert not any("comma-separated" in rule or "45 words" in rule for rule in rules)
        assert caption_rules(family).vocals is True

    def test_the_closing_line_names_the_shape_of_the_output(self):
        assert _style(family="yue2").endswith("on one line: no preamble, no headings, no quotes around the whole text.")
        assert _style(family="stable_audio_3").endswith("Output only the one sentence: no preamble, no headings, no quotes around the whole text.")

    def test_names_dates_places_and_quotes_are_copied_exactly(self):
        system = _style('Anna\'s song "Blue Monday" from 1983 in Oslo', family="yue2")

        assert "Copy names, dates and places exactly as the user wrote them." in _rules(system)
        assert 'Keep every quoted text exactly as written, with its quote marks, even a name: "Blue Monday".' in _rules(system)
        assert "EVERY DETAIL IS IMPORTANT: keep the user's subject, intent and every element they mention." in _rules(system)

    def test_a_song_for_a_family_that_cannot_sing_is_written_as_instrumental_music(self):
        singing = _rules(_messages("a warm folk song", "style", "stable_audio_3")[0]["content"])
        sound = _rules(_messages("a crowd singing", "sound", "stable_audio_3")[0]["content"])
        yue2 = _rules(_messages("a warm folk song", "style", "yue2")[0]["content"])

        assert any(rule.startswith("Write it as instrumental music even if the text says song") for rule in singing)
        assert not any("instrumental music" in rule for rule in sound + yue2)

    def test_the_quote_rule_comes_last_and_covers_a_quoted_name_in_a_sound(self):
        system = _messages('the bell of the "Santa Maria" ringing', "sound", "stable_audio_3", context=AudioEnhanceContext(sound_type="effect", seconds=5))[0]["content"]

        assert _rules(system)[-1] == 'Keep every quoted text exactly as written, with its quote marks, even a name: "Santa Maria".'

    def test_no_quote_rule_without_quoted_text(self):
        assert not any("quoted text" in rule for rule in _rules(_messages("rain", "sound", "stable_audio_3")[0]["content"]))

    def test_a_set_tempo_or_key_is_left_to_the_setting_for_a_family_without_metadata(self):
        for family in ("yue2", "stable_audio_3"):
            both = _rules(_messages("fast disco", "style", family, context=AudioEnhanceContext(tempo=118, key="C"))[0]["content"])
            tempo_only = _rules(_messages("fast disco", "style", family, context=AudioEnhanceContext(tempo=118))[0]["content"])
            auto = _rules(_messages("fast disco", "style", family)[0]["content"])

            assert any(rule.startswith("The tempo is set separately: write no tempo") for rule in both)
            assert any(rule.startswith("The key is set separately") for rule in both)
            assert not any(rule.startswith("The key is set separately") for rule in tempo_only)
            assert not any("is set separately" in rule for rule in auto)

    def test_the_sound_rules_describe_what_is_heard(self):
        rules = _rules(_messages("rain", "sound", "stable_audio_3")[0]["content"])

        assert any("source" in rule and "reverberant" in rule for rule in rules)
        assert any("Describe only what is heard" in rule for rule in rules)

    def test_a_later_family_with_headings_and_metadata_fits_the_same_shape(self, monkeypatch):
        rules = CaptionRules(form=("Write 250 to 450 English words.",), headings=("Global Metadata", "Vocal Details", "Arrangement"), metadata=True, max_tokens=900)
        monkeypatch.setitem(audio_module._CAPTION_RULES, "minimax_music3", rules)

        both = _rules(build_audio_messages("sad folk", AudioEnhanceSettings(), task="style", rules=caption_rules("minimax_music3"), context=AudioEnhanceContext(tempo=96, key="Am"))[0]["content"])
        auto = _rules(build_audio_messages("sad folk", AudioEnhanceSettings(), task="style", rules=caption_rules("minimax_music3"), context=AudioEnhanceContext())[0]["content"])

        assert "Write these headings, each on its own line followed by its text, in this order: Global Metadata, Vocal Details, Arrangement." in both
        assert 'State the tempo as "BPM: 96".' in both and 'State the key as "Key: A minor".' in both
        assert not any(rule.startswith("Never invent") for rule in both)
        assert "Never invent a BPM or key: write none that is not given." in auto
        assert not any(rule.startswith("State the") for rule in auto)
        assert caption_rules("minimax_music3").max_tokens == 900

    def test_a_family_that_does_not_write_metadata_never_gets_the_tempo_or_key(self):
        system = _messages("sad folk", "style", "yue2", context=AudioEnhanceContext(tempo=96, key="Am"))[0]["content"]

        assert "BPM: 96" not in system and "A minor" not in system and "Never invent" not in system


class TestVocalWording:
    def test_stable_audio_music_never_mentions_a_vocal_type_or_the_lyrics(self):
        context = AudioEnhanceContext(lyrics="walking the shore")

        system = _messages("calm piano", "style", "stable_audio_3", context=context)[0]["content"]

        assert "vocal type" not in system
        assert "lyrics (context only" not in system and "walking the shore" not in system
        assert "genre, mood, instruments and production" in system

    def test_a_yue2_instrumental_gets_an_example_without_any_vocal_word(self):
        context = AudioEnhanceContext(instrumental=True)

        rules = _rules(_messages("calm piano", "style", "yue2", context=context)[0]["content"])

        (example,) = [rule for rule in rules if rule.startswith("For example:")]
        assert "vocal" not in example.lower() and "no " not in example
        assert not any("vocal type" in rule for rule in rules)

    def test_a_yue2_song_keeps_the_vocal_example_and_the_vocal_type(self):
        rules = _rules(_style(family="yue2"))

        (example,) = [rule for rule in rules if rule.startswith("For example:")]
        assert "expressive lead vocal" in example and "no " not in example
        assert any("vocal type" in rule for rule in rules)

    def test_a_sound_gets_no_music_example(self):
        assert not any(rule.startswith("For example:") for rule in _rules(_messages("rain", "sound", "yue2")[0]["content"]))


class TestSinger:
    def test_the_singer_is_written_for_a_family_that_sings(self):
        assert "Vocal: a female lead vocal." in _rules(_style(family="yue2", singer="female"))

    def test_the_singer_is_dropped_when_the_song_is_instrumental(self):
        context = AudioEnhanceContext(instrumental=True)

        rules = _rules(_messages("sad folk", "style", "yue2", context=context, singer="female")[0]["content"])

        assert not any(rule.startswith("Vocal:") for rule in rules)
        assert "The song is instrumental: describe no singer, voice, vocals or lyrics." in rules

    def test_the_singer_is_dropped_for_a_model_that_cannot_sing(self):
        assert not any(rule.startswith("Vocal:") for rule in _rules(_style(family="stable_audio_3", singer="choir")))


class TestContext:
    def test_an_empty_sound_is_refused(self):
        enhancer = _FakeEnhancer("never used")

        with pytest.raises(ValueError, match="Enter a sound"):
            enhance_audio_caption(enhancer, " ", AudioEnhanceSettings(), task="sound", family="stable_audio_3", seed=1)
        assert enhancer.messages == []

    def test_an_empty_style_is_refused(self):
        with pytest.raises(ValueError, match="Enter a style to enhance"):
            enhance_audio_caption(_FakeEnhancer(), "", AudioEnhanceSettings(), task="style", family="yue2", seed=1)

    def test_the_start_of_the_lyrics_is_context_and_is_cut_to_the_excerpt(self):
        lyrics = "walking the shore\n" * 100

        rules = _rules(_messages("sad folk", "style", "yue2", context=AudioEnhanceContext(lyrics=lyrics))[0]["content"])

        (rule,) = [rule for rule in rules if rule.startswith("The song's lyrics")]
        assert "never copy them into the style" in rule
        assert len(rule.split("): ", 1)[1]) <= LYRICS_CONTEXT_CHARS
        assert "\n" not in rule

    def test_the_lyrics_are_left_out_of_an_instrumental(self):
        system = _messages("calm", "style", "yue2", context=AudioEnhanceContext(lyrics="la la", instrumental=True))[0]["content"]

        assert "la la" not in system

    def test_a_sound_type_and_a_length_shape_a_sound(self):
        context = AudioEnhanceContext(sound_type="one-shot", seconds=2.0)

        rules = _rules(_messages("door slam", "sound", "stable_audio_3", context=context)[0]["content"])

        assert any(rule.startswith("This is a one-shot") for rule in rules)
        assert "The sound runs about 2 seconds: describe only what fits in that time, and do not write the length." in rules

    def test_an_auto_sound_type_adds_no_type_rule(self):
        rules = _rules(_messages("door slam", "sound", "stable_audio_3", context=AudioEnhanceContext())[0]["content"])

        assert not any(rule.startswith("This is a") for rule in rules)


class TestCleaner:
    def test_marks_and_section_tags_stay(self):
        text = "[Verse 1]\nThe sea is calm\n[sad] I am leaving\n[Chorus]\nla la"

        assert clean_audio_output(text) == text

    def test_reasoning_special_tokens_fences_and_a_label_go(self):
        assert clean_audio_output("<think>hmm [x]</think>Style: [sad] folk, piano<|im_end|>") == "[sad] folk, piano"
        assert clean_audio_output("```text\nfolk, piano\n```") == "folk, piano"
        assert clean_audio_output("<think>never closed [x]") == ""

    def test_a_pair_of_quotes_around_all_of_it_goes_by_default(self):
        assert clean_audio_output('"Folk, piano"') == "Folk, piano"
        assert clean_audio_output('"She said "hi" and left"') == '"She said "hi" and left"'

    def test_speech_marks_stay_when_asked(self):
        assert clean_audio_output('"Welcome back."', strip_quotes=False) == '"Welcome back."'
        assert clean_audio_output('He said "stop" [shout]') == 'He said "stop" [shout]'

    def test_the_base_cleaner_is_not_changed_for_images(self):
        from zvisiongenerator.utils.prompt_enhance import clean_output

        assert clean_output('"A fox"<|im_end|>') == "A fox"


class TestRun:
    def test_the_rewrite_is_cleaned_and_streamed_and_costs_the_budget_of_the_shape(self):
        enhancer = _FakeEnhancer("Style: Folk, piano, no guitar")
        seen: list[str] = []

        result = enhance_audio_caption(enhancer, "sad folk", AudioEnhanceSettings(), task="style", family="yue2", seed=5, on_text=seen.append)

        assert (result.prompt, result.clamped, result.warning) == ("Folk, piano, no guitar", False, None)
        assert seen[0] == "" and seen[-1] == "Folk, piano, no guitar"
        assert enhancer.max_tokens == [caption_rules("yue2").max_tokens] and enhancer.seeds == [5]

    def test_a_budget_can_be_given(self):
        enhancer = _FakeEnhancer("Rain on a tin roof.")

        enhance_audio_caption(enhancer, "rain", AudioEnhanceSettings(), task="sound", family="stable_audio_3", seed=1, max_tokens=77)

        assert enhancer.max_tokens == [77]

    def test_every_shape_has_a_budget_that_fits_its_answer(self):
        assert caption_rules("yue2").max_tokens >= 100 and caption_rules("stable_audio_3").max_tokens >= 100 and caption_rules(None).max_tokens >= 100

    @pytest.mark.parametrize("bad", ["", "   ", "sad folk", "Sad Folk.", "I'm sorry, I can't help with that."])
    def test_an_unusable_answer_is_retried_once_with_the_next_seed(self, bad):
        enhancer = _FakeEnhancer(bad, "Folk, piano")

        result = enhance_audio_caption(enhancer, "sad folk", AudioEnhanceSettings(), task="style", family="yue2", seed=10)

        assert result.prompt == "Folk, piano" and enhancer.seeds == [10, 11]

    def test_two_unusable_answers_end_in_an_error(self):
        with pytest.raises(RuntimeError) as error:
            enhance_audio_caption(_FakeEnhancer("", ""), "sad folk", AudioEnhanceSettings(), task="style", family="yue2", seed=1)
        assert str(error.value) == NO_USABLE_MESSAGE

    def test_a_changed_quote_is_retried_and_then_returned_with_a_warning(self):
        enhancer = _FakeEnhancer("Folk, a song called blue monday", "Folk, a song called blue monday again")

        with pytest.warns(UserWarning, match="quoted text was changed"):
            result = enhance_audio_caption(enhancer, 'folk "Blue Monday"', AudioEnhanceSettings(), task="style", family="yue2", seed=1)

        assert result.prompt == "Folk, a song called blue monday again" and result.warning == QUOTES_CHANGED_WARNING

    def test_a_kept_quote_passes_without_a_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = enhance_audio_caption(_FakeEnhancer('Folk, titled "Blue Monday"'), 'folk "Blue Monday"', AudioEnhanceSettings(), task="style", family="yue2", seed=1)

        assert result.warning is None

    def test_cancelling_stops_the_run(self):
        with pytest.raises(RuntimeError, match="cancelled"):
            enhance_audio_caption(_FakeEnhancer("Folk"), "sad folk", AudioEnhanceSettings(), task="style", family="yue2", seed=1, cancelled=lambda: True)

    def test_an_unknown_task_is_refused(self):
        with pytest.raises(ValueError, match="Unknown audio task"):
            enhance_audio_caption(_FakeEnhancer(), "x", AudioEnhanceSettings(), task="lyrics", family="yue2", seed=1)
