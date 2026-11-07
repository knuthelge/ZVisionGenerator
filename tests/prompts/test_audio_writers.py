"""Tests for the audio writer tasks: the song as sections, the lyrics settings and messages, the checked run and the lyrics task."""

from __future__ import annotations

import json

import pytest

from tests._paths import FIXTURES
from zvisiongenerator.utils import audio_writers as writers
from zvisiongenerator.utils.audio_settings import MARKS
from zvisiongenerator.utils.audio_writers import (
    FILL_FAILED_MESSAGE,
    LYRICS_IMPROVEMENTS,
    LYRICS_UNCHANGED_MESSAGE,
    FITTED_NOTHING_TO_ADD_NOTE,
    NOTHING_TO_ADD_MESSAGE,
    SCRIPT_UNCHANGED_MESSAGE,
    SECTIONS_CHANGED_MESSAGE,
    LyricsSettings,
    NoChangeError,
    ScriptSettings,
    Verdict,
    build_lyrics_messages,
    build_script_messages,
    build_fill_messages,
    check_fill_request,
    check_lyrics_request,
    check_script_request,
    edit_script,
    fill_in,
    fill_settings_from_mapping,
    improve_lyrics,
    lyrics_settings_from_mapping,
    parse_fill_answer,
    parse_song,
    render_fill,
    replace_section,
    run_checked,
    run_writer,
    script_settings_from_mapping,
    script_settings_summary,
    script_settings_to_mapping,
    serialize_song,
    song_has_words,
    writer_token_budget,
)
from zvisiongenerator.utils.prompt_enhance import NO_USABLE_MESSAGE, QUOTES_CHANGED_WARNING, EnhanceResult
from zvisiongenerator.utils.speech_text import strip_marks, unknown_marks
from zvisiongenerator.utils.prompt_enhance_audio import LONG_TEXT_MAX_CHARS, AudioEnhanceContext

SONG = "[Verse]\nThe porch light flickers, the street is wet\nI still keep your coat by the door\n\n[Chorus]\nCome back home\nCome back home to me\n\n[Bridge]\n\n[Outro]\n"


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


def _rules(messages: list[dict[str, str]]) -> list[str]:
    return [line.removeprefix("- ") for line in messages[0]["content"].splitlines() if line.startswith("- ")]


def _settings(**overrides) -> LyricsSettings:
    return lyrics_settings_from_mapping(overrides)


class TestSong:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "just words",
            "[Verse]",
            "[Verse]\nline\n\n\n[Chorus]\nline\n\n",
            "before\n\n[Verse]\nline",
            "[Verse]\n[Chorus]\n\n[Bridge]",
            "  [Verse]  \nline",
            "[Verse]\nline [not a tag]\n[Odd tag",
            "\n\n[Verse]\nx\n",
        ],
    )
    def test_a_song_joins_back_into_the_text_it_was_cut_from(self, text):
        assert serialize_song(parse_song(text)) == text

    def test_the_song_is_cut_at_its_tag_lines(self):
        song = parse_song(SONG)

        assert [block.tag for block in song.blocks] == ["[Verse]", "[Chorus]", "[Bridge]", "[Outro]"]
        assert song.blocks[0].text == "The porch light flickers, the street is wet\nI still keep your coat by the door" and song.blocks[0].blank == 1
        assert (song.blocks[2].text, song.blocks[3].text, song.blocks[3].blank) == ("", "", 1)
        assert parse_song("words\n[Verse]\nx").lead.text == "words" and parse_song("").lead.text == ""

    def test_words_are_found_anywhere_but_not_in_tags_alone(self):
        assert song_has_words(parse_song("[Verse]\nx")) and song_has_words(parse_song("before\n[Verse]"))
        assert not song_has_words(parse_song("[Verse]\n\n[Chorus]\n")) and not song_has_words(parse_song(" \n"))

    def test_a_section_is_replaced_with_its_tag_and_gap_kept(self):
        song = parse_song(SONG)

        assert replace_section(song, 2, "Long night\nStill here\n") == SONG.replace("[Bridge]\n", "[Bridge]\nLong night\nStill here\n")
        assert replace_section(song, 1, "One line") == SONG.replace("Come back home\nCome back home to me", "One line")
        assert replace_section(song, 1, "  ") == SONG.replace("Come back home\nCome back home to me\n", "")


class TestSongMatchesTheWebUi:
    """``tests/fixtures/song_parse_cases.json`` holds what the Web UI's ``parseSong`` makes of each text; the section index must mean the same block on both sides."""

    CASES = json.loads((FIXTURES / "song_parse_cases.json").read_text(encoding="utf-8"))

    @pytest.mark.parametrize("case", CASES, ids=[repr(case["text"])[:40] for case in CASES])
    def test_the_song_is_cut_as_the_web_ui_cuts_it(self, case):
        song = parse_song(case["text"])

        assert {"lead": {"text": song.lead.text, "blank": song.lead.blank}, "blocks": [{"tag": b.tag, "text": b.text, "blank": b.blank} for b in song.blocks]} == case["song"]
        assert serialize_song(song) == case["text"].replace("\r\n", "\n")


class TestSettings:
    def test_the_defaults_improve_the_whole_song(self):
        assert lyrics_settings_from_mapping({}) == LyricsSettings(action="improve", improve=(), note="", section=None)

    def test_every_field_is_read(self):
        settings = lyrics_settings_from_mapping({"improve": ["tighten", "rhymes", "tighten"], "note": "  more   hopeful ", "section": 2})

        assert settings == LyricsSettings(action="improve", improve=("tighten", "rhymes"), note="more hopeful", section=2)

    def test_a_write_names_its_section_and_takes_no_improvements(self):
        assert lyrics_settings_from_mapping({"action": "write", "section": 0, "note": "gentle"}).action == "write"

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            ("tighten", "must be a mapping"),
            ({"tone": "sad"}, "Unknown lyrics setting"),
            ({"action": "compose"}, "action must be one of: improve, write"),
            ({"improve": "tighten"}, "improve must be a list"),
            ({"improve": ["shorter"]}, "Unknown improvement"),
            ({"section": -1}, "section must be a whole number"),
            ({"section": True}, "section must be a whole number"),
            ({"section": "1"}, "section must be a whole number"),
            ({"action": "write"}, "name the section"),
            ({"action": "write", "section": 1, "improve": ["tighten"]}, "Write takes no improvements"),
            ({"section": 1, "improve": ["fit"]}, "works on the whole song"),
            ({"note": "x" * 400}, "Direction is too long"),
        ],
    )
    def test_a_bad_setting_is_refused_in_words_the_user_sees(self, data, message):
        with pytest.raises(ValueError, match=message):
            lyrics_settings_from_mapping(data)

    def test_the_four_improvements_are_the_ones_the_helper_offers(self):
        assert {slug: item.label for slug, item in LYRICS_IMPROVEMENTS.items()} == {"tighten": "Tighten", "fit": "Fit the sections", "rhymes": "Smooth rhymes", "simpler": "Simpler words"}
        assert LYRICS_IMPROVEMENTS["fit"].section is None


class TestRequestCheck:
    def test_lyrics_with_words_pass(self):
        check_lyrics_request(SONG, _settings())
        check_lyrics_request(SONG, _settings(section=0))
        check_lyrics_request(SONG, _settings(action="write", section=2))

    @pytest.mark.parametrize("lyrics", ["", "   ", "[Verse]\n\n[Chorus]\n"])
    def test_a_song_without_words_has_nothing_to_improve(self, lyrics):
        with pytest.raises(ValueError, match="Nothing to improve yet: new lyrics come from Describe it"):
            check_lyrics_request(lyrics, _settings())

    def test_the_lyrics_are_kept_to_the_cap(self):
        check_lyrics_request("a" * LONG_TEXT_MAX_CHARS, _settings())
        with pytest.raises(ValueError, match="too long.*4,000"):
            check_lyrics_request("a" * (LONG_TEXT_MAX_CHARS + 1), _settings())

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            ({"section": 4}, "no section 5"),
            ({"section": 2}, "Section 3 has no words yet: use Write"),
            ({"action": "write", "section": 0}, "Section 1 already has words: use Improve"),
            ({"action": "write", "section": 9}, "no section 10"),
        ],
    )
    def test_a_section_must_fit_the_action(self, settings, message):
        with pytest.raises(ValueError, match=message):
            check_lyrics_request(SONG, _settings(**settings))

    def test_a_write_needs_words_somewhere_in_the_song_to_fit_and_names_write(self):
        with pytest.raises(ValueError, match="Nothing to write yet"):
            check_lyrics_request("[Verse]\n\n[Chorus]\n", _settings(action="write", section=1))
        with pytest.raises(ValueError, match="Nothing to improve yet"):
            check_lyrics_request("[Verse]\n\n[Chorus]\n", _settings())


class TestMessages:
    def test_a_whole_song_keeps_its_tags_and_names_and_follows_its_language(self):
        messages = build_lyrics_messages(SONG, _settings(improve=["tighten"]))

        rules = _rules(messages)
        assert any("exactly as written, each on its own line and in the same order" in rule for rule in rules)
        assert any("section without words stays without words" in rule for rule in rules)
        assert any("same word for word" in rule or "word-for-word repeat" in rule for rule in rules)
        assert "Copy names, dates and places exactly as the user wrote them." in rules
        assert any("language of the lyrics" in rule for rule in rules)
        assert messages[1] == {"role": "user", "content": f"Lyrics:\n{SONG}"}
        assert messages[0]["content"].endswith("no quotes around it.") and "section tags on their own lines" in messages[0]["content"]

    def test_each_improvement_adds_its_own_rule_and_none_is_a_polish(self):
        for slug, item in LYRICS_IMPROVEMENTS.items():
            assert item.whole in _rules(build_lyrics_messages(SONG, _settings(improve=[slug])))
        assert any(rule.startswith("Polish the lyrics") for rule in _rules(build_lyrics_messages(SONG, _settings())))
        assert not any(rule.startswith("Polish") for rule in _rules(build_lyrics_messages(SONG, _settings(improve=["tighten"]))))

    def test_an_improvement_asks_for_real_changes_and_a_write_does_not(self):
        for settings in (_settings(), _settings(improve=["tighten"]), _settings(section=1)):
            assert any("do not return the lyrics unchanged" in rule for rule in _rules(build_lyrics_messages(SONG, settings)))
        assert not any("unchanged" in rule for rule in _rules(build_lyrics_messages(SONG, _settings(action="write", section=2))))

    def test_the_direction_the_style_and_the_quoted_texts_are_context_the_model_keeps(self):
        lyrics = 'He said "hold on, Marie" twice\n[Verse]\nline'
        messages = build_lyrics_messages("[Verse]\n" + lyrics, _settings(note="more hopeful"), style="  warm   folk, piano  ")

        rules = _rules(messages)
        assert "Direction from the user (follow it): more hopeful" in rules
        assert any(rule.startswith("The song's style (context only") and rule.endswith(": warm folk, piano") for rule in rules)
        assert 'Keep every quoted text exactly as written, with its quote marks: "hold on, Marie".' in rules

    def test_the_style_is_cut_and_absent_without_text(self):
        long = _rules(build_lyrics_messages(SONG, _settings(), style="x" * 900))
        none = _rules(build_lyrics_messages(SONG, _settings(), style="  "))

        assert max(len(rule) for rule in long if rule.startswith("The song's style")) < 600
        assert not any(rule.startswith("The song's style") for rule in none)

    def test_a_single_quote_in_lyrics_is_an_apostrophe_not_a_quoted_text(self):
        rules = _rules(build_lyrics_messages("[Verse]\nI'm goin' down, 'cause you're gone", _settings()))

        assert not any("quoted text" in rule for rule in rules)

    def test_one_section_is_improved_with_the_song_as_context(self):
        messages = build_lyrics_messages(SONG, _settings(section=1, improve=["rhymes", "simpler"]))

        rules = _rules(messages)
        assert any("Edit only that section's words" in rule for rule in rules)
        assert LYRICS_IMPROVEMENTS["rhymes"].section in rules and LYRICS_IMPROVEMENTS["simpler"].section in rules
        assert messages[1]["content"] == f"Song:\n{SONG}\n\nSection to edit: [Chorus]\nIts words:\nCome back home\nCome back home to me"
        assert "no section tag" in messages[0]["content"]

    def test_fit_names_the_sections_that_are_out_of_shape(self):
        long_line = " ".join(["word"] * 31)
        verse = "\n".join(["a b c"] * 9)
        song = f"[Verse]\n{verse}\n\n[Chorus]\n{long_line}\n\n[Bridge]\nStay\n"

        user = build_lyrics_messages(song, _settings(improve=["fit"]))[1]["content"]

        assert user.startswith(f"Lyrics:\n{song}\n\nSections to fit:\n")
        assert "Section 1 (Verse) has 9 lines: trim it to about 4." in user
        assert "Section 2 (Chorus) has a line of 31 words: split it into shorter lines." in user
        assert "Section 3 (Bridge) has only 1 word: give it 2 to 4 lines." in user

    def test_fit_compares_a_section_with_the_others_of_its_name(self):
        song = "[Verse 1]\na b c\nd e f\ng h i\nj k l\n\n[Verse 2]\n" + "\n".join(["a b c"] * 7) + "\n"

        user = build_lyrics_messages(song, _settings(improve=["fit"]))[1]["content"]

        assert "Section 2 (Verse 2) has 7 lines while the other verse sections have 4: trim it." in user
        assert "Section 1" not in user

    def test_fit_asks_for_even_lines_when_nothing_is_out_of_shape(self):
        user = build_lyrics_messages(SONG, _settings(improve=["fit"]))[1]["content"]

        assert user.endswith("Sections to fit:\nNo section is clearly out of shape: make the line lengths even, shortening the longest lines.")

    def test_other_improvements_and_sections_get_no_fit_hints(self):
        assert build_lyrics_messages(SONG, _settings(improve=["tighten"]))[1]["content"] == f"Lyrics:\n{SONG}"
        assert "Sections to fit" not in build_lyrics_messages(SONG, _settings(section=0, improve=["rhymes"]))[1]["content"]

    def test_a_write_names_the_empty_section_its_role_and_how_many_lines_fit(self):
        messages = build_lyrics_messages(SONG, _settings(action="write", section=2))

        rules = _rules(messages)
        assert any(rule.startswith("The empty section is Bridge (a contrast to the verses") and rule.endswith("Write about 2 lines.") for rule in rules)
        assert any("fits the rest of the song" in rule for rule in rules)
        assert messages[1]["content"].endswith("Empty section to write: [Bridge]")

    def test_a_write_matches_the_length_of_the_songs_sections_of_the_same_name(self):
        song = "[Verse 1]\na\nb\nc\nd\n\n[Chorus]\nx\ny\n\n[Verse 2]\n"

        assert any(rule.endswith("Write about 4 lines.") for rule in _rules(build_lyrics_messages(song, _settings(action="write", section=2))))

    def test_a_second_chorus_repeats_the_first(self):
        song = "[Verse]\nx\n\n[Chorus]\nsing it loud\n\n[Verse]\ny\n\n[Chorus]\n"

        rules = _rules(build_lyrics_messages(song, _settings(action="write", section=3)))

        assert any("repeats it, so write the same words" in rule for rule in rules)
        assert not any("repeats it" in rule for rule in _rules(build_lyrics_messages(SONG, _settings(action="write", section=2))))

    def test_a_section_of_an_unknown_name_has_no_role(self):
        rules = _rules(build_lyrics_messages("[Verse]\nx\n\n[Hook line]\n", _settings(action="write", section=1)))

        assert "The empty section is Hook line. Write about 2 lines." in rules


class TestRunChecked:
    def test_the_first_clean_answer_is_returned(self):
        enhancer = _FakeEnhancer("good")

        result = run_checked(enhancer, [], lambda out: Verdict(out.upper()), source="x", failure="no", seed=7, temperature=0.5, max_tokens=33)

        assert (result.prompt, result.warning) == ("GOOD", None)
        assert enhancer.seeds == [7] and enhancer.max_tokens == [33]

    @pytest.mark.parametrize("bad", ["", "  ", "…", "same"])
    def test_an_answer_without_words_or_unchanged_is_retried_once_with_the_next_seed(self, bad):
        enhancer = _FakeEnhancer(bad, "better")

        result = run_checked(enhancer, [], lambda out: Verdict(out), source="same", failure="no", seed=10, temperature=0.5, max_tokens=9)

        assert result.prompt == "better" and enhancer.seeds == [10, 11]

    def test_two_answers_with_a_problem_end_in_the_tasks_failure(self):
        with pytest.raises(RuntimeError) as error:
            run_checked(_FakeEnhancer("a", "b"), [], lambda out: Verdict(out, problem="bad"), source="x", failure="Try again.", seed=1, temperature=0.5, max_tokens=9)
        assert str(error.value) == "Try again."

    @pytest.mark.parametrize("answer", ["I'm sorry I missed your call.", "I can't stay.", "I won't go back.", "As an AI language model, hello."])
    def test_an_answer_that_opens_like_a_refusal_is_a_usable_answer(self, answer):
        enhancer = _FakeEnhancer(answer)

        result = run_checked(enhancer, [], lambda out: Verdict(out), source="different", failure="no", seed=1, temperature=0.5, max_tokens=9)

        assert result.prompt == answer and enhancer.seeds == [1]

    def test_a_last_attempt_that_left_the_source_as_it_was_ends_in_the_tasks_no_change_message(self):
        with pytest.raises(RuntimeError) as error:
            run_checked(_FakeEnhancer("", "Same!"), [], lambda out: Verdict(out), source="same", failure="Try again.", no_change="No changes.", seed=1, temperature=0.5, max_tokens=9)
        assert str(error.value) == "No changes."

    def test_the_no_change_message_belongs_to_the_last_attempt_only(self):
        with pytest.raises(RuntimeError) as error:
            run_checked(_FakeEnhancer("same", ""), [], lambda out: Verdict(out), source="same", failure="Try again.", no_change="No changes.", seed=1, temperature=0.5, max_tokens=9)
        assert str(error.value) == NO_USABLE_MESSAGE

    def test_two_unusable_answers_end_in_the_enhancers_usual_message(self):
        with pytest.raises(RuntimeError) as error:
            run_checked(_FakeEnhancer("", "same"), [], lambda out: Verdict(out), source="same", failure="Try again.", seed=1, temperature=0.5, max_tokens=9)
        assert str(error.value) == NO_USABLE_MESSAGE

    def test_a_flawed_answer_is_kept_with_its_notice_when_none_is_clean(self):
        with pytest.warns(UserWarning, match="close"):
            result = run_checked(_FakeEnhancer("one", "two"), [], lambda out: Verdict(out, warning="close"), source="x", failure="no", seed=1, temperature=0.5, max_tokens=9)

        assert (result.prompt, result.warning) == ("two", "close")

    def test_a_clean_second_answer_beats_a_flawed_first(self):
        result = run_checked(_FakeEnhancer("one", "two"), [], lambda out: Verdict(out, warning="close" if out == "one" else None), source="x", failure="no", seed=1, temperature=0.5, max_tokens=9)

        assert (result.prompt, result.warning) == ("two", None)

    def test_a_hint_goes_to_the_model_as_a_reply_to_its_own_answer_before_the_second_attempt(self):
        enhancer = _FakeEnhancer("one", "two")
        base = [{"role": "system", "content": "rules"}, {"role": "user", "content": "text"}]

        run_checked(enhancer, base, lambda out: Verdict(out, hint="Too short.") if out == "one" else Verdict(out), source="x", failure="no", seed=1, temperature=0.5, max_tokens=9)

        assert enhancer.messages[0] == base
        assert enhancer.messages[1] == [*base, {"role": "assistant", "content": "one"}, {"role": "user", "content": "Too short."}]

    def test_no_hint_means_the_same_messages_again(self):
        enhancer = _FakeEnhancer("one", "two")
        base = [{"role": "user", "content": "text"}]

        run_checked(enhancer, base, lambda out: Verdict(out, warning="close") if out == "one" else Verdict(out), source="x", failure="no", seed=1, temperature=0.5, max_tokens=9)

        assert enhancer.messages == [base, base]

    def test_an_answer_with_a_problem_can_carry_a_hint_too(self):
        enhancer = _FakeEnhancer("one", "two")

        run_checked(enhancer, [], lambda out: Verdict(out, problem="bad", hint="Fix it.") if out == "one" else Verdict(out), source="x", failure="no", seed=1, temperature=0.5, max_tokens=9)

        assert enhancer.messages[1][-1] == {"role": "user", "content": "Fix it."}

    def test_a_usable_answer_with_a_hint_is_kept_without_a_notice_when_the_retry_is_no_better(self):
        enhancer = _FakeEnhancer("one", "two")

        result = run_checked(enhancer, [], lambda out: Verdict(out, hint="Closer.", miss=0.2 if out == "one" else 0.3), source="x", failure="no", seed=1, temperature=0.5, max_tokens=9)

        assert (result.prompt, result.warning) == ("one", None)

    def test_the_closer_of_two_flawed_answers_is_kept_and_a_later_one_wins_a_tie(self):
        closer = run_checked(
            _FakeEnhancer("one", "two"), [], lambda out: Verdict(out, warning="far", miss=0.4 if out == "one" else 0.5), source="x", failure="no", seed=1, temperature=0.5, max_tokens=9
        )
        tie = run_checked(_FakeEnhancer("one", "two"), [], lambda out: Verdict(out, warning="far", miss=0.4), source="x", failure="no", seed=1, temperature=0.5, max_tokens=9)

        assert (closer.prompt, tie.prompt) == ("one", "two")

    def test_cancelling_stops_the_run(self):
        with pytest.raises(RuntimeError, match="cancelled"):
            run_checked(_FakeEnhancer("x"), [], lambda out: Verdict(out), source="y", failure="no", seed=1, temperature=0.5, max_tokens=9, cancelled=lambda: True)


IMPROVED = "[Verse]\nPorch light flickers, street is wet\nI keep your coat by the door\n\n[Chorus]\nCome back home\nCome home to me\n\n[Bridge]\n\n[Outro]\n"


class TestImproveLyrics:
    def test_a_whole_song_comes_back_as_the_whole_song_with_its_tags_and_gaps(self):
        enhancer = _FakeEnhancer(IMPROVED)

        result = improve_lyrics(enhancer, SONG, improve=("tighten",), seed=3)

        assert result.prompt == IMPROVED and result.warning is None and not result.clamped
        assert [block.tag for block in parse_song(result.prompt).blocks] == ["[Verse]", "[Chorus]", "[Bridge]", "[Outro]"]

    def test_the_sources_tags_and_gaps_are_kept_whatever_spacing_the_answer_has(self):
        squeezed = "[Verse]\nPorch light flickers, street is wet\nI keep your coat by the door\n[Chorus]\nCome back home\nCome home to me\n[Bridge]\n[Outro]"

        result = improve_lyrics(_FakeEnhancer(squeezed), SONG, seed=1)

        assert result.prompt == IMPROVED

    def test_a_tag_written_as_the_source_had_it_is_kept(self):
        song = "  [Verse]  \nline one\n[Chorus 2]\nline two\n"

        result = improve_lyrics(_FakeEnhancer("[Verse]\nline 1\n[Chorus 2]\nline 2"), song, seed=1)

        assert result.prompt == "  [Verse]  \nline 1\n[Chorus 2]\nline 2\n"

    @pytest.mark.parametrize(
        "answer",
        [
            "[Verse]\nnew\n[Chorus]\nnew\n[Outro]\n",  # three sections for a song of four, two of them with words
            "[Verse]\nnew\n[Chorus]\nnew\n[Bridge]\n[Outro]\n[Verse]\nextra",  # one added
            "Verse: new\nChorus: new",  # the tags written another way
            "[Verse] new words [Chorus] more words",  # tags not on their own lines
            "[Verse]\nall of it in one section",  # sections merged
        ],
    )
    def test_sections_that_cannot_be_matched_are_retried_once_and_then_refused_as_merged_or_dropped(self, answer):
        enhancer = _FakeEnhancer(answer, answer)

        with pytest.raises(RuntimeError) as error:
            improve_lyrics(enhancer, SONG, seed=4)

        assert str(error.value) == SECTIONS_CHANGED_MESSAGE and "merged or dropped" in SECTIONS_CHANGED_MESSAGE and enhancer.seeds == [4, 5]

    @pytest.mark.parametrize(
        "answer",
        [
            "[Verse]\nPorch light flickers, street is wet\nI keep your coat by the door\n\n[Refrain]\nCome back home\nCome home to me\n\n[Bridge]\n\n[Outro]\n",  # renamed
            "[Chorus]\nPorch light flickers, street is wet\nI keep your coat by the door\n[Verse]\nCome back home\nCome home to me\n[Outro]\n[Bridge]",  # swapped
            "Verse 1\n[Verse 1]\nPorch light flickers, street is wet\nI keep your coat by the door\n[Chorus 1]\nCome back home\nCome home to me\n[Bridge 1]\n[Outro 1]",
        ],
    )
    def test_the_names_of_the_answers_sections_do_not_matter_when_their_number_does(self, answer):
        result = improve_lyrics(_FakeEnhancer(answer), SONG, seed=1)

        assert result.prompt == IMPROVED

    def test_a_section_the_answer_left_without_words_keeps_the_words_it_had(self):
        result = improve_lyrics(_FakeEnhancer("[Verse]\nnew verse\n[Chorus]\n[Bridge]\n[Outro]"), SONG, seed=1)

        assert [block.text for block in parse_song(result.prompt).blocks] == ["new verse", parse_song(SONG).blocks[1].text, "", ""]

    def test_tags_the_model_dropped_from_empty_sections_are_mapped_by_the_sections_with_words(self):
        result = improve_lyrics(_FakeEnhancer("[Verse]\nPorch light flickers\n\n[Chorus]\nCome home to me"), SONG, seed=1)

        assert result.prompt == "[Verse]\nPorch light flickers\n\n[Chorus]\nCome home to me\n\n[Bridge]\n\n[Outro]\n"

    def test_a_retry_that_keeps_the_sections_is_returned(self):
        enhancer = _FakeEnhancer("[Verse]\nonly one section", IMPROVED)

        assert improve_lyrics(enhancer, SONG, seed=1).prompt == IMPROVED

    def test_a_title_line_the_model_dropped_is_put_back(self):
        song = "Last Train Home\n[Verse]\nline one\n\n[Chorus]\nline two\n"

        result = improve_lyrics(_FakeEnhancer("[Verse]\nline 1\n\n[Chorus]\nline 2\n"), song, seed=1)

        assert result.prompt == "Last Train Home\n[Verse]\nline 1\n\n[Chorus]\nline 2\n"

    def test_a_song_without_tags_gets_none_the_model_added(self):
        result = improve_lyrics(_FakeEnhancer("[Verse]\nnew words here\n\n[Chorus]\nand more words"), "old words here\n\nand more", seed=1)

        assert result.prompt == "new words here\n\nand more words"
        assert not parse_song(result.prompt).blocks

    def test_the_prompt_of_a_song_without_tags_does_not_ask_for_tags(self):
        enhancer = _FakeEnhancer("changed words here\nand more")

        improve_lyrics(enhancer, "old words here\nand more", seed=1)

        system = enhancer.messages[0][0]["content"].lower()
        assert "tag" not in system and "[verse]" not in system

    def test_an_untagged_answer_that_only_adds_tags_is_no_change(self):
        enhancer = _FakeEnhancer("[Verse]\nold words here\nand more", "[Verse]\nold words here\nand more")

        with pytest.raises(NoChangeError) as error:
            improve_lyrics(enhancer, "old words here\nand more", seed=1)

        assert str(error.value) == LYRICS_UNCHANGED_MESSAGE and enhancer.seeds == [1, 2]

    @pytest.mark.parametrize(
        ("lyrics", "answer"),
        [
            ("I can't stay here\nI can't go on", "I can't stay\nI can't go on"),  # untagged whole song
            ("[Verse]\nI'm sorry for the rain\nI'm sorry for the pain", "[Verse]\nSorry for the rain\nI'm sorry for the pain"),
            ("I won't go back\nI won't look down", "I won't go back there\nI won't look down"),
            ("I cannot sleep\nI cannot dream", "As an AI of the night, I cannot sleep\nI cannot dream"),
        ],
    )
    def test_an_answer_that_opens_like_a_refusal_is_accepted(self, lyrics, answer):
        enhancer = _FakeEnhancer(answer)

        result = improve_lyrics(enhancer, lyrics, seed=1)

        assert result.prompt.strip() == answer and enhancer.seeds == [1]

    def test_a_song_the_model_left_as_it_was_says_no_changes(self):
        enhancer = _FakeEnhancer(SONG, SONG.strip())

        with pytest.raises(NoChangeError) as error:
            improve_lyrics(enhancer, SONG, seed=1)

        assert str(error.value) == LYRICS_UNCHANGED_MESSAGE

    def test_typography_alone_is_no_change(self):
        song = "[Verse]\nShe said \u201chold on\u201d, don\u2019t go\n"

        with pytest.raises(NoChangeError) as error:
            improve_lyrics(_FakeEnhancer('[Verse]\nShe said "hold on", don\'t go', '[Verse]\n\nShe said "hold on", don\'t go'), song, seed=1)

        assert str(error.value) == LYRICS_UNCHANGED_MESSAGE

    def test_a_title_before_the_first_tag_is_dropped_but_the_songs_own_lead_words_stay(self):
        titled = "My Song\n" + IMPROVED
        assert improve_lyrics(_FakeEnhancer(titled), SONG, seed=1).prompt == IMPROVED

        song = "Intro words\n\n[Verse]\nline\n"
        assert improve_lyrics(_FakeEnhancer("Opening words\n\n[Verse]\nlines"), song, seed=1).prompt == "Opening words\n\n[Verse]\nlines\n"

    def test_a_section_without_words_stays_empty_on_an_improvement(self):
        invented = IMPROVED.replace("[Bridge]\n", "[Bridge]\nan invented bridge\n")

        result = improve_lyrics(_FakeEnhancer(invented), SONG, seed=1)

        assert result.prompt == IMPROVED

    def test_a_changed_quote_is_kept_with_a_notice(self):
        song = '[Verse]\nShe said "hold on, Marie" twice\n'

        with pytest.warns(UserWarning, match="quoted text was changed"):
            result = improve_lyrics(_FakeEnhancer('[Verse]\nShe said "wait, Maria" twice', '[Verse]\nShe said "wait there, Maria" twice'), song, seed=1)

        assert result.prompt == '[Verse]\nShe said "wait there, Maria" twice\n' and result.warning == QUOTES_CHANGED_WARNING

    def test_a_quote_written_with_curly_marks_is_not_a_changed_quote(self):
        song = "[Verse]\nShe said \u201chold on, Marie\u201d twice\n"

        result = improve_lyrics(_FakeEnhancer('[Verse]\nShe said "hold on, Marie" two times'), song, seed=1)

        assert result.warning is None

    def test_a_kept_quote_passes_without_a_notice(self):
        song = '[Verse]\nShe said "hold on, Marie" twice\n'

        result = improve_lyrics(_FakeEnhancer('[Verse]\nShe said "hold on, Marie" two times'), song, seed=1)

        assert result.warning is None

    def test_the_answer_is_cleaned_of_fences_and_wrapping_quotes(self):
        fenced = "```\n" + IMPROVED.rstrip("\n") + "\n```"

        assert improve_lyrics(_FakeEnhancer(fenced), SONG, seed=1).prompt == IMPROVED

    def test_the_style_and_the_direction_reach_the_model(self):
        enhancer = _FakeEnhancer(IMPROVED)

        improve_lyrics(enhancer, SONG, improve=("rhymes",), note="more hopeful", style="warm folk", seed=1)

        system = enhancer.messages[0][0]["content"]
        assert "more hopeful" in system and "warm folk" in system and "Smooth rhymes" in system

    def test_an_improvement_the_task_does_not_know_is_refused_before_any_model_call(self):
        enhancer = _FakeEnhancer()

        with pytest.raises(ValueError, match="Unknown improvement"):
            improve_lyrics(enhancer, SONG, improve=("louder",), seed=1)
        with pytest.raises(ValueError, match="Nothing to improve"):
            improve_lyrics(enhancer, "[Verse]\n", seed=1)

        assert enhancer.messages == []

    def test_the_budget_follows_the_length_of_the_lyrics(self):
        enhancer = _FakeEnhancer(IMPROVED)

        improve_lyrics(enhancer, SONG, seed=1)

        assert enhancer.max_tokens == [writer_token_budget(len(SONG))]
        assert writer_token_budget(1) == 256 and writer_token_budget(10**6) == 2400
        assert writer_token_budget(1000) < writer_token_budget(4000) < 2400

    def test_a_budget_can_be_given(self):
        enhancer = _FakeEnhancer(IMPROVED)

        improve_lyrics(enhancer, SONG, seed=1, max_tokens=77)

        assert enhancer.max_tokens == [77]


class TestOneSection:
    def test_a_section_is_improved_and_spliced_into_the_whole_song(self):
        enhancer = _FakeEnhancer("Come home\nCome home to me")

        result = improve_lyrics(enhancer, SONG, improve=("tighten",), section=1, seed=1)

        assert result.prompt == IMPROVED.replace(
            "Porch light flickers, street is wet\nI keep your coat by the door", "The porch light flickers, the street is wet\nI still keep your coat by the door"
        ).replace("Come back home\nCome home to me", "Come home\nCome home to me")
        assert parse_song(result.prompt).blocks[0].text == parse_song(SONG).blocks[0].text

    def test_the_tag_lines_of_the_answer_are_stripped(self):
        result = improve_lyrics(_FakeEnhancer("[Chorus]\nCome home\nCome home to me"), SONG, section=1, seed=1)

        assert result.prompt == SONG.replace("Come back home\nCome back home to me", "Come home\nCome home to me")

    @pytest.mark.parametrize("answer", ["[Chorus]\nnew\n[Bridge]\nmore", "new\n[Bridge]\nmore", "Hmm\n[Chorus]\nnew", "[Chorus]", "[Chorus]\n\n"])
    def test_an_answer_with_other_sections_or_no_words_is_retried_and_refused(self, answer):
        enhancer = _FakeEnhancer(answer, answer)

        with pytest.raises(RuntimeError, match="did not return the section's words"):
            improve_lyrics(enhancer, SONG, section=1, seed=2)

        assert enhancer.seeds == [2, 3]

    def test_the_words_it_already_had_say_no_changes(self):
        with pytest.raises(NoChangeError) as error:
            improve_lyrics(_FakeEnhancer("Come back home\nCome back home to me", "[Chorus]\nCome back home\nCome back home to me"), SONG, section=1, seed=2)

        assert str(error.value) == LYRICS_UNCHANGED_MESSAGE

    @pytest.mark.parametrize("answer", ["I can't let go\nI can't look back", "I'm sorry for the rain\nI'm sorry for the pain", "I won't be broken\nI won't be shaken"])
    def test_a_section_that_opens_like_a_refusal_is_accepted(self, answer):
        song = "[Verse]\nI cannot let go\nI cannot look back\n[Chorus]\nhome"
        enhancer = _FakeEnhancer(answer)

        result = improve_lyrics(enhancer, song, section=0, seed=1)

        assert parse_song(result.prompt).blocks[0].text == answer and enhancer.seeds == [1]

    def test_a_section_from_the_start_of_the_lyrics_and_one_at_the_end_both_splice(self):
        song = "[Verse]\none\n[Chorus]\ntwo"

        assert improve_lyrics(_FakeEnhancer("uno"), song, section=0, seed=1).prompt == "[Verse]\nuno\n[Chorus]\ntwo"
        assert improve_lyrics(_FakeEnhancer("dos"), song, section=1, seed=1).prompt == "[Verse]\none\n[Chorus]\ndos"


class TestWrite:
    def test_only_the_empty_target_section_is_filled(self):
        enhancer = _FakeEnhancer("[Bridge]\nThe night is long\nbut I am still here")

        result = improve_lyrics(enhancer, SONG, section=2, write=True, style="folk", seed=1)

        assert result.prompt == SONG.replace("[Bridge]\n", "[Bridge]\nThe night is long\nbut I am still here\n")
        before, after = parse_song(SONG), parse_song(result.prompt)
        assert [block.text for at, block in enumerate(after.blocks) if at != 2] == [block.text for at, block in enumerate(before.blocks) if at != 2]
        assert [block.tag for block in after.blocks] == [block.tag for block in before.blocks]

    def test_the_last_section_without_a_closing_newline_is_filled(self):
        result = improve_lyrics(_FakeEnhancer("Fade out slowly"), SONG.rstrip("\n"), section=3, write=True, seed=1)

        assert result.prompt.endswith("[Outro]\nFade out slowly")

    def test_a_section_that_has_words_cannot_be_written_over(self):
        enhancer = _FakeEnhancer("x")

        with pytest.raises(ValueError, match="already has words"):
            improve_lyrics(enhancer, SONG, section=0, write=True, seed=1)

        assert enhancer.messages == []

    def test_a_tag_without_words_is_retried_and_then_refused(self):
        with pytest.raises(RuntimeError, match="did not return the section's words"):
            improve_lyrics(_FakeEnhancer("[Bridge]\n", "[Bridge]\n\n"), SONG, section=2, write=True, seed=1)

    def test_an_empty_answer_is_not_usable(self):
        with pytest.raises(RuntimeError) as error:
            improve_lyrics(_FakeEnhancer("", "  "), SONG, section=2, write=True, seed=1)

        assert str(error.value) == NO_USABLE_MESSAGE

    def test_a_written_section_that_opens_like_a_refusal_is_accepted(self):
        result = improve_lyrics(_FakeEnhancer("I can't sleep tonight\nI'm sorry for the light"), SONG, section=2, write=True, seed=1)

        assert parse_song(result.prompt).blocks[2].text == "I can't sleep tonight\nI'm sorry for the light"

    def test_a_written_section_that_drifts_into_other_sections_is_refused(self):
        answer = "[Bridge]\nwords\n[Outro]\nmore"

        with pytest.raises(RuntimeError, match="did not return the section's words"):
            improve_lyrics(_FakeEnhancer(answer, answer), SONG, section=2, write=True, seed=1)

    def test_the_budget_of_a_short_section_is_a_short_one(self):
        enhancer = _FakeEnhancer("two lines\nof words")

        improve_lyrics(enhancer, SONG, section=2, write=True, seed=1)

        assert enhancer.max_tokens == [writer_token_budget(300)] and enhancer.max_tokens[0] < writer_token_budget(len(SONG) * 20)


class TestRunWriter:
    def test_the_lyrics_task_runs_from_its_parsed_settings_and_the_context_style(self):
        enhancer = _FakeEnhancer(IMPROVED)
        settings = lyrics_settings_from_mapping({"improve": ["tighten"], "note": "calmer"})

        result = run_writer("lyrics", enhancer, SONG, settings, context=AudioEnhanceContext(style="warm folk"), seed=1, temperature=0.4, cancelled=None)

        assert result.prompt == IMPROVED
        system = enhancer.messages[0][0]["content"]
        assert "calmer" in system and "warm folk" in system and "Tighten" in system

    def test_a_write_setting_writes(self):
        settings = lyrics_settings_from_mapping({"action": "write", "section": 2})

        result = run_writer("lyrics", _FakeEnhancer("words"), SONG, settings, context=AudioEnhanceContext(), seed=1, temperature=0.4, cancelled=None)

        assert "[Bridge]\nwords" in result.prompt

    def test_an_unknown_writer_task_is_refused(self):
        with pytest.raises(ValueError, match="Unknown writer task"):
            run_writer("poem", _FakeEnhancer(), SONG, None, context=AudioEnhanceContext(), seed=1, temperature=0.4, cancelled=None)


SCRIPT = "I never told anyone. Not even you. The door was open."
MARKED = "[whisper] I never told anyone. …\nNot even you. [as set] The door was open."
MARKED_LONG = "[sad] " + ("The harbour was quiet that morning. " * 20).strip()


def _script_rules(messages: list[dict[str, str]]) -> str:
    return messages[0]["content"]


class TestScriptSettings:
    def test_add_delivery_takes_the_actions_alone(self):
        assert script_settings_from_mapping({"actions": ["delivery"]}) == ScriptSettings(actions=("delivery",))

    def test_fit_takes_whole_seconds_and_a_pace(self):
        assert script_settings_from_mapping({"actions": ["fit"], "seconds": 20}) == ScriptSettings(("fit",), 20, "normal")
        assert script_settings_from_mapping({"actions": ["fit"], "seconds": 600, "pace": "very-slow"}) == ScriptSettings(("fit",), 600, "very-slow")

    def test_both_actions_run_fit_first_whatever_the_order_asked(self):
        both = ScriptSettings(("fit", "delivery"), 20, "slow")
        assert script_settings_from_mapping({"actions": ["fit", "delivery"], "seconds": 20, "pace": "slow"}) == both
        assert script_settings_from_mapping({"actions": ["delivery", "fit"], "seconds": 20, "pace": "slow"}) == both

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            ("fit", "must be a mapping"),
            ({}, "actions must list each"),
            ({"actions": []}, "actions must list each"),
            ({"actions": "fit"}, "actions must list each"),
            ({"actions": ["write"]}, "actions must list each"),
            ({"actions": ["fit", "fit"], "seconds": 20}, "actions must list each"),
            ({"action": "fit", "seconds": 20}, "Unknown script setting"),
            ({"actions": ["fit"]}, "whole number from 3 to 600"),
            ({"actions": ["fit", "delivery"]}, "whole number from 3 to 600"),
            ({"actions": ["fit"], "seconds": 2}, "whole number from 3 to 600"),
            ({"actions": ["fit"], "seconds": 601}, "whole number from 3 to 600"),
            ({"actions": ["fit"], "seconds": 12.5}, "whole number"),
            ({"actions": ["fit"], "seconds": True}, "whole number"),
            ({"actions": ["fit"], "seconds": "20"}, "whole number"),
            ({"actions": ["fit"], "seconds": 20, "pace": "warp"}, "pace must be one of"),
            ({"actions": ["delivery"], "seconds": 20}, "takes no reading time"),
            ({"actions": ["delivery"], "pace": "slow"}, "takes no reading time"),
            ({"actions": ["fit"], "seconds": 20, "note": "calm"}, "Unknown script setting"),
        ],
    )
    def test_a_request_the_helper_cannot_run_is_refused_in_words_the_user_sees(self, data, message):
        with pytest.raises(ValueError, match=message):
            script_settings_from_mapping(data)

    def test_settings_are_stored_with_only_what_they_ask(self):
        assert script_settings_to_mapping(ScriptSettings(("delivery",))) == {"actions": ["delivery"]}
        assert script_settings_to_mapping(ScriptSettings(("fit",), 20)) == {"actions": ["fit"], "seconds": 20}
        assert script_settings_to_mapping(ScriptSettings(("fit", "delivery"), 20, "fast")) == {"actions": ["fit", "delivery"], "seconds": 20, "pace": "fast"}
        assert script_settings_summary({"actions": ["delivery"]}) == ["Add delivery"]
        assert script_settings_summary({"actions": ["fit"], "seconds": 20, "pace": "very-fast"}) == ["Fit to 20 s", "very fast pace"]
        assert script_settings_summary({"actions": ["fit", "delivery"], "seconds": 20}) == ["Fit to 20 s", "Add delivery"]


class TestCheckScriptRequest:
    @pytest.mark.parametrize("text", ["", "   \n", "[sad]", "[sad] … [as set]", "…", ". . ."])
    def test_a_script_without_words_has_nothing_to_work_on(self, text):
        with pytest.raises(ValueError, match="Nothing to work on yet.*Describe it"):
            check_script_request(text)

    def test_a_script_over_the_cap_is_too_long(self):
        check_script_request("a" * LONG_TEXT_MAX_CHARS)

        with pytest.raises(ValueError, match="too long.*4,000"):
            check_script_request("a" * (LONG_TEXT_MAX_CHARS + 1))

    def test_a_script_with_words_passes(self):
        check_script_request("[sad] Hello.")


class TestScriptMessages:
    def test_add_delivery_lists_only_the_marks_the_app_knows_and_forbids_changing_words(self):
        rules = _script_rules(build_script_messages(SCRIPT, "delivery"))

        for mark in ("[sad]", "[whisper]", "[shout]", "[as set]", "[afraid]"):
            assert mark in rules
        assert "Never change the words" in rules and "Never write any other bracketed word" in rules
        assert all(f"[{mark}]" in rules for mark in MARKS) and "long pause" not in rules.lower()

    def test_the_example_is_written_with_real_line_breaks_so_the_model_does_not_copy_a_backslash(self):
        rules = _script_rules(build_script_messages(SCRIPT, "delivery"))

        assert "\\n" not in rules and "With delivery:\nThe door was open. …\n[afraid] Nobody knew why.\n[shout] Run!" in rules

    def test_the_script_is_the_user_message(self):
        messages = build_script_messages(SCRIPT, "delivery")

        assert messages[1] == {"role": "user", "content": f"Script:\n{SCRIPT}"}

    def test_fit_names_the_characters_the_words_and_the_times_to_land_on(self):
        text = "word " * 100  # 499 characters read, 100 words
        rules = _script_rules(build_script_messages(text, "fit", seconds=20))

        assert "The script is 500 characters long, about 100 words, 0:36 to read" in rules
        assert "Shorten it to about 280 characters, about 56 words (56% of its length)" in rules
        assert "reads in about 0:20" in rules and "within 10% of 280" in rules

    def test_fit_lengthens_a_short_script_without_inventing_anything(self):
        rules = _script_rules(build_script_messages("Hello there friend.", "fit", seconds=20))

        assert "Lengthen it to about 280 characters" in rules and "Never add new facts, names, numbers or claims" in rules

    def test_fit_uses_the_pace_for_the_target_and_leaves_the_marks_out_of_the_count(self):
        rules = _script_rules(build_script_messages("[sad] " + "x" * 300, "fit", seconds=20, pace="slow"))

        assert f"The script is {301} characters long" in rules
        assert f"to about {round(20 * 14 * 0.81)} characters" in rules

    def test_fit_keeps_the_marks_with_their_sentences_and_adds_none(self):
        rules = _script_rules(build_script_messages(SCRIPT, "fit", seconds=3))

        assert "Keep each mark" in rules and "Never add a mark or any other bracketed word" in rules and "Copy" not in rules
        assert "Keep names, numbers, dates and quoted text exactly as written" in rules

    def test_an_edit_that_is_close_to_the_length_adjusts_instead_of_cutting(self):
        rules = _script_rules(build_script_messages("x" * 275, "fit", seconds=20))

        assert "Adjust it to about 280 characters" in rules


class TestAddDelivery:
    def test_marks_and_pauses_are_added_and_the_words_are_the_same(self):
        enhancer = _FakeEnhancer(MARKED)

        result = edit_script(enhancer, SCRIPT, actions=["delivery"], seed=5)

        assert result == EnhanceResult(prompt=MARKED, clamped=False)
        assert enhancer.seeds == [5]

    def test_a_pause_or_a_line_break_alone_is_a_real_change(self):
        assert edit_script(_FakeEnhancer("I never told anyone. …\nNot even you. The door was open."), SCRIPT, actions=["delivery"], seed=1).prompt.startswith("I never told anyone. …")
        assert edit_script(_FakeEnhancer("I never told anyone.\nNot even you.\nThe door was open."), SCRIPT, actions=["delivery"], seed=1).prompt.count("\n") == 2

    def test_unknown_marks_are_dropped_and_known_ones_are_written_in_their_form(self):
        enhancer = _FakeEnhancer("[Sad] I never told anyone. [sighs] Not even you. [long pause] The door was open. [as  SET]")  # a mark with nothing after it is dropped too

        result = edit_script(enhancer, SCRIPT, actions=["delivery"], seed=1)

        assert result.prompt == "[sad] I never told anyone. Not even you. The door was open."
        assert unknown_marks(result.prompt) == () and result.warning is None

    @pytest.mark.parametrize(
        "answer",
        [
            "[sad] I never told anybody. Not even you. The door was open.",
            "[sad] I never told anyone. The door was open.",
            "[sad] I never told anyone. Not even you. The door was wide open.",
            "[sad] I never told anyone, not even you. The door was open.",
            "[sad] The door was open. I never told anyone. Not even you.",
        ],
    )
    def test_whatever_words_the_model_changes_the_scripts_own_words_come_back(self, answer):
        enhancer = _FakeEnhancer(answer)

        result = edit_script(enhancer, SCRIPT, actions=["delivery"], seed=3)

        assert result.prompt.replace("[sad] ", "") == SCRIPT and enhancer.seeds == [3]

    def test_the_text_comes_back_exactly_as_written_with_curly_quotes_and_the_users_own_brackets(self):
        script = "\u201cWhere were you?\u201d she whispered. [Host] It\u2019s late. Go home."
        answer = '"Where were you?" she whispered. …\n[sad] It\'s late. [Host] Go home.'

        result = edit_script(_FakeEnhancer(answer), script, actions=["delivery"], seed=1)

        assert " ".join(result.prompt.replace("\u2026", " ").split()) == script
        assert "\u201cWhere were you?\u201d she whispered." in result.prompt and "[Host]" in result.prompt and "\n" in result.prompt

    def test_a_mark_goes_only_at_the_start_of_a_sentence_or_a_line(self):
        script = "I never told anyone. Not even you, my friend. The door was open."

        result = edit_script(_FakeEnhancer("I never told [sad] anyone. Not even [whisper] you, [shout] my friend. [afraid] The door was open."), script, actions=["delivery"], seed=1)

        assert result.prompt == "I never told anyone. Not even you, my friend. [afraid] The door was open."

    def test_no_mark_is_added_where_the_user_already_has_one(self):
        script = "[sad] I never told anyone. Not even you."

        result = edit_script(_FakeEnhancer("[whisper] I never told anyone. [afraid] Not even you."), script, actions=["delivery"], seed=1)

        assert result.prompt == "[sad] I never told anyone. [afraid] Not even you."

    def test_no_pause_is_doubled(self):
        script = "I never told anyone. \u2026 Not even you. The door was open."

        result = edit_script(_FakeEnhancer("I never told anyone. \u2026 \u2026\nNot even you. \u2026 \u2026 The door was open."), script, actions=["delivery"], seed=1)

        assert "\u2026 \u2026" not in result.prompt and result.prompt.count("\u2026") == 2

    def test_an_answer_that_adds_nothing_says_there_is_nothing_to_add(self):
        enhancer = _FakeEnhancer(SCRIPT, "  " + SCRIPT.replace("  ", " ") + "  ")

        with pytest.raises(NoChangeError) as error:
            edit_script(enhancer, SCRIPT, actions=["delivery"], seed=1)

        assert str(error.value) == NOTHING_TO_ADD_MESSAGE and enhancer.seeds == [1, 2]

    def test_a_script_that_opens_like_a_refusal_is_marked_up(self):
        script = "I'm sorry I missed your call. Please leave a message."

        result = edit_script(_FakeEnhancer("I'm sorry I missed your call. \u2026\nPlease leave a message."), script, actions=["delivery"], seed=1)

        assert result.prompt == "I'm sorry I missed your call. \u2026\nPlease leave a message."

    def test_the_marks_and_pauses_the_script_already_has_stay_the_script_s_words(self):
        script = "[sad] I never told anyone. … Not even you."
        enhancer = _FakeEnhancer("[sad] I never told anyone. …\n[whisper] Not even you.")

        assert edit_script(enhancer, script, actions=["delivery"], seed=1).prompt == "[sad] I never told anyone. …\n[whisper] Not even you."

    def test_a_mark_with_nothing_after_it_is_dropped(self):
        assert (
            edit_script(_FakeEnhancer("[sad] I never told anyone. Not even you. The door was open. [whisper]"), SCRIPT, actions=["delivery"], seed=1).prompt
            == "[sad] I never told anyone. Not even you. The door was open."
        )

    def test_an_answer_cleaned_of_fences_and_wrapping_quotes_is_checked(self):
        assert edit_script(_FakeEnhancer(f"```\n{MARKED}\n```"), SCRIPT, actions=["delivery"], seed=1).prompt == MARKED
        assert edit_script(_FakeEnhancer(f'"{MARKED}"'), SCRIPT, actions=["delivery"], seed=1).prompt == MARKED

    def test_a_script_that_is_one_whole_quotation_keeps_its_quotes(self):
        script = '"I never told anyone. Not even you. The door was open."'
        answer = '"I never told anyone. …\nNot even you. [sad] The door was open."'

        assert edit_script(_FakeEnhancer(answer), script, actions=["delivery"], seed=1).prompt == answer

    def test_the_budget_follows_the_length_of_the_script(self):
        enhancer = _FakeEnhancer(MARKED)

        edit_script(enhancer, SCRIPT, actions=["delivery"], seed=1)

        assert enhancer.max_tokens == [writer_token_budget(len(SCRIPT) * 5 // 4 + 120)]

    def test_a_script_with_nothing_to_mark_is_refused_before_any_model_call(self):
        enhancer = _FakeEnhancer(MARKED)

        with pytest.raises(ValueError, match="Nothing to work on yet"):
            edit_script(enhancer, "[sad] …", actions=["delivery"], seed=1)

        assert enhancer.seeds == []

    def test_a_time_on_add_delivery_is_refused(self):
        with pytest.raises(ValueError, match="takes no reading time"):
            edit_script(_FakeEnhancer(MARKED), SCRIPT, actions=["delivery"], target_seconds=20, seed=1)

    @pytest.mark.parametrize(
        "source",
        [
            'Mara said, "Meet me at 7:45 at Pier 9." The ferry costs $12.50, not 15.\n\nDr. O\'Neil laughed: "Ha!" Then he left.',
            "Dear [Sarah], come home. I miss you... really. [Sad] So much.",
            "A [sad] line\nand another one.\n\nThird paragraph, with Ünïcödé and 日本語。",
            "Wait… what? No!  Two  spaces. ",
            "He said [whisper]hello[as set] to her.",
        ],
    )
    def test_whatever_the_model_adds_the_result_has_the_scripts_words_and_no_mark_the_user_did_not_write(self, source):
        """Names, quotes, numbers and line breaks survive; anything the model adds besides known marks, pauses and line breaks is left out."""
        import random

        rng = random.Random(7)
        extras = [" …", " [sad]", " [Whisper]", "\n", " [as set]", " [laughs]", " [x]", " [AS SET]", " [  sad  ]", " [sad"]

        def words(text: str) -> str:
            return " ".join(strip_marks(text).replace("…", " ").split())

        kept = 0
        for _ in range(300):
            answer = " ".join(word + (rng.choice(extras) if rng.random() < 0.25 else "") for word in source.split(" "))
            try:
                result = edit_script(_FakeEnhancer(answer, answer), source, actions=["delivery"], seed=1)
            except RuntimeError:
                continue
            kept += 1
            assert unknown_marks(result.prompt) == unknown_marks(source)
            assert words(result.prompt) == words(source)
        assert kept > 0


class TestFit:
    LONG = ("The harbour was quiet that morning. " * 20).strip()  # 719 characters
    ON_TARGET = "x" * 280

    def test_an_answer_near_the_target_is_returned_at_once_without_a_notice(self):
        # 20 seconds at Normal are 280 characters; within 15% either side is on target.
        for length in (238, 280, 322):
            enhancer = _FakeEnhancer("y" * length)

            result = edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1)

            assert result == EnhanceResult(prompt="y" * length, clamped=False) and len(enhancer.seeds) == 1

    def test_an_answer_between_15_and_25_percent_off_is_tried_again_and_kept_without_a_notice_when_it_stays_there(self):
        enhancer = _FakeEnhancer("y" * 340, "y" * 345)

        result = edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert result == EnhanceResult(prompt="y" * 340, clamped=False) and len(enhancer.seeds) == 2  # the closer of the two

    def test_the_second_attempt_is_told_how_the_first_came_out(self):
        enhancer = _FakeEnhancer("y" * 420, "y" * 285)

        edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1)

        asked = enhancer.messages[1]
        assert asked[:2] == enhancer.messages[0] and asked[2] == {"role": "assistant", "content": "y" * 420}
        assert asked[3]["role"] == "user"
        assert "That script is 420 characters, 0:30 to read: 50% too long" in asked[3]["content"]
        assert "at most 280 characters" in asked[3]["content"] and "Cut about 140 characters more" in asked[3]["content"]

    def test_a_script_that_is_too_short_is_told_to_say_more(self):
        enhancer = _FakeEnhancer("y" * 100, "y" * 290)

        edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert "64% too short" in enhancer.messages[1][3]["content"] and "at least 280 characters" in enhancer.messages[1][3]["content"]
        assert "Add about 180 characters more" in enhancer.messages[1][3]["content"]

    def test_the_script_returned_unchanged_is_told_how_long_it_is(self):
        enhancer = _FakeEnhancer(self.LONG, "y" * 290)

        result = edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert result.prompt == "y" * 290 and "That script is 719 characters" in enhancer.messages[1][3]["content"]

    def test_the_closer_of_two_answers_that_are_off_is_the_one_kept(self):
        enhancer = _FakeEnhancer("y" * 380, "y" * 700)

        with pytest.warns(UserWarning, match="Came out at about 0:27"):
            result = edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert result.prompt == "y" * 380 and result.warning == "Came out at about 0:27."

    def test_marks_with_nothing_after_them_are_dropped(self):
        result = edit_script(_FakeEnhancer("[sad] " + "y" * 280 + " [sad] [whisper]"), "[sad] [whisper] " + self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert result.prompt == "[sad] " + "y" * 280

    def test_an_answer_outside_the_tolerance_is_kept_with_the_time_it_came_out_at_after_a_retry(self):
        enhancer = _FakeEnhancer("y" * 700, "y" * 600)

        with pytest.warns(UserWarning, match="Came out at about 0:43"):
            result = edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert result.prompt == "y" * 600 and result.warning == "Came out at about 0:43."
        assert len(enhancer.seeds) == 2

    def test_a_clean_second_answer_beats_a_first_that_is_off(self):
        enhancer = _FakeEnhancer("y" * 100, "y" * 290)

        assert edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1).warning is None

    def test_the_time_is_counted_at_the_pace(self):
        # 20 seconds at Slow are 227 characters: 260 is on target, 300 is 32% over.
        assert edit_script(_FakeEnhancer("y" * 260), self.LONG, actions=["fit"], target_seconds=20, pace="slow", seed=1).warning is None
        with pytest.warns(UserWarning, match="Came out at about 0:26"):
            result = edit_script(_FakeEnhancer("y" * 300, "y" * 300), self.LONG, actions=["fit"], target_seconds=20, pace="slow", seed=1)
        assert result.warning == "Came out at about 0:26."

    def test_marks_are_not_counted(self):
        answer = "[sad] " + "y" * 280 + " [as set] " + "z" * 0

        assert edit_script(_FakeEnhancer(answer), self.LONG, actions=["fit"], target_seconds=20, seed=1).warning is None

    def test_a_fit_adds_no_mark_the_script_did_not_have_and_drops_unknown_ones(self):
        script = "[sad] " + self.LONG
        answer = "[Sad] " + "y" * 130 + " [whisper] " + "z" * 130 + " [sighs] " + "w" * 10

        result = edit_script(_FakeEnhancer(answer), script, actions=["fit"], target_seconds=20, seed=1)

        assert result.prompt == "[sad] " + "y" * 130 + " " + "z" * 130 + " " + "w" * 10
        assert unknown_marks(result.prompt) == ()

    def test_a_fit_of_a_script_with_no_marks_has_none_in_its_answer(self):
        result = edit_script(_FakeEnhancer("[sad] " + "y" * 280), self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert result.prompt == "y" * 280

    def test_a_changed_quote_is_kept_with_a_notice(self):
        script = 'He said "Meet me at Pier 9." ' + self.LONG
        answer = 'He said "Meet me at the pier." ' + "y" * 250

        with pytest.warns(UserWarning, match="quoted text was changed"):
            result = edit_script(_FakeEnhancer(answer, answer), script, actions=["fit"], target_seconds=20, seed=1)

        assert result.warning == QUOTES_CHANGED_WARNING

    def test_a_kept_quote_passes_without_a_notice(self):
        script = 'He said "Meet me at Pier 9." ' + self.LONG
        answer = 'He said "Meet me at Pier 9." ' + "y" * 250

        assert edit_script(_FakeEnhancer(answer), script, actions=["fit"], target_seconds=20, seed=1).warning is None

    def test_a_script_that_is_one_whole_quotation_keeps_its_quotes_in_a_fit(self):
        script = f'"{self.LONG}"'
        answer = '"' + "y" * 280 + '"'

        with pytest.warns(UserWarning, match="quoted text was changed"):
            result = edit_script(_FakeEnhancer(answer, answer), script, actions=["fit"], target_seconds=20, seed=1)

        assert result.prompt == answer

    def test_the_fit_prompt_counts_pauses_and_line_breaks_but_not_marks(self):
        rules = _script_rules(build_script_messages(self.LONG, "fit", seconds=20))

        assert "Pauses (…) and line breaks may stay. They count toward the length like any character; marks do not." in rules

    def test_the_script_itself_is_not_a_fit(self):
        enhancer = _FakeEnhancer(self.LONG, self.LONG)

        with pytest.raises(NoChangeError) as error:
            edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert str(error.value) == SCRIPT_UNCHANGED_MESSAGE

    @pytest.mark.parametrize("opening", ["I'm sorry", "I can't", "I won't", "As an AI"])
    def test_a_fitted_script_that_opens_like_a_refusal_is_accepted(self, opening):
        answer = f"{opening} say it better than this. " + "x" * 255

        result = edit_script(_FakeEnhancer(answer), "I said it plainly. " * 40, actions=["fit"], target_seconds=20, seed=1)

        assert result.prompt == answer

    def test_a_quote_the_model_straightens_is_not_a_changed_quote(self):
        script = "She said \u201cwait here\u201d. " + "The harbour was quiet. " * 30

        result = edit_script(_FakeEnhancer('She said "wait here". ' + "x" * 260), script, actions=["fit"], target_seconds=20, seed=1)

        assert result.warning is None

    def test_the_budget_follows_the_longer_of_the_script_and_the_target(self):
        enhancer = _FakeEnhancer("y" * 280)

        edit_script(enhancer, "Short script here.", actions=["fit"], target_seconds=20, seed=1)
        edit_script(_enhancer := _FakeEnhancer("y" * 280), self.LONG, actions=["fit"], target_seconds=20, seed=1)

        assert enhancer.max_tokens == [writer_token_budget(280)] and _enhancer.max_tokens == [writer_token_budget(len(self.LONG))]

    @pytest.mark.parametrize("seconds", [None, 2, 601])
    def test_a_time_outside_what_fit_takes_is_refused_before_any_model_call(self, seconds):
        enhancer = _FakeEnhancer("y")

        with pytest.raises(ValueError, match="whole number from 3 to 600"):
            edit_script(enhancer, self.LONG, actions=["fit"], target_seconds=seconds, seed=1)

        assert enhancer.seeds == []


class TestRunScript:
    def test_the_script_task_runs_from_its_parsed_settings(self):
        enhancer = _FakeEnhancer(MARKED)

        result = run_writer("script", enhancer, SCRIPT, ScriptSettings(("delivery",)), context=AudioEnhanceContext(), seed=2, temperature=0.4, cancelled=None)

        assert result.prompt == MARKED and enhancer.seeds == [2]

    def test_a_fit_setting_fits(self):
        enhancer = _FakeEnhancer("y" * 280)

        result = run_writer("script", enhancer, TestFit.LONG, ScriptSettings(("fit",), 20, "normal"), context=AudioEnhanceContext(), seed=2, temperature=0.4, cancelled=None)

        assert result.prompt == "y" * 280

    def test_both_settings_fit_and_then_mark_up_the_fitted_text(self):
        enhancer = _FakeEnhancer("The harbour was quiet. The tide came in.", "The harbour was quiet. …\n[sad] The tide came in.")

        result = run_writer("script", enhancer, TestFit.LONG, ScriptSettings(("fit", "delivery"), 3, "normal"), context=AudioEnhanceContext(), seed=2, temperature=0.4, cancelled=None)

        assert result.prompt == "The harbour was quiet. …\n[sad] The tide came in." and enhancer.seeds == [2, 2]


class TestFitAndAddDelivery:
    FITTED = "The harbour was quiet. The tide came in."
    BOTH = {"actions": ["fit", "delivery"], "target_seconds": 3}

    def test_one_run_fits_first_and_marks_up_the_fitted_words(self):
        enhancer = _FakeEnhancer(self.FITTED, "The harbour was quiet. …\n[sad] The tide came in.")

        result = edit_script(enhancer, TestFit.LONG, seed=4, **self.BOTH)

        assert result == EnhanceResult(prompt="The harbour was quiet. …\n[sad] The tide came in.", clamped=False)
        assert "The script is 719 characters long" in enhancer.messages[0][0]["content"]
        # The delivery step works from the fitted text, not from the script that was fitted.
        assert enhancer.messages[1][1] == {"role": "user", "content": f"Script:\n{self.FITTED}"}
        assert "Never change the words" in enhancer.messages[1][0]["content"]

    def test_the_order_of_the_actions_asked_does_not_change_the_run(self):
        enhancer = _FakeEnhancer(self.FITTED, "[sad] The harbour was quiet. The tide came in.")

        edit_script(enhancer, TestFit.LONG, actions=["delivery", "fit"], target_seconds=3, seed=1)

        assert "edit a script" in enhancer.messages[0][0]["content"] and "add delivery" in enhancer.messages[1][0]["content"]

    def test_a_delivery_that_adds_nothing_keeps_the_fitted_text_with_a_note(self):
        enhancer = _FakeEnhancer(self.FITTED, self.FITTED, self.FITTED)

        result = edit_script(enhancer, TestFit.LONG, seed=1, **self.BOTH)

        assert result == EnhanceResult(prompt=self.FITTED, clamped=False, warning=FITTED_NOTHING_TO_ADD_NOTE)

    def test_the_fit_notice_and_the_nothing_to_add_note_are_both_kept(self):
        enhancer = _FakeEnhancer("y" * 600, "y" * 600, "y" * 600, "y" * 600)

        with pytest.warns(UserWarning):
            result = edit_script(enhancer, TestFit.LONG, seed=1, **self.BOTH)

        assert result.prompt == "y" * 600 and result.warning == f"Came out at about 0:43. {FITTED_NOTHING_TO_ADD_NOTE}"

    def test_a_script_that_already_fits_still_gets_its_delivery(self):
        enhancer = _FakeEnhancer(TestFit.LONG, TestFit.LONG, MARKED_LONG)

        result = edit_script(enhancer, TestFit.LONG, seed=1, actions=["fit", "delivery"], target_seconds=52)

        assert result.prompt == MARKED_LONG

    def test_a_script_that_fits_and_gets_nothing_added_fails_as_a_delivery_that_adds_nothing(self):
        enhancer = _FakeEnhancer(TestFit.LONG, TestFit.LONG, TestFit.LONG, TestFit.LONG)

        with pytest.raises(NoChangeError, match="Nothing to add"):
            edit_script(enhancer, TestFit.LONG, seed=1, actions=["fit", "delivery"], target_seconds=52)

    def test_a_fit_alone_that_changes_nothing_is_still_a_no_change(self):
        enhancer = _FakeEnhancer(TestFit.LONG, TestFit.LONG)

        with pytest.raises(NoChangeError, match="No changes"):
            edit_script(enhancer, TestFit.LONG, seed=1, actions=["fit"], target_seconds=52)


def test_the_writer_tables_agree_on_their_tasks():
    # Fill in works on both kinds and writes several fields, so it has no single kind or field.
    assert set(writers.WRITER_KINDS) == set(writers.WRITER_FIELDS) == set(writers.WRITER_TASKS) - {writers.FILL_TASK}


# ── Fill in ────────────────────────────────────────────────────────────────────

MUSIC_ANSWER = """STYLE: folk, acoustic guitar, wistful female vocal, sparse
TEMPO: 84 BPM
KEY: A minor
LYRICS:
[Verse]
I pack the boat at dawn
The harbour wakes behind me

[Chorus]
Goodbye to the coast
The tide knows my name"""
MUSIC_FIELDS = ("style", "tempo", "key", "lyrics")
SPEECH_ANSWER = "VIBE: sad-confession\nTEXT:\nI should have told you sooner. The letter was never meant for Marta."


def _fill(
    *replies: str,
    kind: str = "music",
    fields: tuple[str, ...] = MUSIC_FIELDS,
    describe: str = "a sad folk song about leaving the coast",
    context: AudioEnhanceContext | None = None,
    family: str | None = "yue2",
):
    enhancer = _FakeEnhancer(*replies)
    result = fill_in(enhancer, describe, kind=kind, fields=fields, context=context or AudioEnhanceContext(), family=family, seed=7, temperature=0.4)
    return result, enhancer


class TestParseFillAnswer:
    def test_every_labelled_block_is_read_and_validated(self):
        answer = parse_fill_answer(MUSIC_ANSWER, MUSIC_FIELDS)

        assert answer.fields == {
            "style": "folk, acoustic guitar, wistful female vocal, sparse",
            "tempo": "84",
            "key": "Am",
            "lyrics": "[Verse]\nI pack the boat at dawn\nThe harbour wakes behind me\n\n[Chorus]\nGoodbye to the coast\nThe tide knows my name",
        }
        assert answer.missing == () and not answer.instrumental

    def test_only_the_fields_asked_for_are_kept(self):
        answer = parse_fill_answer(MUSIC_ANSWER, ("style", "key"))

        assert set(answer.fields) == {"style", "key"} and answer.missing == ()

    @pytest.mark.parametrize(("raw", "expected"), [("84", "84"), ("about 84 bpm", "84"), ("84.4", "84"), ("40", "40"), ("220", "220"), ("6/8 at 120", "120")])
    def test_a_tempo_is_a_whole_number_in_range(self, raw, expected):
        assert parse_fill_answer(f"TEMPO: {raw}", ("tempo",)).fields == {"tempo": expected}

    @pytest.mark.parametrize("raw", ["fast", "39", "221", "900", "", "0"])
    def test_a_tempo_out_of_range_is_dropped_and_named_missing(self, raw):
        answer = parse_fill_answer(f"TEMPO: {raw}", ("tempo",))

        assert answer.fields == {} and answer.missing == ("tempo",)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("A minor", "Am"),
            ("F# minor", "F#m"),
            ("F♯ minor", "F#m"),
            ("Bb major", "Bb"),
            ("B♭ Major", "Bb"),
            ("C", "C"),
            ("Am", "Am"),
            ("bm", "Bm"),
            ("Gb major", "F#"),
            ("G# major", "Ab"),
            ("Db minor", "C#m"),
            ("D# minor", "Ebm"),
            ("A-flat minor", None),
            ("H minor", None),
            ("A dorian", None),
            ("", None),
        ],
    )
    def test_a_key_is_one_of_the_24_in_any_common_spelling(self, raw, expected):
        answer = parse_fill_answer(f"KEY: {raw}", ("key",))

        assert answer.fields == ({"key": expected} if expected else {}) and answer.missing == (() if expected else ("key",))

    @pytest.mark.parametrize(
        ("raw", "expected"), [("calm-narrator", "calm-narrator"), ("Calm narrator", "calm-narrator"), ("whispered_secret.", "whispered-secret"), ("**storyteller**", "storyteller")]
    )
    def test_a_vibe_is_one_the_app_serves(self, raw, expected):
        assert parse_fill_answer(f"VIBE: {raw}", ("vibe",)).fields == {"vibe": expected}

    def test_a_near_miss_of_one_served_vibe_is_that_vibe(self):
        assert parse_fill_answer("VIBE: whisper-secret", ("vibe",)).fields == {"vibe": "whispered-secret"}

    @pytest.mark.parametrize("raw", ["villain", "calm", "sad", ""])
    def test_an_unknown_vibe_is_dropped(self, raw):
        assert parse_fill_answer(f"VIBE: {raw}", ("vibe",)).missing == ("vibe",)

    @pytest.mark.parametrize("raw", [", TEMPO:, KEY:, LYRICS:", "<the style>", "TEMPO: 90", "- ", ""])
    def test_a_style_that_is_the_form_echoed_or_starts_with_punctuation_is_dropped(self, raw):
        answer = parse_fill_answer(f"STYLE:{raw}", ("style",))

        assert answer.fields == {} and answer.missing == ("style",)

    def test_lyrics_get_every_section_tag_on_its_own_line_and_lose_what_came_before_the_first(self):
        answer = parse_fill_answer("LYRICS:\nMy Song Title\n[Verse] The coast is far\nand I am gone\n[Chorus]   Sail on\n", ("lyrics",))

        assert answer.fields["lyrics"] == "[Verse]\nThe coast is far\nand I am gone\n\n[Chorus]\nSail on"
        assert [block.tag for block in parse_song(answer.fields["lyrics"]).blocks] == ["[Verse]", "[Chorus]"]

    @pytest.mark.parametrize("raw", ["just some words with no sections", "[Verse]\n\n[Chorus]", ""])
    def test_lyrics_without_a_section_that_has_words_are_dropped(self, raw):
        answer = parse_fill_answer(f"LYRICS:\n{raw}", ("lyrics",))

        assert answer.fields == {} and answer.missing == ("lyrics",)

    @pytest.mark.parametrize("raw", ["none", "None.", "instrumental", "no lyrics", "n/a", "-"])
    def test_lyrics_none_is_an_empty_song_not_a_missing_field(self, raw):
        answer = parse_fill_answer(f"STYLE: piano\nLYRICS: {raw}", ("style", "lyrics"))

        assert answer.fields == {"style": "piano"} and answer.missing == () and answer.instrumental

    def test_lyrics_are_cut_at_a_section_when_the_song_is_too_long(self):
        verse = "[Verse]\n" + "\n".join(f"line {number} of a long verse" for number in range(120))
        answer = parse_fill_answer("LYRICS:\n" + "\n\n".join([verse] * 8), ("lyrics",))

        assert len(answer.fields["lyrics"]) <= LONG_TEXT_MAX_CHARS and answer.fields["lyrics"].count("[Verse]") < 8

    def test_a_text_keeps_only_the_marks_the_app_knows(self):
        answer = parse_fill_answer("TEXT:\n[sad] I should have told you. [applause] [whisper] Sorry.", ("text",))

        assert answer.fields == {"text": "[sad] I should have told you. [whisper] Sorry."}
        assert not unknown_marks(answer.fields["text"])

    def test_a_text_without_words_is_dropped(self):
        assert parse_fill_answer("TEXT: [sad]", ("text",)).missing == ("text",)

    @pytest.mark.parametrize(
        "answer",
        [
            "**STYLE:** folk\n**TEMPO:** 90",
            "Style: folk\nTempo: 90",
            "- STYLE: folk\n- TEMPO: 90",
            "style : folk\n\nTEMPO:90",
        ],
    )
    def test_labels_may_be_loose(self, answer):
        assert parse_fill_answer(answer, ("style", "tempo")).fields == {"style": "folk", "tempo": "90"}

    def test_a_chatty_windows_answer_is_read_field_by_field(self):
        answer = "Sure, here you go!\r\n\r\n**STYLE:** warm acoustic folk\r\n**TEMPO:** 80 BPM\r\n**KEY:** G major\r\nLYRICS:\r\n[Verse 1] Walking home\r\nunder rain\r\n\r\n[Chorus]\r\nla la\r\n"

        parsed = parse_fill_answer(answer, ("style", "tempo", "key", "lyrics"))

        assert parsed.missing == ()
        assert parsed.fields == {"style": "warm acoustic folk", "tempo": "80", "key": "G", "lyrics": "[Verse 1]\nWalking home\nunder rain\n\n[Chorus]\nla la"}

    def test_a_field_that_is_wrong_does_not_cost_the_others(self):
        answer = "STYLE: lo-fi hip hop\nTEMPO: 9999\nKEY: H sharp lydian\nLYRICS:\n[Verse]\nrain on glass"

        parsed = parse_fill_answer(answer, ("style", "tempo", "key", "lyrics"))

        assert parsed.missing == ("tempo", "key")
        assert parsed.fields == {"style": "lo-fi hip hop", "lyrics": "[Verse]\nrain on glass"}

    def test_a_one_line_field_is_its_first_line(self):
        answer = parse_fill_answer("VIBE: sad-confession\nText (start): I never told anyone.\nTEMPO: 90 BPM\nkey of the song", ("vibe", "tempo"))

        assert answer.fields == {"vibe": "sad-confession", "tempo": "90"}

    def test_a_label_inside_the_last_block_is_part_of_it_once_seen(self):
        answer = parse_fill_answer("STYLE: folk\nLYRICS:\n[Verse]\nStyle: of my own\nText: and more", ("style", "lyrics"))

        assert answer.fields["style"] == "folk" and "Style: of my own" in answer.fields["lyrics"]

    @pytest.mark.parametrize("answer", ["", "I can't help with that.", "STYLE:\nTEMPO:", "no labels at all just prose"])
    def test_a_malformed_answer_fills_nothing(self, answer):
        parsed = parse_fill_answer(answer, MUSIC_FIELDS)

        assert parsed.fields == {} and parsed.missing == MUSIC_FIELDS

    def test_rendering_and_parsing_agree(self):
        answer = parse_fill_answer(MUSIC_ANSWER, MUSIC_FIELDS)

        assert parse_fill_answer(render_fill(answer.fields), MUSIC_FIELDS) == answer
        assert parse_fill_answer(render_fill({"style": "piano"}, instrumental=True), ("style", "lyrics")).instrumental


class TestFillSettings:
    def test_fields_keep_the_order_of_the_answer_and_drop_duplicates(self):
        assert fill_settings_from_mapping({"fields": ["lyrics", "style", "style", "key"]}, kind="music").fields == ("style", "key", "lyrics")
        assert fill_settings_from_mapping({"fields": ["text", "vibe"]}, kind="speech").fields == ("vibe", "text")

    @pytest.mark.parametrize(
        ("data", "kind", "message"),
        [
            ({"fields": ["style"]}, "sound", "fills music and speech, not sound"),
            ({"fields": []}, "music", "fields must list what to fill"),
            ({}, "music", "fields must list what to fill"),
            ({"fields": "style"}, "music", "fields must list what to fill"),
            ({"fields": [3]}, "music", "fields must list what to fill"),
            ({"fields": ["text"]}, "music", "Unknown music field"),
            ({"fields": ["tempo"]}, "speech", "Unknown speech field"),
            ({"fields": ["style"], "note": "x"}, "music", "Unknown fill setting"),
            ("style", "music", "must be a mapping"),
        ],
    )
    def test_malformed_settings_are_refused(self, data, kind, message):
        with pytest.raises(ValueError, match=message):
            fill_settings_from_mapping(data, kind=kind)

    def test_the_idea_must_have_words_and_fit_the_cap(self):
        check_fill_request("a sad song")
        with pytest.raises(ValueError, match="Describe what you want"):
            check_fill_request("   ")
        with pytest.raises(ValueError, match="too long"):
            check_fill_request("x" * 1001)
        check_fill_request("x" * 1000)


class TestFillMessages:
    def test_a_song_asks_for_the_fields_in_order_with_each_ones_rules(self):
        messages = build_fill_messages("a sad folk song", fill_settings_from_mapping({"fields": list(MUSIC_FIELDS)}, kind="music"), context=AudioEnhanceContext(), family="yue2")

        system, user = (message["content"] for message in messages)
        assert "STYLE: <the style>\nTEMPO: <beats per minute>\nKEY: <the key>\nLYRICS:\n<the lyrics" in system
        assert "comma-separated descriptors" in system and "from 40 to 220" in system and "A minor" in system and "F# minor" in system and "Bb major" in system
        assert "section tag alone on its own line" in system and "repeat its words exactly" in system and "LYRICS: none" in system
        assert "Copy names, dates, places and quoted words" in system
        assert user == "Idea:\na sad folk song"

    def test_lyrics_with_only_tags_are_filled_into_those_sections(self):
        settings = fill_settings_from_mapping({"fields": ["style", "lyrics"]}, kind="music")

        system = build_fill_messages("a sad folk song", settings, context=AudioEnhanceContext(lyrics="[Intro]\n\n[Verse 1]\n\n[Chorus]\n"), family="yue2")[0]["content"]

        assert "Use exactly these sections, in this order: [Intro], [Verse 1], [Chorus]." in system
        assert "Use a shape such as" not in system

    def test_lyrics_with_no_tags_get_the_usual_shape(self):
        settings = fill_settings_from_mapping({"fields": ["lyrics"]}, kind="music")

        system = build_fill_messages("a sad folk song", settings, context=AudioEnhanceContext(), family="yue2")[0]["content"]

        assert "Use a shape such as [Verse], [Chorus]" in system and "Use exactly these sections" not in system

    def test_a_speech_asks_for_a_vibe_from_the_served_list_and_a_text(self):
        messages = build_fill_messages("a secret", fill_settings_from_mapping({"fields": ["vibe", "text"]}, kind="speech"), context=AudioEnhanceContext(), family="qwen3_tts")

        system = messages[0]["content"]
        assert "VIBE: <the vibe id>\nTEXT:\n<the words" in system
        for vibe in ("calm-narrator", "storyteller", "excited-host", "announcer", "sad-confession", "whispered-secret"):
            assert vibe in system
        assert "[sad]" in system and "Most texts need none" in system and "60 to 120 words" in system and "tempo" not in system.lower()

    def test_a_quoted_text_in_the_idea_is_a_rule_to_copy_it_exactly(self):
        messages = build_fill_messages(
            'a tale where he says "The scan is clear."', fill_settings_from_mapping({"fields": ["vibe", "text"]}, kind="speech"), context=AudioEnhanceContext(), family="qwen3_tts"
        )

        assert 'Keep every quoted text exactly as written, with its quote marks: "The scan is clear."' in messages[0]["content"]

    def test_the_style_of_a_model_that_cannot_sing_describes_no_voice(self):
        messages = build_fill_messages("a calm piece", fill_settings_from_mapping({"fields": ["style", "tempo", "key"]}, kind="music"), context=AudioEnhanceContext(), family="stable_audio_3")

        system = messages[0]["content"]
        assert "at most 45 words" in system and "never describe a voice" in system and "LYRICS" not in system

    def test_what_the_user_already_set_is_context_and_not_asked_for(self):
        context = AudioEnhanceContext(style="dark synth", lyrics="[Verse]\nOne two three", tempo=100, key="Em")
        messages = build_fill_messages("a night drive", fill_settings_from_mapping({"fields": ["lyrics"]}, kind="music"), context=context, family="yue2")

        system, user = (message["content"] for message in messages)
        assert "STYLE:" not in system and "TEMPO:" not in system and "KEY:" not in system
        assert "Style: dark synth" in user and "Tempo: 100 BPM" in user and "Key: E minor" in user and "One two three" not in user  # lyrics are asked for, so not repeated

    def test_the_text_so_far_is_context_for_a_vibe_alone(self):
        context = AudioEnhanceContext(text="I should have told you sooner.")
        messages = build_fill_messages("read it sadly", fill_settings_from_mapping({"fields": ["vibe"]}, kind="speech"), context=context, family="qwen3_tts")

        assert "Text (start): I should have told you sooner." in messages[1]["content"] and "TEXT:" not in messages[0]["content"]


class TestFillIn:
    def test_one_call_writes_every_field_asked_for(self):
        result, enhancer = _fill(MUSIC_ANSWER)

        assert list(result.fields) == ["style", "tempo", "key", "lyrics"] and result.warning is None
        assert result.fields["tempo"] == "84" and result.fields["key"] == "Am"
        assert len(enhancer.messages) == 1

    def test_a_speech_fill_writes_a_text_and_a_vibe(self):
        result, _ = _fill(SPEECH_ANSWER, kind="speech", fields=("vibe", "text"), describe="a sad confession about the letter to Marta", family="qwen3_tts")

        assert result.fields == {"vibe": "sad-confession", "text": "I should have told you sooner. The letter was never meant for Marta."}

    def test_only_the_fields_asked_for_are_returned_even_when_the_answer_holds_more(self):
        result, _ = _fill(MUSIC_ANSWER, fields=("key", "tempo"))

        assert result.fields == {"tempo": "84", "key": "Am"}

    def test_the_answer_keeps_its_labels_when_it_starts_with_one(self):
        result, _ = _fill("STYLE: folk, guitar\nTEMPO: 90", fields=("style", "tempo"))

        assert result.fields == {"style": "folk, guitar", "tempo": "90"}

    def test_a_field_the_answer_lacks_earns_one_more_try_with_what_was_wrong(self):
        result, enhancer = _fill("STYLE: folk\nTEMPO: 900\nKEY: A minor", "STYLE: folk, guitar\nTEMPO: 84\nKEY: A minor", fields=("style", "tempo", "key"))

        assert result.fields == {"style": "folk, guitar", "tempo": "84", "key": "Am"} and result.warning is None
        assert enhancer.seeds == [7, 8]
        retry = enhancer.messages[1]
        assert retry[-2]["role"] == "assistant" and "unusable value for: Tempo" in retry[-1]["content"]

    def test_a_field_that_stays_wrong_is_left_out_and_named_in_the_warning(self):
        result, enhancer = _fill("STYLE: folk\nTEMPO: fast\nKEY: A minor", "STYLE: folk\nTEMPO: quick\nKEY: A minor", fields=("style", "tempo", "key"))

        assert result.fields == {"style": "folk", "key": "Am"}
        assert result.warning == "Left empty: Tempo."
        assert len(enhancer.messages) == 2

    def test_the_better_of_two_flawed_answers_is_kept(self):
        result, _ = _fill("STYLE: folk\nTEMPO: fast\nKEY: nowhere", "STYLE: folk\nTEMPO: 80\nKEY: nowhere", fields=("style", "tempo", "key"))

        assert result.fields == {"style": "folk", "tempo": "80"} and result.warning == "Left empty: Key."

    def test_a_song_of_only_tags_is_filled_with_lyrics(self):
        context = AudioEnhanceContext(lyrics="[Verse]\n\n[Chorus]\n")

        result, enhancer = _fill("STYLE: folk\nLYRICS:\n[Verse]\nA line\n\n[Chorus]\nA hook", fields=("style", "lyrics"), context=context)

        assert result.fields["lyrics"] == "[Verse]\nA line\n\n[Chorus]\nA hook" and result.warning is None and not result.instrumental
        assert "Use exactly these sections, in this order: [Verse], [Chorus]." in enhancer.messages[0][0]["content"]

    def test_an_instrumental_idea_leaves_the_tags_of_the_song_as_they_are(self):
        context = AudioEnhanceContext(lyrics="[Verse]\n\n[Chorus]\n")

        result, _ = _fill("STYLE: piano, ambient\nLYRICS: none", describe="an instrumental piano piece", fields=("style", "lyrics"), context=context)

        assert "lyrics" not in result.fields and result.instrumental and result.warning is None

    def test_an_instrumental_has_no_lyrics_and_no_warning(self):
        result, enhancer = _fill("STYLE: piano, ambient\nTEMPO: 70\nKEY: C\nLYRICS: none", describe="an instrumental piano piece")

        assert result.fields == {"style": "piano, ambient", "tempo": "70", "key": "C"} and result.warning is None and result.instrumental
        assert len(enhancer.messages) == 1

    def test_a_malformed_answer_is_retried_then_fails_in_words_the_user_sees(self):
        enhancer = _FakeEnhancer("I can't help with that.", "nothing useful here")

        with pytest.raises(RuntimeError) as failure:
            fill_in(enhancer, "a song", kind="music", fields=MUSIC_FIELDS, context=AudioEnhanceContext(), seed=1)

        assert str(failure.value) == FILL_FAILED_MESSAGE and len(enhancer.messages) == 2

    def test_a_quoted_name_that_changed_is_kept_with_a_notice_after_one_more_try(self):
        changed = "VIBE: storyteller\nTEXT: Then he said go away."
        result, enhancer = _fill(changed, changed, kind="speech", fields=("vibe", "text"), describe='a tale where the man says "Go Away"', family="qwen3_tts")

        assert result.warning == QUOTES_CHANGED_WARNING and result.fields["text"] == "Then he said go away." and len(enhancer.messages) == 2
        assert "Keep every quoted text" in enhancer.messages[1][-1]["content"]

    def test_a_quote_in_the_idea_that_the_style_alone_changes_is_no_notice(self):
        result, enhancer = _fill("STYLE: folk", fields=("style",), describe='a song called "Sea Glass"')

        assert result.warning is None and len(enhancer.messages) == 1

    def test_the_token_budget_follows_the_fields(self):
        _, long = _fill(MUSIC_ANSWER)
        _, short = _fill("TEMPO: 90", fields=("tempo",))

        assert short.max_tokens[0] < 200 < long.max_tokens[0]

    def test_cancelling_stops_the_run(self):
        enhancer = _FakeEnhancer(MUSIC_ANSWER)

        with pytest.raises(RuntimeError, match="cancelled"):
            fill_in(enhancer, "a song", kind="music", fields=MUSIC_FIELDS, context=AudioEnhanceContext(), seed=1, cancelled=lambda: True)

    def test_run_writer_dispatches_the_fill_task(self):
        settings = fill_settings_from_mapping({"fields": list(MUSIC_FIELDS)}, kind="music")

        result = run_writer("fill", _FakeEnhancer(MUSIC_ANSWER), "a sad folk song", settings, context=AudioEnhanceContext(), family="yue2", seed=1, temperature=0.4, cancelled=None)

        assert result.fields["key"] == "Am"
