"""Tests for the realignment of an Add delivery answer onto the script's own text."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.speech_realign import realign_delivery


class TestKeepsTheScriptsText:
    def test_the_words_come_back_exactly_as_written_with_the_delivery_the_answer_added(self):
        source = "“Where were you?” she whispered. It’s late."

        result = realign_delivery(source, '"Where were you?" she whispered. …\n[sad] It\'s late.')

        assert result == "“Where were you?” she whispered. …\n[sad] It’s late."

    def test_the_users_own_brackets_stay(self):
        source = "[Host] Welcome back. (pause) Today we talk about bees."

        result = realign_delivery(source, "Welcome back. [afraid] Today we talk about bees.")

        assert result.startswith("[Host] Welcome back. (pause)") and "Today we talk about bees." in result

    def test_words_the_answer_changed_added_or_dropped_do_not_reach_the_result(self):
        source = "I never told anyone. Not even you. The door was open."

        assert realign_delivery(source, "[sad] I never told anybody. Not even you. The door was wide open.") == "[sad] " + source
        assert realign_delivery(source, "[sad] I never told anyone. Extra words here. The door was open.") == "[sad] " + source

    def test_an_answer_that_adds_nothing_gives_the_source_back(self):
        source = "One thing. Another thing."

        assert realign_delivery(source, "One thing! Another thing") == source
        assert realign_delivery(source, "One thing. [sighs] Another thing.") == source


class TestMarks:
    def test_a_mark_is_added_only_at_the_start_of_a_sentence_or_a_line(self):
        source = "I never told anyone. Not even you, my friend.\nThe door was open."

        result = realign_delivery(source, "I never told [sad] anyone. Not even you, [whisper] my friend. [shout] The door was open.")

        assert result == "I never told anyone. Not even you, my friend.\n[shout] The door was open."

    def test_a_mark_after_a_closing_quote_that_ends_a_sentence_counts_as_a_start(self):
        result = realign_delivery('He said "Go." Then he left.', '"He said "Go." [sad] Then he left.')

        assert result == 'He said "Go." [sad] Then he left.'

    def test_no_mark_is_added_where_the_user_already_has_one(self):
        result = realign_delivery("[sad] One. Two.", "[whisper] One. [afraid] Two.")

        assert result == "[sad] One. [afraid] Two."

    def test_a_mark_is_written_in_its_form(self):
        assert realign_delivery("One. Two.", "[SAD] One. [As  Set] Two.") == "[sad] One. [as set] Two."

    def test_a_bracket_that_is_not_a_mark_is_not_added(self):
        assert realign_delivery("One. Two.", "[sighs] One. [Host] Two.") == "One. Two."


class TestPausesAndBreaks:
    def test_pauses_and_line_breaks_go_between_the_words_they_stood_between(self):
        assert realign_delivery("One. Two. Three.", "One. …\nTwo. Three.") == "One. …\nTwo. Three."

    def test_a_pause_is_never_doubled(self):
        source = "One. … Two. Three."

        result = realign_delivery(source, "One. … … Two. … … Three.")

        assert "… …" not in result and result.count("…") == 2

    def test_a_pause_at_the_end_is_kept_once(self):
        assert realign_delivery("One. Two.", "One. Two. …") == "One. Two. …"
        assert realign_delivery("One. Two. …", "One. Two. …") == "One. Two. …"

    def test_a_pause_written_as_three_dots_counts_as_the_scripts_own(self):
        assert realign_delivery("Well... I guess so.", "Well… I guess so.") == "Well... I guess so."
        assert realign_delivery("Hi...", "Hi…") == "Hi..."
        assert realign_delivery("Hi... ", "Hi… ") == "Hi... "

    def test_a_new_pause_is_still_added_next_to_a_script_pause_written_as_three_dots(self):
        assert realign_delivery("Well... I guess so.", "Well… I … guess so.") == "Well... I … guess so."
        assert realign_delivery("Well... I guess so.", "Well… I guess so. …") == "Well... I guess so. …"

    def test_a_line_break_is_not_added_where_the_script_has_one(self):
        assert realign_delivery("One.\nTwo.", "One.\n\nTwo.") == "One.\nTwo."

    def test_nothing_is_put_before_a_lone_punctuation_mark(self):
        result = realign_delivery("Dear [Sarah], come home.", "Dear [Sarah] …\n, come home.")

        assert "…," not in result and result.startswith("Dear [Sarah], come")


@pytest.mark.parametrize("answer", ["", "…", "[sad]", "totally different words"])
def test_an_answer_without_the_scripts_words_changes_nothing(answer):
    assert realign_delivery("One. Two.", answer) == "One. Two."
