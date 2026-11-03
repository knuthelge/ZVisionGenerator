"""Tests for ``ziv audio``: the flags, each refusal, the licence notice, and the commands Copy CLI builds."""

from __future__ import annotations

import json
import shlex
import sys

import pytest

from tests._audio_presets import audio_config, sa3_preset
from tests._factories import load_packaged_config
from tests._paths import REPO_ROOT
from tests.audio._helpers import case_request, clip, fake_backend, patch_runner, sine, voice_folder

from zvisiongenerator import audio_cli
from zvisiongenerator.audio_cli import _build_audio_parser
from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.utils.video_splice import write_pcm16_wav

CASES = json.loads((REPO_ROOT / "tests" / "fixtures" / "audio_cli_cases.json").read_text(encoding="utf-8"))


@pytest.fixture
def voices(tmp_path):
    """Return a folder with a recording (``me.wav``), an *Any* take (``any.flac``) and a cloned take (``cloned.flac``), each 4 s."""
    return voice_folder(tmp_path)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Run ``main`` with the three fixture presets on a CUDA machine and a backend that makes a short clip; return the backend and the patched pieces."""
    backend = fake_backend(clip(0.2))
    patched = patch_runner(monkeypatch, backend)
    monkeypatch.setattr(audio_cli, "load_config", lambda: audio_config())
    monkeypatch.setattr(audio_cli, "ensure_ffmpeg", lambda: None)
    patched.backend = backend
    patched.out = tmp_path / "out"
    return patched


def run_cli(monkeypatch, *argv: str, prog: str = "ziv audio") -> None:
    monkeypatch.setattr(sys, "argv", [prog, *argv])
    audio_cli.main(prog=prog)


def refuses(monkeypatch, capsys, *argv: str) -> str:
    """Run the CLI, expecting it to exit with a usage error, and return the error text."""
    with pytest.raises(SystemExit) as exit_:
        run_cli(monkeypatch, *argv)
    assert exit_.value.code == 2
    return capsys.readouterr().err


def make(env, *argv: str, monkeypatch) -> list[AudioGenerationRequest]:
    run_cli(monkeypatch, *argv, "-o", str(env.out))
    return [call.args[0] for call in env.backend.generate.call_args_list]


class TestFlags:
    def test_the_parser_takes_every_flag(self):
        args = _build_audio_parser().parse_args(
            ["-m", "higgs3", "--prompt", "Hi", "-r", "2", "--seed", "9", "-o", "out", "--seconds", "5", "--voice", "v.wav", "--voice-text", "Hello", "--voice-consent", "--lyrics", "[verse]"]
        )

        assert (args.model, args.prompt, args.runs, args.seed, args.output, args.seconds) == ("higgs3", "Hi", 2, 9, "out", 5.0)
        assert (args.voice, args.voice_text, args.voice_consent, args.lyrics) == ("v.wav", "Hello", True, "[verse]")

    def test_lyrics_and_a_lyrics_file_cannot_be_given_together(self, capsys):
        with pytest.raises(SystemExit):
            _build_audio_parser().parse_args(["--lyrics", "a", "--lyrics-file", "b.txt"])

        assert "not allowed with argument" in capsys.readouterr().err

    def test_the_help_of_lyrics_says_empty_or_tag_only_lyrics_make_an_instrumental(self):
        text = " ".join(_build_audio_parser().format_help().split())

        assert "Without lyrics, or with only [section] tags, the take is an instrumental" in text

    def test_a_song_is_made_from_a_style_and_lyrics_with_the_defaults_for_the_rest(self, env, monkeypatch, tmp_path):
        lyrics = tmp_path / "song.txt"
        lyrics.write_text("[verse]\nla la\n", encoding="utf-8")

        requests = make(env, "-m", "yue2", "--prompt", "synth-pop", "--lyrics-file", str(lyrics), "--seed", "4", monkeypatch=monkeypatch)

        assert requests == [AudioGenerationRequest(preset="yue2", kind="music", prompt="synth-pop", lyrics="[verse]\nla la\n", seconds=None, voice=None, seed=4)]
        assert len(list(env.out.glob("music_*.flac"))) == 1

    @pytest.mark.parametrize(("lyrics", "kind"), [(None, "instrumental"), ("[intro] [verse] [outro]", "instrumental"), ("[verse]\nwords here", "song")])
    def test_no_lyrics_or_only_tags_make_an_instrumental(self, env, monkeypatch, capsys, lyrics, kind):
        extra = ["--lyrics", lyrics] if lyrics is not None else []

        make(env, "-m", "yue2", "--prompt", "calm", *extra, monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        assert ("Making an instrumental" in out) == (kind == "instrumental") and ("Making a song" in out) == (kind == "song")

    @pytest.mark.parametrize(("lyrics", "sections"), [(None, 8), ("[Intro] [Instrumental] [Solo] [Outro]", 4)])
    def test_the_shipped_config_gives_an_empty_or_tags_only_lyrics_box_its_sections(self, env, monkeypatch, capsys, lyrics, sections):
        """The request keeps the lyrics as given (the backend swaps in the default list); the announcement counts the packaged list."""
        monkeypatch.setattr(audio_cli, "load_config", load_packaged_config)
        extra = ["--lyrics", lyrics] if lyrics is not None else []

        (request,) = make(env, "-m", "yue2", "--prompt", "calm piano, slow", *extra, monkeypatch=monkeypatch)

        assert (request.prompt, request.lyrics) == ("calm piano, slow", lyrics or "")
        assert f"Making an instrumental of {sections} sections." in capsys.readouterr().out

    def test_a_sound_takes_ten_seconds_unless_told(self, env, monkeypatch):
        assert make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)[0].seconds == 10.0

    def test_the_runs_are_taken_with_one_load(self, env, monkeypatch):
        requests = make(env, "-m", "sa3", "--prompt", "rain", "--seconds", "5", "-r", "3", monkeypatch=monkeypatch)

        assert len(requests) == 3 and env.backend.load.call_count == 1

    def test_ziv_dispatches_audio_to_this_command(self, env, monkeypatch):
        from zvisiongenerator import cli

        monkeypatch.setattr(sys, "argv", ["ziv", "audio", "-m", "sa3", "--prompt", "rain", "-o", str(env.out)])
        cli.main()

        assert env.backend.generate.call_count == 1

    def test_ziv_help_lists_audio(self, capsys):
        from zvisiongenerator import cli

        cli._build_parser().print_help()

        assert "audio" in capsys.readouterr().out


class TestRefusals:
    def test_a_flag_the_model_does_not_take_is_refused_by_name(self, env, monkeypatch, capsys):
        assert "YuE2 3B sets the length from the lyrics; drop --seconds" in refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "--seconds", "30")

    def test_speech_has_no_length_either(self, env, monkeypatch, capsys):
        assert "Higgs TTS 3 4B reads the whole text; drop --seconds" in refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--seconds", "30")

    @pytest.mark.parametrize(("flag", "model"), [("--lyrics", "sa3"), ("--lyrics", "higgs3")])
    def test_lyrics_are_refused_by_a_model_without_them(self, env, monkeypatch, capsys, flag, model):
        assert f"takes no lyrics; drop {flag}" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", flag, "[verse]")

    def test_a_lyrics_file_is_named_when_it_is_the_flag_given(self, env, monkeypatch, capsys, tmp_path):
        path = tmp_path / "l.txt"
        path.write_text("la")

        assert "drop --lyrics-file" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--lyrics-file", str(path))

    @pytest.mark.parametrize("model", ["yue2", "sa3"])
    def test_a_voice_is_refused_by_a_model_that_cannot_follow_one(self, env, monkeypatch, capsys, model):
        assert "cannot follow a voice; drop --voice" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", "--voice", "me.wav", "--voice-text", "x", "--voice-consent")

    def test_a_voice_without_its_text_is_refused(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "me.wav"), "--voice-consent")

        assert "Add --voice-text: what the voice recording says." in error
        assert env.backend.load.call_count == 0

    def test_a_voice_without_consent_is_refused_with_the_reason(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "me.wav"), "--voice-text", "Hello")

        assert "Add --voice-consent: cloning a voice needs the speaker's permission." in error
        assert env.ensure.call_count == 0

    def test_a_cloned_take_needs_consent_too(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "cloned.flac"), "--voice-text", "Hello")

        assert "Add --voice-consent" in error

    def test_an_any_take_needs_no_consent(self, env, monkeypatch, voices):
        requests = make(env, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "any.flac"), "--voice-text", "Hello", monkeypatch=monkeypatch)

        assert requests[0].voice == VoiceRef(str(voices / "any.flac"), "Hello")

    def test_a_voice_that_is_not_there_is_refused(self, env, monkeypatch, capsys, tmp_path):
        assert "Voice recording not found" in refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(tmp_path / "gone.wav"), "--voice-text", "x", "--voice-consent")

    @pytest.mark.parametrize("flag", [["--voice-text", "x"], ["--voice-consent"]])
    def test_the_voice_flags_need_a_voice(self, env, monkeypatch, capsys, flag):
        assert f"{flag[0]} needs --voice" in refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", *flag)

    def test_a_recording_outside_the_allowed_length_is_refused_before_anything_loads(self, env, monkeypatch, capsys, tmp_path):
        short = tmp_path / "short.wav"
        write_pcm16_wav(sine(1.0, rate=24000), short, 24000)

        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(short), "--voice-text", "x", "--voice-consent")

        assert "between 3 and 30 seconds" in error and env.ensure.call_count == 0

    @pytest.mark.parametrize(("model", "message"), [("yue2", "style of the song"), ("higgs3", "the text to read"), ("sa3", "description of the sound")])
    def test_a_missing_or_blank_prompt_is_refused_in_words_of_the_kind(self, env, monkeypatch, capsys, model, message):
        assert message in refuses(monkeypatch, capsys, "-m", model)
        assert message in refuses(monkeypatch, capsys, "-m", model, "--prompt", "   ")

    def test_speech_over_10000_characters_is_refused_with_the_web_ui_message(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "word " * 2001)

        assert "Shorten the text: it's over 10,000 characters, about 10 minutes of speech." in error
        assert env.ensure.call_count == 0

    def test_speech_of_exactly_10000_characters_is_read_in_parts(self, env, monkeypatch):
        text = ("Short sentence here. " * 600)[:10000]

        assert len(make(env, "-m", "higgs3", "--prompt", text, monkeypatch=monkeypatch)) == 1

    @pytest.mark.parametrize("seconds", ["0.5", "381", "0", "-3", "nan"])
    def test_a_length_outside_the_models_range_is_refused(self, env, monkeypatch, capsys, seconds):
        assert "--seconds must be between 1 and 380" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--seconds", seconds)

    def test_a_lyrics_file_that_cannot_be_read_is_refused(self, env, monkeypatch, capsys, tmp_path):
        assert "Could not read the lyrics file" in refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "--lyrics-file", str(tmp_path / "gone.txt"))

    def test_zero_runs_are_refused(self, env, monkeypatch, capsys):
        assert "--runs must be at least 1" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "-r", "0")


class TestModels:
    def test_a_config_without_audio_models_says_so(self, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "audio_platform_key", lambda: "cuda")
        monkeypatch.setattr(audio_cli, "load_config", lambda: {**load_packaged_config(), "audio_model_presets": {}})

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x")

        assert "No audio models are configured" in error

    def test_the_packaged_config_offers_the_three_models(self, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "audio_platform_key", lambda: "cuda")
        monkeypatch.setattr(audio_cli, "load_config", load_packaged_config)

        assert "Choose one of: yue2, higgs3, sa3." in refuses(monkeypatch, capsys, "--prompt", "x")

    def test_a_model_is_required_and_the_choices_are_listed(self, env, monkeypatch, capsys):
        assert "--model is required. Choose one of: yue2, higgs3, sa3." in refuses(monkeypatch, capsys, "--prompt", "x")

    def test_an_unknown_model_is_refused_with_the_choices(self, env, monkeypatch, capsys):
        assert "Unknown audio model 'nope'. Choose one of: yue2, higgs3, sa3." in refuses(monkeypatch, capsys, "-m", "nope", "--prompt", "x")

    def test_models_that_do_not_run_on_this_platform_are_not_offered(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "audio_platform_key", lambda: "mps")
        monkeypatch.setattr(audio_cli, "load_config", lambda: audio_config(sa3=_cuda_only()))

        assert "None of the configured audio models run on this platform (mps)" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x")


def _cuda_only() -> dict:
    preset = sa3_preset()
    del preset["platforms"]["mps"]
    return preset


class TestLicenceNotice:
    LICENCE = {
        "sa3-licence": {
            "name": "Stability AI Community License",
            "supplier": "Stability AI",
            "summary": "Free under $1M revenue",
            "url": "https://example.com/blob/75aad8836271cb5a5123fb58304fae5d3bb4b0f9/LICENSE",
            "models": ["sa3"],
        }
    }

    def _config(self):
        return {**audio_config(), "licences": self.LICENCE}

    def test_the_terms_are_printed_before_a_missing_model_downloads(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", self._config)
        monkeypatch.setattr(audio_cli, "target_present_offline", lambda target: False)
        env.ensure.side_effect = lambda *a, **k: print("DOWNLOAD STARTS") or "ready"

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        assert "By downloading you agree to these terms:" in out and "Stability AI Community License" in out
        assert out.index("By downloading you agree") < out.index("DOWNLOAD STARTS")

    def test_nothing_is_printed_for_a_model_that_is_already_downloaded(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", self._config)
        monkeypatch.setattr(audio_cli, "target_present_offline", lambda target: True)

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        assert "By downloading" not in capsys.readouterr().out

    def test_a_model_without_a_record_prints_nothing(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", self._config)
        monkeypatch.setattr(audio_cli, "target_present_offline", lambda target: False)

        make(env, "-m", "yue2", "--prompt", "calm", monkeypatch=monkeypatch)

        assert "By downloading" not in capsys.readouterr().out


class TestFailures:
    def test_running_out_of_memory_ends_with_the_models_message(self, env, monkeypatch, capsys):
        import torch

        env.backend.generate.side_effect = torch.OutOfMemoryError("out of memory")

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "-o", str(env.out))

        assert "YuE2 3B ran out of GPU memory. Close other apps that use the GPU and try again or shorten the lyrics." in error
        env.backend.close.assert_called_once()

    def test_a_model_whose_files_are_not_in_the_cache_ends_with_its_own_message(self, env, monkeypatch, capsys):
        env.backend.load.side_effect = FileNotFoundError("Higgs TTS 3 needs bosonai/higgs-tts-3-4b, which is not downloaded.")

        assert "bosonai/higgs-tts-3-4b, which is not downloaded" in refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "-o", str(env.out))


class TestCopyCli:
    """The command Copy CLI builds (``frontend/src/lib/state/cliCommand.ts``, same fixture in its Vitest) makes the job the form would."""

    @pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
    def test_the_copied_command_runs_to_the_request_of_the_form(self, env, monkeypatch, voices, case):
        command = case["command"].replace("/voices/", f"{voices}/")
        assert command.startswith("ziv audio ")

        requests = make(env, *shlex.split(command)[2:], monkeypatch=monkeypatch)

        expected = [case_request(item, voices) for item in case["requests"]]
        assert requests == expected
        assert len(list(env.out.glob("*.flac"))) == len(expected)

    def test_only_a_cloned_voice_carries_the_consent_flag(self):
        flags = {case["name"]: "--voice-consent" in case["command"] for case in CASES}

        assert flags == {
            "a song from a style and lyrics": False,
            "an instrumental from a style alone": False,
            "speech in the model's own voice (Any)": False,
            "speech following a recording, with the speaker's consent": True,
            "speech following an Any take, which needs no consent": False,
            "speech following a cloned take, with consent": True,
            "a sound of a chosen length, two takes": False,
        }
