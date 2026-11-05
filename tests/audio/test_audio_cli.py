"""Tests for ``ziv audio``: the flags, each refusal, the licence notice, and the commands Copy CLI builds."""

from __future__ import annotations

import functools
import json
import shlex
import sys
from unittest.mock import MagicMock

import pytest

from tests._audio_presets import audio_config
from tests._factories import load_packaged_config
from tests._licences import use_licences
from tests._paths import REPO_ROOT
from tests.audio._helpers import case_request, clip, fake_backend, patch_runner, sine, voice_folder

from zvisiongenerator import audio_cli
from zvisiongenerator.audio_cli import _build_audio_parser
from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.downloads import require_checked
from zvisiongenerator.utils.audio_track import TRACK_UNREADABLE
from zvisiongenerator.utils.library_noise import MUTED_ERROR_TEXT
from zvisiongenerator.utils import audio_request as audio_request_module
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
    patched.ensure = MagicMock(return_value="ready")
    monkeypatch.setattr(audio_cli, "ensure_downloaded", patched.ensure)
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


class TestQuit:
    @pytest.mark.parametrize("outcome", ["stopped", "detached"])
    def test_a_quit_during_the_download_says_so_on_its_own_row_and_makes_nothing(self, env, monkeypatch, capsys, outcome):
        env.ensure.return_value = outcome

        run_cli(monkeypatch, "-m", "sa3", "--prompt", "rain", "-o", str(env.out))

        out = capsys.readouterr().out
        assert out.endswith("⏹ Quitting batch...\n") and not out.endswith("\n\n")
        env.backend.load.assert_not_called()

    def test_the_commands_line_is_followed_by_no_blank_row(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "stdin_is_interactive", lambda: True)
        monkeypatch.setattr(audio_cli.SkipSignal, "start", lambda self: None)

        run_cli(monkeypatch, "-m", "sa3", "--prompt", "rain", "-o", str(env.out))

        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("Commands:") and lines[1] != ""


class TestConsoleNoise:
    def test_warnings_and_log_lines_are_routed_through_the_live_line_with_the_known_noise_muted(self, env, monkeypatch):
        route = MagicMock()
        monkeypatch.setattr(audio_cli, "route_stderr_through_live_line", route)

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        route.assert_called_once_with(mute=MUTED_ERROR_TEXT)


class TestFlags:
    def test_lyrics_and_a_lyrics_file_cannot_be_given_together(self, capsys):
        with pytest.raises(SystemExit):
            _build_audio_parser().parse_args(["--lyrics", "a", "--lyrics-file", "b.txt"])

        assert "not allowed with argument" in capsys.readouterr().err

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


class TestRefusals:
    def test_a_flag_the_model_does_not_take_is_refused_by_name(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "--seconds", "30")

        assert "--seconds" in error and "YuE2 3B" in error

    @pytest.mark.parametrize(("flag", "model"), [("--lyrics", "sa3"), ("--lyrics", "higgs3")])
    def test_lyrics_are_refused_by_a_model_without_them(self, env, monkeypatch, capsys, flag, model):
        assert flag in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", flag, "[verse]")

    @pytest.mark.parametrize("model", ["yue2", "sa3"])
    def test_a_voice_is_refused_by_a_model_that_cannot_follow_one(self, env, monkeypatch, capsys, model):
        assert "--voice" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "x", "--voice", "me.wav", "--voice-text", "x", "--voice-consent")

    def test_a_voice_without_its_text_is_refused(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "me.wav"), "--voice-consent")

        assert "--voice-text" in error
        assert env.backend.load.call_count == 0

    def test_a_voice_without_consent_is_refused(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "me.wav"), "--voice-text", "Hello")

        assert "--voice-consent" in error
        assert env.ensure.call_count == 0 and env.backend.load.call_count == 0

    def test_a_cloned_take_needs_consent_too(self, env, monkeypatch, capsys, voices):
        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "cloned.flac"), "--voice-text", "Hello")

        assert "--voice-consent" in error

    def test_an_any_take_needs_no_consent(self, env, monkeypatch, voices):
        requests = make(env, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "any.flac"), "--voice-text", "Hello", monkeypatch=monkeypatch)

        assert requests[0].voice == VoiceRef(str(voices / "any.flac"), "Hello")

    def test_a_relative_voice_path_is_recorded_in_full(self, env, monkeypatch, voices):
        monkeypatch.chdir(voices)

        requests = make(env, "-m", "higgs3", "--prompt", "x", "--voice", "any.flac", "--voice-text", "Hello", monkeypatch=monkeypatch)

        assert requests[0].voice == VoiceRef(str((voices / "any.flac").resolve()), "Hello")

    def test_ffmpeg_is_asked_for_before_the_recording_is_read(self, env, monkeypatch, capsys, voices):
        """A machine without ffmpeg is offered the install, not a consent refusal or a probe failure from the missing tool."""
        probes: list[str] = []

        def declined() -> None:
            raise SystemExit(1)

        monkeypatch.setattr(audio_cli, "ensure_ffmpeg", declined)
        monkeypatch.setattr(audio_cli, "voice_needs_consent", lambda path: probes.append(str(path)) or True)
        monkeypatch.setattr(audio_request_module, "probe_track_seconds", lambda path: probes.append(str(path)) or 4.0)

        with pytest.raises(SystemExit) as exit_:
            run_cli(monkeypatch, "-m", "higgs3", "--prompt", "x", "--voice", str(voices / "me.wav"), "--voice-text", "Hello")

        assert exit_.value.code == 1
        assert probes == []
        assert "--voice-consent" not in capsys.readouterr().err

    def test_a_voice_that_is_not_there_is_refused(self, env, monkeypatch, capsys, tmp_path):
        assert "gone.wav" in refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(tmp_path / "gone.wav"), "--voice-text", "x", "--voice-consent")

    def test_a_voice_that_is_not_an_audio_file_is_refused_by_name(self, env, monkeypatch, capsys, tmp_path):
        notes = tmp_path / "notes.txt"
        notes.write_text("not audio", encoding="utf-8")

        def unreadable(path):
            raise ValueError(TRACK_UNREADABLE)

        monkeypatch.setattr(audio_request_module, "probe_track_seconds", unreadable)

        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(notes), "--voice-text", "x", "--voice-consent")

        assert f"{TRACK_UNREADABLE} in {notes}" in error and env.ensure.call_count == 0

    @pytest.mark.parametrize("flag", [["--voice-text", "x"], ["--voice-consent"]])
    def test_the_voice_flags_need_a_voice(self, env, monkeypatch, capsys, flag):
        assert f"{flag[0]} needs --voice" in refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", *flag)

    def test_a_recording_outside_the_allowed_length_is_refused_before_anything_loads(self, env, monkeypatch, capsys, tmp_path):
        short = tmp_path / "short.wav"
        write_pcm16_wav(sine(1.0, rate=24000), short, 24000)

        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "--voice", str(short), "--voice-text", "x", "--voice-consent")

        assert "3" in error and "30" in error and env.ensure.call_count == 0

    @pytest.mark.parametrize("model", ["yue2", "higgs3", "sa3"])
    def test_a_missing_or_blank_prompt_is_refused_by_flag_name(self, env, monkeypatch, capsys, model):
        assert "--prompt" in refuses(monkeypatch, capsys, "-m", model)
        assert "--prompt" in refuses(monkeypatch, capsys, "-m", model, "--prompt", "   ")

    def test_speech_over_10000_characters_is_refused(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "word " * 2001)

        assert "10,000" in error  # the limit is the config's, shown with a thousands separator
        assert env.ensure.call_count == 0

    @pytest.mark.parametrize("seconds", ["0.5", "381", "0", "-3", "nan"])
    def test_a_length_outside_the_models_range_is_refused(self, env, monkeypatch, capsys, seconds):
        error = refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--seconds", seconds)

        assert "--seconds" in error and "380" in error

    def test_a_lyrics_file_that_cannot_be_read_is_refused(self, env, monkeypatch, capsys, tmp_path):
        assert "gone.txt" in refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "--lyrics-file", str(tmp_path / "gone.txt"))

    @pytest.mark.parametrize("seed", ["-1", str(2**63)])
    def test_a_seed_no_model_takes_is_refused(self, env, monkeypatch, capsys, seed):
        assert "--seed" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "--seed", seed)
        env.backend.load.assert_not_called()

    def test_zero_runs_are_refused(self, env, monkeypatch, capsys):
        assert "--runs" in refuses(monkeypatch, capsys, "-m", "sa3", "--prompt", "x", "-r", "0")


class TestModels:
    def test_a_config_without_audio_models_says_so(self, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "audio_platform_key", lambda: "cuda")
        monkeypatch.setattr(audio_cli, "load_config", lambda: {**load_packaged_config(), "audio_model_presets": {}})

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x")

        assert "audio_model_presets" in error

    def test_a_model_is_required_and_the_choices_are_listed(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "--prompt", "x")

        assert "--model" in error and all(model in error for model in ("yue2", "higgs3", "sa3"))

    def test_an_unknown_model_is_refused_with_the_choices(self, env, monkeypatch, capsys):
        error = refuses(monkeypatch, capsys, "-m", "nope", "--prompt", "x")

        assert "nope" in error and all(model in error for model in ("yue2", "higgs3", "sa3"))


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

    @pytest.fixture(autouse=True)
    def _licence_records(self, monkeypatch, tmp_path):
        use_licences(monkeypatch, tmp_path, self.LICENCE)

    def test_the_terms_are_printed_before_a_missing_model_downloads(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", audio_config)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: False)
        env.ensure.side_effect = lambda *a, **k: print("DOWNLOAD STARTS") or "ready"

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        assert "By downloading you agree to these terms:" in out and "Stability AI Community License" in out
        assert out.index("By downloading you agree") < out.index("DOWNLOAD STARTS")

    def test_nothing_is_printed_for_a_model_that_is_already_downloaded(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", audio_config)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: True)

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        assert "By downloading" not in capsys.readouterr().out

    def test_a_model_without_a_record_prints_nothing(self, env, monkeypatch, capsys):
        monkeypatch.setattr(audio_cli, "load_config", audio_config)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: False)

        make(env, "-m", "yue2", "--prompt", "calm", monkeypatch=monkeypatch)

        assert "By downloading" not in capsys.readouterr().out


class TestMemoryWarning:
    def test_a_card_that_is_too_small_is_warned_before_the_licence_notice_and_the_download(self, env, monkeypatch, capsys):
        from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

        monkeypatch.setattr(audio_cli, "accelerator_memory_budget_or_none", lambda: MemoryBudget(8 * GIB, system_bytes=64 * GIB))
        order: list[str] = []
        monkeypatch.setattr(audio_cli, "print_licence_notice", lambda *a, **k: order.append("licence"))
        env.ensure.side_effect = lambda *a, **k: order.append("download") or "ready"

        with pytest.warns(UserWarning, match="Higgs TTS 3 4B.*8.2 GB.*8 GB") as warned:
            make(env, "-m", "higgs3", "--prompt", "Hello there.", monkeypatch=monkeypatch)

        assert len(warned) == 1 and order == ["licence", "download"]

    def test_a_card_that_fits_is_not_warned(self, env, monkeypatch, recwarn):
        from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

        monkeypatch.setattr(audio_cli, "accelerator_memory_budget_or_none", lambda: MemoryBudget(24 * GIB, system_bytes=64 * GIB))

        make(env, "-m", "higgs3", "--prompt", "Hello there.", monkeypatch=monkeypatch)

        assert not [w for w in recwarn if "GPU" in str(w.message)]


class TestDownload:
    def test_the_models_files_are_fetched_before_it_loads(self, env, monkeypatch):
        order: list[str] = []
        env.ensure.side_effect = lambda *a, **k: order.append("download") or "ready"
        env.backend.load.side_effect = lambda *a, **k: order.append("load")

        make(env, "-m", "sa3", "--prompt", "rain", monkeypatch=monkeypatch)

        assert order == ["download", "load"]
        assert [target.repo_id for target in env.ensure.call_args.args[0]][0] == "Comfy-Org/stable-audio-3"

    @pytest.mark.parametrize("outcome", ["stopped", "detached"])
    def test_quitting_during_the_download_makes_nothing(self, env, monkeypatch, outcome):
        env.ensure.return_value = outcome

        run_cli(monkeypatch, "-m", "sa3", "--prompt", "rain", "-o", str(env.out))

        env.backend.load.assert_not_called()


class TestFailures:
    def test_running_out_of_memory_names_the_model(self, env, monkeypatch, capsys):
        import torch

        env.backend.generate.side_effect = torch.OutOfMemoryError("out of memory")

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "-o", str(env.out))

        assert "YuE2 3B" in error and "GPU memory" in error
        env.backend.close.assert_called_once()

    def test_hugging_face_out_of_reach_ends_with_a_message_about_the_connection(self, env, monkeypatch, capsys):
        env.ensure.return_value = "unchecked"
        monkeypatch.setattr(audio_cli, "require_checked", functools.partial(require_checked, present=lambda target: False))

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "-o", str(env.out))

        assert "Hugging Face" in error and "YuE2 3B" in error
        env.backend.load.assert_not_called()

    def test_files_missing_after_a_download_that_was_checked_do_not_blame_the_connection(self, env, monkeypatch, capsys):
        monkeypatch.setattr("zvisiongenerator.audio_runner.target_present_offline", lambda target: False)

        error = refuses(monkeypatch, capsys, "-m", "yue2", "--prompt", "x", "-o", str(env.out))

        assert "YuE2 3B is not downloaded" in error and "Hugging Face" not in error
        env.backend.load.assert_not_called()

    def test_a_model_whose_files_are_not_in_the_cache_ends_with_its_own_message(self, env, monkeypatch, capsys):
        env.backend.load.side_effect = FileNotFoundError("Higgs TTS 3 needs bosonai/higgs-tts-3-4b, which is not downloaded.")

        assert "bosonai/higgs-tts-3-4b" in refuses(monkeypatch, capsys, "-m", "higgs3", "--prompt", "x", "-o", str(env.out))


class TestOnAMac:
    """On a Mac, Higgs runs as its ``mps`` entry names it: the MLX backend, and only the model repository to download."""

    def test_higgs_downloads_the_model_repository_only_and_runs_on_mlx(self, env, monkeypatch):
        from zvisiongenerator import audio_runner

        patch_runner(monkeypatch, env.backend, platform="mps")
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: False)

        make(env, "-m", "higgs3", "--prompt", "Hello there.", monkeypatch=monkeypatch)

        assert [target.repo_id for target in env.ensure.call_args.args[0]] == ["bosonai/higgs-tts-3-4b"]
        audio_runner.get_audio_backend.assert_called_once_with("higgs3_mlx")
        assert env.backend.load.call_args.args[0].family == "higgs3_mlx"


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
