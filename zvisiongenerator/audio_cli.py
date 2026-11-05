"""Audio CLI — entry point for the ziv-audio command: music, speech and sound."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

from zvisiongenerator.audio_runner import QUIT_MESSAGE, run_audio_batch
from zvisiongenerator.backends import accelerator_memory_budget_or_none, audio_platform_key
from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.downloads import DETACHED, STOPPED, CliDownloadReporter, DownloadFlights, audio_plan, describe_refusal, ensure_downloaded, require_checked
from zvisiongenerator.licence_cli import print_licence_notice
from zvisiongenerator.utils.app_log import setup_logging
from zvisiongenerator.utils.audio_memory import audio_memory_fit, memory_warning_text
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.audio_track import TRACK_UNREADABLE
from zvisiongenerator.utils.audio_request import DEFAULT_SOUND_SECONDS, check_seed, check_speech_request, find_audio_preset, resolve_sound_seconds
from zvisiongenerator.utils.audio_voice import voice_needs_consent
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.ffmpeg import ensure_ffmpeg
from zvisiongenerator.utils.interactive import SkipSignal, stdin_is_interactive
from zvisiongenerator.utils.library_noise import MUTED_ERROR_TEXT, mute_library_warnings
from zvisiongenerator.utils.live_line import route_stderr_through_live_line
from zvisiongenerator.utils.music_mode import INSTRUMENTAL, music_mode

logger = logging.getLogger(__name__)

_PROMPT_HELP = {
    "music": "Add --prompt: the style of the song, such as 'upbeat synth-pop, female vocal'.",
    "speech": "Add --prompt: the text to read.",
    "sound": "Add --prompt: a description of the sound, such as 'rain on a tin roof'.",
}


_NO_SECONDS_REASON = {
    "music": "sets the length from the lyrics",
    "speech": "reads the whole text",
    "sound": "sets its own length",
}


def _build_audio_parser(*, prog: str = "ziv-audio") -> argparse.ArgumentParser:
    """Build the argument parser for ziv-audio."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Z-Vision Audio Generator — songs and instrumentals, speech and sound.",
        epilog=f"Example usage: {prog} -m yue2 --prompt 'upbeat synth-pop, female vocal' --lyrics-file song.txt",
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Audio model id from the config (required), such as yue2, higgs3 or sa3.")
    parser.add_argument("--prompt", type=str, default=None, help="Required. The style of a song, the text to read, or the description of a sound.")
    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of takes.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for random).")
    parser.add_argument("-o", "--output", type=str, default=".", help="Output directory.")
    lyrics = parser.add_mutually_exclusive_group()
    lyrics.add_argument(
        "--lyrics",
        type=str,
        default=None,
        help="Lyrics of a song, with [verse] and [chorus] tags. Without lyrics, or with only [section] tags, the take is an instrumental; the tags set its length. Music models only.",
    )
    lyrics.add_argument("--lyrics-file", type=str, default=None, metavar="PATH", help="Read the lyrics from a text file. Same as --lyrics.")
    parser.add_argument("--seconds", type=float, default=None, help=f"Length of a sound in seconds (default: {DEFAULT_SOUND_SECONDS:g}). Models that set their own length refuse it.")
    parser.add_argument(
        "--voice", type=str, default=None, metavar="PATH", help="An audio recording of the voice to follow, 3 to 30 seconds. Speech models that can clone a voice only. Needs --voice-text."
    )
    parser.add_argument("--voice-text", type=str, default=None, help="What the --voice recording says.")
    parser.add_argument(
        "--voice-consent",
        action="store_true",
        default=False,
        help="Confirm the speaker of --voice allows cloning their voice. Needed with --voice, except for a speech take this app made in the model's own voice.",
    )
    return parser


def _find_preset(parser: argparse.ArgumentParser, config: dict[str, Any], model: str | None) -> AudioPreset:
    """Return the preset *model* names, or exit with what is wrong: no models configured, no id, or an unknown id."""
    try:
        return find_audio_preset(config, model, missing="--model is required")
    except ValueError as e:
        parser.error(str(e))


def _read_lyrics(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Return the lyrics from ``--lyrics`` or ``--lyrics-file``, or an empty string when neither is given."""
    if args.lyrics_file is None:
        return args.lyrics or ""
    try:
        return Path(args.lyrics_file).expanduser().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        parser.error(f"Could not read the lyrics file {args.lyrics_file}: {e}")


def _check_flags(parser: argparse.ArgumentParser, args: argparse.Namespace, preset: AudioPreset) -> None:
    """Refuse a flag the model does not take, and the voice flags that do not fit together."""
    lyrics_flag = "--lyrics" if args.lyrics is not None else "--lyrics-file"
    given_lyrics = args.lyrics is not None or args.lyrics_file is not None
    if given_lyrics and not preset.takes_lyrics:
        parser.error(f"{preset.label} takes no lyrics; drop {lyrics_flag}")
    if args.seconds is not None and not preset.sets_length:
        parser.error(f"{preset.label} {_NO_SECONDS_REASON[preset.kind]}; drop --seconds")
    if args.voice is not None and not preset.follows_voice:
        parser.error(f"{preset.label} cannot follow a voice; drop --voice")
    if args.voice is None:
        for given, flag in ((args.voice_text is not None, "--voice-text"), (args.voice_consent, "--voice-consent")):
            if given:
                parser.error(f"{flag} needs --voice")


def _resolve_voice(parser: argparse.ArgumentParser, args: argparse.Namespace) -> VoiceRef | None:
    """Return the voice to follow, after checking the recording exists, says what it says, and is allowed to be cloned."""
    if args.voice is None:
        return None
    path = Path(args.voice).expanduser().resolve()  # recorded in the take's settings, so it must work from any folder
    if not path.is_file():
        parser.error(f"Voice recording not found: {path}")
    if args.voice_text is None or not args.voice_text.strip():
        parser.error("Add --voice-text: what the voice recording says.")
    if not args.voice_consent and voice_needs_consent(path):
        parser.error("Add --voice-consent: cloning a voice needs the speaker's permission.")
    return VoiceRef(str(path), args.voice_text.strip())


def _resolve_seconds(parser: argparse.ArgumentParser, args: argparse.Namespace, preset: AudioPreset) -> float | None:
    """Return the length of a sound (the default when not given), or ``None`` for a model that sets its own; refuse a length outside the model's range."""
    try:
        return resolve_sound_seconds(args.seconds, preset, name="--seconds")
    except ValueError as e:
        parser.error(str(e))


def _announce_music(request: AudioGenerationRequest, preset: AudioPreset, platform_key: str) -> None:
    """Say whether the lyrics make a song or an instrumental, when the preset gives the section list that decides it."""
    settings = preset.settings_for(platform_key)
    sections = settings.get("instrumental_sections")
    if preset.kind != "music" or not sections:
        return
    _, kind, count = music_mode(request.lyrics, sections)
    print(f"Making an instrumental of {count} sections." if kind == INSTRUMENTAL else "Making a song from the lyrics.", flush=True)


def _warn_if_memory_short(preset: AudioPreset, platform_key: str) -> None:
    """Warn when the model's recorded GPU peak is too large or tight for this machine; the run goes on, as the download may still be wanted."""
    text = memory_warning_text(preset.label, audio_memory_fit(preset, platform_key, accelerator_memory_budget_or_none()))
    if text:
        warnings.warn(text, stacklevel=2)


def _download_model(preset: AudioPreset, skip: SkipSignal) -> bool:
    """Fetch the model's files, with a live progress line on a terminal; return ``False`` when the user quit meanwhile.

    Raises:
        RuntimeError: When Hugging Face could not be reached and the files are not on disk.
    """
    targets = audio_plan(preset)
    reporter = CliDownloadReporter()
    try:
        outcome = ensure_downloaded(targets, flights=DownloadFlights(), report=reporter, cancelled=lambda: skip.pending() == "quit")
    finally:
        reporter.close()
    require_checked(outcome, targets, preset.label)
    if outcome in (STOPPED, DETACHED):
        print(QUIT_MESSAGE)  # the reporter ended its line, so this starts on a fresh row
        return False
    return True


def main(*, prog: str = "ziv-audio") -> None:
    """Entry point for ziv-audio CLI."""
    parser = _build_audio_parser(prog=prog)
    args = parser.parse_args()
    # Warnings and log lines end the progress line before they print; set up first, so the log handlers write through it.
    route_stderr_through_live_line(mute=MUTED_ERROR_TEXT)
    mute_library_warnings()
    setup_logging("audio")

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    try:
        check_seed(args.seed)
    except ValueError as e:
        parser.error(f"--seed: {e}")
    try:
        config = load_config()
    except ValueError as e:
        parser.error(str(e))
    preset = _find_preset(parser, config, args.model)
    try:
        platform_key = audio_platform_key()
    except RuntimeError as e:
        parser.error(str(e))
    preset = preset.on_platform(platform_key)
    prompt = (args.prompt or "").strip()
    if not prompt:
        parser.error(_PROMPT_HELP[preset.kind])
    _check_flags(parser, args, preset)
    lyrics = _read_lyrics(parser, args)
    # Before anything reads a recording: ffmpeg is what reads it, and a machine without it is offered the install first.
    ensure_ffmpeg()
    voice = _resolve_voice(parser, args)
    request = AudioGenerationRequest(
        preset=preset.id,
        kind=preset.kind,
        prompt=args.prompt if preset.kind == "speech" else prompt,
        lyrics=lyrics,
        seconds=_resolve_seconds(parser, args, preset),
        voice=voice,
    )
    if preset.kind == "speech":
        try:
            check_speech_request(request, preset.limits)
        except ValueError as e:
            # The library's message cannot say which file; the command line knows.
            parser.error(f"{e} in {args.voice}" if str(e) == TRACK_UNREADABLE else str(e))
    output = str(Path(args.output).expanduser())
    _announce_music(request, preset, platform_key)

    # Before anything downloads: a card that is too small should be known before the model's files are fetched.
    _warn_if_memory_short(preset, platform_key)
    # The terms of a model that has them are named before its files download.
    try:
        print_licence_notice(preset.id, lambda: audio_plan(preset))
    except ValueError as e:
        parser.error(str(e))

    skip = SkipSignal()
    skip.start()
    if stdin_is_interactive():
        print("Commands: [n] skip  [q] quit  [p] pause  [r] repeat")
    try:
        if _download_model(preset, skip):
            run_audio_batch(request, preset, config, runs=args.runs, seed=args.seed, output_dir=output, skip_signal=skip)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
        logger.exception("Audio generation failed")
        parser.error(describe_refusal(e) or str(e))
    finally:
        skip.stop()


if __name__ == "__main__":
    main()
