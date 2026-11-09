"""Audio CLI — entry point for the ziv-audio command: music, speech and sound."""

from __future__ import annotations

import argparse
import logging
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from zvisiongenerator.audio_preflight import run_audio_preflight
from zvisiongenerator.audio_runner import QUIT_MESSAGE, run_audio_batch
from zvisiongenerator.backends import accelerator_memory_budget_or_none, audio_platform_key
from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef
from zvisiongenerator.downloads import DETACHED, STOPPED, CliDownloadReporter, DownloadFlights, describe_refusal, ensure_downloaded, require_checked
from zvisiongenerator.downloads.plans import audio_plan
from zvisiongenerator.enhance_cli import add_enhance_arguments, parse_enhance_args
from zvisiongenerator.job_controls import seed_range
from zvisiongenerator.licence_cli import print_licence_notice
from zvisiongenerator.utils.app_log import setup_logging
from zvisiongenerator.utils.audio_memory import audio_memory_fit, memory_warning_text
from zvisiongenerator.utils.audio_preset import AUDIO_KINDS, AudioPreset, parse_audio_presets
from zvisiongenerator.utils.audio_track import TRACK_UNREADABLE
from zvisiongenerator.utils.audio_request import check_seed, check_speech_request, find_audio_preset, resolve_audio_seconds, resolve_speaker
from zvisiongenerator.utils.audio_settings import (
    AUTO_SOUND_SECONDS,
    DEFAULT_MUSIC_SECONDS,
    DELIVERY_FIELDS,
    EMOTIONS,
    LEVELS,
    MANNERS,
    MUSIC_KEYS,
    PACES,
    SOUND_TYPES,
    TEMPO_MAX,
    TEMPO_MIN,
    VIBES,
    parse_delivery,
    parse_key,
    parse_sound_type,
    parse_tempo,
    parse_vibe,
)
from zvisiongenerator.utils.audio_voice import voice_needs_consent
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.ffmpeg import ensure_ffmpeg
from zvisiongenerator.utils.interactive import SkipSignal, stdin_is_interactive
from zvisiongenerator.utils.library_noise import MUTED_ERROR_TEXT, mute_library_warnings
from zvisiongenerator.utils.live_line import route_stderr_through_live_line
from zvisiongenerator.utils.music_mode import INSTRUMENTAL, music_mode
from zvisiongenerator.utils.prompt_enhance_audio import AudioEnhanceSettings, audio_enhance_context, caption_task
from zvisiongenerator.utils.voice_library import SavedVoice, find_voice, list_voices

logger = logging.getLogger(__name__)

_PROMPT_HELP = {
    "music": "Add --prompt: the style of the song, such as 'upbeat synth-pop, female vocal'.",
    "speech": "Add --prompt: the text to read.",
    "sound": "Add --prompt: a description of the sound, such as 'rain on a tin roof'.",
}


_NO_SECONDS_REASON = {
    "music": "sets its length from the lyrics",
    "speech": "reads the whole text",
    "sound": "sets its own length",
}


_DELIVERY_FLAGS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("emotion", EMOTIONS, "The emotion"),
    ("manner", MANNERS, "The manner"),
    ("pace", PACES, "The pace"),
    ("pitch", LEVELS, "The pitch"),
    ("expressiveness", LEVELS, "The expressiveness"),
)
_DELIVERY_DEFAULTS = {field: getattr(SpeechDelivery(), field) for field in DELIVERY_FIELDS}


_SPEECH_ENHANCE_REFUSAL = "Speech has no Enhance: its text is read as you wrote it. The Web UI's Script helper fits a script to a time or marks it up; drop {flag}"


_MARKS_SENTENCE = {"emotion": "Marks in the text such as [sad] change it from there on.", "manner": "[whisper] and [shout] in the text change it from there on."}


def _auto_lengths() -> str:
    """Return the Auto lengths of a sound for the help: the default and the length of each type, such as ``12 seconds (ambience 12, effect 5, one-shot 2)``."""
    by_type = ", ".join(f"{name} {seconds:g}" for name, seconds in AUTO_SOUND_SECONDS.items() if name != "auto")
    return f"{AUTO_SOUND_SECONDS['auto']:g} seconds long ({by_type} with --sound-type)"


def _speaker_choices(voices: Mapping[str, Sequence[str]]) -> str:
    """Return the sentence of ``--speaker``'s help that lists each speech model's voice ids, such as ``qwen3_tts: ryan, aiden.``; empty when none are known."""
    if not voices:
        return "the model's voices are listed when the id is wrong."
    return "Voices: " + "; ".join(f"{model}: {', '.join(ids)}" for model, ids in voices.items()) + "."


def _builtin_voices(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return the built-in voice ids of each speech model in *config*, for the help; empty when the config cannot be read."""
    try:
        return {preset.id: tuple(voice.id for voice in preset.voices) for preset in parse_audio_presets(config) if preset.voices}
    except ValueError:
        return {}  # the run reports a bad config itself


def _build_audio_parser(*, prog: str = "ziv-audio", voices: Mapping[str, Sequence[str]] | None = None) -> argparse.ArgumentParser:
    """Build the argument parser for ziv-audio; *voices* (the built-in voice ids by speech model) are listed in ``--speaker``'s help."""
    voices = voices or {}
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Z-Vision Audio Generator — songs and instrumentals, speech and sound.",
        epilog=f"Example usage: {prog} -m yue2 --prompt 'upbeat synth-pop, female vocal' --lyrics-file song.txt",
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Audio model id from the config (required), such as yue2, qwen3_tts or sa3.")
    parser.add_argument("--prompt", type=str, default=None, help="Required. The style of a song, the text to read, or the description of a sound.")
    parser.add_argument(
        "--kind",
        type=str,
        default=None,
        metavar="KIND",
        help=f"What to make: {', '.join(AUDIO_KINDS)} (default: the model's first kind). A model that makes one kind needs no --kind.",
    )
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
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help=(
            f"Length in seconds. Left out, a sound is {_auto_lengths()} and music from a model without lyrics is {DEFAULT_MUSIC_SECONDS:g}. Models that set their own length (YuE2 from the lyrics, speech from the text) refuse it."
        ),
    )
    parser.add_argument(
        "--tempo",
        type=str,
        default=None,
        metavar="BPM",
        help=f"The tempo of a song in beats per minute, {TEMPO_MIN} to {TEMPO_MAX} (default: the model's choice). Music only.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        metavar="ID",
        help=f"The key of a song (default: the model's choice). Music only. One of: {', '.join(MUSIC_KEYS)}; a trailing m is minor, # is a sharp and b a flat.",
    )
    parser.add_argument(
        "--sound-type",
        type=str,
        default=None,
        metavar="TYPE",
        help=f"The type of a sound, which sets its Auto length (default: auto). Sound only. One of: {', '.join(SOUND_TYPES)}.",
    )
    parser.add_argument(
        "--vibe",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Start the delivery of speech in a built-in voice from a Vibe, which sets all five delivery options at once; "
            f"--emotion, --manner, --pace, --pitch and --expressiveness then change one. One of: {', '.join(VIBES)}."
        ),
    )
    for field, choices, what in _DELIVERY_FLAGS:
        parser.add_argument(
            f"--{field}",
            type=str,
            default=None,
            metavar=field.upper(),
            help=f"{what} of speech in a built-in voice (default: {_DELIVERY_DEFAULTS[field]}). One of: {', '.join(choices)}. {_MARKS_SENTENCE.get(field, '')}".rstrip(),
        )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        metavar="ID",
        help=f"The built-in voice a speech model reads in (default: the model's first voice). Speech models with built-in voices only; {_speaker_choices(voices)} Use --voice to follow a recording instead.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        metavar="PATH|NAME",
        help="An audio recording to follow (3 to 30 seconds), or the name of a saved voice (save one in the Web UI). A recording needs --voice-text. Speech models that can clone a voice only.",
    )
    parser.add_argument("--voice-text", type=str, default=None, help="What the --voice recording says. Not for a saved voice.")
    parser.add_argument(
        "--voice-consent",
        action="store_true",
        default=False,
        help="Confirm the speaker of the --voice recording allows cloning their voice. Needed with a recording, except for a speech take this app made in a built-in voice; not for a saved voice.",
    )
    add_enhance_arguments(parser, mode="audio")
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


def _resolve_kind(parser: argparse.ArgumentParser, args: argparse.Namespace, preset: AudioPreset) -> str:
    """Return the kind to make: ``--kind``, else the model's first kind; refuse a kind the model does not make, naming the ones it does."""
    if args.kind is None:
        return preset.default_kind
    if args.kind not in AUDIO_KINDS:
        parser.error(f"--kind must be one of: {', '.join(AUDIO_KINDS)}.")
    if not preset.has_kind(args.kind):
        parser.error(f"--kind {args.kind}: {preset.label} makes {' and '.join(preset.kinds)}, not {args.kind}.")
    return args.kind


def _no_setting(preset: AudioPreset, kind: str, setting: str, flag: str) -> str:
    """Return the refusal of *flag* for a *kind* take: the model has no *setting* (said for the kind when the model also makes others)."""
    where = f" for {kind}" if len(preset.kinds) > 1 else ""
    return f"{preset.label} has no {setting}{where}; drop {flag}"


def _given_delivery_flags(args: argparse.Namespace) -> list[str]:
    """Return the delivery flags the command gives: ``--vibe``, then ``--emotion``... in the order of the settings."""
    given = ["--vibe"] if args.vibe is not None else []
    return [*given, *(f"--{field}" for field in DELIVERY_FIELDS if getattr(args, field) is not None)]


def _check_flags(parser: argparse.ArgumentParser, args: argparse.Namespace, preset: AudioPreset, kind: str) -> None:
    """Refuse a flag the model does not take as a *kind* take, and the voice flags that do not fit together."""
    lyrics_flag = "--lyrics" if args.lyrics is not None else "--lyrics-file"
    given_lyrics = args.lyrics is not None or args.lyrics_file is not None
    if given_lyrics and not preset.takes_lyrics(kind):
        parser.error(f"{preset.label} takes no lyrics; drop {lyrics_flag}")
    if args.seconds is not None and not preset.sets_length(kind):
        parser.error(f"{preset.label} {_NO_SECONDS_REASON[kind]}; drop --seconds")
    for flag, given, setting in (("--tempo", args.tempo is not None, "tempo"), ("--key", args.key is not None, "key")):
        if given and kind != "music":
            parser.error(_no_setting(preset, kind, setting, flag))
    if args.sound_type is not None and kind != "sound":
        parser.error(_no_setting(preset, kind, "sound type", "--sound-type"))
    if delivery_flags := _given_delivery_flags(args):
        if kind != "speech":
            parser.error(_no_setting(preset, kind, "delivery", delivery_flags[0]))
        if args.voice is not None:
            parser.error(f"Delivery applies to built-in voices; drop {delivery_flags[0]} to follow --voice")
    if args.voice is not None and not preset.follows_voice(kind):
        parser.error(f"{preset.label} cannot follow a voice; drop --voice")
    if args.speaker is not None and args.voice is not None:
        parser.error("--speaker picks a built-in voice; drop it to follow --voice")
    if args.speaker is not None and not preset.voices:
        parser.error(f"{preset.label} has no built-in voices; drop --speaker")
    if args.speaker is not None and kind != "speech":
        parser.error(_no_setting(preset, kind, "voice", "--speaker"))
    if args.voice is None:
        for given, flag in ((args.voice_text is not None, "--voice-text"), (args.voice_consent, "--voice-consent")):
            if given:
                parser.error(f"{flag} needs --voice")


def _parsed[T](parser: argparse.ArgumentParser, flag: str, parse: Callable[[object], T], value: object) -> T:
    """Return *value* read by the shared *parse* (the one the Web UI uses), or exit with its message under the name of *flag*."""
    try:
        return parse(value)
    except ValueError as e:
        parser.error(f"{flag}: {e}")


def _resolve_delivery(parser: argparse.ArgumentParser, args: argparse.Namespace, kind: str, voice: VoiceRef | None) -> SpeechDelivery | None:
    """Return how a speech take in a built-in voice is delivered: the neutral delivery, or ``--vibe``'s five values, with each flag given changing its own; ``None`` for a recording or any other kind."""
    if kind != "speech" or voice is not None:
        return None
    vibe = _parsed(parser, "--vibe", parse_vibe, args.vibe)
    start = VIBES[vibe] if vibe is not None else SpeechDelivery()
    given = {field: getattr(args, field) for field in DELIVERY_FIELDS}
    try:
        return parse_delivery({field: getattr(start, field) if value is None else value for field, value in given.items()})
    except ValueError as e:  # names the setting ("Emotion must be one of: ...")
        parser.error(str(e))


def _saved_voice_ref(parser: argparse.ArgumentParser, args: argparse.Namespace, voice: SavedVoice) -> VoiceRef:
    """Return the reference of the saved *voice*, refusing the flags that only a new recording needs (its transcript and consent are saved with it)."""
    if args.voice_text is not None:
        parser.error(f'--voice-text: the saved voice "{voice.name}" has its own transcript; drop it')
    if args.voice_consent:
        parser.error(f'--voice-consent: the saved voice "{voice.name}" was confirmed when it was saved; drop it')
    return voice.voice_ref()


def _unknown_voice_message(name: str, saved: Sequence[SavedVoice]) -> str:
    """Return the refusal of a ``--voice`` that is neither a file nor a saved voice, naming the saved voices."""
    listing = f"Saved voices: {', '.join(voice.name for voice in saved)}." if saved else "You have no saved voices yet."
    return f'No recording or saved voice named "{name}". {listing}'


def _recording_ref(parser: argparse.ArgumentParser, args: argparse.Namespace, path: Path) -> VoiceRef:
    """Return the reference of the recording at *path*, after checking it says what it says and is allowed to be cloned."""
    if args.voice_text is None or not args.voice_text.strip():
        parser.error("Add --voice-text: what the voice recording says.")
    if not args.voice_consent and voice_needs_consent(path):
        parser.error("Add --voice-consent: cloning a voice needs the speaker's permission.")
    return VoiceRef(str(path), args.voice_text.strip())


def _resolve_voice(parser: argparse.ArgumentParser, args: argparse.Namespace) -> VoiceRef | None:
    """Return the voice to follow: ``--voice`` as a recording when a file by that name exists, else as a saved voice (by name or slug); refuse one that is neither."""
    if args.voice is None:
        return None
    path = Path(args.voice).expanduser().resolve()  # recorded in the take's settings, so it must work from any folder
    if path.is_file():
        return _recording_ref(parser, args, path)
    if (saved := find_voice(args.voice)) is not None:
        return _saved_voice_ref(parser, args, saved)
    parser.error(_unknown_voice_message(args.voice, list_voices()))


def _resolve_speaker(parser: argparse.ArgumentParser, args: argparse.Namespace, preset: AudioPreset, kind: str) -> str | None:
    """Return the built-in voice a speech take without ``--voice`` reads in (the model's default when ``--speaker`` is not given), or ``None``; refuse an id the model lacks."""
    if kind != "speech" or args.voice is not None:
        return None
    try:
        return resolve_speaker(preset, args.speaker)
    except ValueError:
        parser.error(f"Unknown voice '{args.speaker}' for {preset.label}. Choose one of: {', '.join(voice.id for voice in preset.voices)}.")


def _resolve_seconds(parser: argparse.ArgumentParser, args: argparse.Namespace, preset: AudioPreset, kind: str, sound_type: str | None) -> float | None:
    """Return the length of a *kind* take (the default when not given), or ``None`` for a (model, kind) that sets its own; refuse a length outside the model's range."""
    try:
        return resolve_audio_seconds(args.seconds, preset, kind, sound_type, name="--seconds")
    except ValueError as e:
        parser.error(str(e))


def _resolve_enhance(parser: argparse.ArgumentParser, args: argparse.Namespace, kind: str) -> tuple[AudioEnhanceSettings | None, str | None]:
    """Return the Enhance options of a *kind* take and the task that applies them (``sound`` or ``style``), or ``(None, None)`` when the text is not enhanced.

    Speech is read as written, so Enhance is refused for it. ``--enhance-model`` only picks the enhancer of an Enhance that runs.
    """
    task = caption_task(kind)
    asked = [flag for flag, given in (("--enhance", args.enhance is not None), ("--enhance-note", args.enhance_note is not None), ("--enhance-format", args.enhance_format is not None)) if given]
    if asked and task is None:
        parser.error(_SPEECH_ENHANCE_REFUSAL.format(flag=asked[0]))
    parse_enhance_args(parser, args, mode="audio", task=task)
    if args.enhance is None:
        if args.enhance_model is not None:
            parser.error("--enhance-model picks the enhancer for --enhance; add --enhance or drop it")
        return None, None
    return args.enhance, task


def _is_sung(request: AudioGenerationRequest, preset: AudioPreset, platform_key: str) -> bool:
    """Return whether the take is a song with words the model sings: the model takes lyrics and they make a song, not an instrumental."""
    if not preset.takes_lyrics(request.kind):
        return False
    sections = preset.settings_for(platform_key).get("instrumental_sections")
    if not sections:
        return bool(request.lyrics.strip())
    return music_mode(request.lyrics, sections)[1] != INSTRUMENTAL


def _enhance_seed(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Return the seed of the rewrite: ``--seed`` when given (the same command writes the same text), else a random one."""
    return args.seed if args.seed is not None else random.randint(*seed_range(config))


def _announce_music(request: AudioGenerationRequest, preset: AudioPreset, platform_key: str) -> None:
    """Say whether the lyrics make a song or an instrumental, when the preset gives the section list that decides it."""
    settings = preset.settings_for(platform_key)
    sections = settings.get("instrumental_sections")
    if not preset.takes_lyrics(request.kind) or not sections:
        return
    _, kind, count = music_mode(request.lyrics, sections)
    print(f"Making an instrumental of {count} sections." if kind == INSTRUMENTAL else "Making a song from the lyrics.", flush=True)


def _warn_if_memory_short(preset: AudioPreset, platform_key: str) -> None:
    """Warn when the model's recorded GPU peak is too large or tight for this machine; the run goes on, as the download may still be wanted."""
    text = memory_warning_text(preset.label, audio_memory_fit(preset, platform_key, accelerator_memory_budget_or_none()))
    if text:
        warnings.warn(text, stacklevel=2)


def _download_model(preset: AudioPreset, skip: SkipSignal, *, voice: bool = False) -> bool:
    """Fetch the model's files, with a live progress line on a terminal; return ``False`` when the user quit meanwhile.

    A take that follows a voice recording (*voice*, from ``--voice``) also fetches the model's cloning files.

    Raises:
        RuntimeError: When Hugging Face could not be reached and the files are not on disk.
    """
    targets = audio_plan(preset, voice=voice)
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
    try:
        config, config_error = load_config(), None
    except ValueError as e:  # reported below, once the arguments have been read
        config, config_error = {}, str(e)
    parser = _build_audio_parser(prog=prog, voices=_builtin_voices(config))
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
    if config_error is not None:
        parser.error(config_error)
    preset = _find_preset(parser, config, args.model)
    try:
        platform_key = audio_platform_key()
    except RuntimeError as e:
        parser.error(str(e))
    preset = preset.on_platform(platform_key)
    kind = _resolve_kind(parser, args, preset)
    prompt = (args.prompt or "").strip()
    if not prompt:
        parser.error(_PROMPT_HELP[kind])
    _check_flags(parser, args, preset, kind)
    enhance, enhance_task = _resolve_enhance(parser, args, kind)
    lyrics = _read_lyrics(parser, args)
    # Before anything reads a recording: ffmpeg is what reads it, and a machine without it is offered the install first.
    ensure_ffmpeg()
    voice = _resolve_voice(parser, args)
    sound_type = _parsed(parser, "--sound-type", parse_sound_type, args.sound_type) if kind == "sound" else None
    seconds = _resolve_seconds(parser, args, preset, kind, sound_type)
    request = AudioGenerationRequest(
        preset=preset.id,
        kind=kind,
        prompt=args.prompt if kind == "speech" else prompt,
        lyrics=lyrics,
        seconds=seconds,
        requested_seconds=seconds if args.seconds is not None else None,
        voice=voice,
        speaker=_resolve_speaker(parser, args, preset, kind),
        tempo=_parsed(parser, "--tempo", parse_tempo, args.tempo) if kind == "music" else None,
        key=_parsed(parser, "--key", parse_key, args.key) if kind == "music" else None,
        delivery=_resolve_delivery(parser, args, kind, voice),
        sound_type=sound_type,
    )
    if kind == "speech":
        try:
            check_speech_request(request, preset.limits)
        except ValueError as e:
            # The library's message cannot say which file; the command line knows.
            parser.error(f"{e} in {args.voice}" if str(e) == TRACK_UNREADABLE else str(e))
    sings = _is_sung(request, preset, platform_key) if kind == "music" else False
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
        if enhance is not None:
            # Before the audio model is fetched or loaded, so the enhancer and the model are never resident together.
            context = audio_enhance_context(request, task=enhance_task, sings=sings)
            preflight = run_audio_preflight(
                request,
                enhance,
                task=enhance_task,
                family=preset.family,
                context=context,
                own_text=args.prompt,
                config=config,
                seed=_enhance_seed(args, config),
                enhance_model=args.enhance_model,
                control=skip,
            )
            if preflight.cancelled:
                print(QUIT_MESSAGE)
                return
            request = preflight.request
        if _download_model(preset, skip, voice=voice is not None):
            run_audio_batch(request, preset, config, runs=args.runs, seed=args.seed, output_dir=output, skip_signal=skip)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
        logger.exception("Audio generation failed")
        parser.error(describe_refusal(e) or str(e))
    finally:
        skip.stop()


if __name__ == "__main__":
    main()
