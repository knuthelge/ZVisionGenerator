"""Rewrite a video prompt as a scene and an action: word plans, chat messages, output parsing and the model run (no model I/O)."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.prompt_compose import strip_quoted_lines
from zvisiongenerator.utils.prompt_enhance import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DETAILS_AXIS,
    MOOD_AXIS,
    NO_USABLE_MESSAGE,
    STYLE_AXIS,
    EnhanceContext,
    EnhanceSettings,
    LengthPlan,
    action_rules,
    clean_output,
    context_rules,
    direction_rules,
    duration_beats,
    generate_raw,
    is_unusable,
    length_instruction,
    normalize_text,
    perception_rule,
    plan_length,
    resolve_parts,
    strip_reasoning,
    validate_settings,
    word_count,
)

if TYPE_CHECKING:
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer

PAIR_ACTION_FLOOR_WORDS = 12
SCENE_HEAVY_RATIO = 0.8
SCENE_HEAVY_WARNING = "The scene uses most of the word budget; the action may be cut short."
PARTS_FALLBACK_WARNING = "parts=action needs a scene to keep; with no scene the prompt is split into scene and action (parts=both)."
PAIR_FALLBACK_WARNING = "The enhancer did not return a scene and an action in the requested format; its rewrite is used as the action and the scene is left as written."

# A marker starts a line or follows whitespace, ``*``, ``#`` or ``-`` (so ``**Scene:**`` and ``## Scene:`` both count).
_MARKER_LEAD = r"(?<![^\s*#\-])[*#\-\s]*"
_MARKER_TAIL = r"\**\s*:\**"
_SCENE_RE = re.compile(f"{_MARKER_LEAD}SCENE{_MARKER_TAIL}", re.I)
_ACTION_RE = re.compile(f"{_MARKER_LEAD}(?:ACTION|PROMPT){_MARKER_TAIL}", re.I)
_PARTIAL_TAIL_RE = re.compile(r"(?<![^\s*#\-])([*#\-]*)\s*([A-Za-z]*)\s*[*#\-]*$")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")
_QUOTED_RE = re.compile(r'"([^"]+)"|“([^”]+)”')
_SCENE_WORDS = ("SCENE", "ACTION", "PROMPT")
_ACTION_WORDS = ("ACTION", "PROMPT")
_NON_WORD_RE = re.compile(r"\W+")

CONTINUE_NATURALLY = "Continue the scene naturally."
# Word-set Jaccard similarity from which a rewrite counts as a repeat of the previous prompt.
_PREVIOUS_CLOSENESS = 0.9
# A quoted line of at least this many words also counts as repeated when a longer new line contains it.
_MIN_CONTAINED_LINE_WORDS = 3

# Action-only rewrites see the scene as context; small models tend to repeat its details unless told exactly how to refer to it.
_KEEP_SCENE_OUT_RULE = (
    'Refer to people and things already in the scene by a pronoun or a short reference ("she", "the dog"). Never repeat their clothing, appearance or the setting; the scene already says it.'
)


@dataclass(frozen=True)
class EnhancePairResult:
    """The enhanced scene and action of a video prompt, whether the length request was clamped, and an optional notice.

    ``scene`` is ``""`` when the scene was not rewritten (an action-only run, or the format fallback): callers then keep
    the user's scene and never copy it into an enhanced field. ``format_fallback`` is set when the model did not answer
    in the requested scene and action format and its rewrite was used as the action; ``warning`` then ends with
    ``PAIR_FALLBACK_WARNING``.
    """

    scene: str
    prompt: str
    clamped: bool
    warning: str | None = None
    format_fallback: bool = False


@dataclass(frozen=True)
class PairLengthPlan:
    """The word budget of a scene and action rewrite: the total, each part, and whether the scene crowds out the action."""

    total: LengthPlan
    scene: LengthPlan
    action: LengthPlan
    scene_heavy: bool


def repeats_source(action: str, *, previous: str) -> bool:
    """Return whether *action* repeats the earlier prompt *previous*: one of its quoted lines, or nearly all of its words.

    Quoted lines are compared by their words only (case, punctuation and spacing do not matter). A line counts
    as repeated when a quoted span of *action* equals it or, for lines of three or more words, contains it.
    Closeness is the word-set Jaccard similarity of *action* and *previous* without its quoted sentences.
    """
    action_spans = [_word_key(straight or curly) for straight, curly in _QUOTED_RE.findall(action)]
    for straight, curly in _QUOTED_RE.findall(previous):
        line = _word_key(straight or curly)
        if line and any(_contains_line(span, line) for span in action_spans):
            return True
    action_words = set(_word_key(action).split())
    previous_words = set(_word_key(strip_quoted_lines(previous)).split())
    union = action_words | previous_words
    return bool(union) and len(action_words & previous_words) / len(union) >= _PREVIOUS_CLOSENESS


def _word_key(text: str) -> str:
    """Return *text* lowercased with punctuation and whitespace runs collapsed to single spaces."""
    return _NON_WORD_RE.sub(" ", text.lower()).strip()


def _contains_line(span: str, line: str) -> bool:
    """Return whether the quoted *span* is *line* or, for a longer line, contains it as whole words."""
    return span == line or (len(line.split()) >= _MIN_CONTAINED_LINE_WORDS and f" {line} " in f" {span} ")


def plan_pair_length(scene_words: int, prompt_words: int, length: str, *, ceiling: int, length_cfg: dict[str, Any] | None = None) -> PairLengthPlan:
    """Split the word budget of a video rewrite between its scene and its action.

    The total follows ``plan_length`` over both inputs. The scene gets its share of the total (a third when
    either input is empty); the action gets the rest, never fewer than ``PAIR_ACTION_FLOOR_WORDS`` words, which
    are taken from the scene. Both upper bounds together stay within *ceiling*, the scene trimmed first.
    """
    total = plan_length(scene_words + prompt_words, length, ceiling=ceiling, length_cfg=length_cfg)
    if scene_words > 0 and prompt_words > 0:
        scene_target = round(total.target * scene_words / (scene_words + prompt_words))
    else:
        scene_target = round(total.target / 3)
    action_target = max(total.target - max(1, scene_target), min(PAIR_ACTION_FLOOR_WORDS, ceiling))
    scene_target = max(1, total.target - action_target)
    action = _part_plan(length, action_target, hi_cap=max(ceiling, action_target), clamped=total.clamped)
    scene = _part_plan(length, scene_target, hi_cap=max(1, ceiling - action.hi), clamped=total.clamped)
    return PairLengthPlan(total=total, scene=scene, action=action, scene_heavy=scene_words >= SCENE_HEAVY_RATIO * ceiling)


def _part_plan(length: str, target: int, *, hi_cap: int, clamped: bool) -> LengthPlan:
    """Return the plan for one part: a ±15% range (at least 3 words) around *target*, capped at *hi_cap*."""
    span = max(3, round(target * 0.15))
    hi = max(1, min(target + span, hi_cap))
    target = min(target, hi)
    return LengthPlan(length=length, target=target, lo=max(1, target - span), hi=hi, clamped=clamped)


def build_pair_messages(scene: str, prompt: str, settings: EnhanceSettings, *, plan: PairLengthPlan, context: EnhanceContext | None = None) -> list[dict[str, str]]:
    """Build the system + user chat messages that ask for a ``SCENE:`` and an ``ACTION:`` part.

    Style, mood and details shape the scene; motion and sound shape the action; the direction applies to both.
    With a reference image the scene stays short and the action describes the change.
    """
    context = context or EnhanceContext()
    shorter = settings.length == "shorter" and not plan.total.clamped
    action_role = "ACTION: what happens, in time order, with camera moves"
    action_role += ", sound and quoted dialogue." if context.audio else "."
    rules = [
        "EVERY DETAIL IS IMPORTANT: keep the user's subject, intent and every element they mention.",
        "Never add new people, animals, characters or major objects.",
        perception_rule("video", context),
        *context_rules("video", replace(context, reference_image=False, seconds=None if context.extends else context.seconds)),
        "SCENE: who and where, their look, the style and the camera setup. No events, no camera moves, no quoted speech.",
        action_role,
        *(_extend_pair_rules(context, scene=scene) if context.extends else [_pair_workflow_rule(context)]),
        f"For the SCENE: {STYLE_AXIS.option(settings.style).instruction}",
        f"For the SCENE: {MOOD_AXIS.option(settings.mood).instruction}",
    ]
    if settings.details:
        aspects = ", ".join(DETAILS_AXIS.option(slug).instruction for slug in settings.details)
        rules.append(f"For the SCENE: keep these aspects while cutting: {aspects}." if shorter else f"For the SCENE: add detail on: {aspects}.")
    rules.extend(action_rules(settings, context, for_pair=True))
    rules.extend(direction_rules(settings, both_parts=True))
    if not prompt.strip() and not context.extends:
        rules.append("The Prompt is (none): write an ACTION that suits the scene.")
    rules.append(length_instruction(plan.scene, "SCENE length"))
    rules.append(length_instruction(plan.action, "ACTION length"))
    if settings.length == "same" or plan.total.clamped:
        rules.append("Apply the style, mood and details; don't return the input unchanged.")
    system = "You rewrite prompts for a text-to-video model. The user gives a Scene and a Prompt; you write a SCENE and an ACTION.\nRules:\n" + "\n".join(f"- {rule}" for rule in rules)
    system += "\nReply with `SCENE:` then `ACTION:`, each followed by plain prose: no preamble, no headings, no quotes around a part."
    if context.extends and scene.strip():
        user = f"Scene: {scene.strip()}\nWhat happens next: {_extend_user_message(prompt, context)}"
    elif context.extends:
        user = _extend_user_message(prompt, context)
    else:
        user = f"Scene: {scene.strip() or '(none)'}\nPrompt: {prompt.strip() or '(none)'}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _continues_naturally(prompt: str, context: EnhanceContext) -> bool:
    """Return whether an extension has no new direction: the prompt is blank or still the old prompt of a source without a scene."""
    return not prompt.strip() or (context.prompt_is_original and bool(context.extract_from))


def _extend_user_message(prompt: str, context: EnhanceContext) -> str:
    """Return the user message of an extension: the user's prompt, or ``CONTINUE_NATURALLY`` when there is no new direction."""
    return CONTINUE_NATURALLY if _continues_naturally(prompt, context) else prompt.strip()


def _previous_rule(context: EnhanceContext) -> str | None:
    """Return the "what just happened" line for an extension, or None without a previous prompt."""
    if not (context.previous or "").strip():
        return None
    return f"What just happened (do not repeat its actions or quoted lines): {context.previous.strip()}"


def _extend_rules(context: EnhanceContext) -> list[str]:
    """Return the rules for writing the part that follows an existing clip."""
    rules = [
        "Start from where the clip ends: the people and objects are in their final positions and poses.",
        "The people in the scene are present; describe speakers as before.",
    ]
    if context.orientation is not None:
        rules.append(f"Frame: {context.orientation}; compose for that shape.")
    if context.new_seconds is not None:
        window = f" of a {max(1, round(context.seconds))} second window" if context.seconds is not None else ""
        beats = duration_beats(context.new_seconds)
        rules.append(f"Write the next action in time order, fitted to {max(1, round(context.new_seconds))} seconds{window}: {beats}.")
    else:
        rules.append("Write the next action in time order.")
    return rules


def _extend_pair_rules(context: EnhanceContext, *, scene: str = "") -> list[str]:
    """Return the scene-and-action rules for an extension: the scene is the user's, or extracted from the earlier prompt when blank."""
    rules = []
    has_scene = bool(scene.strip())
    if has_scene:
        rules.append("The user's Scene is the clip's scene: rewrite it, keeping its facts, and write the ACTION for what happens next. Do not repeat the earlier action or quoted lines.")
    elif context.extract_from:
        rules.append(
            f"Extract the scene (who, where, look, style, camera) from this earlier prompt: {context.extract_from.strip()}. "
            "Write the action for what happens next. Do not repeat the earlier prompt's actions or quoted lines."
        )
    previous = _previous_rule(context)
    if previous is not None and (has_scene or (context.previous or "").strip() != (context.extract_from or "").strip()):
        rules.append(previous)
    return [*rules, *_extend_rules(context)]


def _action_only_messages(preamble: list[str], situation: list[str], settings: EnhanceSettings, plan: LengthPlan, context: EnhanceContext, user: str) -> list[dict[str, str]]:
    """Build the system + user chat messages of a rewrite that writes only the action beside a kept scene.

    Args:
        preamble: The opening lines of the system message: the task and the kept scene.
        situation: The rules about the clip the action is written for, between the shared opening rules and the motion, sound and direction rules.
        user: The user message.
    """
    rules = [
        "Keep the user's intent and every element they mention.",
        "Never add new people, animals, characters or major objects.",
        _KEEP_SCENE_OUT_RULE,
        perception_rule("video", context),
        *situation,
        *action_rules(settings, context),
        *direction_rules(settings),
        length_instruction(plan, "Length"),
    ]
    system = "\n".join(preamble) + "\nRules:\n" + "\n".join(f"- {rule}" for rule in rules)
    system += "\nOutput only the action as plain prose: no preamble, no headings, no scene description, no quotes around it."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_extend_messages(scene: str, prompt: str, settings: EnhanceSettings, *, plan: LengthPlan, context: EnhanceContext) -> list[dict[str, str]]:
    """Build the system + user chat messages that ask for only the action that follows an existing clip.

    The *scene* is context the model must not restate, and the source's prompt (``context.previous``) is "what
    just happened"; neither is in the user message, which is the user's *prompt* or ``CONTINUE_NATURALLY``.
    Motion, sound and the direction shape the action; style, mood and details belong to the kept scene and are left out.
    """
    preamble = [
        "You write only the action for the next part of a clip for a text-to-video model. The user tells you what happens next.",
        f"The scene (kept as is, do not restate): {scene.strip()}",
    ]
    previous = _previous_rule(context)
    if previous is not None:
        preamble.append(previous)
    return _action_only_messages(preamble, _extend_rules(context), settings, plan, context, _extend_user_message(prompt, context))


def build_action_messages(scene: str, prompt: str, settings: EnhanceSettings, *, plan: LengthPlan, context: EnhanceContext) -> list[dict[str, str]]:
    """Build the system + user chat messages that ask for only the action of a video that is not an extension, beside a kept scene.

    The *scene* is fixed context the model must not restate or change. The user message is the user's *prompt*, or a
    request for an action that suits the scene when it is blank. With a reference image the action describes the
    change from that first frame. Motion, sound and the direction shape the action; style, mood and details belong
    to the kept scene and are left out.
    """
    preamble = [
        "You write only the action of a clip for a text-to-video model. The user tells you what happens.",
        f"The scene (kept exactly as written, do not restate or change it): {scene.strip()}",
    ]
    situation = [*context_rules("video", context), "Write what happens in time order, fitted to the scene."]
    return _action_only_messages(preamble, situation, settings, plan, context, prompt.strip() or "(none): write an action that suits the scene.")


def _pair_workflow_rule(context: EnhanceContext) -> str:
    """Return the rule on how the user's text maps onto scene and action for this workflow."""
    if context.reference_image:
        return (
            "The first frame is supplied as an image. Keep the SCENE short: the style and what the image does not show, without re-describing what it shows. "
            "Make the ACTION the motion, camera movement and changes over time."
        )
    return (
        "Sort the user's text: who, where, look, style and camera setup go in the SCENE; events, camera moves and sound go in the ACTION. "
        "The user may have written everything in the Prompt, or mixed both boxes: split it, keep the Scene's facts, and expand both parts."
    )


def parse_scene_action(text: str) -> tuple[str, str] | None:
    """Split model output into its ``SCENE:`` and ``ACTION:`` parts, or return None when either is missing.

    Reasoning blocks are dropped first. The scene marker is the first ``SCENE:``; the action marker is the first
    ``ACTION:`` (or its alias ``PROMPT:``) after it, inline or on a new line, with optional bold markers. Anything
    after the action marker, including the word "scene", is action text.
    """
    body = strip_reasoning(text)
    scene_match = _SCENE_RE.search(body)
    if scene_match is None:
        return None
    action_match = _ACTION_RE.search(body, scene_match.end())
    if action_match is None:
        return None
    scene = clean_output(body[scene_match.end() : action_match.start()])
    action = clean_output(body[action_match.end() :])
    return (scene, action) if scene and action else None


def split_scene_action(raw: str, *, action_only: bool = False) -> tuple[str, str]:
    """Split in-progress model text into a scene and an action for live display.

    Reasoning blocks are hidden. A trailing, still-incomplete marker (``ACT``, ``**Sc``) is held back so it
    never flashes as text. Text before a ``SCENE:`` marker shows as the action; after the action marker nothing is
    treated as a marker. Use ``parse_scene_action`` for the finished text.

    Args:
        action_only: Treat all of *raw* as the action (workflows that write no scene).
    """
    if action_only:
        return "", raw
    text = strip_reasoning(raw)
    scene_match = _SCENE_RE.search(text)
    if scene_match is None:
        return "", _hold_back_partial_marker(text, _SCENE_WORDS).strip()
    action_match = _ACTION_RE.search(text, scene_match.end())
    if action_match is None:
        return _hold_back_partial_marker(text, _ACTION_WORDS)[scene_match.end() :].strip(), ""
    return text[scene_match.end() : action_match.start()].strip(), text[action_match.end() :].strip()


def _hold_back_partial_marker(text: str, words: tuple[str, ...]) -> str:
    """Return *text* without a trailing fragment that may still become one of the marker *words*."""
    match = _PARTIAL_TAIL_RE.search(text)
    if match is None:
        return text
    stars, letters = match.group(1), match.group(2).upper()
    if letters and not any(word.startswith(letters) for word in words):
        return text
    return text[: match.start()] if letters or stars else text


def _quotes_known(scene: str, user_scene: str) -> bool:
    """Return whether every quoted span in *scene* already appears in *user_scene*."""
    known = normalize_text(user_scene)
    return all(normalize_text(straight or curly) in known for straight, curly in _QUOTED_RE.findall(scene))


def _accept_pair(raw: str, user_scene: str, prompt: str, *, previous: str | None = None) -> tuple[str, str] | None:
    """Return the parsed scene and action of *raw* when they pass validation, else None.

    An echoed scene is fine; the action must not echo *prompt*, refuse, or repeat *previous* (see ``repeats_source``);
    the scene may only quote what the user's scene already quotes.
    """
    parsed = parse_scene_action(raw)
    if parsed is None:
        return None
    scene, action = parsed
    if is_unusable(action, prompt) or _repeats(action, previous) or not _quotes_known(scene, user_scene):
        return None
    return parsed


def _repeats(action: str, previous: str | None) -> bool:
    """Return whether *action* repeats the earlier prompt *previous*; False without one."""
    return bool(previous and previous.strip()) and repeats_source(action, previous=previous)


def _fallback_action(raw: str) -> str:
    """Return the best action in output that is not a valid pair: the action segment, else the text after the scene segment, else all of it.

    The scene segment (up to the first blank line, else the first line) never ends up in the action.
    """
    body = strip_reasoning(raw)
    scene_match = _SCENE_RE.search(body)
    action_match = _ACTION_RE.search(body, scene_match.end() if scene_match else 0)
    if action_match is not None:
        return clean_output(body[action_match.end() :])
    if scene_match is None:
        return clean_output(body)
    after = body[scene_match.end() :].lstrip()
    blank = _BLANK_LINE_RE.search(after)
    return clean_output(after[blank.end() :] if blank else after.partition("\n")[2])


def enhance_video_prompt(
    enhancer: PromptEnhancer,
    scene: str | None,
    prompt: str | None,
    settings: EnhanceSettings,
    *,
    seed: int,
    ceiling: int,
    context: EnhanceContext | None = None,
    length_cfg: dict[str, Any] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    on_text: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> EnhancePairResult:
    """Rewrite a video's *scene* and *prompt* into an enhanced scene and action with one model call.

    Retries once with the next seed when the output has no valid ``SCENE:``/``ACTION:`` pair. If the retry is
    no better, its action (see ``_fallback_action``) is used with the user's scene kept, and the result carries a warning.
    Expand ``{a|b}`` choices before calling.

    ``settings.parts`` (see ``resolve_parts``) picks what is written. For ``"action"`` with a *scene*, the scene is kept
    exactly and the model writes only an action that fits it, as plain text with no markers, and the result's ``scene``
    is ``""`` (it was not rewritten). Otherwise the output is a ``SCENE:``/``ACTION:`` pair; a blank *scene* is
    extracted or written new, so ``"action"`` without one falls back to this. Unset ``parts`` means ``"action"`` when
    extending a clip that has a scene and ``"both"`` otherwise.

    When ``context.extends`` the video continues an existing clip and both inputs may be blank (the model then
    continues the scene naturally). The model then also gets the clip's previous prompt, and output that repeats
    ``context.previous`` is retried.

    Args:
        scene: Who and where, look, style and camera; may be blank (the prompt is then split into both parts).
        prompt: What happens; may be blank when *scene* is not.
        context: Job facts (audio track, clip length, frame shape, reference image, extension).
        on_text: Receives the full in-progress raw text (``SCENE: … ACTION: …``; see ``split_scene_action``, with
            ``action_only=True`` when only the action is written) at most every ``TEXT_UPDATE_INTERVAL_SECONDS``,
            then the final text; a retry starts over from an empty string.
        cancelled: Polled per token; when it returns True generation stops and ``RuntimeError`` is raised.

    Raises:
        ValueError: For invalid settings, or when both *scene* and *prompt* are blank and the job does not extend.
        RuntimeError: When cancelled, or when no usable action came out of either attempt.
    """
    scene = (scene or "").strip()
    prompt = (prompt or "").strip()
    context = context or EnhanceContext()
    extends = context.extends
    if not extends and not scene and not prompt:
        raise ValueError("Enter a scene or a prompt to enhance.")
    validate_settings(settings, mode="video", split=extends or not scene)
    if resolve_parts(settings, scene=scene, extends=extends) == "action":
        return _enhance_action_only(
            enhancer, scene, prompt, settings, seed=seed, ceiling=ceiling, context=context, length_cfg=length_cfg, temperature=temperature, max_tokens=max_tokens, on_text=on_text, cancelled=cancelled
        )
    scene_words, prompt_words = _pair_word_counts(scene, prompt, context)
    plan = plan_pair_length(scene_words, prompt_words, settings.length, ceiling=ceiling, length_cfg=length_cfg)
    warning = SCENE_HEAVY_WARNING if plan.scene_heavy else None
    if warning is not None:
        warnings.warn(warning, stacklevel=2)
    messages = build_pair_messages(scene, prompt, settings, plan=plan, context=context)
    # While an extension has no new direction, the prompt is not the user's text to echo (it may be the old prompt).
    echo = "" if extends and _continues_naturally(prompt, context) else prompt
    previous = (context.previous or context.extract_from) if extends else None
    raw = ""
    for attempt in range(2):
        raw = generate_raw(enhancer, messages, seed=seed + attempt, max_tokens=max_tokens, temperature=temperature, on_text=on_text, cancelled=cancelled)
        pair = _accept_pair(raw, scene, echo, previous=previous)
        if on_text is not None and (pair is not None or attempt == 1):
            on_text(clean_output(raw))
        if pair is not None:
            return EnhancePairResult(scene=pair[0], prompt=pair[1], clamped=plan.total.clamped, warning=warning)
    action = _fallback_action(raw)
    if is_unusable(action, echo) or _repeats(action, previous):
        raise RuntimeError(NO_USABLE_MESSAGE)
    return EnhancePairResult(scene="", prompt=action, clamped=plan.total.clamped, warning=" ".join(part for part in (warning, PAIR_FALLBACK_WARNING) if part), format_fallback=True)


def _pair_word_counts(scene: str, prompt: str, context: EnhanceContext) -> tuple[int, int]:
    """Return the scene and prompt word counts that size a scene-and-action rewrite.

    An extension without a scene is sized by the earlier prompt it extracts the scene from, plus the user's
    direction unless that direction is blank or the earlier prompt itself.
    """
    if context.extends and not scene and context.extract_from:
        direction = 0 if _continues_naturally(prompt, context) else word_count(prompt)
        return 0, word_count(context.extract_from) + direction
    return word_count(scene), word_count(prompt)


def _kept_scene_action_plan(scene_words: int, prompt_words: int, length: str, *, ceiling: int, length_cfg: dict[str, Any] | None, extends: bool) -> tuple[LengthPlan, bool]:
    """Return the word plan of an action written beside a kept scene, and whether the scene crowds it out.

    The action is at least ``PAIR_ACTION_FLOOR_WORDS`` words and no more than what the scene leaves under *ceiling*.
    Its wanted size is two thirds of the total for the action that follows a clip (*extends*), and otherwise the share
    of the total that a scene-and-action rewrite would give it.
    """
    if extends:
        total = plan_length(scene_words + prompt_words, length, ceiling=ceiling, length_cfg=length_cfg)
        wanted = round(2 * total.target / 3)
    else:
        pair = plan_pair_length(scene_words, prompt_words, length, ceiling=ceiling, length_cfg=length_cfg)
        total, wanted = pair.total, pair.action.target
    room = ceiling - scene_words
    target = max(PAIR_ACTION_FLOOR_WORDS, min(wanted, room))
    return _part_plan(length, target, hi_cap=max(room, target), clamped=total.clamped), scene_words >= SCENE_HEAVY_RATIO * ceiling


def _strip_action_marker(text: str) -> str:
    """Return *text* without a leading ``ACTION:`` marker that a model may add although none was asked for."""
    match = _ACTION_RE.match(text)
    return text[match.end() :].strip() if match else text


def _action_only_output(raw: str) -> str:
    """Return the action in the output of an action-only rewrite, or "" when it is unusable.

    A model that ignores "no markers" and writes ``SCENE: … ACTION: …`` contributes only its action segment;
    a ``SCENE:`` without an action segment gives "".
    """
    if _SCENE_RE.search(strip_reasoning(raw)):
        parsed = parse_scene_action(raw)
        return parsed[1] if parsed else ""
    return _strip_action_marker(clean_output(raw))


def _enhance_action_only(
    enhancer: PromptEnhancer,
    scene: str,
    prompt: str,
    settings: EnhanceSettings,
    *,
    seed: int,
    ceiling: int,
    context: EnhanceContext,
    length_cfg: dict[str, Any] | None,
    temperature: float,
    max_tokens: int,
    on_text: Callable[[str], None] | None,
    cancelled: Callable[[], bool] | None,
) -> EnhancePairResult:
    """Write only the action (the one that follows an existing clip, when extending), keeping *scene* as is; retry once on an unusable or repeating output.

    The result's ``scene`` is ``""``: the scene was not rewritten.

    Raises:
        RuntimeError: When cancelled, or when both attempts are unusable.
    """
    plan, scene_heavy = _kept_scene_action_plan(word_count(scene), word_count(prompt), settings.length, ceiling=ceiling, length_cfg=length_cfg, extends=context.extends)
    warning = SCENE_HEAVY_WARNING if scene_heavy else None
    if warning is not None:
        warnings.warn(warning, stacklevel=3)
    build = build_extend_messages if context.extends else build_action_messages
    messages = build(scene, prompt, settings, plan=plan, context=context)
    for attempt in range(2):
        raw = generate_raw(enhancer, messages, seed=seed + attempt, max_tokens=max_tokens, temperature=temperature, on_text=on_text, cancelled=cancelled)
        output = _action_only_output(raw)
        if not (is_unusable(output, prompt) or _repeats(output, context.previous)):
            if on_text is not None:
                on_text(output)
            return EnhancePairResult(scene="", prompt=output, clamped=plan.clamped, warning=warning)
    raise RuntimeError(NO_USABLE_MESSAGE)
