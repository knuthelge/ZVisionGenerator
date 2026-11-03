/**
 * The maths of the Aurora live preview of an audio take: how far a take is, which stage of the sky it is in, where its
 * ribbons run, which colours it reads from the page's tokens, and how the ribbons land on the take's waveform. Everything here is pure, so it is
 * tested without a canvas; `audioAuroraDraw` paints it and `AudioAurora.svelte` runs the clock.
 */
import { waveBars, type WaveBar } from '$lib/components/atoms/Waveform.svelte';
import { clamp, hash, lerp, type RGB } from '$lib/components/atoms/mascotArt';
import { formatClock } from '$lib/state/clock';
import type { AudioProgress, AudioWorkflow } from '$lib/types';

/** How long the ribbons take to collapse onto the take's outline and the bars to open, in milliseconds. */
export const LANDING_MS = 650;
/** The shortest time between two drawn frames: 30 frames a second at most. */
export const FRAME_MS = 1000 / 30;
/** Where the finished waveform sits in the slot as fractions of its width and height (`.job-preview-wave`). */
export const WAVE_INSET = { x: 0.06, y: 0.14, w: 0.88, h: 0.72 } as const;

/**
 * YuE2 and Higgs make this many tokens (frames) per second of sound, so a token count is a time. The `audio_progress` event
 * carries counts only (`step`, `total`), not seconds, so the rate lives here.
 */
export const TOKENS_PER_SECOND = 25;

/** The phase ids the backends report in `audio_progress` (`backends/audio/*.py`): a song's plan and tokens, and the decode every kind ends with. */
export const PHASE_PLAN = 'Plan';
export const PHASE_SONG_TOKENS = 'Song tokens';
export const PHASE_DECODE = 'Decode';

/** One phase of a song, in the order the model reports them: the name the card and the strip call it, and its share of the take as `[start, width]` (set from a few real runs). */
export const MUSIC_PHASES: readonly { id: string; label: string; share: readonly [start: number, width: number] }[] = [
  { id: PHASE_PLAN, label: 'Planning', share: [0, 0.06] },
  { id: PHASE_SONG_TOKENS, label: 'Writing the song', share: [0.06, 0.66] },
  { id: 'Synthesis', label: 'Rendering', share: [0.72, 0.24] },
  { id: PHASE_DECODE, label: 'Finishing', share: [0.96, 0.04] },
];
/** The tokens at which a song's plan is about 63% done: its length is not known, so progress runs toward 1 without reaching it. */
const PLAN_TOKENS_SCALE = 300;
/** The seconds of a song at which its tokens are about 63% done. */
const SONG_SECONDS_SCALE = 80;
/** The share of a speech part its tokens take; its decode is the rest. */
const SPEECH_TOKENS_SHARE = 0.95;

/** The share of a phase done, or `fallback` when it reports no total. */
function phaseShare(progress: AudioProgress, fallback: number): number {
  return progress.total !== null && progress.total > 0 ? clamp(progress.step / progress.total) : fallback;
}

/** Estimate how far the current take is, from 0 to 1, from the phase it reports: Music by phase shares, Speech by the frames of its parts, Sound by its steps. */
export function auroraProgress(kind: AudioWorkflow, progress: AudioProgress | null): number {
  if (progress === null) return 0;
  if (kind === 'music') {
    const phase = MUSIC_PHASES.find(({ id }) => id === progress.phase);
    if (!phase) return 0;
    const [start, width] = phase.share;
    const done =
      progress.phase === PHASE_PLAN ? 1 - Math.exp(-progress.step / PLAN_TOKENS_SCALE)
      : progress.phase === PHASE_SONG_TOKENS ? 1 - Math.exp(-progress.step / TOKENS_PER_SECOND / SONG_SECONDS_SCALE)
      : phaseShare(progress, 0.5);
    return start + width * clamp(done);
  }
  if (kind === 'speech') {
    const parts = Math.max(1, progress.parts);
    const part = clamp(progress.part, 1, parts) - 1;
    const share = phaseShare(progress, 0);
    const within = progress.phase === PHASE_DECODE ? SPEECH_TOKENS_SHARE + (1 - SPEECH_TOKENS_SHARE) * share : SPEECH_TOKENS_SHARE * share;
    return (part + within) / parts;
  }
  if (progress.phase === PHASE_DECODE) return 1;
  return phaseShare(progress, 0);
}

/** Where the playhead is, from 0 to 1 across the strip: how much of the take exists so far. A sound is made all at once, so its head is at the end. */
export function auroraHead(kind: AudioWorkflow, progress: AudioProgress | null): number {
  if (kind === 'sound') return 1;
  if (progress === null) return 0;
  if (kind === 'speech') return auroraProgress(kind, progress);
  if (progress.phase === PHASE_PLAN) return 0.04 * (1 - Math.exp(-progress.step / PLAN_TOKENS_SCALE));
  if (progress.phase === PHASE_SONG_TOKENS) return 1 - Math.exp(-progress.step / TOKENS_PER_SECOND / SONG_SECONDS_SCALE);
  return 1;
}

/** Move `current` toward `target` over `dtSeconds`, so a value that jumps with an event glides instead. A `rate` of 2.5 closes about 90% of the gap in a second. */
export function easeToward(current: number, target: number, dtSeconds: number, rate: number): number {
  return current + (target - current) * (1 - Math.exp(-dtSeconds * rate));
}

/** The caption of a finished take's waveform, with its length when known: the same words wherever the waveform shows. */
export function lastTakeCaption(takeSeconds: number | null | undefined): string {
  return takeSeconds == null ? 'Last finished take' : `Last finished take · ${formatClock(takeSeconds)}`;
}

/** The corner note of the strip: what the take is doing now, or the finished-take caption with its length once it is made. */
export function auroraNote(kind: AudioWorkflow, progress: AudioProgress | null, takeSeconds: number | null = null): string {
  if (progress === null) return lastTakeCaption(takeSeconds);
  if (kind === 'music') {
    const label = MUSIC_PHASES.find(({ id }) => id === progress.phase)?.label ?? 'Finishing';
    if (progress.phase === PHASE_SONG_TOKENS) return `${label} · ${formatClock(progress.step / TOKENS_PER_SECOND)}`;
    if (progress.phase === 'Synthesis' && progress.total) return `${label} · ${Math.round(phaseShare(progress, 0) * 100)}%`;
    return label;
  }
  if (kind === 'speech') {
    const where = progress.parts > 1 ? ` part ${progress.part} of ${progress.parts}` : '';
    return progress.phase === PHASE_DECODE ? `Finishing${where}` : `Reading${where} · ${formatClock(progress.step / TOKENS_PER_SECOND)}`;
  }
  if (progress.phase === PHASE_DECODE) return 'Finishing';
  return progress.total ? `Step ${progress.step} of ${progress.total}` : 'Sampling';
}

// ── Colours ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The colours the preview draws with, by role. */
export type AuroraColors = Readonly<Record<'text' | 'primary' | 'hover' | 'subtle' | 'coral' | 'amber' | 'wave' | 'surface' | 'haze', RGB>>;

/** The CSS custom property each colour role reads. */
const AURORA_TOKENS: Readonly<Record<keyof AuroraColors, string>> = {
  text: '--color-text-primary',
  primary: '--color-primary-main',
  hover: '--color-primary-hover',
  subtle: '--color-primary-subtle',
  coral: '--color-accent-coral',
  amber: '--color-accent-amber',
  wave: '--color-zinc-500',
  surface: '--color-bg-surface',
  haze: '--color-zinc-600',
};

/** A neutral grey, drawn for a token that cannot be read, so a missing token is dim rather than an error. */
const UNREADABLE: RGB = [143, 147, 154];

/** Parse a `#rrggbb` token into channels; null for anything else. */
export function parseCssColor(value: string): RGB | null {
  const long = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/.exec(value.trim().toLowerCase());
  return long ? ([1, 2, 3].map((i) => parseInt(long[i]!, 16)) as unknown as RGB) : null;
}

/** Read the preview's colours with *read*, which returns a CSS custom property's value (`getComputedStyle(el).getPropertyValue`). */
export function readAuroraColors(read: (name: string) => string): AuroraColors {
  const out = {} as Record<keyof AuroraColors, RGB>;
  for (const role of Object.keys(AURORA_TOKENS) as (keyof AuroraColors)[]) out[role] = parseCssColor(read(AURORA_TOKENS[role])) ?? UNREADABLE;
  return out;
}

/** Format a colour with an alpha as a CSS `rgba()` value. */
export function rgba(color: RGB, alpha = 1): string {
  return `rgba(${Math.round(color[0])},${Math.round(color[1])},${Math.round(color[2])},${clamp(alpha).toFixed(3)})`;
}

// ── Ribbons ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One ribbon of the aurora: its colour role, and how it sits and drifts. Speeds are radians per millisecond. */
interface AuroraBand {
  color: 'amber' | 'primary' | 'coral';
  /** The ribbon's centre above or below the middle, as a fraction of the strip's height. */
  offset: number;
  /** How far the centre sways, as a fraction of the height. */
  sway: number;
  /** How many waves of sway fit across the strip. */
  waves: number;
  speed: number;
  /** The ribbon's half thickness at full strength, as a fraction of the height. */
  thickness: number;
  phase: number;
}

/** The ribbons of each kind: music is amber, teal and coral bands; speech a coral ribbon with a thin teal harmonic; sound a wide teal mist. */
export const AURORA_BANDS: Readonly<Record<AudioWorkflow, readonly AuroraBand[]>> = {
  music: [
    { color: 'amber', offset: 0.11, sway: 0.05, waves: 1.1, speed: 0.00035, thickness: 0.22, phase: 0 },
    { color: 'primary', offset: 0, sway: 0.07, waves: 1.7, speed: -0.0005, thickness: 0.32, phase: 1 },
    { color: 'coral', offset: -0.12, sway: 0.05, waves: 2.4, speed: 0.0006, thickness: 0.17, phase: 2 },
  ],
  speech: [
    { color: 'coral', offset: 0.02, sway: 0.06, waves: 2.2, speed: 0.0005, thickness: 0.34, phase: 0 },
    { color: 'primary', offset: -0.08, sway: 0.04, waves: 3.1, speed: -0.0006, thickness: 0.14, phase: 2 },
  ],
  sound: [
    { color: 'primary', offset: 0, sway: 0.05, waves: 0.8, speed: 0.0003, thickness: 0.44, phase: 0 },
    { color: 'amber', offset: 0.09, sway: 0.04, waves: 1.3, speed: -0.00025, thickness: 0.18, phase: 3 },
  ],
};

/** Smooth step from 0 at *edge0* to 1 at *edge1*. */
export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function speechShape(i: number): number {
  let pos = 0;
  let word = 0;
  while (pos <= i) {
    const len = 3 + Math.floor(hash(word, 2, 1) * 4);
    const gap = word % 7 === 6 ? 3 : 1;
    if (i < pos + len) return 0.22 + 0.62 * Math.sin((Math.PI * (i - pos + 0.5)) / len) * (0.6 + 0.4 * hash(word, 2, 2));
    if (i < pos + len + gap) return 0.05;
    pos += len + gap;
    word += 1;
  }
  return 0.05;
}

/** The stand-in loudness of bar *index* (0 to 99) while a take is made: the kind's character, not the take, whose real shape only exists at the end. */
export function standIn(kind: AudioWorkflow, index: number): number {
  const i = ((Math.floor(index) % 100) + 100) % 100;
  if (kind === 'music') return clamp(0.36 + 0.24 * Math.sin(i * 0.085 + 0.5) ** 2 + (i % 4 === 0 ? 0.16 : 0) + 0.12 * hash(i, 7, 1));
  if (kind === 'speech') return speechShape(i);
  return clamp(0.5 + 0.1 * Math.sin(i * 0.17) + 0.2 * hash(i, 3, 3));
}

function standInAt(kind: AudioWorkflow, x: number): number {
  const p = clamp(x, 0, 0.9999) * 99;
  const i = Math.floor(p);
  return lerp(standIn(kind, i), standIn(kind, i + 1), p - i);
}

/** The stand-in loudness at *x* (0 to 1 across the strip), averaged over its neighbours so a curve through it stays smooth. */
function softStandIn(kind: AudioWorkflow, x: number): number {
  let sum = 0;
  for (let d = -3; d <= 3; d += 1) sum += standInAt(kind, clamp(x + d * 0.012, 0, 0.999));
  return sum / 7;
}

/** A ribbon at one point across the strip. */
interface RibbonPoint {
  /** The ribbon's centre, as a fraction of the strip's height from the top. */
  centre: number;
  /** Its half thickness, as a fraction of the height. */
  half: number;
}

/** The inputs of a ribbon's shape at an instant. */
export interface RibbonState {
  kind: AudioWorkflow;
  /** The clock in milliseconds; the ribbons drift with it. */
  time: number;
  /** The eased playhead, 0 to 1: the ribbons glow behind it and are faint ahead of it. */
  head: number;
  /** The eased brightness of a sound, 0 to 1: it rises with the steps. */
  level: number;
  /** How far the ribbons fold into one dim, calm ribbon, 0 to 1 (the model is loading). Default 0. */
  rest?: number;
  /** How much of the take ahead of the playhead is promised: 0 leaves it faint, 1 almost as bright as the written part. Default 0. */
  promise?: number;
  /** A scale of every ribbon's thickness, for a far layer. Default 1. */
  thick?: number;
  /** A shift of every ribbon up (negative) or down, as a fraction of the height. Default 0. */
  shift?: number;
  /** A scale of how fast the ribbons drift. Default 1. */
  speed?: number;
}

/** Return where *band* is at *x* (0 to 1 across the strip). Nothing here beats or flashes: it is a slow drift (10 to 30 s a cycle). */
export function ribbonPoint(band: AuroraBand, x: number, state: RibbonState): RibbonPoint {
  const { kind, time, head, level } = state;
  const { rest = 0, promise = 0, thick = 1, shift = 0, speed = 1 } = state;
  const sound = kind === 'sound';
  const floor = lerp(0.18, 0.8, promise);
  const mask = sound ? 1 : floor + (1 - floor) * (1 - smoothstep(head - 0.03, head + 0.07, x));
  const raw =
    kind === 'speech' ? softStandIn('speech', (x + time * 0.00001) % 1) * 1.2
    : kind === 'music' ? softStandIn('music', x)
    : softStandIn('sound', x) * (0.8 + 0.2 * Math.sin(time * 0.0003 + x * 4));
  const envelope = lerp(raw, 0.45, rest);
  const strength = sound ? 0.25 + 0.75 * level : 1;
  // At rest only the teal ribbon stays, thinner; the others fold away.
  const keep = band.color === 'primary' ? 1 - 0.5 * rest : 1 - rest;
  return {
    centre: 0.5 + shift + band.offset * (1 - rest) + band.sway * (1 - 0.5 * rest) * Math.sin(2 * Math.PI * x * band.waves + time * band.speed * speed + band.phase),
    half: band.thickness * (0.3 + 0.7 * envelope) * mask * strength * keep * thick,
  };
}

/** The faint vertical rays that rise from the first ribbon: their position across the strip, drifting very slowly. */
export function rayPosition(index: number, time: number): number {
  const direction = index % 2 ? 1 : -1;
  return (((hash(index, 4, 4) + time * 0.000006 * direction) % 1) + 1) % 1;
}

/** A star's twinkle, 0.1 to 1, over a cycle of about 8 seconds. */
export function starTwinkle(index: number, time: number): number {
  return 0.55 + 0.45 * Math.sin(time * 0.0008 + index * 1.7);
}

// ── Stages ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** What the sky shows: the model loading, the plan, the writing, the synthesis, and the decode. */
export type AuroraStage = 'wait' | 'plan' | 'write' | 'synth' | 'decode';

/** The stages in order. */
export const AURORA_STAGES: readonly AuroraStage[] = ['wait', 'plan', 'write', 'synth', 'decode'];

/** The stage each phase the models report belongs to; a phase not listed here is writing. Sound's `Sampling` is the writing of its sky: its horizon rises with the steps. */
const PHASE_STAGE: Readonly<Record<string, AuroraStage>> = {
  [PHASE_PLAN]: 'plan',
  [PHASE_SONG_TOKENS]: 'write',
  'Speech tokens': 'write',
  Sampling: 'write',
  Synthesis: 'synth',
  [PHASE_DECODE]: 'decode',
};

/** The plan tokens by which all the stars are out. */
const PLAN_STARS_TOKENS = 600;

/** How long a change of stage cross-fades, in seconds of real time. */
export const STAGE_FADE_S = 0.75;

/** How far a decoded speech part stays tightened toward its outline while the next one is written. */
const DECODE_HOLD = 0.5;

/** Read the stage of a take from its latest `audio_progress`: no progress yet is `wait`. */
export function auroraStage(progress: AudioProgress | null): AuroraStage {
  return progress === null ? 'wait' : (PHASE_STAGE[progress.phase] ?? 'write');
}

/** How much of each stage the sky shows; the weights add up to 1 once settled. */
export type StageMix = Readonly<Record<AuroraStage, number>>;

/** The mix that shows only *stage*. */
export function stageMixOf(stage: AuroraStage): StageMix {
  return { wait: stage === 'wait' ? 1 : 0, plan: stage === 'plan' ? 1 : 0, write: stage === 'write' ? 1 : 0, synth: stage === 'synth' ? 1 : 0, decode: stage === 'decode' ? 1 : 0 };
}

/** Move *mix* toward *stage* over *dtSeconds*: every weight changes by at most `dt / STAGE_FADE_S`, so a change of stage takes `STAGE_FADE_S` of real time whatever the frame rate. */
export function stepStageMix(mix: StageMix, stage: AuroraStage, dtSeconds: number): StageMix {
  const most = dtSeconds / STAGE_FADE_S;
  const next = { ...mix };
  for (const id of AURORA_STAGES) {
    const target = id === stage ? 1 : 0;
    next[id] = most >= 1 ? target : mix[id] + clamp(target - mix[id], -most, most);
  }
  return next;
}

/** The cross-fade weights of *mix*: each eased with a smooth step, and scaled to add up to 1. */
export function stageWeights(mix: StageMix): StageMix {
  const eased = {} as Record<AuroraStage, number>;
  let sum = 0;
  for (const id of AURORA_STAGES) {
    eased[id] = smoothstep(0, 1, mix[id]);
    sum += eased[id];
  }
  for (const id of AURORA_STAGES) eased[id] /= sum || 1;
  return eased;
}

/** What the sky and the ribbons do in one stage, each 0 to 1. */
export interface SkyLook {
  /** How many of the stars are out. */
  star: number;
  /** The veil over the sky. */
  haze: number;
  /** The glow along the bottom edge. */
  horizon: number;
  /** The far, faint layer of ribbons. */
  depth: number;
  /** The single line along the waveform's centre. */
  line: number;
  /** The overall brightness of the ribbons. */
  glow: number;
  /** How far the ribbons fold into one dim ribbon. */
  rest: number;
  /** How much of the ribbons show as outlines only. */
  outline: number;
  /** How much of the take ahead of the playhead is outlined. */
  promise: number;
  /** Extra light in the ribbons. */
  rich: number;
  /** How far the ribbons are pulled toward the take's outline. */
  tight: number;
}

/** The look of each stage: haze while it waits, stars coming out while it plans, a horizon that rises as it writes, a far layer of ribbons while it renders, and a clear sky with one centre line while it decodes. The ribbons keep their look. */
const STAGE_LOOKS: Readonly<Record<AuroraStage, SkyLook>> = {
  wait: { star: 0, haze: 0.6, horizon: 0, depth: 0, line: 0, glow: 0.5, rest: 1, outline: 0, promise: 0, rich: 0, tight: 0 },
  plan: { star: 1, haze: 0.32, horizon: 0.2, depth: 0, line: 0, glow: 0.8, rest: 0, outline: 0.3, promise: 0.4, rich: 0, tight: 0 },
  write: { star: 1, haze: 0.1, horizon: 0.3, depth: 0, line: 0, glow: 1, rest: 0, outline: 0, promise: 0, rich: 0, tight: 0 },
  synth: { star: 1, haze: 0, horizon: 1, depth: 1, line: 0, glow: 1, rest: 0, outline: 0, promise: 0, rich: 0.3, tight: 0 },
  decode: { star: 0.45, haze: 0, horizon: 0.15, depth: 0.25, line: 1, glow: 0.8, rest: 0, outline: 0, promise: 0, rich: 0.15, tight: DECODE_HOLD },
};

/** The eased numbers the look of a take depends on. */
interface LookInput {
  kind: AudioWorkflow;
  /** The eased playhead and, for a sound, the eased brightness (0 to 1). */
  head: number;
  level: number;
  /** The eased share of the plan done; 1 once the plan is over, 0 before it begins. */
  plan: number;
}

/** Where the stars are headed for a take: a song's come out with its plan, 0 while it waits and 1 after; the other kinds have no plan. */
export function planTarget(kind: AudioWorkflow, stage: AuroraStage, progress: AudioProgress | null): number {
  if (kind !== 'music' || stage === 'wait') return 0;
  return stage === 'plan' ? clamp((progress?.step ?? 0) / PLAN_STARS_TOKENS) : 1;
}

/** Blend the looks of the stages by their *weights*. While writing, the horizon rises with what is written and, for speech and sound, the first stretch brings the stars out. */
export function stageLook(weights: StageMix, input: LookInput): SkyLook {
  const { kind, head, level, plan } = input;
  const written = kind === 'sound' ? level : head;
  const write: SkyLook = { ...STAGE_LOOKS.write, horizon: 0.3 + 0.55 * written, star: kind === 'music' ? 1 : clamp(0.4 + written * 3) };
  const looks: Record<AuroraStage, SkyLook> = { ...STAGE_LOOKS, write, plan: { ...STAGE_LOOKS.plan, star: plan } };
  const out = { ...STAGE_LOOKS.wait };
  for (const key of Object.keys(out) as (keyof SkyLook)[]) {
    out[key] = 0;
    for (const id of AURORA_STAGES) out[key] += weights[id] * looks[id][key];
  }
  return out;
}

/** How far ribbons at *x* (0 to 1 across the strip) are pulled toward the take's outline: *amount* for the part of a speech being decoded (or for everything, when there is one part), a steady hold behind it where the parts are done, and nothing ahead. */
export function tightnessAt(x: number, amount: number, part: number, parts: number): number {
  if (parts <= 1) return amount;
  const start = (clamp(part, 1, parts) - 1) / parts;
  const end = clamp(part, 1, parts) / parts;
  return x < start ? DECODE_HOLD : x <= end + 0.01 ? amount : 0;
}

/** How much stronger the sky's effects are drawn on a narrow slot, from 1 at 520 px and wider to 1.8 at 366 px (a phone) and narrower: the haze and the horizon are too subtle to tell the stages apart there. */
export function skyBoost(slotWidth: number): number {
  return 1 + 0.8 * clamp((520 - slotWidth) / (520 - 366));
}

// ── Landing ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** How far a landing has come, 0 to 1, *elapsedMs* after the take was ready. */
export function landingProgress(elapsedMs: number): number {
  return clamp(elapsedMs / LANDING_MS);
}

/** The strands of a landing at *land* (0 to 1): the ribbons collapse, fade out, and the bars take the take's colour. */
export function landingPhases(land: number): { collapse: number; fadeOut: number; settle: number } {
  return { collapse: smoothstep(0, 0.55, land), fadeOut: smoothstep(0.5, 1, land), settle: smoothstep(0.6, 1, land) };
}

/** How open bar *index* of *count* is at *land*, 0 to 1: the bars open from the centre outward. */
export function landingGrow(index: number, count: number, land: number): number {
  const middle = (count - 1) / 2;
  const distance = Math.abs(index - middle) / (count / 2);
  return smoothstep(0, 1, (land - 0.4 - distance * 0.3) / 0.25);
}

/** The bars of the landing at *land*. At 1 they are exactly the bars `<Waveform>` draws for *peaks*, so the swap to it cannot be seen. */
export function landedBars(peaks: readonly number[], land: number): WaveBar[] {
  const bars = waveBars(peaks);
  if (land >= 1) return bars;
  return bars.map((bar, index) => {
    const grow = landingGrow(index, bars.length, land);
    if (grow <= 0.001) return { x: bar.x, y: 50, height: 0 };
    const height = Math.min(bar.height, Math.max(3, bar.height * grow));
    return { x: bar.x, y: (100 - height) / 2, height };
  });
}

/** The heights of the take's bars as fractions of the waveform's height; build it once per landing and read it with {@link outlineHalf}. */
export function peakOutline(peaks: readonly number[]): number[] {
  return waveBars(peaks).map((bar) => bar.height / 100);
}

/** The take's outline at *x* (0 to 1 across the strip) as a half height in fractions of the strip's height; 0 outside the waveform's inset. The ribbons collapse onto it. */
export function outlineHalf(outline: readonly number[], x: number): number {
  const at = ((x - WAVE_INSET.x) / WAVE_INSET.w) * outline.length - 0.5;
  if (at < -0.5 || at > outline.length - 0.5) return 0;
  const i = clamp(Math.floor(at), 0, outline.length - 1);
  const j = clamp(i + 1, 0, outline.length - 1);
  return (lerp(outline[i]!, outline[j]!, clamp(at - i)) / 2) * WAVE_INSET.h;
}
