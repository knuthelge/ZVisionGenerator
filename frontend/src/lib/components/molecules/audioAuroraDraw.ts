/**
 * Paints one frame of the Aurora live preview of an audio take on a 2D canvas: a sky that changes with the stage of the
 * take (haze, stars, a horizon glow, a far layer of ribbons, a clear sky), ribbons of light that glow behind the
 * playhead, and, once the take is ready, the ribbons collapsing onto its outline while the bars open from the centre. All its numbers come from `audioAurora`; this file only draws. Canvas 2D only, no blur: every glow is a stack of
 * translucent shapes, and the scratch arrays are built once per painter, so a frame allocates nothing it keeps.
 */
import { BAR_RADIUS, BAR_WIDTH } from '$lib/components/atoms/Waveform.svelte';
import { hash, lerp, mix } from '$lib/components/atoms/mascotArt';
import type { AudioWorkflow } from '$lib/types';
import {
  AURORA_BANDS,
  easeToward,
  landedBars,
  landingPhases,
  outlineHalf,
  peakOutline,
  rayPosition,
  rgba,
  ribbonPoint,
  smoothstep,
  stageLook,
  stageMixOf,
  stageWeights,
  standIn,
  starTwinkle,
  stepStageMix,
  tightnessAt,
  WAVE_INSET,
  type AuroraColors,
  type RibbonState,
  type SkyLook,
  type StageMix,
  type AuroraStage,
} from './audioAurora';

const STAR_COUNT = 40;
const RAY_COUNT = 26;
/** Points along a ribbon, fewer on a phone. */
const RIBBON_POINTS = { wide: 72, phone: 48 } as const;
/** How the glow of a ribbon is layered: its width as a share of the ribbon, and the opacity of that layer. */
const RIBBON_LAYERS = [
  [1, 0.1],
  [0.55, 0.16],
  [0.18, 0.38],
] as const;
/** How fast the playhead and a sound's brightness glide to their targets (about 90% of the way in one second). */
const HEAD_RATE = 2.5;
const LEVEL_RATE = 1.5;
/** How fast the stars come out with the plan. */
const PLAN_RATE = 1.5;
/** The far layer of ribbons: higher, thinner, slower and faint. */
const FAR_LAYER = { shift: -0.24, thick: 0.45, alpha: 0.4, speed: 0.5, time: 0.6, timeOffset: 4000 } as const;

/** What one frame draws. */
export interface PaintInput {
  kind: AudioWorkflow;
  colors: AuroraColors;
  /** The clock in milliseconds, which the drift follows. */
  time: number;
  /** Seconds since the last frame. */
  dt: number;
  /** Where the playhead and the brightness of a sound are headed, 0 to 1. */
  head: number;
  level: number;
  /** The stage of the take, which the sky shows. */
  stage: AuroraStage;
  /** Where the stars are headed (`planTarget`). */
  plan: number;
  /** The part of a speech being made and how many there are; 1 of 1 for the other kinds and for a finished take. */
  part: number;
  parts: number;
  /** How much stronger the sky is drawn on a narrow slot (`skyBoost`). */
  boost: number;
  /** Jump to the targets instead of gliding (reduced motion). The first frame of a take always jumps. */
  snap: boolean;
  /** Device pixel ratio, which sizes the one-pixel details. */
  px: number;
  phone: boolean;
  /** How far the landing has come, 0 to 1; null while the take is made. */
  land: number | null;
  /** The finished take's peaks, used from the landing on. */
  peaks: readonly number[];
}

/** A painter keeps the eased playhead and the scratch arrays between frames. */
interface AuroraPainter {
  paint(ctx: CanvasRenderingContext2D, width: number, height: number, input: PaintInput): void;
  /** Forget the eased state, for a new take. */
  reset(): void;
}

function roundedBar(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radiusX: number, radiusY: number): void {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, width, height, [{ x: radiusX, y: radiusY }]);
  else ctx.rect(x, y, width, height);
  ctx.fill();
}

/** Draw the stars: `amount` (0 to 1) is how many are out, each coming in as the amount passes its own moment. */
function drawStars(ctx: CanvasRenderingContext2D, width: number, height: number, input: PaintInput, amount: number, alpha: number): void {
  const size = Math.max(input.px, Math.round(input.px * input.boost));
  for (let j = 0; j < STAR_COUNT; j += 1) {
    const at = hash(j, 5, 5) * 0.85;
    const out = smoothstep(at, at + 0.15, amount);
    if (out < 0.01) continue;
    ctx.fillStyle = rgba(input.colors.text, 0.26 * starTwinkle(j, input.time) * out * alpha);
    ctx.fillRect(Math.round(hash(j, 1, 1) * width), Math.round(hash(j, 2, 1) * height * 0.95), size, size);
  }
}

/** Draw a glow along the bottom edge that rises with `horizon` (0 to 1). */
function drawHorizon(ctx: CanvasRenderingContext2D, width: number, height: number, input: PaintInput, horizon: number, alpha: number): void {
  const strength = horizon * input.boost;
  const reach = Math.min(0.9, 0.55 * horizon * (1 + (input.boost - 1) * 0.6));
  const top = height * (1 - reach);
  const glow = ctx.createLinearGradient(0, height, 0, top);
  glow.addColorStop(0, rgba(input.colors.primary, 0.13 * strength * alpha));
  glow.addColorStop(0.5, rgba(input.colors.subtle, 0.12 * strength * alpha));
  glow.addColorStop(1, rgba(input.colors.subtle, 0));
  ctx.fillStyle = glow;
  ctx.fillRect(0, top, width, height - top);
}

/** Draw the veil of the waiting sky: thickest near the bottom, with a grey mist along it. */
function drawHaze(ctx: CanvasRenderingContext2D, width: number, height: number, input: PaintInput, haze: number): void {
  const strength = haze * input.boost;
  const veil = ctx.createLinearGradient(0, 0, 0, height);
  veil.addColorStop(0, rgba(input.colors.surface, 0.35 * strength));
  veil.addColorStop(1, rgba(input.colors.surface, 0.8 * strength));
  ctx.fillStyle = veil;
  ctx.fillRect(0, 0, width, height);
  const mist = ctx.createLinearGradient(0, height * 0.55, 0, height);
  mist.addColorStop(0, rgba(input.colors.haze, 0));
  mist.addColorStop(1, rgba(input.colors.haze, 0.08 * strength));
  ctx.fillStyle = mist;
  ctx.fillRect(0, height * 0.55, width, height * 0.45);
}

/** Draw the one line along the waveform's centre that the clear sky of the decode keeps. */
function drawCentreLine(ctx: CanvasRenderingContext2D, width: number, height: number, input: PaintInput, line: number, alpha: number): void {
  ctx.fillStyle = rgba(input.colors.haze, 0.4 * line * input.boost * alpha);
  const thick = Math.max(input.px, Math.round(input.px * input.boost));
  ctx.fillRect(width * WAVE_INSET.x, Math.round(height / 2 - (thick - input.px) / 2), width * WAVE_INSET.w, thick);
}

function drawPlayhead(ctx: CanvasRenderingContext2D, width: number, height: number, head: number, input: PaintInput, strength: number): void {
  const x = head * width;
  const glow = ctx.createLinearGradient(x - 18 * input.px, 0, x + 2 * input.px, 0);
  glow.addColorStop(0, rgba(input.colors.hover, 0));
  glow.addColorStop(1, rgba(input.colors.hover, 0.3 * strength));
  ctx.fillStyle = glow;
  ctx.fillRect(x - 18 * input.px, 0, 20 * input.px, height);
}

/** Draw the take's bars as `<Waveform>` does, scaled into its inset, in a colour between the run's teal and the waveform's grey. */
function drawLandedBars(ctx: CanvasRenderingContext2D, width: number, height: number, input: PaintInput, settle: number): void {
  const left = width * WAVE_INSET.x;
  const top = height * WAVE_INSET.y;
  const sx = (width * WAVE_INSET.w) / 400;
  const sy = (height * WAVE_INSET.h) / 100;
  ctx.fillStyle = rgba(mix(input.colors.primary, input.colors.wave, settle));
  for (const bar of landedBars(input.peaks, input.land ?? 1)) {
    if (bar.height <= 0) continue;
    roundedBar(ctx, left + bar.x * sx, top + bar.y * sy, BAR_WIDTH * sx, bar.height * sy, BAR_RADIUS * sx, BAR_RADIUS * sy);
  }
}

/** The peaks the ribbons tighten toward while the take is made and its own are not known: the kind's stand-in loudness. */
const standInOf = (kind: AudioWorkflow): number[] => Array.from({ length: 100 }, (_, index) => standIn(kind, index) * 255);
const STAND_IN_PEAKS: Readonly<Record<AudioWorkflow, number[]>> = { music: standInOf('music'), speech: standInOf('speech'), sound: standInOf('sound') };

/** How one set of ribbons is drawn on top of the look of the stage. */
interface RibbonPass {
  /** The clock the drift follows. */
  time: number;
  shift: number;
  thick: number;
  speed: number;
  /** A scale of every opacity. */
  alpha: number;
  /** A scale of the rays; 0 for none. */
  rays: number;
  /** Pull the ribbons toward the take's outline by the look's `tight`, as far as each part of a speech is done. */
  tighten: boolean;
}

/** Create a painter. */
export function createAuroraPainter(): AuroraPainter {
  const top = new Float32Array(RIBBON_POINTS.wide);
  const bottom = new Float32Array(RIBBON_POINTS.wide);
  const middle = new Float32Array(RIBBON_POINTS.wide);
  let fresh = true;
  let head = 0;
  let level = 0;
  let plan = 0;
  let mix: StageMix = stageMixOf('wait');
  let outline: number[] = [];
  let outlineOf: readonly number[] | null = null;

  /** Draw the ribbons of the kind once, as *pass* sets them, in *look*, at the landing phases *phases*. */
  function drawRibbons(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    input: PaintInput,
    look: SkyLook,
    phases: { collapse: number; fadeOut: number },
    pass: RibbonPass,
  ): void {
    const { kind, colors, px, land } = input;
    const visible = 1 - phases.fadeOut;
    const points = input.phone ? RIBBON_POINTS.phone : RIBBON_POINTS.wide;
    const state: RibbonState = { kind, time: pass.time, head, level, rest: look.rest, promise: look.promise, thick: pass.thick, shift: pass.shift, speed: pass.speed };
    const brightness = visible * (1 - 0.6 * phases.collapse) * lerp(look.glow, 1, phases.collapse) * (1 - 0.82 * look.outline * (1 - phases.collapse)) * pass.alpha;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    AURORA_BANDS[kind].forEach((band, bandIndex) => {
      const color = colors[band.color];
      for (let j = 0; j < points; j += 1) {
        const x = j / (points - 1);
        const point = ribbonPoint(band, x, state);
        let centre = point.centre;
        let half = point.half;
        const outlineScale = bandIndex === 0 ? 1 : 0.6;
        const tight = pass.tighten ? tightnessAt(x, look.tight, input.part, input.parts) : 0;
        if (tight > 0) {
          centre = lerp(centre, 0.5, tight);
          half = lerp(half, outlineHalf(outline, x) * outlineScale, tight);
        }
        if (land !== null) {
          centre = lerp(centre, 0.5, phases.collapse);
          half = lerp(half, outlineHalf(outline, x) * outlineScale, phases.collapse);
        }
        middle[j] = centre * height;
        top[j] = (centre - half) * height;
        bottom[j] = (centre + half) * height;
      }
      const trace = (layer: number): void => {
        ctx.beginPath();
        for (let j = 0; j < points; j += 1) {
          const y = lerp(middle[j]!, top[j]!, layer);
          if (j) ctx.lineTo((j / (points - 1)) * width, y);
          else ctx.moveTo(0, y);
        }
        for (let j = points - 1; j >= 0; j -= 1) ctx.lineTo((j / (points - 1)) * width, lerp(middle[j]!, bottom[j]!, layer));
        ctx.closePath();
      };
      for (const [layer, alpha] of RIBBON_LAYERS) {
        trace(layer);
        ctx.fillStyle = rgba(color, alpha * brightness);
        ctx.fill();
        if (look.rich > 0.01) {
          // Extra light in the ribbon: the same shape once more.
          ctx.fillStyle = rgba(color, 0.45 * alpha * visible * (1 - phases.collapse) * pass.alpha * look.rich);
          ctx.fill();
        }
      }
      if (look.outline > 0.02) {
        // Faint outlines: the two edges of the ribbon as hairlines.
        ctx.lineWidth = px;
        ctx.strokeStyle = rgba(color, 0.42 * look.outline * visible * pass.alpha);
        for (const edge of [top, bottom]) {
          ctx.beginPath();
          for (let j = 0; j < points; j += 1) {
            if (j) ctx.lineTo((j / (points - 1)) * width, edge[j]!);
            else ctx.moveTo(0, edge[j]!);
          }
          ctx.stroke();
        }
      }
      const rayAlpha = 0.05 * visible * pass.alpha * (1 - look.outline) * (1 - look.rest) * pass.rays;
      if (bandIndex === 0 && rayAlpha > 0.002) {
        // Faint rays rise from the first ribbon, drifting very slowly.
        for (let c = 0; c < RAY_COUNT; c += 1) {
          const x = rayPosition(c, pass.time);
          const j = Math.round(x * (points - 1));
          const rayHeight = (bottom[j]! - top[j]!) * 1.4;
          if (rayHeight < 4 * px) continue;
          ctx.fillStyle = rgba(color, rayAlpha);
          ctx.fillRect(x * width, top[j]! - rayHeight * 0.6, 1.5 * px, rayHeight);
        }
      }
    });
    ctx.restore();
  }

  function paint(ctx: CanvasRenderingContext2D, width: number, height: number, input: PaintInput): void {
    const { kind, time, land } = input;
    const jump = input.snap || fresh;
    fresh = false;
    head = jump ? input.head : easeToward(head, input.head, input.dt, HEAD_RATE);
    level = jump ? input.level : easeToward(level, input.level, input.dt, LEVEL_RATE);
    plan = jump ? input.plan : easeToward(plan, input.plan, input.dt, PLAN_RATE);
    mix = jump ? stageMixOf(input.stage) : stepStageMix(mix, input.stage, input.dt);
    const weights = stageWeights(mix);
    const look = stageLook(weights, { kind, head, level, plan });
    ctx.clearRect(0, 0, width, height);
    const landing = land !== null;
    const outlinePeaks = landing ? input.peaks : STAND_IN_PEAKS[kind];
    if (outlineOf !== outlinePeaks) {
      outline = peakOutline(outlinePeaks);
      outlineOf = outlinePeaks;
    }
    const phases = landing ? landingPhases(land) : { collapse: 0, fadeOut: 0, settle: 0 };
    const visible = 1 - phases.fadeOut;

    drawStars(ctx, width, height, input, look.star, visible);
    if (look.horizon > 0.01) drawHorizon(ctx, width, height, input, look.horizon, visible);
    if (look.depth > 0.01) {
      // The far layer: higher, thinner and slower, so the sky has depth while the song is rendered.
      const far: RibbonPass = { time: time * FAR_LAYER.time + FAR_LAYER.timeOffset, shift: FAR_LAYER.shift, thick: FAR_LAYER.thick, speed: FAR_LAYER.speed, alpha: FAR_LAYER.alpha * look.depth * input.boost, rays: 0, tighten: false };
      drawRibbons(ctx, width, height, input, { ...look, rich: 0 }, phases, far);
    }
    drawRibbons(ctx, width, height, input, look, phases, { time, shift: 0, thick: 1, speed: 1, alpha: 1, rays: 1, tighten: true });
    if (!landing && kind !== 'sound' && head < 0.995 && weights.write > 0.01) drawPlayhead(ctx, width, height, head, input, weights.write);
    if (look.line > 0.01) drawCentreLine(ctx, width, height, input, look.line, visible);
    if (look.haze > 0.01) drawHaze(ctx, width, height, input, look.haze);
    if (landing) drawLandedBars(ctx, width, height, input, phases.settle);
  }

  function reset(): void {
    fresh = true;
    outlineOf = null;
  }

  return { paint, reset };
}
