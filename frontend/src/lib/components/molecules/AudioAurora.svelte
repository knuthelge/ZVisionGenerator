<script lang="ts">
  import { untrack } from 'svelte';
  import { Waveform } from '$lib/components/atoms';
  import type { AudioProgress, AudioWorkflow } from '$lib/types';
  import { auroraHead, auroraNote, auroraProgress, auroraStage, FRAME_MS, landingProgress, planTarget, readAuroraColors, skyBoost, type AuroraColors, type AuroraStage } from './audioAurora';
  import { createAuroraPainter } from './audioAuroraDraw';

  interface Props {
    kind: AudioWorkflow;
    /** The take's latest `audio_progress`; null until the first one, while the ribbons wait at zero progress. */
    progress?: AudioProgress | null;
    /** The finished take's peaks once it is ready; null while it is made. */
    peaks?: readonly number[] | null;
    /** The finished take's length in seconds, for the corner note. */
    seconds?: number | null;
    /** The corner note while there is no progress yet, such as "Loading the model". */
    waitingNote?: string;
    /** Paused or ended: the ribbons hold their last frame instead of drifting. */
    idle?: boolean;
  }

  let { kind, progress = null, peaks = null, seconds = null, waitingNote = 'Starting', idle = false }: Props = $props();

  const painter = createAuroraPainter();
  let canvas = $state<HTMLCanvasElement | null>(null);
  // A preview that opens on a take that is already ready has nothing to land.
  let landed = $state(untrack(() => peaks !== null));
  let reduced = $state(false);
  let hidden = $state(false);
  let onScreen = $state(true);
  // Reduced motion redraws on progress only, so the drift clock advances by a fixed amount per event.
  let stillClock = 0;

  const head = $derived(auroraHead(kind, progress));
  const level = $derived(auroraProgress(kind, progress));
  // The sky's stage: no progress yet is the waiting sky, and a take that is ready is the decode's clear sky. A speech's part is the one being read.
  const stage = $derived<AuroraStage>(peaks !== null ? 'decode' : auroraStage(progress));
  const plan = $derived(planTarget(kind, stage, progress));
  const part = $derived(peaks !== null || kind !== 'speech' || progress === null ? 1 : progress.part);
  const parts = $derived(peaks !== null || kind !== 'speech' || progress === null ? 1 : progress.parts);
  const note = $derived(peaks !== null ? auroraNote(kind, null, seconds) : progress === null ? waitingNote : auroraNote(kind, progress));

  let colors: AuroraColors | null = null;
  // How far the landing in progress has come; null while the take is made.
  let landing: number | null = null;

  /** Draw one frame at clock *now* (ms), *dt* seconds after the last, mid-landing at *land* (0 to 1) or null while the take is made. */
  function draw(now: number, dt: number, snap: boolean, land: number | null): void {
    const el = canvas;
    if (!el || !colors) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.round(el.clientWidth * ratio);
    const height = Math.round(el.clientHeight * ratio);
    if (!width || !height) return;
    if (el.width !== width || el.height !== height) {
      el.width = width;
      el.height = height;
    }
    const ctx = el.getContext('2d');
    if (!ctx) return;
    painter.paint(ctx, width, height, { kind, colors, time: now, dt, head, level, stage, plan, part, parts, boost: skyBoost(el.clientWidth), snap, px: ratio, phone: el.clientWidth < 420, land, peaks: peaks ?? [] });
  }

  // The things the frame clock depends on: the tab, the viewport and the motion preference.
  $effect(() => {
    const el = canvas;
    if (!el) return;
    colors = readAuroraColors((name) => getComputedStyle(el).getPropertyValue(name));
    const motion = typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : null;
    reduced = motion?.matches ?? false;
    hidden = document.hidden;
    const onMotion = () => { reduced = motion?.matches ?? false; };
    const onVisibility = () => { hidden = document.hidden; };
    motion?.addEventListener?.('change', onMotion);
    document.addEventListener('visibilitychange', onVisibility);
    const visible = typeof IntersectionObserver === 'function' ? new IntersectionObserver((entries) => { onScreen = entries.at(-1)?.isIntersecting ?? true; }) : null;
    visible?.observe(el);
    const resized = typeof ResizeObserver === 'function' ? new ResizeObserver(() => draw(reduced ? stillClock : performance.now(), 0, reduced, landing)) : null;
    resized?.observe(el);
    return () => {
      motion?.removeEventListener?.('change', onMotion);
      document.removeEventListener('visibilitychange', onVisibility);
      visible?.disconnect();
      resized?.disconnect();
    };
  });

  // A new take (progress is still going, with no peaks yet) opens the canvas again.
  $effect(() => {
    if (peaks === null && landed) {
      landed = false;
      painter.reset();
    }
  });

  // Reduced motion: one still per progress event.
  $effect(() => {
    void head;
    void level;
    void kind;
    void stage;
    void plan;
    if (!reduced || peaks !== null || landed || !canvas) return;
    stillClock += 997;
    draw(stillClock, 0, true, null);
  });

  // The frame clock: drifting while the take is made, then one landing. It stops while hidden, off screen or idle, and after the landing.
  $effect(() => {
    const el = canvas;
    if (!el || landed) return;
    const ready = peaks !== null;
    if (ready && (reduced || hidden || !onScreen)) {
      landed = true; // nothing to see, or nothing to animate: go straight to the waveform
      return;
    }
    if (reduced || (!ready && (idle || hidden || !onScreen))) return;
    let frame = 0;
    let last = 0;
    let landStart: number | null = null;
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      if (last && now - last < FRAME_MS - 1) return;
      const dt = Math.min(0.1, last ? (now - last) / 1000 : FRAME_MS / 1000);
      last = now;
      if (ready) {
        landStart ??= now;
        landing = landingProgress(now - landStart);
        draw(now, dt, false, landing);
        if (landing >= 1) {
          cancelAnimationFrame(frame);
          landing = null;
          landed = true;
        }
      } else {
        draw(now, dt, false, null);
      }
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      landing = null;
    };
  });
</script>

{#if landed}
  <!-- The same box and bars as the finished waveform the slot shows between takes, so the hand-over is not seen. -->
  <span class="aurora-wave"><Waveform peaks={peaks ?? []} /></span>
{:else}
  <canvas bind:this={canvas} aria-hidden="true" data-testid="audio-aurora"></canvas>
{/if}
<!-- svelte-ignore a11y_figcaption_parent -->
<figcaption class="media-tag aurora-tag">{note}</figcaption>

<style>
  canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%; }
  .aurora-wave { position: absolute; inset: 14% 6%; display: block; }
  .aurora-tag { position: absolute; top: 8px; left: 8px; }
</style>
