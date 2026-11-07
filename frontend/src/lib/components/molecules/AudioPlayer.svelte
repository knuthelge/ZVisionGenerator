<script lang="ts">
  import { Waveform } from '$lib/components/atoms';
  import { seekFraction } from '$lib/state/audioAsset';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { claimPlayback, releasePlayback } from '$lib/state/nowPlaying';

  interface Props {
    /** Where the browser loads the take from. */
    src: string;
    /** The take's peaks (100 values, 0 to 255). */
    peaks: readonly number[];
    /** The file's name, for the native controls' label. */
    filename: string;
    /** The take's length in seconds, used until the browser has read it from the file. */
    duration?: number | null;
    /** `stage`: the Workspace's panel, with room for the corner actions. `viewer`: the viewer's larger panel. `pane`: the file browser's preview, with a low waveform and no corner actions. */
    size?: 'stage' | 'viewer' | 'pane';
    class?: string;
  }

  let { src, peaks, filename, duration = null, size = 'stage', class: extraClass = '' }: Props = $props();

  let audio = $state<HTMLAudioElement | null>(null);
  let wave = $state<HTMLDivElement | null>(null);
  // How much of the take has played (0 to 1); it follows the audio element's `timeupdate` and a seek.
  let played = $state(0);
  let scrubbing = false;

  function takeLength(): number {
    const read = audio?.duration;
    if (typeof read === 'number' && Number.isFinite(read) && read > 0) return read;
    return duration ?? 0;
  }

  function sync(): void {
    const total = takeLength();
    played = audio && total > 0 ? Math.min(1, Math.max(0, audio.currentTime / total)) : 0;
  }

  function seekTo(clientX: number): void {
    const total = takeLength();
    if (!audio || !wave || !(total > 0)) return;
    const fraction = seekFraction(clientX, wave.getBoundingClientRect());
    audio.currentTime = fraction * total;
    played = fraction;
  }

  function startScrub(event: PointerEvent): void {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    scrubbing = true;
    wave?.setPointerCapture?.(event.pointerId);
    seekTo(event.clientX);
  }

  function moveScrub(event: PointerEvent): void {
    if (scrubbing) seekTo(event.clientX);
  }

  function endScrub(event: PointerEvent): void {
    scrubbing = false;
    if (wave?.hasPointerCapture?.(event.pointerId)) wave.releasePointerCapture(event.pointerId);
  }

  // A take never keeps playing when its player goes: the panel leaves (paging, closing, another take) or another file loads.
  // Each file has its own audio element (the `{#key}` below), so another take never inherits the last one's playing state.
  $effect(() => {
    const element = audio;
    played = 0;
    return () => {
      if (!element) return;
      element.pause();
      releasePlayback(element);
    };
  });
</script>

<!-- The waveform is for the pointer only; the native controls under it are the one set of controls, for keys and screen readers.
     `data-own-drag` keeps a sideways drag on it from paging the stage or the viewer. -->
<div class="audio-player {extraClass}" data-size={size} data-compact={layout.compact} data-own-drag data-testid="audio-player">
  <!-- svelte-ignore a11y_no_static_element_interactions because the native controls below carry the keyboard and screen-reader access -->
  <div
    bind:this={wave}
    class="audio-wave"
    aria-hidden="true"
    data-testid="audio-wave"
    onpointerdown={startScrub}
    onpointermove={moveScrub}
    onpointerup={endScrub}
    onpointercancel={endScrub}
  >
    <Waveform {peaks} {played} />
    <span class="playhead" data-testid="audio-playhead" style:left="{played * 100}%"></span>
  </div>
  <!-- A new element for each file: a source change on a playing element goes back to paused without a `pause` event, which can
       leave the browser's own button on Pause. A fresh element starts on Play. A fresh copy of the same file keeps its element. -->
  {#key src}
    <!-- svelte-ignore a11y_media_has_caption because a take has no captions to show -->
    <audio
      bind:this={audio}
      {src}
      controls
      preload="metadata"
      aria-label="Play {filename}"
      onplay={() => audio && claimPlayback(audio)}
      ontimeupdate={sync}
      onseeked={sync}
      onloadedmetadata={sync}
      onended={sync}
    ></audio>
  {/key}
</div>

<style>
  .audio-player { display: flex; flex-direction: column; gap: 12px; width: 100%; padding: 44px 16px 12px; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  .audio-player[data-size='viewer'] { max-width: 960px; padding: 16px 16px 12px; }
  .audio-wave { position: relative; display: block; width: 100%; height: 140px; cursor: pointer; touch-action: pan-y; user-select: none; -webkit-user-select: none; }
  .audio-player[data-size='viewer'] .audio-wave { height: 200px; }
  .playhead { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px; border-radius: 1px; background: var(--color-text-primary); pointer-events: none; }
  audio { display: block; width: 100%; height: 40px; }
  @media (pointer: coarse) { audio { height: 44px; } }
  /* A phone: a shorter waveform and tighter padding; the stage panel keeps the top room its 36px corner actions need. */
  .audio-player[data-compact='true'] { padding: 52px 12px 10px; }
  .audio-player[data-compact='true'][data-size='viewer'] { padding: 12px 12px 10px; }
  .audio-player[data-compact='true'] .audio-wave { height: 120px; }
  /* The file browser's preview pane, strip and sheet: a low waveform in a narrow box, on every screen. */
  .audio-player[data-size='pane'], .audio-player[data-size='pane'][data-compact='true'] { gap: 8px; padding: 10px 10px 8px; }
  .audio-player[data-size='pane'] .audio-wave, .audio-player[data-size='pane'][data-compact='true'] .audio-wave { height: 88px; }
</style>
