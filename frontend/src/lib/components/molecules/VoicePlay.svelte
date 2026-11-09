<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { claimPlayback, releasePlayback } from '$lib/state/nowPlaying';

  interface Props {
    /** The clip's address on the server; null while there is none, which turns Play off. */
    src: string | null;
    /** A recording held in the browser: it plays from an object URL and wins over `src`. */
    file?: File | null;
    /** What the button is named after: "Play Morgan". */
    label: string;
    class?: string;
  }

  let { src, file = null, label, class: extraClass = '' }: Props = $props();

  let fileUrl = $state<string | null>(null);
  let audio = $state<HTMLAudioElement | null>(null);
  let playing = $state(false);

  // The object URL lives as long as the file is the one shown.
  $effect(() => {
    const made = file ? URL.createObjectURL(file) : null;
    fileUrl = made;
    return () => { if (made) URL.revokeObjectURL(made); };
  });

  const url = $derived(file ? fileUrl : src);

  function toggle(): void {
    if (!audio) return;
    if (audio.paused) void audio.play()?.catch(() => { playing = false; });
    else audio.pause();
  }

  // A clip never keeps playing when its button goes or another file loads; each file has its own audio element (the `{#key}` below).
  $effect(() => {
    const element = audio;
    playing = false;
    return () => {
      if (!element) return;
      element.pause();
      releasePlayback(element);
    };
  });
</script>

<!-- Only Play: the button is the control (a screen reader gets a toggle button), and the pressed state shows Pause. -->
<button
  type="button"
  class="ui-btn ui-btn-icon audio-button {extraClass}"
  aria-label="Play {label}"
  aria-pressed={playing}
  disabled={url === null}
  data-compact={layout.compact}
  data-testid="audio-row-play"
  onclick={toggle}
>
  <Icon name={playing ? 'pause' : 'play'} size={layout.compact ? 18 : 14} />
</button>
{#if url !== null}
  {#key url}
    <!-- svelte-ignore a11y_media_has_caption because a clip has no captions to show -->
    <audio
      bind:this={audio}
      src={url}
      preload="none"
      onplay={() => { playing = true; if (audio) claimPlayback(audio); }}
      onpause={() => { playing = false; }}
      onended={() => { playing = false; }}
    ></audio>
  {/key}
{/if}

<style>
  /* The 22 px (--spacing-control-sm) Play of a row; 32 px for a finger, 44 px on a phone. */
  .audio-button { flex: none; width: var(--spacing-control-sm); height: var(--spacing-control-sm); }
  @media (pointer: coarse) { .audio-button { width: 32px; height: 32px; } }
  .audio-button[data-compact='true'] { width: 44px; height: 44px; }
</style>
