<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { PathBrowser } from '$lib/components/molecules';
  import { galleryStore } from '$lib/state/gallery.svelte';
  import { AUDIO_ACCEPT } from '$lib/state/audioTrack';
  import { IMAGE_ACCEPT } from '$lib/state/references';
  import type { GalleryAsset } from '$lib/types';

  interface Props {
    /** The slot takes several files at once (the references). */
    multiple?: boolean;
    /** The path the server browser opens at. */
    browseFrom?: string | null;
    /** Files chosen from this device. */
    onfiles: (files: File[]) => void;
    /** A path on the server: browsed or the latest output. */
    onpath: (path: string) => void;
    /** The id of the first button, for a caller that moves focus to it. */
    deviceId?: string;
    /** What the sources hold: images, or an audio track (no latest output, the server's audio files to browse). */
    kind?: 'image' | 'audio';
    /** `list` is the menu of rows; `buttons` is one compact row of buttons that always shows the latest output, disabled while there is none. */
    layout?: 'list' | 'buttons';
    /** `buttons` layout: the latest output as the asset, for a caller that needs more than its path. Without it the path goes to `onpath`. */
    onlatest?: (asset: GalleryAsset) => void;
  }

  let { multiple = false, browseFrom = null, onfiles, onpath, deviceId, kind = 'image', layout = 'list', onlatest }: Props = $props();

  const audio = $derived(kind === 'audio');
  const buttons = $derived(layout === 'buttons');

  let fileInput = $state<HTMLInputElement | null>(null);
  let browseOpen = $state(false);
  /** The list offers the newest asset when it is an image; the buttons offer the newest image that has a file. */
  const latestAsset = $derived.by(() => {
    if (audio) return null;
    if (buttons) return galleryStore.assets.find((asset) => asset.media_type === 'image' && asset.file_path) ?? null;
    const newest = galleryStore.assets[0];
    return newest?.media_type === 'image' && newest.file_path ? newest : null;
  });
  const latest = $derived(latestAsset?.file_path ?? null);

  function useLatest(): void {
    if (!latestAsset || !latest) return;
    if (onlatest) onlatest(latestAsset);
    else onpath(latest);
  }

  function picked(input: HTMLInputElement): void {
    const chosen = [...(input.files ?? [])];
    // Forget the choice at once, so choosing the same files again still fires a change.
    input.value = '';
    if (chosen.length > 0) onfiles(chosen);
  }
</script>

<div class={buttons ? 'src-row' : 'img-sources'} role="group" aria-label={audio ? 'Where the audio track comes from' : 'Where the image comes from'}>
  {#if buttons}
    <button id={deviceId} type="button" class="ui-btn" onclick={() => { browseOpen = true; }}><Icon name="folder" size={14} />Browse</button>
    <button type="button" class="ui-btn" onclick={() => fileInput?.click()}><Icon name="upload" size={14} />Upload</button>
    <button type="button" class="ui-btn" disabled={latest === null} title={latest ? undefined : 'No image has been made yet'} onclick={useLatest}><Icon name="clock" size={14} />Latest output</button>
  {:else}
    <button id={deviceId} type="button" class="src-item" onclick={() => fileInput?.click()}><Icon name="upload" size={16} />Choose from this device<small>{audio ? 'audio files' : multiple ? 'one or more' : 'photos or files'}</small></button>
    <button type="button" class="src-item" onclick={() => { browseOpen = true; }}><Icon name="folder" size={16} />Browse the server<small>{audio ? 'audio files' : 'output and reference folders'}</small></button>
    {#if latest}
      <button type="button" class="src-item" onclick={useLatest}><Icon name="film" size={16} />Use the latest output</button>
    {/if}
  {/if}
  <!-- No name: the chosen file is attached on submit only while it is the slot's image (see WorkspacePage). -->
  <input bind:this={fileInput} type="file" accept={audio ? AUDIO_ACCEPT : IMAGE_ACCEPT} {multiple} class="hidden" tabindex="-1" aria-hidden="true" onchange={(e) => picked(e.currentTarget)}>
</div>
<PathBrowser bind:open={browseOpen} purpose={audio ? 'audio_track' : 'reference_image'} initialPath={browseFrom} onchoose={onpath} />

<style>
  .src-row { display: flex; flex-wrap: wrap; justify-content: var(--src-row-justify, center); gap: 6px; }
  .img-sources { display: flex; flex-direction: column; gap: 2px; }
  .src-item { display: flex; align-items: center; gap: 12px; min-height: 48px; padding: 0 12px; border-radius: var(--radius-sm); font-size: 15px; white-space: nowrap; text-align: left; color: var(--color-text-primary); }
  .src-item:active { background: var(--color-bg-surface-hover); }
  .src-item small { flex: none; max-width: 42%; margin-left: auto; overflow: hidden; text-overflow: ellipsis; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-muted); }
</style>
