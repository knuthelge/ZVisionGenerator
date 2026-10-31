<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { PathBrowser } from '$lib/components/molecules';
  import { galleryStore } from '$lib/state/gallery.svelte';
  import { IMAGE_ACCEPT } from '$lib/state/references';

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
  }

  let { multiple = false, browseFrom = null, onfiles, onpath, deviceId }: Props = $props();

  let fileInput = $state<HTMLInputElement | null>(null);
  let browseOpen = $state(false);
  const latest = $derived(galleryStore.assets[0]?.media_type === 'image' ? (galleryStore.assets[0].file_path ?? null) : null);

  function picked(input: HTMLInputElement): void {
    const chosen = [...(input.files ?? [])];
    // Forget the choice at once, so choosing the same files again still fires a change.
    input.value = '';
    if (chosen.length > 0) onfiles(chosen);
  }
</script>

<div class="img-sources" role="group" aria-label="Where the image comes from">
  <button id={deviceId} type="button" class="src-item" onclick={() => fileInput?.click()}><Icon name="upload" size={16} />Choose from this device<small>{multiple ? 'one or more' : 'photos or files'}</small></button>
  <!-- No name: the chosen file is attached on submit only while it is the slot's image (see WorkspacePage). -->
  <input bind:this={fileInput} type="file" accept={IMAGE_ACCEPT} {multiple} class="hidden" tabindex="-1" aria-hidden="true" onchange={(e) => picked(e.currentTarget)}>
  <button type="button" class="src-item" onclick={() => { browseOpen = true; }}><Icon name="folder" size={16} />Browse the server<small>output and reference folders</small></button>
  {#if latest}
    <button type="button" class="src-item" onclick={() => onpath(latest)}><Icon name="film" size={16} />Use the latest output</button>
  {/if}
</div>
<PathBrowser bind:open={browseOpen} purpose="reference_image" initialPath={browseFrom} onchoose={onpath} />

<style>
  .img-sources { display: flex; flex-direction: column; gap: 2px; }
  .src-item { display: flex; align-items: center; gap: 12px; min-height: 48px; padding: 0 12px; border-radius: var(--radius-sm); font-size: 15px; white-space: nowrap; text-align: left; color: var(--color-text-primary); }
  .src-item:active { background: var(--color-bg-surface-hover); }
  .src-item small { flex: none; max-width: 42%; margin-left: auto; overflow: hidden; text-overflow: ellipsis; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-muted); }
</style>
