<script lang="ts">
  import { Icon } from '$lib/components/atoms';
    import { splitImageFiles, unsupportedMessage } from '$lib/state/references';
  import { addToast } from '$lib/state/toasts.svelte';
  import ImageSources from './ImageSources.svelte';
  import ReferenceThumbContent from './ReferenceThumbContent.svelte';

  interface Props {
    /** What the slot is called, for the thumbnail's alt text. */
    title: string;
    /** The image's name, or empty while the slot is empty. */
    name: string;
    path: string | null;
    file: File | null;
    /** Shown in the name's place while the slot is empty. */
    emptyText: string;
    clearLabel: string;
    /** The image is gone; `error` says what to do. */
    missing?: boolean;
    error?: string;
    /** The image is kept but not used: the thumbnail fades to grey. */
    unused?: boolean;
    /** The one line under the picker, if any. */
    help?: string;
    /** An image uploaded from this device (the first supported file of a choice). */
    onfile: (file: File) => void;
    /** A path on the server: browsed or the latest output. */
    onpath: (path: string) => void;
    onclear: () => void;
  }

  let { title, name, path, file, emptyText, clearLabel, missing = false, error, unused = false, help, onfile, onpath, onclear }: Props = $props();

  function takeFiles(files: File[]): void {
    const { supported, unsupported } = splitImageFiles(files);
    if (supported[0]) onfile(supported[0]);
    else if (unsupported > 0) addToast(unsupportedMessage(unsupported), 'info');
  }
</script>

<div class="picker" class:unused data-unused={unused}>
  <div class="picker-row">
    <span class="picker-thumb" class:missing data-testid="slot-thumb" data-missing={missing}>{#if missing}<Icon name="alert" size={20} />{:else}<ReferenceThumbContent {file} {path} alt="{title} preview" />{/if}</span>
    <span class="picker-name" class:muted={!name} class:missing title={name}>{name || emptyText}</span>
    {#if name}<button type="button" class="ui-btn ui-btn-icon" aria-label={clearLabel} onclick={onclear}><Icon name="close" size={16} /></button>{/if}
  </div>
  {#if error}<p class="ui-help ui-help-error">{error}</p>{/if}
  <ImageSources browseFrom={path} onfiles={takeFiles} {onpath} />
  {#if help}<p class="ui-help">{help}</p>{/if}
</div>

<style>
  .picker { display: flex; flex-direction: column; gap: 12px; }
  .picker-row { display: flex; align-items: center; gap: 12px; min-height: 56px; font-size: 15px; }
  .picker-thumb { display: grid; flex: none; place-items: center; width: 56px; height: 56px; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .picker-thumb :global(img) { width: 100%; height: 100%; object-fit: cover; }
  .picker-thumb.missing { background: var(--color-error-surface); color: var(--color-error); }
  .picker-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .picker-name.muted { color: var(--color-text-muted); }
  .picker-name.missing { color: var(--color-error); }
  /* A last frame that is kept but not sent: the picture fades to grey, the words stay readable. */
  .picker.unused .picker-thumb :global(img) { opacity: 0.4; filter: grayscale(1); }
  .picker.unused .picker-name { color: var(--color-text-muted); }
</style>
