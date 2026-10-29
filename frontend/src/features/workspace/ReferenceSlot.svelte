<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { InspectorRow, PathBrowser, PathInput } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import ReferenceThumbContent from './ReferenceThumbContent.svelte';
  import { IMAGE_ACCEPT, splitImageFiles, unsupportedMessage } from '$lib/state/references';
  import { addToast } from '$lib/state/toasts.svelte';

  /** The accessible names of the slot's buttons and thumbnail. */
  interface ReferenceSlotNames {
    browse: string;
    upload: string;
    clear: string;
    preview: string;
  }

  interface Props {
    /** Prefix of the slot's element ids (`<idBase>-browse`, `-file`, `-path` and `-label`). */
    idBase: string;
    /** The image row's label. */
    label: string;
    names: ReferenceSlotNames;
    /** The path field's form name; the path is submitted under it. */
    pathName?: string;
    placeholder?: string;
    /** Shown in the name column while the slot is empty. */
    emptyText: string;
    /** The empty text reads as optional rather than as a prompt to act. */
    emptyMuted?: boolean;
    /** The host path chosen or typed; null when none. */
    path: string | null;
    /** An image uploaded from this device, attached on submit only while it is the slot's image. */
    file: File | null;
    /** Preview for a path that points at a known asset. */
    previewUrl: string | null;
    onfilechange: (file: File | null) => void;
    onpathchange: (path: string | null) => void;
    /** Show the Path row under the image row; a slot whose paths are submitted elsewhere hides it. */
    pathRow?: boolean;
    /** The glyph in the thumbnail while the slot is empty. */
    emptyIcon?: IconName;
    /** Show the clear button; an empty add slot has nothing to clear. */
    clearable?: boolean;
    /** The upload takes several files at once and hands them to `onfileschange` (in order) instead of `onfilechange`. */
    onfileschange?: (files: File[]) => void;
    /** Shown in place of the name while a file is dragged over the slot. */
    dropText?: string;
    /** The image's file is gone: the thumbnail and name show it, and `error` explains. */
    missing?: boolean;
    /** The explanation under a missing image; the slot's group points at it. */
    error?: string;
  }

  let {
    idBase, label, names, pathName, placeholder, emptyText, emptyMuted = false, path, file, previewUrl, onfilechange, onpathchange,
    pathRow = true, emptyIcon = 'reference', clearable = true, onfileschange, dropText, missing = false, error,
  }: Props = $props();

  let dragOver = $state(false);
  let fileInput = $state<HTMLInputElement | null>(null);
  let browsing = $state(false);

  // Forget a browsed file once it stops being the slot's image, so choosing it again fires a change.
  $effect(() => {
    if (!file && fileInput) fileInput.value = '';
  });

  const name = $derived(file?.name ?? (path ? fileName(path) : ''));

  /** Take a host path as the image, replacing an uploaded file. */
  function choosePath(chosen: string): void {
    onfilechange(null);
    onpathchange(chosen);
  }

  function clear(): void {
    onfilechange(null);
    onpathchange(null);
  }

  function handleDrop(event: DragEvent): void {
    event.preventDefault();
    dragOver = false;
    const dropped = event.dataTransfer?.files ?? [];
    // A slot that takes several files hands them all over, in order, and its owner turns away what it cannot use.
    if (onfileschange) {
      if (dropped.length > 0) onfileschange([...dropped]);
      return;
    }
    const { supported, unsupported } = splitImageFiles(dropped);
    if (supported[0]) onfilechange(supported[0]);
    else if (unsupported > 0) addToast(unsupportedMessage(unsupported), 'info');
  }

  function handleFiles(input: HTMLInputElement): void {
    if (!onfileschange) {
      onfilechange(input.files?.[0] ?? null);
      return;
    }
    const chosen = [...(input.files ?? [])];
    // Forget the choice at once, so choosing the same files again still fires a change.
    input.value = '';
    if (chosen.length > 0) onfileschange(chosen);
  }
</script>

<div
  class="reference-drop"
  role="group"
  aria-labelledby="{idBase}-label"
  aria-describedby={error ? `${idBase}-error` : undefined}
  data-dragover={dragOver}
  ondragover={(e) => { e.preventDefault(); dragOver = true; }}
  ondragleave={() => { dragOver = false; }}
  ondrop={handleDrop}
>
  <InspectorRow {label} labelId="{idBase}-label">
    <div class="reference-thumb" class:missing>
      {#if missing}
        <Icon name="alert" size={16} />
      {:else}
        <ReferenceThumbContent {file} url={previewUrl} alt={names.preview} {emptyIcon} />
      {/if}
    </div>
    {#if dragOver && dropText}
      <span class="reference-name drop">{dropText}</span>
    {:else}
      <span class="reference-name" class:empty={!name && emptyMuted} class:missing title={name}>{name || emptyText}</span>
    {/if}
    <button id="{idBase}-browse" type="button" class="ui-btn ui-btn-row" title="Browse for an image" aria-label={names.browse} onclick={() => { browsing = true; }}><Icon name="folder" size={14} /></button>
    <!-- No name: the chosen file is attached on submit only while it is the slot's image (see WorkspacePage). -->
    <input id="{idBase}-file" bind:this={fileInput} type="file" accept={IMAGE_ACCEPT} multiple={onfileschange !== undefined} class="hidden" onchange={(e) => handleFiles(e.currentTarget)}>
    <label for="{idBase}-file" class="ui-btn ui-btn-row" title="Upload an image from this device" aria-label={names.upload}><Icon name="upload" size={14} /></label>
    {#if clearable}
      <button id="{idBase}-clear" type="button" class="ui-btn ui-btn-row" aria-label={names.clear} title="Clear" onclick={clear}><Icon name="close" size={14} /></button>
    {/if}
  </InspectorRow>
  {#if error}<p id="{idBase}-error" class="row-hint ui-help-error">{error}</p>{/if}
  {#if pathRow}
    <InspectorRow label="Path" forId="{idBase}-path">
      <PathInput
        id="{idBase}-path"
        name={pathName}
        purpose="reference_image"
        class="inspector-text"
        plain
        {placeholder}
        value={path ?? ''}
        oninput={(text) => onpathchange(text || null)}
        onpick={choosePath}
      />
    </InspectorRow>
  {/if}
</div>
<PathBrowser bind:open={browsing} purpose="reference_image" initialPath={path} onchoose={choosePath} />

<style>
  .reference-drop { border-radius: var(--radius-sm); }
  .reference-drop[data-dragover='true'] { outline: 2px dashed var(--color-primary-main); outline-offset: -2px; }
  .reference-thumb { display: grid; flex: none; place-items: center; width: 40px; height: 40px; margin: 4px 0; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .reference-thumb :global(img) { width: 100%; height: 100%; object-fit: cover; }
  .reference-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); }
  .reference-name.empty { color: var(--color-text-muted); }
  .reference-name.drop { color: var(--color-primary-main); }
  /* A file that is gone: the thumbnail takes the error surface and the name the error colour. */
  .reference-thumb.missing { background: var(--color-error-surface); color: var(--color-error); }
  .reference-name.missing { color: var(--color-error); }
</style>
