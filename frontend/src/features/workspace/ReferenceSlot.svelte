<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { InspectorRow, PathBrowser, PathInput } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';

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
    pathName: string;
    placeholder: string;
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
  }

  let { idBase, label, names, pathName, placeholder, emptyText, emptyMuted = false, path, file, previewUrl, onfilechange, onpathchange }: Props = $props();

  let dragOver = $state(false);
  let filePreviewUrl = $state<string | null>(null);
  let fileInput = $state<HTMLInputElement | null>(null);
  let browsing = $state(false);

  // Revoke the previous object URL whenever the file changes or the slot unmounts.
  $effect(() => {
    const url = file ? URL.createObjectURL(file) : null;
    filePreviewUrl = url;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  });

  // Forget a browsed file once it stops being the slot's image, so choosing it again fires a change.
  $effect(() => {
    if (!file && fileInput) fileInput.value = '';
  });

  const preview = $derived(filePreviewUrl ?? previewUrl);
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
    const dropped = event.dataTransfer?.files[0] ?? null;
    if (dropped && dropped.type.startsWith('image/')) onfilechange(dropped);
  }
</script>

<div
  class="reference-drop"
  role="group"
  aria-labelledby="{idBase}-label"
  data-dragover={dragOver}
  ondragover={(e) => { e.preventDefault(); dragOver = true; }}
  ondragleave={() => { dragOver = false; }}
  ondrop={handleDrop}
>
  <InspectorRow {label} labelId="{idBase}-label">
    <div class="reference-thumb">
      {#if preview}
        <img src={preview} alt={names.preview}>
      {:else}
        <Icon name="reference" size={16} />
      {/if}
    </div>
    <span class="reference-name" class:empty={!name && emptyMuted} title={name}>{name || emptyText}</span>
    <button id="{idBase}-browse" type="button" class="ui-btn ui-btn-row" title="Browse for an image" aria-label={names.browse} onclick={() => { browsing = true; }}><Icon name="folder" size={14} /></button>
    <!-- No name: the chosen file is attached on submit only while it is the slot's image (see WorkspacePage). -->
    <input id="{idBase}-file" bind:this={fileInput} type="file" accept="image/png,image/jpeg,image/webp" class="hidden" onchange={(e) => onfilechange(e.currentTarget.files?.[0] ?? null)}>
    <label for="{idBase}-file" class="ui-btn ui-btn-row" title="Upload an image from this device" aria-label={names.upload}><Icon name="upload" size={14} /></label>
    <button type="button" class="ui-btn ui-btn-row" aria-label={names.clear} title="Clear" onclick={clear}><Icon name="close" size={14} /></button>
  </InspectorRow>
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
</div>
<PathBrowser bind:open={browsing} purpose="reference_image" initialPath={path} onchoose={choosePath} />

<style>
  .reference-drop { border-radius: var(--radius-sm); }
  .reference-drop[data-dragover='true'] { outline: 2px dashed var(--color-primary-main); outline-offset: -2px; }
  .reference-thumb { display: grid; flex: none; place-items: center; width: 40px; height: 40px; margin: 4px 0; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .reference-thumb img { width: 100%; height: 100%; object-fit: cover; }
  .reference-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); }
  .reference-name.empty { color: var(--color-text-muted); }
</style>
