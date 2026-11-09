<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { InspectorRow, PathBrowser } from '$lib/components/molecules';
  import ReferenceThumbContent from './ReferenceThumbContent.svelte';
  import { AUDIO_ACCEPT, takeAudioFiles } from '$lib/state/audioTrack';
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
    /** Prefix of the slot's element ids (`<idBase>-browse`, `-file`, `-clear` and `-label`). */
    idBase: string;
    /** The image row's label. */
    label: string;
    names: ReferenceSlotNames;
    /** Shown in the name column while the slot is empty. */
    emptyText: string;
    /** The empty text reads as optional rather than as a prompt to act. */
    emptyMuted?: boolean;
    /** The host path chosen; null when none. */
    path: string | null;
    /** An image uploaded from this device, attached on submit only while it is the slot's image. */
    file: File | null;
    onfilechange: (file: File | null) => void;
    onpathchange: (path: string | null) => void;
    /** The glyph in the thumbnail while the slot is empty. */
    emptyIcon?: IconName;
    /** Show the clear button; a slot with no image has nothing to clear. */
    clearable?: boolean;
    /** The upload takes several files at once and hands them to `onfileschange` (in order) instead of `onfilechange`. */
    onfileschange?: (files: File[]) => void;
    /** Shown in place of the name while a file is dragged over the slot. */
    dropText?: string;
    /** The image's file is gone: the thumbnail and name show it, and `error` explains. */
    missing?: boolean;
    /** The explanation under a missing image; the slot's group points at it. */
    error?: string;
    /** The image's name; empty while the slot holds none. */
    name?: string;
    /** A short fact shown after the name that stays whole when the name is cut short (an audio track's length). */
    detail?: string;
    /** The image is kept but not used: the thumbnail fades to grey and the label and name turn muted. */
    unused?: boolean;
    /** What the slot holds: an image (a picture of it) or an audio track (the waveform icon in the thumbnail's place). */
    kind?: 'image' | 'audio';
    /** Show the thumbnail before the name; off for a row that is just a name and its facts (a voice recording). */
    thumb?: boolean;
    /** Browse and upload stay beside a file that is chosen; off for a row whose clear button is the way to another file (a voice recording). A slot with no usable file offers them either way. */
    swappable?: boolean;
  }

  let {
    idBase, label, names, emptyText, emptyMuted = false, path, file, onfilechange, onpathchange,
    emptyIcon = 'reference', clearable = true, onfileschange, dropText, missing = false, error, name = '', detail = '', unused = false, kind = 'image', thumb = true, swappable = true,
  }: Props = $props();

  const audio = $derived(kind === 'audio');
  const noun = $derived(audio ? 'an audio file' : 'an image');

  let dragOver = $state(false);
  let fileInput = $state<HTMLInputElement | null>(null);
  let browsing = $state(false);

  // Forget a browsed file once it stops being the slot's image, so choosing it again fires a change.
  $effect(() => {
    if (!file && fileInput) fileInput.value = '';
  });

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
    if (audio) {
      takeAudioFiles(dropped, onfilechange);
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
  class:unused
  data-unused={unused}
  data-kind={kind}
  role="group"
  aria-label={name ? `${label}: ${name}` : label}
  aria-describedby={error ? `${idBase}-error` : undefined}
  data-dragover={dragOver}
  ondragover={(e) => { e.preventDefault(); dragOver = true; }}
  ondragleave={() => { dragOver = false; }}
  ondrop={handleDrop}
>
  <InspectorRow {label} labelId="{idBase}-label">
    {#if thumb}
      <div class="reference-thumb" class:missing data-testid="slot-thumb" data-missing={missing}>
        {#if missing}
          <Icon name="alert" size={16} />
        {:else if audio}
          <Icon name="waveform" size={18} />
        {:else}
          <ReferenceThumbContent {file} {path} alt={names.preview} {emptyIcon} />
        {/if}
      </div>
    {/if}
    {#if dragOver && dropText}
      <span class="reference-name drop">{dropText}</span>
    {:else}
      <span class="reference-name" class:empty={!name && emptyMuted} class:missing title={name}>{name || emptyText}</span>
      {#if detail}<span class="reference-detail" data-testid="{idBase}-detail">{detail}</span>{/if}
    {/if}
    {#if swappable || !name || missing}
      <button id="{idBase}-browse" type="button" class="ui-btn ui-btn-row" title="Browse for {noun}" aria-label={names.browse} onclick={() => { browsing = true; }}><Icon name="folder" size={14} /></button>
      <!-- No name: the chosen file is attached on submit only while it is the slot's image (see WorkspacePage). -->
      <input id="{idBase}-file" bind:this={fileInput} type="file" accept={audio ? AUDIO_ACCEPT : IMAGE_ACCEPT} multiple={onfileschange !== undefined} class="hidden" onchange={(e) => handleFiles(e.currentTarget)}>
      <label for="{idBase}-file" class="ui-btn ui-btn-row" title="Upload {noun} from this device" aria-label={names.upload}><Icon name="upload" size={14} /></label>
    {/if}
    {#if clearable}
      <button id="{idBase}-clear" type="button" class="ui-btn ui-btn-row" aria-label={names.clear} title="Clear" onclick={clear}><Icon name="close" size={14} /></button>
    {/if}
  </InspectorRow>
  {#if error}<p id="{idBase}-error" class="row-hint ui-help-error">{error}</p>{/if}
</div>
<PathBrowser bind:open={browsing} purpose={audio ? 'audio_track' : 'reference_image'} initialPath={path} onchoose={choosePath} />

<style>
  .reference-drop { border-radius: var(--radius-sm); }
  .reference-drop[data-dragover='true'] { outline: 2px dashed var(--color-primary-main); outline-offset: -2px; }
  .reference-thumb { display: grid; flex: none; place-items: center; width: 40px; height: 40px; margin: 4px 0; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .reference-thumb :global(img) { width: 100%; height: 100%; object-fit: cover; }
  .reference-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); }
  .reference-detail { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; }
  /* A slot that is kept but not used: the picture fades to grey, the words stay readable. */
  .reference-drop.unused .reference-thumb :global(img) { opacity: 0.4; filter: grayscale(1); }
  .reference-drop.unused :global(.inspector-label), .reference-drop.unused .reference-name { color: var(--color-text-muted); }
  .reference-name.empty { color: var(--color-text-muted); }
  .reference-name.drop { color: var(--color-primary-main); }
  /* A file that is gone: the thumbnail takes the error surface and the name the error colour. */
  .reference-thumb.missing { background: var(--color-error-surface); color: var(--color-error); }
  .reference-name.missing { color: var(--color-error); }
  /* A finger gets 44px targets on the track row. */
  @media (pointer: coarse) {
    .reference-drop[data-kind='audio'] :global(.ui-btn-row) { width: 44px; height: 44px; }
  }
</style>
