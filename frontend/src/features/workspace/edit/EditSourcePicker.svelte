<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { splitImageFiles, unsupportedMessage } from '$lib/state/references';
  import { addToast } from '$lib/state/toasts.svelte';
  import ImageSources from '../ImageSources.svelte';
  import { chooseSource } from './editActions';
  import { sourceOfAsset, sourceOfFile, sourceOfPath } from './editSource';

  interface Props {
    /** Phone card: the buttons sit in one row under a single line of text. */
    compact?: boolean;
  }

  let { compact = false }: Props = $props();

  let dragOver = $state(false);

  function takeFiles(files: FileList | File[]): void {
    const { supported, unsupported } = splitImageFiles(files);
    if (supported[0]) chooseSource(sourceOfFile(supported[0]));
    else if (unsupported > 0) addToast(unsupportedMessage(unsupported), 'info');
  }

  function handleDrop(event: DragEvent): void {
    event.preventDefault();
    dragOver = false;
    if (event.dataTransfer?.files) takeFiles(event.dataTransfer.files);
  }
</script>

<div
  class="edit-drop"
  class:compact
  data-dragover={dragOver}
  data-testid="edit-source-picker"
  role="group"
  aria-label="Choose an image to edit"
  ondragover={(e) => { e.preventDefault(); dragOver = true; }}
  ondragleave={() => { dragOver = false; }}
  ondrop={handleDrop}
>
  <p class="edit-drop-text"><Icon name="reference" size={16} />{dragOver ? 'Drop to choose' : 'Choose an image to edit'}</p>
  <ImageSources layout="buttons" onfiles={takeFiles} onpath={(path) => chooseSource(sourceOfPath(path))} onlatest={(asset) => chooseSource(sourceOfAsset(asset))} />
</div>

<style>
  /* The dashed edge of a drop zone: one of the few outlines the design keeps. */
  .edit-drop { display: flex; flex-direction: column; align-items: center; gap: 10px; margin: 6px 12px; padding: 14px 8px; border: 1px dashed var(--color-border-strong); border-radius: var(--radius-md); }
  .edit-drop[data-dragover='true'] { border-color: var(--color-primary-main); }
  .edit-drop-text { display: flex; align-items: center; gap: 8px; margin: 0; font-size: var(--text-ui); color: var(--color-text-secondary); }
  .edit-drop[data-dragover='true'] .edit-drop-text { color: var(--color-primary-main); }
  .edit-drop.compact { flex-direction: row; flex-wrap: wrap; justify-content: space-between; margin: 0; padding: 10px 12px; --src-row-justify: flex-start; }
</style>
