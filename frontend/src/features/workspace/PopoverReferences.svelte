<script lang="ts">
  import { tick } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { galleryStore } from '$lib/state/gallery.svelte';
  import { splitImageFiles } from '$lib/state/references';
  import type { ImagesView } from './imagesView';
  import ImageSources from './ImageSources.svelte';
  import { referenceEditor } from './referenceEditor';
  import ReferenceThumbContent from './ReferenceThumbContent.svelte';

  interface Props {
    /** The reference list of the images view: the rows the list shows and Generate sends, every row of the draft, and how many the model takes. */
    references: NonNullable<ImagesView['references']>;
  }

  let { references }: Props = $props();
  const { rows, max, hasRoom } = $derived(references);

  // Each row has a small thumbnail, its number over its name and a remove button. A row whose image is gone also has Replace:
  // it points the add items below at that row, so the new image keeps its number.
  const editor = referenceEditor(() => max);
  let replacing = $state<number | null>(null);
  // The row being replaced, as long as it is still there and still gone.
  const target = $derived(replacing !== null && rows[replacing]?.gone ? replacing : null);

  async function remove(index: number): Promise<void> {
    editor.remove(index);
    await tick();
    // Focus stays in the popover: the remove button now in this position, else the first way to add one.
    (document.getElementById(`pop-ref-${index + 1}-remove`) ?? document.getElementById('pop-ref-add-device'))?.focus();
  }

  function chooseFiles(files: File[]): void {
    if (target === null) {
      editor.appendFiles(files);
      return;
    }
    const [first] = splitImageFiles(files).supported;
    if (first) editor.replaceWithFile(target, first);
    else editor.appendFiles(files);
  }

  /** Take a server path as the row being replaced, or as the next row when none is. */
  function choosePath(path: string): void {
    if (target === null) editor.append([{ path, file: null }]);
    else editor.replaceWithPath(target, path);
  }
</script>

{#if rows.length > 0}
  <div class="refs">
    {#each rows as row, index (index)}
      {@const number = index + 1}
      <div class="refrow" role="group" aria-label="Reference {number}">
        <span class="ref-thumb" class:missing={row.gone}>{#if row.gone}<Icon name="alert" size={16} />{:else}<ReferenceThumbContent file={row.file} url={galleryStore.previewUrlFor(row.path)} alt="Reference {number} preview" />{/if}</span>
        <span class="ref-text">
          <span class="ref-num">Reference {number}</span>
          <span class="ref-name" class:missing={row.gone} title={row.name}>{row.name}</span>
          {#if row.error}<span class="ref-error">{row.error}</span>{/if}
        </span>
        {#if row.gone}<button id="pop-ref-{number}-replace" type="button" class="ui-btn ui-btn-sm" aria-pressed={target === index} aria-label="Replace reference {number}" onclick={() => { replacing = target === index ? null : index; }}>Replace</button>{/if}
        <button id="pop-ref-{number}-remove" type="button" class="ui-btn ui-btn-icon" aria-label="Remove reference {number}" onclick={() => void remove(index)}><Icon name="close" size={16} /></button>
      </div>
    {/each}
  </div>
{/if}
{#if hasRoom || target !== null}
  <div class="add">
    <h5 class="ui-area-label add-label">{target === null ? `Add reference ${rows.length + 1}` : `Replace reference ${target + 1}`}</h5>
    <ImageSources deviceId="pop-ref-add-device" multiple={target === null} onfiles={chooseFiles} onpath={choosePath} />
  </div>
{:else}
  <p class="ui-help">That's all {max} this model takes. Remove one to add another.</p>
{/if}
<p class="ui-help">Refer to them in the prompt as image 1, image 2…</p>

<style>
  .refs { display: flex; flex-direction: column; }
  .refrow { display: flex; align-items: center; gap: 12px; min-height: 52px; font-size: 15px; }
  .refrow + .refrow { border-top: 1px solid var(--color-border-subtle); }
  .ref-thumb { display: grid; flex: none; place-items: center; width: 40px; height: 40px; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .ref-thumb :global(img) { width: 100%; height: 100%; object-fit: cover; }
  .ref-thumb.missing { background: var(--color-error-surface); color: var(--color-error); }
  .ref-text { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 1px; }
  .ref-num { font-size: var(--text-ui); color: var(--color-text-muted); }
  .ref-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ref-name.missing, .ref-error { color: var(--color-error); }
  .ref-error { font-size: var(--text-meta); line-height: 1.4; }
  .add-label { padding: 0 0 4px; }
</style>
