<script lang="ts">
  import { tick } from 'svelte';
  import { InspectorSection } from '$lib/components/molecules';
  import { draft } from '$lib/state/draft.svelte';
  import { galleryStore } from '$lib/state/gallery.svelte';
  import { isMissingRow, missingReason, referenceRows } from '$lib/state/references';
  import { referenceEditor } from './referenceEditor';
  import ReferenceSlot from './ReferenceSlot.svelte';

  interface Props {
    /** How many references the model takes. */
    max: number;
    /** The files uploaded from this device, by row; the Workspace keeps them and sends them on submit. */
    files: (File | null)[];
    onfileschange: (files: (File | null)[]) => void;
  }

  let { max, files, onfileschange }: Props = $props();

  // A list longer than the model takes keeps its other rows out of sight (and out of Generate), and a row removed here brings one back.
  const allRows = $derived(referenceRows(draft.state.referenceImagePaths, files));
  const rows = $derived(allRows.slice(0, max));
  const missing = $derived(draft.state.missingReferencePaths);
  const hasRoom = $derived(allRows.length < max);

  const editor = referenceEditor({ rows: () => allRows, max: () => max, onfileschange: (next) => onfileschange(next) });

  async function remove(index: number): Promise<void> {
    editor.remove(index);
    await tick();
    // Focus stays in the list: the remove button now in this position, else the add row.
    (document.getElementById(`ws-ref-${index + 1}-clear`) ?? document.getElementById('ws-ref-add-browse'))?.focus();
  }
</script>

<InspectorSection id="references" title="References" summary={rows.length > 0 ? `${rows.length} of ${max}` : 'None'} active={rows.length > 0}>
  {#each rows as row, index (index)}
    {@const number = index + 1}
    {@const gone = isMissingRow(row, missing)}
    <ReferenceSlot
      idBase="ws-ref-{number}"
      label="Image {number}"
      names={{ browse: `Browse for image ${number}`, upload: `Upload image ${number} from this device`, clear: `Remove image ${number}`, preview: `Image ${number} preview` }}
      pathRow={false}
      emptyText=""
      path={row.path || null}
      file={row.file}
      previewUrl={galleryStore.previewUrlFor(row.path)}
      dropText="Drop to replace"
      missing={gone}
      error={gone ? missingReason(row) : undefined}
      onfilechange={(file) => editor.replaceWithFile(index, file)}
      onpathchange={(path) => (path === null ? void remove(index) : editor.replaceWithPath(index, path))}
    />
  {/each}
  {#if hasRoom}
    <ReferenceSlot
      idBase="ws-ref-add"
      label="Image {rows.length + 1}"
      names={{ browse: `Browse for image ${rows.length + 1}`, upload: `Upload image ${rows.length + 1} from this device`, clear: '', preview: '' }}
      pathRow={false}
      emptyText="Add reference"
      emptyMuted
      emptyIcon="plus"
      clearable={false}
      path={null}
      file={null}
      previewUrl={null}
      onfilechange={() => {}}
      onfileschange={(chosen) => editor.appendFiles(chosen)}
      onpathchange={(path) => { if (path) editor.append([{ path, file: null }]); }}
    />
  {/if}
  {#if rows.length > 0}<p class="row-hint">Refer to them in the prompt as image 1, image 2…</p>{/if}
</InspectorSection>
