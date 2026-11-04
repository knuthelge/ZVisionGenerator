<script lang="ts">
  import { tick } from 'svelte';
  import { InspectorNumber } from '$lib/components/atoms';
  import { InspectorRow, InspectorSection } from '$lib/components/molecules';
  import type { ScrubOptions } from '$lib/actions/scrub';
  import { defaultsFor, draft } from '$lib/state/draft.svelte';
  import { rowChanged } from './settingRows';
  import { galleryStore } from '$lib/state/gallery.svelte';
  import type { SlotValue } from '$lib/state/imageSlots';
  import type { WorkspaceContext } from '$lib/types';
  import { imagesSummary } from './imagesSummary';
  import type { ImagesView } from './imagesView';
  import { referenceEditor } from './referenceEditor';
  import ReferenceSlot from './ReferenceSlot.svelte';
  import { rangeFill, SPEC } from './settingsSpec';

  interface Props {
    context: WorkspaceContext;
    /** The images view of the open tab, computed once by the Workspace. */
    images: ImagesView;
    /** A submit is in flight. */
    busy: boolean;
  }

  let { context, images: view, busy }: Props = $props();

  const s = $derived(draft.state);
  const video = $derived(view.tab === 'video');
  const tab = $derived(view.tab);

  const start = $derived(view.start);
  const last = $derived(view.last);
  const references = $derived(view.references);
  const refRows = $derived(references?.rows ?? []);
  const editor = referenceEditor(() => references?.max ?? 0);

  const startNoun = $derived(video ? 'a first frame' : 'a start image');

  // --- Strength ---------------------------------------------------------------
  const defaults = $derived(defaultsFor(context, s, draft.extendSource));
  const strengthChanged = $derived(rowChanged('strength', s, defaults));
  const strengthScrub = $derived<ScrubOptions>({ ...SPEC.strength, value: s.referenceImageStrength, onchange: setStrength, disabled: busy });

  function setStrength(value: number): void {
    draft.update('referenceImageStrength', value);
  }

  // --- Summary ----------------------------------------------------------------
  const summary = $derived(imagesSummary(view));

  // Every write goes through the draft's setters, so the workflow stays derived from the start image.
  function setStart(value: SlotValue): void {
    draft.setStartImage(tab, value, context);
  }

  // Focus stays in the list: the remove button now in this position, else the add row.
  async function removeReference(index: number): Promise<void> {
    editor.remove(index);
    await tick();
    (document.getElementById(`ws-ref-${index + 1}-clear`) ?? document.getElementById('ws-ref-add-browse'))?.focus();
  }
</script>

{#if view.visible}
  <InspectorSection id="images" title="Images" summary={summary.text} active={summary.tone === 'active'} error={summary.tone === 'error'}>
    {#if start}
      <div class="image-group" role="group" aria-labelledby="ws-start-group">
        <h4 id="ws-start-group" class="ui-area-label image-group-label">{video ? 'First frame' : 'Start image'}</h4>
        <ReferenceSlot
          idBase="ws-image"
          label="Image"
          names={{ browse: `Browse for ${startNoun}`, upload: `Upload ${startNoun} from this device`, clear: video ? 'Clear first frame' : 'Clear start image', preview: video ? 'First frame preview' : 'Start image preview' }}
          emptyText="Browse or drop"
          clearable={start.set}
          path={start.path}
          file={start.file}
          previewUrl={galleryStore.previewUrlFor(start.path)}
          name={start.name}
          dropText={start.set ? 'Drop to replace' : 'Drop an image'}
          missing={start.gone}
          error={start.error ?? undefined}
          onfilechange={(file) => { if (file !== null) setStart({ file }); }}
          onpathchange={(path) => setStart(path === null ? null : { path })}
        />
        {#if view.strength}
          <InspectorRow label="Strength" forId="ws-image-strength" scrubber={strengthScrub} changed={strengthChanged} onreset={() => setStrength(defaults.referenceImageStrength)}>
            <InspectorNumber id="ws-image-strength" value={s.referenceImageStrength} {...SPEC.strength} onchange={(v) => v !== null && setStrength(v)} />
            <input class="mini-range ui-range" type="range" tabindex="-1" aria-hidden="true" {...SPEC.strength} value={s.referenceImageStrength} style:--fill={rangeFill(s.referenceImageStrength, SPEC.strength)} oninput={(e) => setStrength(Number(e.currentTarget.value))}>
          </InspectorRow>
        {/if}
      </div>
    {/if}

    {#if references}
      <div class="image-group" role="group" aria-labelledby="ws-references-group">
        <h4 id="ws-references-group" class="ui-area-label image-group-label">References</h4>
        {#each refRows as row, index (index)}
          {@const number = index + 1}
          <ReferenceSlot
            idBase="ws-ref-{number}"
            label="Reference {number}"
            names={{ browse: `Browse for reference ${number}`, upload: `Upload reference ${number} from this device`, clear: `Remove reference ${number}`, preview: `Reference ${number} preview` }}
            emptyText=""
            path={row.path || null}
            file={row.file}
            name={row.name}
            previewUrl={galleryStore.previewUrlFor(row.path)}
            dropText="Drop to replace"
            missing={row.gone}
            error={row.error ?? undefined}
            onfilechange={(file) => editor.replaceWithFile(index, file)}
            onpathchange={(path) => (path === null ? void removeReference(index) : editor.replaceWithPath(index, path))}
          />
        {/each}
        {#if references.hasRoom}
          <ReferenceSlot
            idBase="ws-ref-add"
            label="Reference {refRows.length + 1}"
            names={{ browse: `Browse for reference ${refRows.length + 1}`, upload: `Upload reference ${refRows.length + 1} from this device`, clear: '', preview: '' }}
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
        {#if refRows.length > 0}<p class="row-hint">Refer to them in the prompt as image 1, image 2…</p>{/if}
      </div>
    {/if}

    {#if last}
      <div class="image-group" role="group" aria-labelledby="ws-last-group">
        <h4 id="ws-last-group" class="ui-area-label image-group-label">Last frame</h4>
        <ReferenceSlot
          idBase="ws-end-image"
          label="Image"
          names={{ browse: 'Browse for a last frame', upload: 'Upload a last frame from this device', clear: 'Clear last frame', preview: 'Last frame preview' }}
          emptyText="Optional"
          emptyMuted
          clearable={last.set}
          path={last.path}
          file={last.file}
          previewUrl={galleryStore.previewUrlFor(last.path)}
          name={last.name}
          dropText={last.set ? 'Drop to replace' : 'Drop an image'}
          unused={!last.used}
          missing={last.gone}
          error={last.error ?? undefined}
          onfilechange={(file) => { if (file !== null) draft.setLastFrame({ file }); }}
          onpathchange={(path) => draft.setLastFrame(path === null ? null : { path })}
        />
        {#if last.set}
          <p class="row-hint">{last.used ? 'The clip ends on this image; different scenes dissolve into each other.' : 'Not used without a first frame'}</p>
        {/if}
      </div>
    {/if}
  </InspectorSection>
{/if}

<style>
  .image-group { display: flex; flex-direction: column; }
  /* A group inside the section: the area label's face, sentence case and a notch quieter than the section's own title. */
  .image-group-label { padding: 10px 12px 2px; font-weight: 600; letter-spacing: 0; text-transform: none; }
  .image-group + .image-group { margin-top: 4px; }
  .mini-range { flex: none; width: 90px; }
  @media (pointer: coarse) { .mini-range { width: 72px; } }
</style>
