<script lang="ts">
  import { InspectorNumber } from '$lib/components/atoms';
  import { draft } from '$lib/state/draft.svelte';
  import { galleryStore } from '$lib/state/gallery.svelte';
  import type { SlotValue } from '$lib/state/imageSlots';
  import type { WorkspaceContext } from '$lib/types';
  import type { ImagesView } from './imagesView';
  import PopoverImagePicker from './PopoverImagePicker.svelte';
  import PopoverReferences from './PopoverReferences.svelte';
  import { rangeFill, SPEC } from './settingsSpec';

  interface Props {
    context: WorkspaceContext;
    /** The images view of the open tab, computed once by the Workspace. */
    images: ImagesView;
  }

  let { context, images: view }: Props = $props();

  const s = $derived(draft.state);
  const tab = $derived(view.tab);
  const video = $derived(tab === 'video');
  const start = $derived(view.start);
  const last = $derived(view.last);
  const references = $derived(view.references);

  // Every write goes through the draft's setters, so the workflow stays derived from the start image.
  function setStart(value: SlotValue): void {
    draft.setStartImage(tab, value, context);
  }
</script>

<div class="img-pop">
  {#if start}
    <div class="pop-group" role="group" aria-labelledby="pop-start-group">
      <h4 id="pop-start-group" class="ui-area-label pop-group-label">{video ? 'First frame' : 'Start image'}</h4>
      <PopoverImagePicker
        title={video ? 'First frame' : 'Start image'}
        name={start.name}
        path={start.path}
        file={start.file}
        previewUrl={galleryStore.previewUrlFor(start.path)}
        emptyText={video ? 'No first frame yet' : 'No start image yet'}
        clearLabel={video ? 'Clear first frame' : 'Clear start image'}
        missing={start.gone}
        error={start.error ?? undefined}
        onfile={(file) => setStart({ file })}
        onpath={(path) => setStart({ path })}
        onclear={() => setStart(null)}
      />
    </div>
  {/if}

  {#if view.strength}
    <div class="pop-group" role="group" aria-labelledby="pop-strength-group">
      <h4 id="pop-strength-group" class="ui-area-label pop-group-label">Strength</h4>
      <InspectorNumber ariaLabel="Strength" value={s.referenceImageStrength} {...SPEC.strength} stepper size="lg" onchange={(v) => v !== null && draft.update('referenceImageStrength', v)} />
      <input class="ui-range pop-range" type="range" aria-label="Strength" {...SPEC.strength} value={s.referenceImageStrength} style:--fill={rangeFill(s.referenceImageStrength, SPEC.strength)} oninput={(e) => draft.update('referenceImageStrength', Number(e.currentTarget.value))}>
      <p class="ui-help">How far from the start image the result may go.</p>
    </div>
  {/if}

  {#if references}
    <div class="pop-group" role="group" aria-labelledby="pop-references-group">
      <h4 id="pop-references-group" class="ui-area-label pop-group-label">References</h4>
      <PopoverReferences {references} />
    </div>
  {/if}

  {#if last}
    <div class="pop-group" role="group" aria-labelledby="pop-last-group">
      <h4 id="pop-last-group" class="ui-area-label pop-group-label">Last frame</h4>
      <PopoverImagePicker
        title="Last frame"
        name={last.name}
        path={last.path}
        file={last.file}
        previewUrl={galleryStore.previewUrlFor(last.path)}
        emptyText="No last frame"
        clearLabel="Clear last frame"
        unused={!last.used}
        missing={last.gone}
        error={last.error ?? undefined}
        help={last.set && !last.used ? 'Not used without a first frame' : 'Optional. The clip ends on this image; different scenes dissolve into each other.'}
        onfile={(file) => draft.setLastFrame({ file })}
        onpath={(path) => draft.setLastFrame({ path })}
        onclear={() => draft.setLastFrame(null)}
      />
    </div>
  {/if}
</div>

<style>
  .img-pop { display: flex; flex-direction: column; gap: 20px; }
  .pop-group { display: flex; flex-direction: column; gap: 12px; padding: 0 2px; }
  .pop-group-label { padding: 0; }
  .pop-range { width: 100%; height: 36px; }
  .pop-range::-webkit-slider-thumb { width: 24px; height: 24px; margin-top: -10px; }
  .pop-range::-moz-range-thumb { width: 24px; height: 24px; }
</style>
