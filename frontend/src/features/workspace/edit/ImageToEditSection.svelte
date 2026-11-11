<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { InspectorRow, InspectorSection } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import type { WorkspaceContext } from '$lib/types';
  import { currentEditFacts } from './currentEdit';
  import EditSourcePicker from './EditSourcePicker.svelte';
  import MaskThumb from './MaskThumb.svelte';
  import { clearSource } from './editActions';
  import { imageEdit, painterLayer } from './editStore.svelte';
  import { extendSummary, paintSummary, sectionSummary } from './editView';

  interface Props {
    context: WorkspaceContext;
    /** A submit is in flight. */
    busy: boolean;
  }

  let { context, busy }: Props = $props();

  const source = $derived(imageEdit.source);
  const facts = $derived(currentEditFacts(context));
  const paint = $derived(paintSummary(facts));
  const extend = $derived(extendSummary(facts));
  const size = $derived(source?.width && source?.height ? `${source.width} × ${source.height}` : '');
  const buttonLabel = $derived(paint ? 'Change paint' : extend ? 'Change extension' : 'Paint or extend');
</script>

<InspectorSection id="image-to-edit" title="Image to edit" summary={sectionSummary(source ? (source.name || fileName(source.path ?? '')) : null)} active={source !== null}>
  {#if source}
    <InspectorRow label="Image">
      <MaskThumb size={40} />
      <span class="edit-name" title={source.path ?? source.name}>{source.name}</span>
      {#if size}<span class="edit-size">{size}</span>{/if}
      <button type="button" class="ui-btn ui-btn-row" aria-label="Clear the image to edit" title={busy ? 'Available once the job is submitted.' : 'Clear'} disabled={busy} onclick={clearSource}><Icon name="close" size={14} /></button>
    </InspectorRow>
    <div class="edit-summary" aria-live="polite" data-testid="edit-summary">
      {#if paint}<p>{paint}</p>{/if}
      {#if extend}<p>{extend}</p>{/if}
    </div>
    <div class="edit-action">
      <button type="button" class="ui-btn" data-testid="edit-open-painter" disabled={busy} onclick={(e) => painterLayer.show(e.currentTarget)}><Icon name="brush" size={14} />{buttonLabel}</button>
    </div>
  {:else}
    <EditSourcePicker />
  {/if}
</InspectorSection>

<style>
  .edit-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); }
  .edit-size { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; }
  .edit-summary { display: flex; flex-direction: column; gap: 2px; padding: 0 12px; }
  .edit-summary:not(:empty) { padding-bottom: 6px; }
  .edit-summary p { margin: 0; font-size: var(--text-ui); color: var(--color-text-secondary); }
  .edit-action { padding: 2px 12px 6px; }
  .edit-action .ui-btn { width: 100%; }
</style>
