<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { InspectorRow, InspectorSection } from '$lib/components/molecules';
  import { currentEditFacts } from './currentEdit';
  import EditSourcePicker from './EditSourcePicker.svelte';
  import MaskThumb from './MaskThumb.svelte';
  import { clearSource } from './editActions';
  import { imageEdit, painterLayer } from './editStore.svelte';
  import ModeLine from './ModeLine.svelte';
  import { modeSummary, wholePictureEdit } from './editView';

  interface Props {
    /** A submit is in flight. */
    busy: boolean;
  }

  let { busy }: Props = $props();

  const source = $derived(imageEdit.source);
  const facts = $derived(currentEditFacts());
  const summary = $derived(modeSummary(facts));
  const size = $derived(source?.width && source?.height ? `${source.width} × ${source.height}` : '');
</script>

<InspectorSection id="image-to-edit" title="Image to edit" summary={source?.name ?? 'None'} active={source !== null}>
  {#if source}
    <InspectorRow label="Image">
      <MaskThumb size={40} whole={wholePictureEdit(facts)} />
      <span class="edit-name" title={source.path ?? source.name}>{source.name}</span>
      {#if size}<span class="edit-size">{size}</span>{/if}
      <button type="button" class="ui-btn ui-btn-row" aria-label="Clear the image to edit" title={busy ? 'Available once the job is submitted.' : 'Clear'} disabled={busy} onclick={clearSource}><Icon name="close" size={14} /></button>
    </InspectorRow>
    <div class="edit-summary" aria-live="polite" data-testid="edit-summary">
      <ModeLine {summary} />
    </div>
    <div class="edit-action">
      <button type="button" class="ui-btn" data-testid="edit-open-painter" disabled={busy} onclick={(e) => painterLayer.show(e.currentTarget)}><Icon name="brush" size={14} />Open painter</button>
    </div>
  {:else}
    <EditSourcePicker />
  {/if}
</InspectorSection>

<style>
  .edit-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); }
  .edit-size { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; }
  .edit-summary { padding: 2px 12px 8px; }
  /* The sidebar is narrow: a long detail wraps instead of being cut. */
  .edit-summary :global(.mode-line) { align-items: flex-start; }
  .edit-summary :global(.mode-detail) { overflow: visible; white-space: normal; }
  .edit-action { padding: 2px 12px 6px; }
  .edit-action .ui-btn { width: 100%; }
</style>
