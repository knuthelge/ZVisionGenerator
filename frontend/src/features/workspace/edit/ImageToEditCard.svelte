<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { WorkspaceContext } from '$lib/types';
  import { currentEditFacts } from './currentEdit';
  import EditSourcePicker from './EditSourcePicker.svelte';
  import MaskThumb from './MaskThumb.svelte';
  import { clearSource } from './editActions';
  import { imageEdit, painterLayer } from './editStore.svelte';
  import { cardSummary } from './editView';

  interface Props {
    context: WorkspaceContext;
  }

  let { context }: Props = $props();

  const source = $derived(imageEdit.source);
  const facts = $derived(currentEditFacts(context));
  const buttonLabel = $derived(facts.painted > 0 || facts.extension ? 'Change' : 'Paint or extend');
</script>

<!-- The phone's Image to edit: a card above the settings chips, one row once an image is chosen. -->
<section class="edit-card" class:empty={!source} aria-label="Image to edit" data-testid="edit-card">
  {#if source}
    <MaskThumb size={44} />
    <div class="edit-card-text">
      <b class="edit-card-name" title={source.name}>{source.name}</b>
      <span class="edit-card-summary" aria-live="polite">{cardSummary(facts)}</span>
    </div>
    <button type="button" class="ui-btn edit-card-clear" aria-label="Clear the image to edit" onclick={clearSource}><Icon name="close" size={14} /></button>
    <button type="button" class="ui-btn edit-card-open" data-testid="edit-open-painter" onclick={(e) => painterLayer.show(e.currentTarget)}><Icon name="brush" size={14} />{buttonLabel}</button>
  {:else}
    <EditSourcePicker compact />
  {/if}
</section>

<style>
  .edit-card { display: flex; align-items: center; gap: 10px; margin: 8px max(12px, env(safe-area-inset-right, 0px)) 0 max(12px, env(safe-area-inset-left, 0px)); padding: 8px; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  /* Without an image the dashed zone is the card. */
  .edit-card.empty { padding: 0; background: transparent; }
  .edit-card-text { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 2px; }
  .edit-card-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); font-weight: 600; }
  .edit-card-summary { display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; font-size: var(--text-ui); line-height: 1.3; color: var(--color-text-secondary); }
  .edit-card-open { height: var(--spacing-bar); }
  .edit-card-clear { width: var(--spacing-control); padding: 0; background: transparent; color: var(--color-text-muted); }
  .edit-card :global(.edit-drop) { flex: 1; }
</style>
