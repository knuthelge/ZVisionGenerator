<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { currentEditFacts } from './currentEdit';
  import EditSourcePicker from './EditSourcePicker.svelte';
  import MaskThumb from './MaskThumb.svelte';
  import { clearSource } from './editActions';
  import { imageEdit, painterLayer } from './editStore.svelte';
  import ModeLine from './ModeLine.svelte';
  import { modeSummary, wholePictureEdit } from './editView';

  const source = $derived(imageEdit.source);
  const facts = $derived(currentEditFacts());
  const summary = $derived(modeSummary(facts));
</script>

<!-- The phone's Image to edit: a card above the settings chips, one row once an image is chosen. -->
<section class="edit-card" class:empty={!source} aria-label="Image to edit">
  {#if source}
    <!-- The thumbnail, name and mode line are one button: a tap anywhere on them opens the painter. -->
    <button type="button" class="edit-card-open" data-testid="edit-open-painter" aria-label="Open the painter. {summary.label}: {summary.spoken ?? summary.detail}" onclick={(e) => painterLayer.show(e.currentTarget)}>
      <MaskThumb size={44} whole={wholePictureEdit(facts)} />
      <span class="edit-card-text">
        <b class="edit-card-name" title={source.name}>{source.name}</b>
        <span class="edit-card-summary" aria-live="polite"><ModeLine {summary} /></span>
      </span>
      <Icon name="chevright" size={16} />
    </button>
    <button type="button" class="ui-btn edit-card-clear" aria-label="Clear the image to edit" onclick={clearSource}><Icon name="close" size={14} /></button>
  {:else}
    <EditSourcePicker compact />
  {/if}
</section>

<style>
  .edit-card { display: flex; align-items: center; gap: 4px; margin: 8px max(12px, env(safe-area-inset-right, 0px)) 0 max(12px, env(safe-area-inset-left, 0px)); padding: 4px; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  /* Without an image the dashed zone is the card. */
  .edit-card.empty { padding: 0; background: transparent; }
  .edit-card-open { display: flex; flex: 1; min-width: 0; align-items: center; gap: 10px; min-height: var(--spacing-bar); padding: 4px; border-radius: var(--radius-sm); background: transparent; color: var(--color-text-muted); text-align: left; }
  @media (hover: hover) { .edit-card-open:hover { background: var(--color-bg-surface-hover); } }
  @media (hover: none) { .edit-card-open:active { background: var(--color-bg-surface-hover); } }
  .edit-card-text { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 4px; }
  .edit-card-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-primary); }
  .edit-card-summary { display: block; min-width: 0; }
  /* A detail that does not fit beside the badge goes under it whole, so no state is cut off on a narrow phone. */
  .edit-card-summary :global(.mode-line) { flex-wrap: wrap; gap: 2px 8px; }
  .edit-card-summary :global(.mode-detail) { overflow: visible; white-space: normal; }
  .edit-card-clear { width: var(--spacing-control); padding: 0; background: transparent; color: var(--color-text-muted); }
  .edit-card :global(.edit-drop) { flex: 1; }
</style>
