<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { scrub, type ScrubOptions } from '$lib/actions/scrub';

  interface Props {
    label: string;
    /** Id of the input the label focuses. */
    forId?: string;
    /** Makes the label draggable to change a number. */
    scrubber?: ScrubOptions;
    /** The value differs from the model default. */
    changed?: boolean;
    /** Resets this row to its default; omitted rows have no reset. */
    onreset?: () => void;
    /** Indents the row under the one above. */
    sub?: boolean;
    children: Snippet;
  }

  let { label, forId, scrubber, changed = false, onreset, sub = false, children }: Props = $props();
</script>

<div class="inspector-row" class:sub data-changed={changed}>
  {#snippet labelText()}
    <span class="inspector-label-text">{label}</span>
    {#if changed}<span class="inspector-changed" aria-label="changed from default"></span>{/if}
  {/snippet}
  {#if scrubber}
    <label for={forId} class="inspector-label" class:scrubbable={!scrubber.disabled} title="Drag to change" use:scrub={scrubber}>
      {@render labelText()}
    </label>
  {:else}
    <svelte:element this={forId ? 'label' : 'span'} for={forId} class="inspector-label">
      {@render labelText()}
    </svelte:element>
  {/if}
  <div class="inspector-value">
    {@render children()}
    {#if onreset}
      <button
        type="button"
        class="inspector-reset"
        aria-label="Reset {label} to default"
        title="Reset to default"
        tabindex={changed ? 0 : -1}
        aria-hidden={!changed}
        onclick={onreset}
      ><Icon name="reset" size={12} /></button>
    {/if}
  </div>
</div>

<style>
  .inspector-row { display: grid; grid-template-columns: 96px minmax(0, 1fr); align-items: center; gap: 8px; min-height: 30px; padding: 0 8px 0 12px; }
  .inspector-row:hover { background: color-mix(in srgb, var(--color-bg-surface) 60%, transparent); }
  .inspector-label { display: flex; align-items: center; gap: 5px; overflow: hidden; font-size: var(--text-ui); color: var(--color-text-secondary); white-space: nowrap; user-select: none; }
  .inspector-row.sub .inspector-label { padding-left: 12px; }
  .inspector-row[data-changed='true'] .inspector-label { color: var(--color-text-primary); }
  .inspector-label.scrubbable { cursor: ew-resize; touch-action: none; }
  .inspector-label.scrubbable:hover .inspector-label-text { text-decoration: underline dotted; text-underline-offset: 3px; }
  .inspector-label-text { overflow: hidden; text-overflow: ellipsis; }
  .inspector-changed { flex: none; width: 6px; height: 6px; border-radius: 50%; background: var(--color-primary-main); }
  .inspector-value { display: flex; min-width: 0; align-items: center; gap: 4px; }
  /* The reset slot is always reserved so values never shift when it appears. */
  .inspector-reset { display: inline-grid; flex: none; place-items: center; width: 20px; height: 20px; margin-left: auto; border-radius: 4px; color: var(--color-text-muted); visibility: hidden; }
  .inspector-row[data-changed='true'] .inspector-reset { visibility: visible; }
  .inspector-reset:hover { background: var(--color-bg-surface); color: var(--color-primary-main); }
  .inspector-reset:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 1px; }
</style>
