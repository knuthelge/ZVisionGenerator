<script lang="ts" module>
  const STORAGE_KEY = 'ziv-inspector-sections-v1';

  function loadOpenState(): Record<string, boolean> {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? (JSON.parse(raw) as Record<string, boolean>) : {};
    } catch {
      return {};
    }
  }

  // Shared across sections so each remembers its own open/closed choice between visits.
  const openState = $state<Record<string, boolean>>(loadOpenState());

  function setOpen(id: string, open: boolean): void {
    openState[id] = open;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(openState));
    } catch {
      // storage unavailable — keep the in-memory choice only
    }
  }
</script>

<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';

  interface Props {
    /** Stable key used to remember whether the section is open. */
    id: string;
    title: string;
    /** Short state summary shown in the header, visible even when collapsed. */
    summary?: string;
    /** Highlights the summary (e.g. the feature is switched on). */
    active?: boolean;
    /** Marks the summary as an error (e.g. an image the run needs is gone); it wins over `active`. */
    error?: boolean;
    /**
     * Makes the section controlled, for an accordion the parent owns: it is open when this is true and says so through `ontoggle`,
     * and it keeps no memory of its own. The header is then a heading pinned to the top of its scroll area, the body is a labelled
     * region, and a closed body is not rendered.
     */
    open?: boolean;
    /** Called when the header is activated on a controlled section. */
    ontoggle?: () => void;
    children: Snippet;
  }

  let { id, title, summary = '', active = false, error = false, open: controlledOpen, ontoggle, children }: Props = $props();

  const controlled = $derived(controlledOpen !== undefined);
  const open = $derived(controlledOpen ?? openState[id] ?? true);
  const bodyId = $derived(`inspector-${id}`);
  const headId = $derived(`inspector-head-${id}`);

  function toggle(): void {
    if (controlled) ontoggle?.();
    else setOpen(id, !open);
  }
</script>

{#snippet head()}
  <button
    type="button"
    id={controlled ? headId : undefined}
    class="inspector-head"
    aria-expanded={open}
    aria-controls={bodyId}
    aria-label={controlled && summary ? `${title}, ${summary}` : undefined}
    onclick={toggle}
  >
    <Icon name="chevright" size={12} class="inspector-chev" />
    <span>{title}</span>
    {#if summary}<span class="inspector-summary" class:active={active && !error} class:error data-tone={error ? 'error' : active ? 'active' : 'muted'}>{summary}</span>{/if}
  </button>
{/snippet}

<section class="inspector-section" data-section={id} data-controlled={controlled ? '' : undefined}>
  {#if controlled}
    <h3 class="inspector-heading">{@render head()}</h3>
    <div id={bodyId} class="inspector-body" role="region" aria-labelledby={headId} hidden={!open}>
      {#if open}{@render children()}{/if}
    </div>
  {:else}
    {@render head()}
    <div id={bodyId} class="inspector-body" hidden={!open}>
      {@render children()}
    </div>
  {/if}
</section>

<style>
  /* Header band: the raised shade, no border above or below. */
  .inspector-head {
    display: flex;
    width: 100%;
    align-items: center;
    gap: 6px;
    min-height: 30px;
    padding: 0 12px;
    background: var(--color-bg-raised);
    font-family: var(--font-heading);
    font-size: var(--text-meta);
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-text-muted);
    transition: color 0.12s ease, background-color 0.12s ease;
  }
  @media (hover: hover) { .inspector-head:hover { background: color-mix(in srgb, var(--color-bg-raised) 70%, var(--color-bg-surface-hover)); color: var(--color-text-primary); } }
  .inspector-head:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .inspector-head :global(.inspector-chev) { transition: transform 0.12s ease; }
  .inspector-head[aria-expanded='true'] :global(.inspector-chev) { transform: rotate(90deg); }
  .inspector-summary {
    max-width: 190px;
    margin-left: auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: var(--font-sans);
    font-size: var(--text-meta);
    font-weight: 400;
    letter-spacing: 0;
    text-transform: none;
  }
  .inspector-summary.active { color: var(--color-primary-main); }
  .inspector-summary.error { color: var(--color-error); }
  .inspector-body { padding-block: 4px 6px; }
  /* A controlled section is a panel: the header is pinned while the body scrolls under it, with the panel's own gutter, and the
     body is the caller's. */
  .inspector-heading { position: sticky; top: 0; z-index: 2; margin: 0; }
  .inspector-section[data-controlled]:not(:first-child) .inspector-head { box-shadow: 0 -1px 0 var(--color-border-subtle); }
  [data-controlled] .inspector-head { min-height: var(--spacing-row); padding: 0 16px; color: var(--color-text-secondary); }
  [data-controlled] .inspector-head[aria-expanded='true'] { color: var(--color-text-primary); }
  [data-controlled] .inspector-head :global(.inspector-chev) { color: var(--color-text-muted); }
  [data-controlled] .inspector-summary { color: var(--color-text-muted); }
  [data-controlled] .inspector-summary.active { color: var(--color-primary-main); }
  [data-controlled] .inspector-body { padding: 0; }
  @media (prefers-reduced-motion: reduce) { .inspector-head :global(.inspector-chev) { transition: none; } }
  .inspector-body[hidden] { display: none; }
</style>
