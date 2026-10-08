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
    children: Snippet;
  }

  let { id, title, summary = '', active = false, children }: Props = $props();

  const open = $derived(openState[id] ?? true);
  const bodyId = $derived(`inspector-${id}`);
</script>

<section class="inspector-section" data-section={id}>
  <button
    type="button"
    class="inspector-head"
    aria-expanded={open}
    aria-controls={bodyId}
    onclick={() => setOpen(id, !open)}
  >
    <Icon name="chevright" size={12} class="inspector-chev" />
    <span>{title}</span>
    {#if summary}<span class="inspector-summary" class:active>{summary}</span>{/if}
  </button>
  <div id={bodyId} class="inspector-body" hidden={!open}>
    {@render children()}
  </div>
</section>

<style>
  /* Header band: a border above and below and the raised shade; adjacent headers share one line. */
  .inspector-section + :global(.inspector-section) { margin-top: -1px; }
  .inspector-head {
    display: flex;
    width: 100%;
    align-items: center;
    gap: 6px;
    min-height: 30px;
    padding: 0 12px;
    border-block: 1px solid var(--color-border-subtle);
    background: var(--color-bg-raised);
    font-family: var(--font-heading);
    font-size: var(--text-meta);
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-text-muted);
    transition: color 0.12s ease, background-color 0.12s ease;
  }
  .inspector-head:hover { background: color-mix(in srgb, var(--color-bg-raised) 70%, var(--color-bg-surface-hover)); color: var(--color-text-primary); }
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
  .inspector-body { padding-block: 4px 6px; }
  .inspector-body[hidden] { display: none; }
</style>
