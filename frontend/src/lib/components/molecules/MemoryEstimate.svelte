<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { MEMORY_VARIANTS, memoryEstimateSpeech, memoryEstimateView, memoryRowSpeech } from '$lib/state/memoryEstimate';
  import type { JobMemory } from '$lib/types';

  /**
   * The memory estimate: a title naming what it is about, what is judged, one sentence about the current choice, then the table of this
   * job's levels against the budgets. It sits in the memory tag's popover. Every word is the server's; this lays them out.
   * Built from spans only while it is for reading: a popover may sit inside a button. With `onselect`, the rows become buttons
   * (the current level is pressed): a click, tap, Enter or Space on a row chooses that level.
   */
  interface Props {
    memory: JobMemory;
    /** The level the job uses (`none`, `8`, `4`, `on`, `off`); the answer's own current one when absent. */
    level?: string | null;
    /** Chooses the level of a row, by its key; makes rows choosable. Not called for the level already in use. */
    onselect?: (key: string) => void;
    /** The row keys `onselect` may take; every row when absent. The others stay plain. */
    selectable?: readonly string[];
  }

  let { memory, level = null, onselect, selectable }: Props = $props();

  const view = $derived(memoryEstimateView(memory, level));
  const single = $derived(view.control === null);
  const choosable = $derived(onselect && !single ? new Set(selectable ?? view.rows.map((row) => row.key)) : new Set<string>());
  const interactive = $derived(view.rows.some((row) => choosable.has(row.key)));

  function choose(key: string, current: boolean): void {
    if (!current) onselect?.(key);
  }
</script>

<span class="mem-est" data-status={view.status}>
  <span class="sr-only">{memoryEstimateSpeech(memory, view, !interactive)}</span>
  <span class="mem-body" aria-hidden={interactive ? undefined : 'true'}>
    <span class="mem-title" aria-hidden="true" data-testid="memory-title">{view.title}</span>
    {#if view.details}<span class="mem-subject" aria-hidden="true" data-testid="memory-judged">{view.details}</span>{/if}
    {#if memory.download_size_label !== null}
      <span class="mem-subject mem-download" aria-hidden="true" data-testid="memory-download"><Icon name="download" size={11} />Not downloaded · about {memory.download_size_label}</span>
    {/if}
    <span class="mem-sentence" aria-hidden="true" data-testid="memory-sentence">{view.sentence}</span>
    {#each memory.notes ?? [] as note (note)}
      <span class="mem-subject mem-note" aria-hidden="true" data-testid="memory-note">{note}</span>
    {/each}

    {#if view.heading}<span class="mem-options-label" aria-hidden="true" data-testid="memory-options-label">{view.heading}</span>{/if}
    <span class="mem-table" class:single class:labelled={view.heading !== null} data-testid="memory-options">
      <span class="mem-row mem-row-head" aria-hidden="true">
        {#if !single}<span class="mem-level">{view.control}</span>{/if}
        <span></span>
        {#each view.columns as column (column.label)}<span class="num">{column.label} · {column.budget} GB</span>{/each}
      </span>
      {#each view.rows as row (row.key)}
        {@const pick = interactive && choosable.has(row.key)}
        <!-- A row is a button only while the table is for choosing (the click handler is set only then); otherwise a span, which a popover inside a button needs. -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <svelte:element
          this={interactive ? 'button' : 'span'}
          class="mem-row"
          class:current={row.current}
          class:choosable={pick}
          type={interactive ? 'button' : undefined}
          aria-pressed={interactive ? row.current : undefined}
          disabled={interactive && !pick ? true : undefined}
          aria-label={interactive ? memoryRowSpeech(view, row, false) : undefined}
          onclick={pick ? () => choose(row.key, row.current) : undefined}
          data-testid="memory-row"
          data-key={row.key}
          data-current={row.current ? '' : undefined}
        >
          {#if !single}<span class="mem-level">{row.label}</span>{/if}
          <span class="mem-word mem-{MEMORY_VARIANTS[row.status]}">{row.word}</span>
          <span class="num">{row.gpu}</span>
          {#if row.ram !== undefined}<span class="num">{row.ram}</span>{/if}
        </svelte:element>
      {/each}
    </span>
  </span>
</span>

<style>
  /* Every value comes from the existing tokens: the overlay surface, the three text greys, the status colours. */
  .mem-est { display: block; min-width: 272px; max-width: 100%; padding: 12px 14px 10px; color: var(--color-text-primary); font-size: var(--text-ui); line-height: 16px; text-align: left; white-space: normal; }
  .mem-body { display: flex; flex-direction: column; }

  /* The frame: the title, what is judged (quiet), then one sentence about the current choice. */
  .mem-title { min-width: 0; font-weight: 600; color: var(--color-text-primary); }
  .mem-subject { margin-top: 2px; font-size: var(--text-meta); line-height: 14px; color: var(--color-text-muted); overflow-wrap: anywhere; }
  .mem-download { display: flex; align-items: center; gap: 5px; }
  .mem-sentence { margin-top: 8px; color: var(--color-text-primary); }
  .mem-note { margin-top: 4px; }

  /* The table, under its own small label: the level, its status word (the prominent part), then quiet figures under
     their budgets. One grid, so every row's columns line up; each row is a subgrid so the current one can carry its tint. */
  .mem-options-label { margin-top: 14px; font-size: var(--text-meta); line-height: 14px; font-weight: 600; color: var(--color-text-secondary); }
  .mem-table { display: grid; grid-template-columns: max-content minmax(0, 1fr) repeat(2, max-content); column-gap: 16px; margin: 10px -6px 0; }
  .mem-table.labelled { margin-top: 4px; }
  .mem-table.single { grid-template-columns: minmax(0, 1fr) repeat(2, max-content); }
  .mem-row { display: grid; grid-column: 1 / -1; grid-template-columns: subgrid; align-items: center; height: 24px; padding: 0 6px; border-radius: var(--radius-xs); }
  /* A row that is a button reads as the plain row it is. */
  button.mem-row { border: 0; background: transparent; font: inherit; color: inherit; text-align: inherit; cursor: default; }
  .mem-row-head { height: 18px; white-space: nowrap; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--color-text-muted); }
  .mem-level { color: var(--color-text-secondary); white-space: nowrap; }
  .mem-row-head .mem-level { color: inherit; }
  .num { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; color: var(--color-text-muted); }
  .mem-row-head .num { color: inherit; }
  .mem-row.current { background: color-mix(in srgb, var(--color-text-primary) 9%, transparent); }
  .mem-row.current .mem-level { font-weight: 600; color: var(--color-text-primary); }
  .mem-row.current .num { color: var(--color-text-secondary); }
  /* A row the user can choose: the pointer says so, hover tints it lighter than the current row, and the focus ring sits inside the row. */
  .mem-row.choosable { cursor: pointer; }
  .mem-row.choosable:focus-visible { outline-offset: -2px; }
  @media (hover: hover) { .mem-row.choosable:not(.current):hover { background: color-mix(in srgb, var(--color-text-primary) 6%, transparent); } }
  .mem-row.choosable:not(.current):active { background: color-mix(in srgb, var(--color-text-primary) 9%, transparent); }
  .mem-word { font-size: var(--text-ui); font-weight: 600; white-space: nowrap; }
  .mem-success { color: var(--color-success); }
  .mem-warning { color: var(--color-warning); }
  .mem-error { color: var(--color-error); }

  /* Touch: one step larger, as the app's other phone and tablet text is. */
  @media (pointer: coarse) {
    .mem-est { min-width: 280px; padding: 14px 16px 12px; font-size: var(--text-content); line-height: 18px; }
    .mem-subject, .mem-options-label { font-size: var(--text-ui); line-height: 16px; }
    .mem-row { height: 30px; }
    .mem-row.choosable { height: 32px; }
    .mem-row-head { height: 20px; font-size: var(--text-meta); }
    .mem-word { font-size: var(--text-content); }
  }
</style>
