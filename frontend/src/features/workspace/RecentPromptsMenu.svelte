<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { popover } from '$lib/actions/popover';
  import { errorText } from '$lib/api/client';
  import { clearRecentPrompts, getRecentPrompts } from '$lib/api/recentPrompts';
  import { Icon, Spinner } from '$lib/components/atoms';
  import { requestConfirm, Sheet } from '$lib/components/molecules';
  import { promptFirstLine, timeAgo } from '$lib/format';
  import { INPUT_DEBOUNCE_MS } from '$lib/state/debounce';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { RecentPrompt, WorkflowMode } from '$lib/types';

  interface Props {
    open: boolean;
    /** The button that opens the menu; the popover lines up with it. */
    anchor: HTMLElement | null;
    /** Where focus goes when the menu closes without a pick: the button, or the prompt box after Alt+↑. */
    returnFocus: HTMLElement | null;
    /** The workflow's mode; the menu lists and clears only the prompts of this mode. */
    mode: WorkflowMode;
    /** The current model takes JSON captions, so JSON rows can be picked. */
    jsonCapable: boolean;
    onpick: (entry: RecentPrompt) => void;
    onclose: () => void;
  }

  let { open, anchor, returnFocus, mode, jsonCapable, onpick, onclose }: Props = $props();

  const JSON_REASON = 'Needs a model that takes JSON captions';

  type Phase = 'loading' | 'error' | 'empty' | 'nomatch' | 'rows';

  let filter = $state('');
  let entries = $state<RecentPrompt[]>([]);
  /** What the list area shows. */
  let phase = $state<Phase>('loading');
  let error = $state<string | null>(null);
  let active = $state(-1);
  let inputEl = $state<HTMLInputElement | null>(null);
  let requestId = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  /** The filter the shown rows answer; null until the first result is in. */
  let shownFilter: string | null = null;
  /** Enter was pressed before the shown rows answered the typed filter: the top enabled row of its result is picked. */
  let pickOnLoad = false;

  /** The menu was opened with Alt+↑ from the prompt box, so a keyboard is in use even on a touch screen. */
  const byKeyboard = $derived(returnFocus !== anchor);

  function isDisabled(entry: RecentPrompt): boolean {
    return entry.format === 'json' && !jsonCapable;
  }

  function firstEnabled(list: readonly RecentPrompt[]): number {
    return list.findIndex((entry) => !isDisabled(entry));
  }

  /** The whole text on one line; the row's two-line clamp decides how much shows. */
  function oneLine(text: string): string {
    return text.replace(/\s+/g, ' ').trim();
  }

  function rowText(entry: RecentPrompt): string {
    if (entry.format === 'json') return oneLine(entry.prompt);
    return oneLine(entry.prompt) || oneLine(entry.scene);
  }

  /** The hover text: the whole prompt, with the scene and the negative prompt on lines of their own. */
  function rowTitle(entry: RecentPrompt): string {
    const lines = [entry.prompt.trim()];
    if (entry.prompt && entry.scene) lines.push(`Scene: ${entry.scene.trim()}`);
    else if (!entry.prompt && entry.scene) lines.push(entry.scene.trim());
    if (entry.negative_prompt) lines.push(`Negative: ${entry.negative_prompt.trim()}`);
    return lines.filter(Boolean).join('\n');
  }

  function rowSub(entry: RecentPrompt): string {
    if (isDisabled(entry)) return JSON_REASON;
    if (entry.prompt && entry.scene) return `Scene: ${promptFirstLine(entry.scene)}`;
    if (entry.negative_prompt) return `Negative: ${promptFirstLine(entry.negative_prompt)}`;
    return '';
  }

  async function load(query: string): Promise<void> {
    const id = ++requestId;
    try {
      const { entries: next } = await getRecentPrompts(query, mode);
      if (id !== requestId) return;
      entries = next;
      shownFilter = query;
      phase = next.length ? 'rows' : query.trim() ? 'nomatch' : 'empty';
      active = firstEnabled(next);
      if (pickOnLoad) pick(entries[active]);
    } catch (err) {
      if (id !== requestId) return;
      entries = [];
      error = errorText(err, 'Could not load the recent prompts.');
      phase = 'error';
    } finally {
      if (id === requestId) pickOnLoad = false;
    }
  }

  // Every opening starts from the whole list and a clean filter.
  $effect(() => {
    if (!open) return;
    untrack(() => {
      filter = '';
      entries = [];
      shownFilter = null;
      phase = 'loading';
      active = -1;
      void load('');
    });
    return () => {
      requestId += 1;
      clearTimeout(timer);
      pickOnLoad = false;
    };
  });

  // With a keyboard the filter takes focus; a finger would have the touch keyboard cover the list.
  $effect(() => {
    if (open && inputEl && !layout.compact && (byKeyboard || !layout.coarse)) inputEl.focus({ preventScroll: true });
  });

  // Keep the active row in view.
  $effect(() => {
    if (!open || active < 0) return;
    void tick().then(() => document.getElementById(`ws-recent-${active}`)?.scrollIntoView?.({ block: 'nearest' }));
  });

  function onFilterInput(value: string): void {
    filter = value;
    pickOnLoad = false;
    clearTimeout(timer);
    timer = setTimeout(() => void load(value), INPUT_DEBOUNCE_MS);
  }

  function move(step: 1 | -1): void {
    for (let i = active + step; i >= 0 && i < entries.length; i += step) {
      if (!isDisabled(entries[i])) {
        active = i;
        return;
      }
    }
  }

  function pick(entry: RecentPrompt | undefined): void {
    if (!entry || isDisabled(entry)) return;
    onpick(entry);
  }

  function onFilterKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowDown') { event.preventDefault(); move(1); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); move(-1); }
    else if (event.key === 'Enter') { event.preventDefault(); pickWhenCurrent(); }
  }

  // The shown rows may answer an earlier filter: fetch the typed one now, without waiting out the debounce, and pick from it.
  function pickWhenCurrent(): void {
    if (shownFilter === filter) {
      pick(entries[active]);
      return;
    }
    pickOnLoad = true;
    clearTimeout(timer);
    void load(filter);
  }

  // Tab goes from the filter to Clear recents; past it the menu closes and focus goes back to its button.
  function onMenuKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Tab' || event.shiftKey) return;
    const stops = (event.currentTarget as HTMLElement).querySelectorAll<HTMLElement>('input, button:not(:disabled)');
    if (event.target !== stops[stops.length - 1]) return;
    event.preventDefault();
    onclose();
    returnFocus?.focus();
  }

  async function clearAll(): Promise<void> {
    onclose();
    const approved = await requestConfirm({
      question: `Clear all recent ${mode} prompts?`,
      info: `Every ${mode} prompt kept for the Recent prompts list is removed, on every device. Your ${mode === 'video' ? 'image' : 'video'} prompts, job log and files stay.`,
      confirmLabel: 'Clear recents',
      cancelLabel: 'Keep',
      danger: true,
    });
    if (!approved) {
      returnFocus?.focus();
      return;
    }
    try {
      await clearRecentPrompts(mode);
    } catch (err) {
      addToast(errorText(err, 'Could not clear the recent prompts.'), 'error');
      return;
    }
    addToast('Recent prompts cleared', 'success');
  }
</script>

{#snippet filterField()}
  <div class="recents-filter">
    <Icon name="search" size={14} class="recents-filter-icon" />
    <input
      bind:this={inputEl}
      class="ui-field recents-filter-input"
      type="search"
      role="combobox"
      aria-label="Filter recent prompts"
      aria-expanded="true"
      aria-controls="ws-recents-list"
      aria-autocomplete="list"
      aria-activedescendant={active >= 0 && phase === 'rows' ? `ws-recent-${active}` : undefined}
      placeholder="Filter recent prompts"
      autocomplete="off"
      spellcheck="false"
      value={filter}
      oninput={(e) => onFilterInput(e.currentTarget.value)}
      onkeydown={onFilterKeydown}
    />
  </div>
{/snippet}

{#snippet list()}
  <div class="recents-list" id="ws-recents-list" role="listbox" aria-label="Recent prompts" aria-busy={phase === 'loading'} data-state={phase}>
    {#if phase === 'loading'}
      <div class="recents-state"><Spinner size="md" /><span>Loading recent prompts…</span></div>
    {:else if phase === 'error'}
      <div class="recents-state recents-error" role="alert">
        <span class="recents-error-text"><Icon name="alert" size={14} />{error}</span>
        <button type="button" class="ui-btn ui-btn-sm" onclick={() => void load(filter)}>Try again</button>
      </div>
    {:else if phase === 'empty'}
      <div class="recents-state">
        <Icon name="clock" size={20} />
        <span class="recents-state-title">No recent prompts yet</span>
        <span>{mode === 'video' ? 'Every scene and action you generate is kept here.' : 'Every image prompt you generate is kept here.'}</span>
      </div>
    {:else if phase === 'nomatch'}
      <div class="recents-state">
        <Icon name="search" size={20} />
        <span class="recents-state-title">No recent prompts match</span>
        <span class="recents-state-query">“{filter.trim()}”</span>
      </div>
    {:else}
      {#each entries as entry, i (i)}
        {@const disabled = isDisabled(entry)}
        {@const sub = rowSub(entry)}
        <div
          id={`ws-recent-${i}`}
          role="option"
          class="recents-row"
          class:active={!layout.compact && i === active}
          aria-selected={!layout.compact && i === active}
          aria-disabled={disabled ? 'true' : undefined}
          tabindex="-1"
          title={rowTitle(entry)}
          onclick={() => pick(entry)}
          onkeydown={(e) => { if (e.key === 'Enter') pick(entry); }}
        >
          <span class="recents-line">
            <span class="recents-text" class:mono={entry.format === 'json'}>{rowText(entry)}</span>
            <span class="recents-age">{timeAgo(entry.used_at, Date.now())}</span>
          </span>
          {#if entry.format === 'json' || sub}
            <span class="recents-line recents-meta">
              {#if entry.format === 'json'}<span class="ui-badge recents-json">{'{ }'} JSON</span>{/if}
              {#if sub}<span class="recents-sub" class:reason={disabled}>{sub}</span>{/if}
            </span>
          {/if}
        </div>
      {/each}
    {/if}
  </div>
{/snippet}

{#snippet clearButton()}
  <button type="button" class="ui-btn ui-btn-sm ui-btn-danger" disabled={phase !== 'rows' && phase !== 'nomatch'} onclick={() => void clearAll()}><Icon name="trash" size={14} />Clear recents</button>
{/snippet}

{#if layout.compact}
  <Sheet {open} title="Recent prompts" fill role="dialog" layer="menu" onclose={() => { if (open) onclose(); }}>
    <div class="recents-sheet">
      {@render filterField()}
      {@render list()}
    </div>
    {#snippet footer()}
      {@render clearButton()}
    {/snippet}
  </Sheet>
{:else if open}
  <div
    use:popover={{ anchor, returnFocus, align: 'end', onclose, followViewport: layout.coarse }}
    class="recents ui-overlay"
    role="dialog"
    aria-label="Recent prompts"
    tabindex="-1"
    onkeydown={onMenuKeydown}
  >
    {@render filterField()}
    {@render list()}
    <div class="recents-foot">{@render clearButton()}</div>
  </div>
{/if}

<style>
  /* Popover (wide and tall): one fixed frame from the moment it opens; every state changes inside the list. */
  .recents { --recents-row-h: 48px; z-index: var(--z-menu); display: flex; flex-direction: column; width: min(384px, calc(100vw - 16px)); overflow: hidden; }
  .recents-filter { position: relative; flex: none; padding: 8px; border-bottom: 1px solid var(--color-border-subtle); }
  .recents-filter :global(.recents-filter-icon) { position: absolute; top: 50%; left: 16px; translate: 0 -50%; color: var(--color-text-muted); pointer-events: none; }
  .recents-filter-input { padding-left: 28px; }
  .recents-filter-input::-webkit-search-cancel-button { appearance: none; }
  .recents-list { display: flex; flex: 1 1 auto; min-height: 0; flex-direction: column; gap: 0; overflow-y: auto; overscroll-behavior: contain; padding: 4px; }
  .recents .recents-list { flex: none; height: min(calc(6 * var(--recents-row-h) + 8px), 60dvh); }
  .recents-foot { display: flex; flex: none; justify-content: flex-end; padding: 6px 8px; border-top: 1px solid var(--color-border-subtle); }

  .recents-row { display: flex; flex: none; flex-direction: column; justify-content: center; gap: 3px; min-height: var(--recents-row-h); padding: 6px 8px; border-radius: var(--radius-sm); cursor: pointer; outline: none; }
  @media (hover: hover) { .recents-row:hover:not([aria-disabled='true']) { background: var(--color-bg-surface-hover); } }
  .recents-row.active { background: var(--color-bg-surface-hover); }
  .recents-row[aria-disabled='true'] { cursor: not-allowed; }
  .recents-row[aria-disabled='true'] .recents-text { color: var(--color-text-muted); }
  .recents-line { display: flex; align-items: center; gap: 6px; min-width: 0; }
  .recents-line:first-child { align-items: flex-start; }
  .recents-text { flex: 1; min-width: 0; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; overflow-wrap: anywhere; font-size: var(--text-content); color: var(--color-text-primary); }
  .recents-text.mono { font-family: var(--font-mono); font-size: var(--text-ui); }
  .recents-age { flex: none; font-size: var(--text-meta); color: var(--color-text-muted); font-variant-numeric: tabular-nums; }
  .recents-json { font-family: var(--font-mono); font-weight: 500; }
  .recents-sub { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-meta); color: var(--color-text-muted); }
  .recents-sub.reason { color: var(--color-text-secondary); }

  .recents-state { display: flex; flex: 1; flex-direction: column; align-items: center; justify-content: center; gap: 6px; padding: 16px; text-align: center; font-size: var(--text-ui); color: var(--color-text-muted); }
  .recents-state-title { font-family: var(--font-heading); font-size: var(--text-content); font-weight: 800; color: var(--color-text-primary); }
  .recents-state-query { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .recents-error-text { display: inline-flex; align-items: flex-start; gap: 6px; color: var(--color-error); text-align: left; }
  .recents-error-text :global(svg) { flex: none; margin-top: 1px; }

  /* Touch: finger-sized rows at every width. */
  @media (pointer: coarse) { .recents { --recents-row-h: 56px; } }

  /* Phone sheet: full height, so the filter never moves as results change; rows like the action sheet's. */
  .recents-sheet { --recents-row-h: 60px; display: flex; flex: 1; min-height: 0; flex-direction: column; }
  .recents-sheet .recents-filter { padding: 4px max(12px, env(safe-area-inset-right, 0px)) 8px max(12px, env(safe-area-inset-left, 0px)); }
  .recents-sheet .recents-filter :global(.recents-filter-icon) { left: 22px; translate: 0 calc(-50% - 2px); }
  .recents-sheet .recents-filter-input { height: var(--spacing-control); padding-left: 32px; }
  .recents-sheet .recents-list { padding: 4px max(8px, env(safe-area-inset-right, 0px)) 8px max(8px, env(safe-area-inset-left, 0px)); }
  .recents-sheet .recents-row { padding: 8px 12px; }
  .recents-sheet .recents-text { font-size: 15px; }
  .recents-sheet .recents-text.mono { font-size: 13px; }
  .recents-sheet .recents-age, .recents-sheet .recents-sub { font-size: var(--text-ui); }
</style>
