<script lang="ts">
  import { popover } from '$lib/actions/popover';
  import { browsePlaces, listDirectory } from '$lib/api';
  import { Icon } from '$lib/components/atoms';
  import { completedPath, completionTarget, matchCompletions, readRecentPaths } from '$lib/state/pathBrowser';
  import type { BrowseEntry, BrowsePlace, BrowsePurposeName } from '$lib/types';
  import { fileName } from './assetDetails';

  interface Props {
    id: string;
    name?: string;
    value: string;
    purpose: BrowsePurposeName;
    placeholder?: string;
    disabled?: boolean;
    required?: boolean;
    /** Classes for the input; `plain` drops the standard field look for a caller that styles it itself. */
    class?: string;
    plain?: boolean;
    ariaDescribedby?: string;
    ariaInvalid?: boolean;
    ariaBusy?: boolean;
    /** The text changed, by typing or by taking a completion. */
    oninput: (value: string) => void;
    /** A completion or recent path named a file to take now (Enter or click on it). */
    onpick?: (path: string) => void;
    /** Keys the completion list did not use, such as Enter with nothing highlighted. */
    onkeydown?: (event: KeyboardEvent) => void;
    onchange?: (event: Event) => void;
    onblur?: (event: FocusEvent) => void;
    oninvalid?: (event: Event) => void;
  }

  let {
    id,
    name,
    value,
    purpose,
    placeholder,
    disabled = false,
    required = false,
    class: extraClass = '',
    plain = false,
    ariaDescribedby,
    ariaInvalid,
    ariaBusy,
    oninput,
    onpick,
    onkeydown,
    onchange,
    onblur,
    oninvalid,
  }: Props = $props();

  type Suggestion = { key: string; label: string; detail: string; icon: 'folder' | 'file' | 'clock' | 'home'; apply: () => void };

  const DEBOUNCE_MS = 120;

  let inputEl = $state<HTMLInputElement | null>(null);
  let open = $state(false);
  let active = $state(-1);
  let suggestions = $state<Suggestion[]>([]);
  // The text the suggestions were made for; typing makes them stale until the next refresh.
  let suggestedFor = '';
  let places: BrowsePlace[] | null = null;
  let cache = new Map<string, BrowseEntry[]>();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let version = 0;
  const listId = $derived(`${id}-completions`);

  $effect(() => () => clearTimeout(timer));

  function close(): void {
    open = false;
    active = -1;
    clearTimeout(timer);
    version += 1;
  }

  function setText(text: string): void {
    if (inputEl) inputEl.value = text;
    oninput(text);
  }

  function schedule(text: string): void {
    clearTimeout(timer);
    timer = setTimeout(() => void refresh(text), DEBOUNCE_MS);
  }

  async function refresh(text: string): Promise<void> {
    const current = ++version;
    const next = text === '' ? await startSuggestions() : await folderSuggestions(text);
    if (current !== version) return;
    suggestions = next;
    suggestedFor = text;
    active = -1;
    open = next.length > 0 && document.activeElement === inputEl;
  }

  async function folderSuggestions(text: string): Promise<Suggestion[]> {
    const target = completionTarget(text);
    if (!target) return [];
    // Dot names (`~/.ziv`) are listed only once the typed name starts with a dot.
    const hidden = target.prefix.startsWith('.');
    const key = `${hidden ? 'h' : ''}:${target.dir}`;
    let entries = cache.get(key);
    if (!entries) {
      try {
        entries = (await listDirectory(purpose, target.dir, { hidden })).entries;
      } catch {
        return [];
      }
      cache.set(key, entries);
    }
    const matches = matchCompletions(entries, target.prefix);
    // A name typed in full leaves nothing to complete.
    if (matches.length === 1 && matches[0].kind === 'file' && matches[0].name === target.prefix) return [];
    return matches.map((entry) => ({
      key: entry.path,
      label: entry.kind === 'directory' ? `${entry.name}/` : entry.name,
      detail: '',
      icon: entry.kind === 'directory' ? 'folder' : 'file',
      apply: () => applyCompletion(completedPath(target.dir, entry), entry),
    }));
  }

  async function startSuggestions(): Promise<Suggestion[]> {
    if (places === null) {
      try {
        places = (await browsePlaces(purpose)).places;
      } catch {
        places = [];
      }
    }
    const recent = readRecentPaths(purpose).map<Suggestion>((path) => ({
      key: `recent:${path}`,
      label: fileName(path),
      detail: path,
      icon: 'clock',
      apply: () => { setText(path); close(); onpick?.(path); },
    }));
    const shortcuts = places.map<Suggestion>((place) => ({
      key: `place:${place.id}`,
      label: place.label,
      detail: place.path,
      icon: place.id === 'home' ? 'home' : 'folder',
      apply: () => { const text = place.path.endsWith('/') || place.path.endsWith('\\') ? place.path : `${place.path}/`; setText(text); void refresh(text); },
    }));
    return [...recent, ...shortcuts];
  }

  function applyCompletion(text: string, entry: BrowseEntry): void {
    setText(text);
    if (entry.kind === 'directory') {
      void refresh(text);
    } else {
      close();
      onpick?.(text);
    }
  }

  function handleInput(event: Event): void {
    const text = (event.currentTarget as HTMLInputElement).value;
    oninput(text);
    schedule(text);
  }

  // Suggestions open only while typing or on ↓, so focusing the box (or returning to it from a dialog) covers nothing.
  function handleFocus(): void {
    cache = new Map();
  }

  function handleKeydown(event: KeyboardEvent): void {
    if (!open && event.key === 'ArrowDown') {
      event.preventDefault();
      void refresh(inputEl?.value ?? '');
      return;
    }
    if (open && suggestions.length > 0) {
      if (event.key === 'ArrowDown') { event.preventDefault(); active = (active + 1) % suggestions.length; return; }
      if (event.key === 'ArrowUp') { event.preventDefault(); active = (active - 1 + suggestions.length) % suggestions.length; return; }
      if (event.key === 'Tab' && !event.shiftKey && (inputEl?.value ?? '') !== '') {
        event.preventDefault();
        const text = inputEl?.value ?? '';
        // Typed since the list was made: complete from a fresh list, never from names that no longer match.
        if (text !== suggestedFor) void refresh(text).then(() => { if (suggestedFor === text) suggestions[0]?.apply(); });
        else suggestions[Math.max(active, 0)].apply();
        return;
      }
      if (event.key === 'Enter' && active >= 0 && (inputEl?.value ?? '') === suggestedFor) { event.preventDefault(); suggestions[active].apply(); return; }
    }
    if (event.key === 'Enter') close();
    onkeydown?.(event);
  }

  function handleBlur(event: FocusEvent): void {
    close();
    onblur?.(event);
  }
</script>

<input
  bind:this={inputEl}
  {id}
  {name}
  type="text"
  class="{plain ? '' : 'ui-field'} {extraClass}"
  {value}
  {placeholder}
  {disabled}
  {required}
  spellcheck="false"
  autocomplete="off"
  role="combobox"
  aria-autocomplete="list"
  aria-expanded={open}
  aria-controls={open ? listId : undefined}
  aria-activedescendant={open && active >= 0 ? `${listId}-${active}` : undefined}
  aria-describedby={ariaDescribedby}
  aria-invalid={ariaInvalid ? 'true' : undefined}
  aria-busy={ariaBusy ? 'true' : undefined}
  oninput={handleInput}
  onfocus={handleFocus}
  onkeydown={handleKeydown}
  {onchange}
  onblur={handleBlur}
  {oninvalid}
>

{#if open}
  <div
    use:popover={{ anchor: inputEl, align: 'start', onclose: close }}
    id={listId}
    class="completions ui-overlay"
    role="listbox"
    aria-label="Path suggestions"
    style="min-width: {inputEl?.offsetWidth ?? 240}px"
  >
    {#each suggestions as suggestion, index (suggestion.key)}
      <!-- The input handles the keyboard for its options (aria-activedescendant). -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <div
        id="{listId}-{index}"
        class="completion"
        class:completion-active={index === active}
        role="option"
        tabindex="-1"
        aria-selected={index === active}
        onpointerdown={(event) => event.preventDefault()}
        onclick={() => suggestion.apply()}
      >
        <Icon name={suggestion.icon} size={14} />
        <span class="label">{suggestion.label}</span>
        {#if suggestion.detail}<span class="detail">{suggestion.detail}</span>{/if}
      </div>
    {/each}
    <p class="hint"><kbd class="ui-kbd">Tab</kbd> completes · <kbd class="ui-kbd">↑↓</kbd> move</p>
  </div>
{/if}

<style>
  .completions { z-index: 220; max-width: min(560px, calc(100vw - 16px)); max-height: min(50vh, 20rem); overflow-y: auto; padding: 4px; }
  .completion { display: flex; align-items: center; gap: 8px; height: var(--spacing-row); padding: 0 8px; border-radius: var(--radius-sm); font-size: var(--text-ui); color: var(--color-text-secondary); cursor: pointer; }
  .completion:hover, .completion-active { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .label { font-family: var(--font-mono); white-space: nowrap; }
  .detail { margin-left: auto; padding-left: 16px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .hint { padding: 6px 8px 2px; font-size: var(--text-meta); color: var(--color-text-muted); }
</style>
