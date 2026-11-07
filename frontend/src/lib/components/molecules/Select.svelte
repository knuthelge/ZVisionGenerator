<script lang="ts" module>
  export interface SelectOption {
    value: string;
    label: string;
    disabled?: boolean;
    /** A second line under the label, in the muted shade. */
    hint?: string;
    /** The column the option sits in when the list lays its groups out side by side (`groupColumns`). */
    group?: string;
  }

  /** How the closed control looks: a form field, a toolbar shell, or unstyled for the caller to style. */
  export type SelectVariant = 'field' | 'shell' | 'bare';

  const VARIANT_CLASS: Record<SelectVariant, string> = {
    field: 'ui-field',
    shell: 'ui-select-shell h-control rounded-sm px-2 text-ui',
    bare: '',
  };

  const TYPEAHEAD_RESET_MS = 600;

  let nextId = 0;
</script>

<script lang="ts">
  import type { Snippet } from 'svelte';
  import { tick } from 'svelte';
  import { popover } from '$lib/actions/popover';
  import { layout } from '$lib/state/layoutMode.svelte';
  import Icon from '../atoms/Icon.svelte';

  interface Props {
    id?: string;
    name?: string;
    value?: string;
    options: SelectOption[];
    /** Shown while no option is chosen; also an empty, unpickable first option so `required` can fail. */
    placeholder?: string;
    disabled?: boolean;
    required?: boolean;
    error?: string | null;
    /** Accessible name when there is no visible label. */
    ariaLabel?: string;
    /** Name of the open list; defaults to the accessible name or the field's label. */
    label?: string;
    variant?: SelectVariant;
    /** Classes for the outer box, e.g. its width in a flex row. */
    class?: string;
    /** Classes for the button that shows the choice. */
    triggerClass?: string;
    testId?: string;
    /** Custom content for the closed control; gets the chosen option. */
    trigger?: Snippet<[SelectOption | undefined]>;
    /**
     * Lay the options that have a `group` out as one titled column per group, side by side, under the options that have none.
     * Up and Down walk the options in their order, Left and Right move to the same row of the next column.
     */
    groupColumns?: boolean;
    onchange?: (value: string) => void;
  }

  let {
    id,
    name,
    value = $bindable(''),
    options,
    placeholder,
    disabled = false,
    required = false,
    error = null,
    ariaLabel,
    label,
    variant = 'field',
    class: extraClass = '',
    triggerClass = '',
    testId,
    trigger,
    groupColumns = false,
    onchange,
  }: Props = $props();

  const listId = `ui-select-${++nextId}`;

  let open = $state(false);
  let active = $state(-1);
  let minWidth = $state(0);
  let triggerEl = $state<HTMLButtonElement | null>(null);
  let listEl = $state<HTMLDivElement | null>(null);
  let nativeEl = $state<HTMLSelectElement | null>(null);
  let typed = '';
  let typedTimer: ReturnType<typeof setTimeout> | undefined;

  const selected = $derived(options.find((option) => option.value === value));
  const listLabel = $derived(label ?? ariaLabel ?? triggerEl?.labels?.[0]?.textContent?.trim() ?? 'Choose');

  /** In column mode: the options with no group (listed first, full width), and the groups, each with the options' indexes. */
  const columns = $derived.by(() => {
    if (!groupColumns) return null;
    const plain: number[] = [];
    const groups = new Map<string, number[]>();
    options.forEach((option, index) => {
      if (option.group === undefined) plain.push(index);
      else groups.set(option.group, [...(groups.get(option.group) ?? []), index]);
    });
    return { plain, groups: [...groups.entries()].map(([name, indexes]) => ({ name, indexes })) };
  });

  /** The enabled option in the column beside the active one, at the same row (or the nearest below it); -1 when there is none. */
  function sideways(direction: 1 | -1): number {
    if (!columns) return -1;
    const here = columns.groups.findIndex((group) => group.indexes.includes(active));
    const next = columns.groups[here + direction];
    if (here < 0 || !next) return -1;
    const row = columns.groups[here].indexes.indexOf(active);
    const target = next.indexes[Math.min(row, next.indexes.length - 1)];
    return options[target].disabled ? -1 : target;
  }

  function enabledIndex(from: number, step: 1 | -1): number {
    for (let index = from; index >= 0 && index < options.length; index += step) {
      if (!options[index].disabled) return index;
    }
    return -1;
  }

  async function show(): Promise<void> {
    if (disabled || open) return;
    minWidth = triggerEl?.offsetWidth ?? 0;
    const current = options.findIndex((option) => option.value === value && !option.disabled);
    active = current >= 0 ? current : enabledIndex(0, 1);
    open = true;
    await tick();
    listEl?.focus();
    revealActive();
  }

  function close(returnFocus = true): void {
    if (!open) return;
    open = false;
    if (returnFocus) triggerEl?.focus();
  }

  function choose(index: number): void {
    const option = options[index];
    if (!option || option.disabled) return;
    close();
    if (option.value === value) return;
    value = option.value;
    onchange?.(option.value);
  }

  function revealActive(): void {
    listEl?.querySelector<HTMLElement>(`#${listId}-${active}`)?.scrollIntoView?.({ block: 'nearest' });
  }

  function moveTo(index: number): void {
    if (index < 0) return;
    active = index;
    revealActive();
  }

  /** Jump to the next option whose label starts with what was typed in the last moment. */
  function typeahead(key: string): void {
    clearTimeout(typedTimer);
    typed += key.toLowerCase();
    typedTimer = setTimeout(() => { typed = ''; }, TYPEAHEAD_RESET_MS);
    const start = typed.length === 1 ? active + 1 : Math.max(active, 0);
    for (let offset = 0; offset < options.length; offset++) {
      const index = (start + offset) % options.length;
      if (!options[index].disabled && options[index].label.toLowerCase().startsWith(typed)) {
        moveTo(index);
        return;
      }
    }
  }

  function onTriggerKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      void show();
    }
  }

  // Escape is handled by the popover action, which closes the list.
  function onListKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowDown') { event.preventDefault(); moveTo(enabledIndex(active + 1, 1)); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); moveTo(enabledIndex(active - 1, -1)); }
    else if (event.key === 'ArrowRight' && columns) { event.preventDefault(); moveTo(sideways(1)); }
    else if (event.key === 'ArrowLeft' && columns) { event.preventDefault(); moveTo(sideways(-1)); }
    else if (event.key === 'Home') { event.preventDefault(); moveTo(enabledIndex(0, 1)); }
    else if (event.key === 'End') { event.preventDefault(); moveTo(enabledIndex(options.length - 1, -1)); }
    else if (event.key === 'Enter') { event.preventDefault(); choose(active); }
    // Space picks on release: picking on press moves focus to the button, and Firefox would click it on the release.
    else if (event.key === ' ') { event.preventDefault(); }
    // Focus goes back to the button first, so Tab moves on from the field rather than from the list at the end of <body>.
    else if (event.key === 'Tab') { close(); }
    else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) { event.preventDefault(); typeahead(event.key); }
  }

  function onListKeyup(event: KeyboardEvent): void {
    if (event.key === ' ') { event.preventDefault(); choose(active); }
  }

  // The hidden native select carries the value into forms and validation; a change made to it directly is taken over.
  function onNativeChange(event: Event): void {
    value = (event.currentTarget as HTMLSelectElement).value;
    onchange?.(value);
  }

  // A required field left empty is focused by the browser; the visible button takes that focus.
  function onNativeFocus(): void {
    triggerEl?.focus();
  }

  // A form reset puts the hidden select on its first option once the reset event has run. The value is the caller's
  // (a page clears its own state on reset), so put it back afterwards: what is submitted is what the button shows.
  // A task, not a microtask: a reset button's reset runs after the event's microtasks.
  $effect(() => {
    const form = nativeEl?.form;
    if (!form) return;
    const reconcileAfterReset = (): void => {
      setTimeout(() => {
        if (nativeEl) nativeEl.value = value;
      });
    };
    form.addEventListener('reset', reconcileAfterReset);
    return () => form.removeEventListener('reset', reconcileAfterReset);
  });
</script>

<div class="ui-select relative min-w-0 {extraClass}" data-testid={testId}>
  <button
    bind:this={triggerEl}
    {id}
    type="button"
    role="combobox"
    class="ui-select-trigger {VARIANT_CLASS[variant]} {triggerClass}"
    aria-haspopup="listbox"
    aria-expanded={open}
    aria-controls={open ? listId : undefined}
    aria-label={ariaLabel}
    aria-invalid={error ? 'true' : undefined}
    aria-describedby={error && id ? `${id}-error` : undefined}
    data-value={value}
    {disabled}
    onclick={() => (open ? close() : void show())}
    onkeydown={onTriggerKeydown}
  >
    {#if trigger}
      {@render trigger(selected)}
    {:else}
      <span class="ui-select-value" class:placeholder={!selected}>{selected?.label ?? placeholder ?? ''}</span>
    {/if}
    <Icon name="chevdown" size={14} class="pointer-events-none shrink-0 text-text-muted" />
  </button>
  <select bind:this={nativeEl} class="ui-select-native" tabindex="-1" aria-hidden="true" {name} {required} {disabled} {value} onchange={onNativeChange} onfocus={onNativeFocus}>
    {#if placeholder}
      <option value="" disabled>{placeholder}</option>
    {/if}
    {#each options as option (option.value)}
      <option value={option.value} disabled={option.disabled}>{option.label}</option>
    {/each}
  </select>
</div>
{#if error}
  <p id="{id}-error" class="ui-help ui-help-error mt-1">{error}</p>
{/if}

{#snippet optionRow(option: SelectOption, index: number)}
  <!-- Focus stays on the listbox, which takes the keys and points at the active option (aria-activedescendant). -->
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_interactive_supports_focus -->
  <div
    id="{listId}-{index}"
    class="ui-select-option"
    role="option"
    aria-selected={option.value === value}
    aria-disabled={option.disabled ? 'true' : undefined}
    data-active={index === active}
    data-value={option.value}
    onpointermove={() => { if (!option.disabled) active = index; }}
    onclick={() => choose(index)}
  >
    <span class="min-w-0 flex-1 truncate">
      {option.label}
      {#if option.hint}<small class="ui-select-hint">{option.hint}</small>{/if}
    </span>
    {#if option.value === value}<Icon name="check" size={14} class="shrink-0 text-primary-main" />{/if}
  </div>
{/snippet}

{#if open}
  <!-- Right at the field on every screen. On a touch screen the rows are finger-sized, and the list stays open when
       the keyboard sliding away scrolls or resizes the page. -->
  <div class="ui-select-popover ui-overlay" style:min-width="{minWidth}px" use:popover={{ anchor: triggerEl, align: 'start', followViewport: layout.coarse, onclose: () => close(false) }}>
    <div
      bind:this={listEl}
      id={listId}
      class="ui-select-list"
      class:touch={layout.coarse}
      class:columns={columns !== null}
      role="listbox"
      tabindex="-1"
      aria-label={listLabel}
      aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined}
      onkeydown={onListKeydown}
    onkeyup={onListKeyup}
    >
      {#if columns}
        {#each columns.plain as index (options[index].value)}
          {@render optionRow(options[index], index)}
        {/each}
        <div class="ui-select-columns" class:many={columns.groups.length > 2} style:--columns={columns.groups.length}>
          {#each columns.groups as group (group.name)}
            <div class="ui-select-column" role="group" aria-label={group.name}>
              <p class="ui-select-group ui-area-label" aria-hidden="true">{group.name}</p>
              {#each group.indexes as index (options[index].value)}
                {@render optionRow(options[index], index)}
              {/each}
            </div>
          {/each}
        </div>
      {:else}
        {#each options as option, index (option.value)}
          {@render optionRow(option, index)}
        {/each}
      {/if}
    </div>
  </div>
{/if}

<style>
  .ui-select-trigger { display: flex; width: 100%; align-items: center; justify-content: space-between; gap: 8px; text-align: left; cursor: pointer; }
  .ui-select-trigger:disabled { cursor: not-allowed; }
  .ui-select-value { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .ui-select-value.placeholder { color: var(--color-zinc-600); }
  /* Under the button, so a browser's "please choose" bubble for a required field points at it. */
  .ui-select-native { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; pointer-events: none; }

  .ui-select-popover { z-index: var(--z-menu); max-width: min(28rem, calc(100vw - 16px)); }
  .ui-select-list { max-height: min(60vh, 22rem); overflow-y: auto; padding: 4px; outline: none; }
  /* Twelve rows to a column: tall enough to see the whole list without scrolling where the screen allows. */
  .ui-select-list.columns { max-height: min(75vh, 30rem); }
  .ui-select-option { display: flex; align-items: center; gap: 8px; border-radius: var(--radius-sm); padding: 6px 8px; font-size: var(--text-ui); color: var(--color-text-secondary); cursor: pointer; user-select: none; }
  .ui-select-option[aria-selected='true'] { color: var(--color-text-primary); }
  .ui-select-option[data-active='true'] { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .ui-select-list:focus-visible .ui-select-option[data-active='true'] { box-shadow: inset 0 0 0 2px var(--color-primary-main); }
  .ui-select-option[aria-disabled='true'] { opacity: 0.5; cursor: not-allowed; }
  .ui-select-hint { display: block; font-size: var(--text-meta); color: var(--color-text-muted); }
  .ui-select-columns { display: grid; grid-template-columns: repeat(var(--columns), minmax(0, 1fr)); gap: 0 4px; min-width: 15rem; }
  /* More than two groups side by side do not fit a finger's rows or a phone's width: they wrap to two columns. */
  .touch .ui-select-columns.many { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  @media (max-width: 30rem) { .ui-select-columns.many { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  .ui-select-group { padding: 8px 8px 4px; }
  .touch .ui-select-option { min-height: 44px; padding: 0 12px; font-size: 15px; }
  /* A finger has no pointer to follow, so the active row only shows while keys move it; the tick marks the choice. */
  .touch:not(:focus-visible) .ui-select-option[data-active='true'] { background: none; }
</style>
