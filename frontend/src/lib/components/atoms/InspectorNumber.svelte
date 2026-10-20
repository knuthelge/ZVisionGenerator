<script lang="ts">
  import { flushSync } from 'svelte';
  import { holdRepeat } from '$lib/actions/holdRepeat';
  import { nudgeValue, snapToSpec, type NumberSpec } from '$lib/actions/scrub';
  import { layout } from '$lib/state/layoutMode.svelte';

  interface Props extends NumberSpec {
    id?: string;
    name?: string;
    value: number | null;
    placeholder?: string;
    /** Empty input means `null` (e.g. "auto" or "random"); otherwise empty input is ignored. */
    nullable?: boolean;
    disabled?: boolean;
    /** Report typed values only on commit, for a value the parent reshapes (e.g. converts and snaps). */
    commitOnly?: boolean;
    ariaLabel?: string;
    /** Tooltip, e.g. why the field is disabled. */
    title?: string;
    /**
     * − and + buttons beside the field, with press-and-hold repeat. `auto` shows them for a coarse pointer, where
     * labels cannot be scrubbed; `false` keeps a plain field (W × H and Seed, which have presets and dice instead).
     */
    stepper?: boolean | 'auto';
    /** `lg`: a 48px control with a 22px value, for a popover that holds one setting. */
    size?: 'md' | 'lg';
    onchange: (value: number | null) => void;
  }

  let {
    id,
    name,
    value,
    step,
    min,
    max,
    placeholder,
    nullable = false,
    disabled = false,
    commitOnly = false,
    ariaLabel,
    title,
    stepper = 'auto',
    size = 'md',
    onchange,
  }: Props = $props();

  const showStepper = $derived(stepper === true || (stepper === 'auto' && layout.coarse));
  const stepLabel = $derived(ariaLabel ?? '');

  function onInput(event: Event): void {
    if (commitOnly) return;
    const raw = (event.currentTarget as HTMLInputElement).value;
    if (raw === '') {
      if (nullable) onchange(null);
      return;
    }
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) onchange(parsed);
  }

  /** Report a settled value; a commit-only one reaches the form only now, and Enter commits and submits in one task. */
  function report(next: number | null): void {
    if (next === value) return;
    if (commitOnly) flushSync(() => onchange(next));
    else onchange(next);
  }

  /** Show the value the parent kept: it may reshape a change, and an unchanged value would not re-render. */
  function showKept(input: HTMLInputElement): void {
    input.value = value === null ? '' : String(value);
  }

  // Typed values settle on a valid step within bounds once the field is committed (blur or Enter),
  // so the form never submits a value the backend or the browser would reject.
  function onCommit(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    if (input.value === '') {
      // A blank means `null` only for nullable fields; others keep the value they hold.
      if (nullable && commitOnly) report(null);
      showKept(input);
      return;
    }
    const parsed = Number(input.value);
    if (!Number.isFinite(parsed)) return;
    report(snapToSpec(parsed, { step, min, max }));
    showKept(input);
  }

  // Arrow keys step by `step`, Shift by ten steps. The browser's own step and range checks are off
  // (step="any", no min/max): a hidden collapsed field that failed them would block submit silently.
  // Bounds are applied here instead, when nudging and when a typed value is committed.
  function onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    // A commit-only field may hold typed text it has not reported yet; nudge from that.
    const input = event.currentTarget as HTMLInputElement;
    const typed = Number(input.value);
    const start = commitOnly && input.value !== '' && Number.isFinite(typed) ? typed : value;
    report(nudgeValue(start, event.key === 'ArrowUp' ? 1 : -1, { step, min, max }, event.shiftKey));
    showKept(input);
  }

  /** A stepper press: one step from the kept value, within bounds. */
  function nudge(direction: 1 | -1): void {
    if (disabled) return;
    report(nudgeValue(value, direction, { step, min, max }));
  }
</script>

<span class="inspector-number-wrap" class:with-steps={showStepper} class:lg={size === 'lg'}>
  {#if showStepper}
    <button
      type="button"
      class="inspector-step"
      tabindex="-1"
      aria-label="Decrease {stepLabel}"
      data-step="down"
      {disabled}
      use:holdRepeat={() => nudge(-1)}
    >−</button>
  {/if}
  <input
    {id}
    {name}
    type="number"
    step="any"
    {placeholder}
    {disabled}
    aria-label={ariaLabel}
    {title}
    class="inspector-number"
    value={value ?? ''}
    oninput={onInput}
    onchange={onCommit}
    onkeydown={onKeydown}
  >
  {#if showStepper}
    <button
      type="button"
      class="inspector-step"
      tabindex="-1"
      aria-label="Increase {stepLabel}"
      data-step="up"
      {disabled}
      use:holdRepeat={() => nudge(1)}
    >+</button>
  {/if}
</span>

<style>
  .inspector-number-wrap { display: flex; flex: 1; min-width: 0; align-items: center; }
  .inspector-number {
    width: 100%;
    min-width: 0;
    height: 24px;
    padding: 0 6px;
    border: 1px solid transparent;
    border-radius: 4px;
    background: transparent;
    font-family: var(--font-mono);
    font-size: var(--text-ui);
    color: var(--color-text-primary);
    appearance: textfield;
    -moz-appearance: textfield;
  }
  .inspector-number::-webkit-inner-spin-button,
  .inspector-number::-webkit-outer-spin-button { margin: 0; -webkit-appearance: none; }
  @media (hover: hover) { .inspector-number:hover { background: var(--color-bg-surface-hover); } }
  .inspector-number:focus { outline: none; border-color: var(--color-primary-main); background: var(--color-bg-surface); }
  .inspector-number:disabled { opacity: 0.4; }
  .inspector-number::placeholder { color: var(--color-zinc-600); }

  /* Touch: the field and its − and + make one filled control, 32px tall, with a 16px value so iOS does not zoom in. */
  .with-steps { border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); }
  .with-steps .inspector-number { height: 32px; min-width: 52px; background: transparent; text-align: center; font-size: 14px; }
  .with-steps .inspector-number:focus { background: var(--color-bg-surface); }
  .inspector-step {
    display: grid;
    flex: none;
    place-items: center;
    width: 36px;
    height: 32px;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 16px;
    line-height: 1;
    color: var(--color-text-secondary);
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
    transition: background-color 0.12s ease, color 0.12s ease;
  }
  .inspector-step:active:not(:disabled) { background: var(--color-bg-control-hover); color: var(--color-primary-main); }
  .inspector-step:disabled { opacity: 0.4; cursor: not-allowed; }
  @media (pointer: coarse) {
    .inspector-number { height: 32px; font-size: 14px; }
  }
  /* One setting in a popover: a 48px control the thumb can work. */
  .lg { border-radius: var(--radius-md); background: var(--color-bg-surface-hover); }
  .lg .inspector-number { height: 48px; padding: 0 12px; font-size: 22px; text-align: center; background: transparent; }
  .lg .inspector-number::placeholder { font-size: 15px; }
  .lg .inspector-step { width: 64px; height: 48px; font-size: 22px; border-radius: var(--radius-md); }
</style>
