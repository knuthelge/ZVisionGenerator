<script lang="ts">
  import { nudgeValue, snapToSpec, type NumberSpec } from '$lib/actions/scrub';

  interface Props extends NumberSpec {
    id?: string;
    name?: string;
    value: number | null;
    placeholder?: string;
    /** Empty input means `null` (e.g. "auto" or "random"); otherwise empty input is ignored. */
    nullable?: boolean;
    disabled?: boolean;
    ariaLabel?: string;
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
    ariaLabel,
    onchange,
  }: Props = $props();

  function onInput(event: Event): void {
    const raw = (event.currentTarget as HTMLInputElement).value;
    if (raw === '') {
      if (nullable) onchange(null);
      return;
    }
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) onchange(parsed);
  }

  // Typed values settle on a valid step within bounds once the field is committed (blur or Enter),
  // so the form never submits a value the backend or the browser would reject.
  function onCommit(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    if (input.value === '') return;
    const parsed = Number(input.value);
    if (!Number.isFinite(parsed)) return;
    const snapped = snapToSpec(parsed, { step, min, max });
    input.value = String(snapped);
    if (snapped !== value) onchange(snapped);
  }

  // Arrow keys step by `step`, Shift by ten steps. The browser's own step and range checks are off
  // (step="any", no min/max): a hidden collapsed field that failed them would block submit silently.
  // Bounds are applied here instead, when nudging and when a typed value is committed.
  function onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    onchange(nudgeValue(value, event.key === 'ArrowUp' ? 1 : -1, { step, min, max }, event.shiftKey));
  }
</script>

<input
  {id}
  {name}
  type="number"
  step="any"
  {placeholder}
  {disabled}
  aria-label={ariaLabel}
  class="inspector-number"
  value={value ?? ''}
  oninput={onInput}
  onchange={onCommit}
  onkeydown={onKeydown}
>

<style>
  .inspector-number {
    width: 100%;
    min-width: 0;
    height: 24px;
    padding: 0 6px;
    border: 1px solid transparent;
    border-radius: 4px;
    background: transparent;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--color-text-primary);
    appearance: textfield;
    -moz-appearance: textfield;
  }
  .inspector-number::-webkit-inner-spin-button,
  .inspector-number::-webkit-outer-spin-button { margin: 0; -webkit-appearance: none; }
  .inspector-number:hover { border-color: var(--color-border-strong); background: var(--color-zinc-900); }
  .inspector-number:focus { outline: none; border-color: var(--color-primary-main); background: var(--color-zinc-900); }
  .inspector-number:disabled { opacity: 0.4; }
  .inspector-number::placeholder { color: var(--color-zinc-600); }
</style>
