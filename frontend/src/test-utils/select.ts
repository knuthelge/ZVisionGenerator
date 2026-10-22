import { flushSync } from 'svelte';

/** Return the open option list of the Select whose button is *box*, opening it first. */
export function openOptions(box: HTMLElement): HTMLElement[] {
  if (box.getAttribute('aria-expanded') !== 'true') flushSync(() => box.click());
  const list = document.getElementById(box.getAttribute('aria-controls') ?? '');
  if (!list) throw new Error('Select did not open its option list');
  return Array.from(list.querySelectorAll<HTMLElement>('[role="option"]'));
}

/** Return the values a Select offers, leaving it closed. */
export function optionValues(box: HTMLElement): string[] {
  const values = openOptions(box).map((option) => option.dataset.value ?? '');
  flushSync(() => box.click());
  return values;
}

/** Return the labels a Select offers, leaving it closed. */
export function optionLabels(box: HTMLElement): string[] {
  const labels = openOptions(box).map((option) => option.textContent?.trim() ?? '');
  flushSync(() => box.click());
  return labels;
}

/** Choose *value* in the Select whose button is *box*, the way a pointer would: open it and click the option. */
export function pickOption(box: HTMLElement, value: string): void {
  const option = openOptions(box).find((item) => item.dataset.value === value);
  if (!option) throw new Error(`Select has no option "${value}"`);
  flushSync(() => option.click());
}
