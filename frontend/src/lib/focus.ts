import { tick } from 'svelte';

/**
 * Put the focus on *target* once the page has updated, when it has fallen to the page itself: the control that held it was removed
 * (a Run button that became Stop, a panel that closed). A focus that has moved elsewhere on purpose is left alone.
 */
export async function refocusIfLost(target: () => HTMLElement | null | undefined): Promise<void> {
  await tick();
  const active = document.activeElement;
  if (active === null || active === document.body) target()?.focus();
}
