import { afterEach, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { Component } from 'svelte';

type Mounted = { app: Record<string, unknown>; target: HTMLElement };

const live: Mounted[] = [];

/**
 * Mount *component* on the document body; every mount is torn down after each test.
 *
 * The component goes into a fresh element unless *into* names one (a form, say), which is attached to the body if it is not already.
 */
export function mountComponent(
  // Tests pass loose props (mocks, partial objects) that a component's own prop types would reject.
  component: Component<any>,
  props?: Record<string, unknown>,
  into?: HTMLElement,
): { target: HTMLElement; app: Record<string, unknown>; unmount: () => Promise<void> } {
  const target = into ?? document.createElement('div');
  if (!target.isConnected) document.body.appendChild(target);
  const app = flushSync(() => mount(component, { target, props } as never)) as Record<string, unknown>;
  const entry: Mounted = { app, target };
  live.push(entry);
  return {
    target,
    app,
    unmount: async () => {
      const index = live.indexOf(entry);
      if (index === -1) return;
      live.splice(index, 1);
      await unmount(app);
      target.remove();
    },
  };
}

afterEach(async () => {
  for (const { app, target } of live.splice(0)) {
    await unmount(app);
    target.remove();
  }
  document.body.innerHTML = '';
});

/**
 * Let pending promises, timers and Svelte effects run.
 *
 * With fake timers the clock is advanced by *ms* per round; otherwise the rounds wait on real timers, so pass a
 * non-zero *ms* only when fake timers are installed.
 */
export async function settle(ms = 0, rounds = 3): Promise<void> {
  for (let round = 0; round < rounds; round += 1) {
    if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(round === 0 ? ms : 0);
    else await new Promise((resolve) => setTimeout(resolve, round === 0 ? ms : 0));
    flushSync();
  }
}

/** Return the first button inside *root* whose trimmed text starts with *label*, or throw. */
export function buttonByText(label: string, root: ParentNode = document): HTMLButtonElement {
  const found = Array.from(root.querySelectorAll('button')).find((item) => item.textContent?.trim().startsWith(label));
  if (!found) throw new Error(`No button "${label}"`);
  return found;
}
