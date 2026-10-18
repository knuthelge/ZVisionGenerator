// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import SettingsStrip from './SettingsStrip.svelte';
import type { SettingChip } from './settingsChips';

const chips: SettingChip[] = [
  { key: 'steps', label: 'Steps', value: '32', changed: true },
  { key: 'seed', label: 'Seed', value: 'random', changed: false },
  { key: 'model', label: '', value: 'Z-Image Turbo', changed: false, warn: true },
];

describe('SettingsStrip', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
  });

  it('shows one chip per setting with its value, the changed dot and All with the count', () => {
    const onopen = vi.fn();
    const onall = vi.fn();
    app = flushSync(() => mount(SettingsStrip, { target, props: { chips, activeKey: null, changedCount: 1, onopen, onall } }));

    const steps = target.querySelector('[data-chip="steps"]') as HTMLButtonElement;
    expect(steps.textContent).toContain('32');
    expect(steps.dataset.changed).toBe('true');
    expect((target.querySelector('[data-chip="seed"]') as HTMLElement).dataset.changed).toBe('false');

    steps.click();
    expect(onopen).toHaveBeenCalledWith('steps');

    const all = target.querySelector('button[aria-label="All settings"]') as HTMLButtonElement;
    expect(all.textContent).toContain('1');
    all.click();
    expect(onall).toHaveBeenCalledOnce();
  });

  it('marks the open chip expanded', () => {
    app = flushSync(() => mount(SettingsStrip, { target, props: { chips, activeKey: 'seed', changedCount: 0, onopen: vi.fn(), onall: vi.fn() } }));
    expect((target.querySelector('[data-chip="seed"]') as HTMLElement).getAttribute('aria-expanded')).toBe('true');
    expect((target.querySelector('[data-chip="steps"]') as HTMLElement).getAttribute('aria-expanded')).toBe('false');
  });
});
