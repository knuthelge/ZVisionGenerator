// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/state/layoutMode.svelte', () => ({
  layout: { compact: true, mode: 'compact', coarse: true },
}));

import { openOptions, pickOption } from '../../../test-utils/select';
import Select from './Select.svelte';

describe('Select on a phone', () => {
  let app: Record<string, unknown> | null = null;

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    document.body.innerHTML = '';
  });

  it('opens the list at the field, not as a sheet, and picks with a tap', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const onchange = vi.fn();
    app = flushSync(() => mount(Select, {
      target,
      props: { ariaLabel: 'Sort', value: 'newest', options: [{ value: 'newest', label: 'Newest first' }, { value: 'oldest', label: 'Oldest first' }], onchange },
    }));
    const box = target.querySelector('[role="combobox"]') as HTMLButtonElement;

    const list = openOptions(box)[0].closest('[role="listbox"]') as HTMLElement;
    expect(list.getAttribute('aria-label')).toBe('Sort');
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();

    pickOption(box, 'oldest');
    expect(onchange).toHaveBeenCalledExactlyOnceWith('oldest');
  });
});
