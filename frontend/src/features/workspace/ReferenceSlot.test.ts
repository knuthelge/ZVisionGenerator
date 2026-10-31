// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ReferenceSlot from './ReferenceSlot.svelte';

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

const NAMES = { browse: 'Browse for a start image', upload: 'Upload a start image from this device', clear: 'Remove start image', preview: 'Start image preview' };

describe('ReferenceSlot', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  function show(props: Record<string, unknown>): void {
    app = flushSync(() => mount(ReferenceSlot, {
      target,
      props: { idBase: 'ws-image', label: 'Start image', names: NAMES, pathRow: false, emptyText: '', path: null, file: null, previewUrl: null, onfilechange: vi.fn(), onpathchange: vi.fn(), ...props },
    }));
  }

  const group = () => target.querySelector('[role="group"]') as HTMLElement;
  const nameText = () => target.querySelector('.reference-name')?.textContent;

  beforeEach(() => {
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test-image') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
  });

  it('names its group by the label alone while it holds no image', () => {
    show({});
    expect(group().getAttribute('aria-label')).toBe('Start image');
  });
});
