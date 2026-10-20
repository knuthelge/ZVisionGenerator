// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/state/layoutMode.svelte', () => ({
  layout: { compact: true, mode: 'compact', coarse: true },
}));

import Modal from './Modal.svelte';

describe('Modal on a compact layout', () => {
  let app: Record<string, unknown> | null = null;

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    document.body.innerHTML = '';
  });

  it('renders a side dialog as the same bottom sheet as a centred one', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    app = flushSync(() => mount(Modal, { target, props: { open: true, title: 'Job log', placement: 'side', fill: true } }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    flushSync();

    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull();
    expect(document.body.querySelector('[data-placement]')).toBeNull();
  });
});
