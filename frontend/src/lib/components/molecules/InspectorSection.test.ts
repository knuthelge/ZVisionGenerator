// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { createRawSnippet } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import InspectorSection from './InspectorSection.svelte';

describe('InspectorSection', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  const children = createRawSnippet(() => ({ render: () => '<p>body</p>' }));
  const summary = () => target.querySelector('.inspector-summary') as HTMLElement;

  function show(props: { summary?: string; active?: boolean; error?: boolean }): void {
    app = flushSync(() => mount(InspectorSection, { target, props: { id: 'test-section', title: 'Images', children, ...props } }));
  }

  beforeEach(() => {
    localStorage.clear();
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

  it('shows its summary in the header, muted by default', () => {
    show({ summary: 'None' });
    expect(summary().textContent).toBe('None');
    expect(summary().dataset.tone).toBe('muted');
  });

  it('marks an active summary', () => {
    show({ summary: 'First frame', active: true });
    expect(summary().dataset.tone).toBe('active');
  });

  it('marks an error summary, which wins over active', () => {
    show({ summary: 'Start image missing', active: true, error: true });
    expect(summary().dataset.tone).toBe('error');
  });
});
