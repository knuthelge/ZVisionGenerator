// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { createRawSnippet } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import InspectorSection from './InspectorSection.svelte';

describe('InspectorSection', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  const children = createRawSnippet(() => ({ render: () => '<p>body</p>' }));
  const summary = () => target.querySelector('.inspector-summary') as HTMLElement;

  function show(props: { summary?: string; active?: boolean; error?: boolean; open?: boolean; ontoggle?: () => void }): void {
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

  describe('controlled', () => {
    const head = () => target.querySelector('button.inspector-head') as HTMLButtonElement;
    const body = () => target.querySelector('.inspector-body') as HTMLElement;

    it('is a heading with a button that names its summary, and a labelled region', () => {
      show({ summary: 'Blue +30', open: true });
      expect(head().closest('h3')).not.toBeNull();
      expect(head().getAttribute('aria-label')).toBe('Images, Blue +30');
      expect(head().getAttribute('aria-controls')).toBe(body().id);
      expect(body().getAttribute('role')).toBe('region');
      expect(body().getAttribute('aria-labelledby')).toBe(head().id);
    });

    it('follows open, calls ontoggle and keeps no memory of its own', () => {
      const ontoggle = vi.fn();
      show({ open: false, ontoggle });
      expect(head().getAttribute('aria-expanded')).toBe('false');
      head().click();
      expect(ontoggle).toHaveBeenCalledOnce();
      expect(head().getAttribute('aria-expanded')).toBe('false');
      expect(localStorage.getItem('ziv-inspector-sections-v1')).toBeNull();
    });

    it('renders its body only while open', () => {
      show({ open: false });
      expect(body().hidden).toBe(true);
      expect(body().textContent).toBe('');
    });

    it('shows its body when open', () => {
      show({ open: true });
      expect(body().hidden).toBe(false);
      expect(body().textContent).toBe('body');
    });
  });
});
