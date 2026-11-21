// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { imageMemory } from '../../../test-utils/memory';
import ModelStatusBadges, { NOT_DOWNLOADED_TOOLTIP } from './ModelStatusBadges.svelte';

describe('ModelStatusBadges', () => {
  let target: HTMLDivElement;
  let component: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(() => {
    if (component) {
      unmount(component);
      component = null;
    }
    target.remove();
  });

  function render(props: Record<string, unknown>): void {
    component = mount(ModelStatusBadges, { target, props });
    flushSync();
  }

  const tag = () => target.querySelector('[data-testid="model-memory"]');

  it('shows the reference job at the selected level as a plain word, with no popover', () => {
    render({ downloaded: true, memory: imageMemory('none'), level: '8' });

    expect(tag()?.getAttribute('data-status')).toBe('fits');
    expect(tag()?.textContent?.trim()).toContain('Fits');
    expect(target.querySelector('[role="tooltip"]')).toBeNull();
    expect(target.querySelector('[data-testid="model-download-status"]')).toBeNull();
  });

  it('judges the answer\'s own level when none is asked for', () => {
    render({ memory: imageMemory('4') });

    expect(tag()?.getAttribute('data-status')).toBe('too_large');
  });

  it('flags models that are not downloaded', () => {
    render({ downloaded: false, memory: null });

    const status = target.querySelector('[data-testid="model-download-status"]');
    expect(status?.textContent).toContain('Not downloaded');
    expect(status?.querySelector('[role="tooltip"]')?.textContent).toBe(NOT_DOWNLOADED_TOOLTIP);
    expect(status?.getAttribute('tabindex')).toBe('0');
    expect(tag()).toBeNull();
  });

  it('shows the memory of a model that is not downloaded beside its download status', () => {
    render({ downloaded: false, memory: imageMemory('8'), level: '8' });

    expect(target.querySelector('[data-testid="model-download-status"]')).not.toBeNull();
    expect(tag()?.getAttribute('data-status')).toBe('fits');
  });

  it('shows the download status instead of "Not downloaded" while a download runs', () => {
    render({ downloaded: false, downloadStatus: 'Downloading 42%' });

    const status = target.querySelector('[data-testid="model-download-status"]');
    expect(status?.textContent).toContain('Downloading 42%');
    expect(status?.textContent).not.toContain('Not downloaded');
  });

  it('ignores a download status for a model that is downloaded', () => {
    render({ downloaded: true, downloadStatus: 'Downloading 42%' });

    expect(target.querySelector('[data-testid="model-download-status"]')).toBeNull();
  });

  it('renders nothing when status is unknown', () => {
    render({ downloaded: null, memory: null });

    expect(target.textContent?.trim()).toBe('');
  });
});
