// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { MemoryFit } from '$lib/types';

import ModelStatusBadges, { NOT_DOWNLOADED_TOOLTIP, memoryFitFor, memoryFitTitle } from './ModelStatusBadges.svelte';
import * as molecules from './index';

const FIT: MemoryFit = {
  budget_gb: 10.7,
  by_quantize: {
    none: { status: 'too_large', required_gb: 20.8 },
    '8': { status: 'tight', required_gb: 9.5 },
    '4': { status: 'fits', required_gb: 7.1 }
  }
};

describe('memoryFitFor', () => {
  it('selects the estimate for the chosen quantize level', () => {
    expect(memoryFitFor(FIT, null)?.status).toBe('too_large');
    expect(memoryFitFor(FIT, 8)?.status).toBe('tight');
    expect(memoryFitFor(FIT, 4)?.status).toBe('fits');
  });

  it('shows nothing rather than another setting when the selected level was not estimated', () => {
    expect(memoryFitFor({ budget_gb: 10, by_quantize: { none: { status: 'fits', required_gb: 3 } } }, 8)).toBeNull();
    expect(memoryFitFor(null, null)).toBeNull();
  });

  it('switches to the all-resident estimate when low memory is off', () => {
    const video: MemoryFit = {
      budget_gb: 10.7,
      by_quantize: { none: { status: 'tight', required_gb: 9.8 } },
      without_low_memory: { status: 'too_large', required_gb: 16.4 },
    };

    expect(memoryFitFor(video, null, true)?.status).toBe('tight');
    expect(memoryFitFor(video, null, false)?.status).toBe('too_large');
    expect(memoryFitFor(FIT, null, false)?.status).toBe('too_large'); // image models have no low-memory variant
  });
});

describe('memoryFitTitle on a discrete GPU', () => {
  const CUDA_FIT: MemoryFit = {
    kind: 'discrete',
    budget_gb: 10,
    system_budget_gb: 30,
    by_quantize: {
      none: { status: 'tight', required_gb: 4.2, system_gb: 33.6 },
      '8': { status: 'fits', required_gb: 4.2, system_gb: 17.6 }
    }
  };

  it('reports the system memory need of the selected level and of each quantized level', () => {
    const title = memoryFitTitle(CUDA_FIT, CUDA_FIT.by_quantize.none);

    expect(title).toContain('33.6 GB');
    expect(title).toContain('17.6 GB');
  });
});

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

  it('is exported from the molecules barrel', () => {
    expect(molecules.ModelStatusBadges).toBe(ModelStatusBadges);
  });

  it('shows the memory fit for the selected quantize level', () => {
    render({ downloaded: true, memoryFit: FIT, quantize: 8 });

    const fit = target.querySelector('[data-testid="model-memory-fit"]');
    expect(fit?.querySelector('[data-status]')?.getAttribute('data-status')).toBe('tight');
    expect(fit?.querySelector('[data-status]')?.textContent?.trim()).toBe('Tight');
    expect(fit?.querySelector('[role="tooltip"]')?.textContent).toContain('9.5 GB');
    expect(fit?.querySelector('[role="tooltip"]')?.textContent).toContain('can slow down while generating');
    expect(target.querySelector('[data-testid="model-download-status"]')).toBeNull();
  });

  it('flags models that are not downloaded', () => {
    render({ downloaded: false, memoryFit: null });

    const status = target.querySelector('[data-testid="model-download-status"]');
    expect(status?.textContent).toContain('Not downloaded');
    expect(status?.querySelector('[role="tooltip"]')?.textContent).toBe(NOT_DOWNLOADED_TOOLTIP);
    expect(status?.getAttribute('tabindex')).toBe('0');
    expect(target.querySelector('[data-testid="model-memory-fit"]')).toBeNull();
  });

  it('right-aligns the tooltip when asked and appends the estimate note', () => {
    render({ memoryFit: { ...FIT, note: 'Excludes the optional upscale pass.' }, tooltipAlign: 'end' });

    const trigger = target.querySelector('[data-testid="model-memory-fit"]') as HTMLElement;
    trigger.dispatchEvent(new Event('pointerenter'));
    flushSync();

    const tooltip = trigger.querySelector('[role="tooltip"]') as HTMLElement;
    expect(tooltip.style.left).not.toBe('');
    expect(tooltip.textContent).toContain('Excludes the optional upscale pass.');
  });

  it('renders nothing when status is unknown', () => {
    render({ downloaded: null, memoryFit: null });

    expect(target.textContent?.trim()).toBe('');
  });
});
