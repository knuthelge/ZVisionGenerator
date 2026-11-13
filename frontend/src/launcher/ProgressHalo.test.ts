import { describe, expect, it } from 'vitest';

import { mountComponent } from '../test-utils/mount';

import ProgressHalo from './ProgressHalo.svelte';

function halo(props: Record<string, unknown>): SVGSVGElement {
  const { target } = mountComponent(ProgressHalo, props);
  return target.querySelector('[data-testid="halo"]') as SVGSVGElement;
}

describe('ProgressHalo', () => {
  it('is a determinate progress bar while progress is known', () => {
    const svg = halo({ kind: 'progress', fraction: 0.41, label: 'Install progress', valuetext: 'about half' });

    expect(svg.getAttribute('role')).toBe('progressbar');
    expect(svg.getAttribute('aria-label')).toBe('Install progress');
    expect(svg.getAttribute('aria-valuenow')).toBe('41');
    expect(svg.getAttribute('aria-valuemin')).toBe('0');
    expect(svg.getAttribute('aria-valuemax')).toBe('100');
    expect(svg.getAttribute('aria-valuetext')).toBe('about half');
    expect(svg.hasAttribute('aria-hidden')).toBe(false);
  });

  it('falls back to the percentage for its value text', () => {
    expect(halo({ kind: 'progress', fraction: 0.5, label: 'Install progress' }).getAttribute('aria-valuetext')).toContain('50');
  });

  it('is an indeterminate, busy progress bar while the length is unknown', () => {
    const svg = halo({ kind: 'busy', label: 'Updating' });

    expect(svg.getAttribute('role')).toBe('progressbar');
    expect(svg.getAttribute('aria-label')).toBe('Updating');
    expect(svg.getAttribute('aria-busy')).toBe('true');
    expect(svg.hasAttribute('aria-valuenow')).toBe(false);
    expect(svg.hasAttribute('aria-valuetext')).toBe(false);
  });

  it.each(['rest', 'live', 'error'] as const)('is hidden from assistive technology at %s', (kind) => {
    const svg = halo({ kind, fraction: 0.4, label: 'Install progress' });

    expect(svg.getAttribute('aria-hidden')).toBe('true');
    expect(svg.hasAttribute('role')).toBe(false);
    expect(svg.hasAttribute('aria-valuenow')).toBe(false);
  });

  it('draws no arc without a fraction', () => {
    const arc = halo({ kind: 'error', label: 'x' }).querySelector('[data-testid="halo-arc"]') as SVGCircleElement;
    expect(arc.style.opacity).toBe('0');
  });
});
