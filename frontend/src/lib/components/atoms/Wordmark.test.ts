import { describe, expect, it } from 'vitest';

import { mountComponent } from '../../../test-utils/mount';

import Wordmark from './Wordmark.svelte';

describe('Wordmark', () => {
  it('is a drawing hidden from assistive technology, leaving the name to its container', () => {
    const { target } = mountComponent(Wordmark);

    const svg = target.querySelector('svg') as SVGSVGElement;
    expect(svg.getAttribute('aria-hidden')).toBe('true');
    expect(svg.hasAttribute('aria-label')).toBe(false);
    expect(svg.textContent?.trim()).toBe('');
  });
});
