import { flushSync } from 'svelte';
import { describe, expect, it } from 'vitest';

import { mountComponent } from '../../../test-utils/mount';

import Mascot from './Mascot.svelte';
import MascotSpot from './MascotSpot.svelte';

describe('Mascot', () => {
  it('defaults to the idle mood', () => {
    const { target } = mountComponent(Mascot);

    const svg = target.querySelector('svg') as SVGSVGElement;
    expect(svg.dataset.mood).toBe('idle');
    expect(svg.getAttribute('aria-label')).toBe('Z-Vision mascot');
    expect(svg.getAttribute('width')).toBe('120');
  });

  it('shows the mood it is given at the size it is given', () => {
    const { target } = mountComponent(Mascot, { mood: 'thinking', size: 48 });

    const svg = target.querySelector('svg') as SVGSVGElement;
    expect(svg.dataset.mood).toBe('thinking');
    expect(svg.getAttribute('width')).toBe('48');
  });

  it('renders a travelling spot around the mascot', () => {
    const { target } = mountComponent(MascotSpot, { mood: 'curious', tool: 'clapper', size: 64, class: 'dock' });

    const svg = target.querySelector('svg') as SVGSVGElement;
    expect(svg.dataset.mood).toBe('curious');
    expect(svg.dataset.tool).toBe('clapper');
    expect(svg.getAttribute('width')).toBe('64');
  });

  it('holds a brush by default', () => {
    const { target } = mountComponent(Mascot);

    expect((target.querySelector('svg') as SVGSVGElement).dataset.tool).toBe('brush');
  });

  it('bounces once when clicked, ignoring clicks until the bounce ends', () => {
    const { target } = mountComponent(Mascot);
    const svg = target.querySelector('svg') as SVGSVGElement;
    expect(svg.hasAttribute('data-boop')).toBe(false);

    flushSync(() => svg.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(svg.hasAttribute('data-boop')).toBe(true);

    const bounce = svg.querySelector('.boop') as SVGGElement;
    flushSync(() => bounce.dispatchEvent(new Event('animationend', { bubbles: true })));
    expect(svg.hasAttribute('data-boop')).toBe(false);
  });
});
