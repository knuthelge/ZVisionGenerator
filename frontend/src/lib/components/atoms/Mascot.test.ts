// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import Mascot from './Mascot.svelte';
import MascotSpot from './MascotSpot.svelte';

describe('Mascot', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
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

  it('defaults to the idle mood', () => {
    app = flushSync(() => mount(Mascot, { target }));

    const svg = target.querySelector('svg') as SVGSVGElement;
    expect(svg.dataset.mood).toBe('idle');
    expect(svg.getAttribute('aria-label')).toBe('Z-Vision mascot');
    expect(svg.getAttribute('width')).toBe('120');
  });

  it.each([
    ['thinking', 'Z-Vision mascot is thinking'],
    ['creating', 'Z-Vision mascot is creating an image'],
    ['cheerful', 'Z-Vision mascot is cheering'],
    ['waving', 'Z-Vision mascot is waving hello'],
    ['curious', 'Z-Vision mascot is watching you type'],
    ['sleeping', 'Z-Vision mascot is sleeping'],
    ['sad', 'Z-Vision mascot is sad'],
    ['surprised', 'Z-Vision mascot is surprised'],
  ] as const)('exposes the %s mood', (mood, label) => {
    app = flushSync(() => mount(Mascot, { target, props: { mood, size: 48 } }));

    const svg = target.querySelector('svg') as SVGSVGElement;
    expect(svg.dataset.mood).toBe(mood);
    expect(svg.getAttribute('aria-label')).toBe(label);
    expect(svg.getAttribute('width')).toBe('48');
  });

  it('renders a travelling spot around the mascot', () => {
    app = flushSync(() => mount(MascotSpot, { target, props: { mood: 'curious', size: 64, class: 'dock' } }));

    const svg = target.querySelector('div.dock > svg.mascot') as SVGSVGElement;
    expect(svg.dataset.mood).toBe('curious');
    expect(svg.getAttribute('width')).toBe('64');
  });
});
