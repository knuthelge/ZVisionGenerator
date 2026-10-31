import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { mountComponent } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

import Mascot, { type MascotMood } from './Mascot.svelte';
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

const MOODS: MascotMood[] = ['idle', 'thinking', 'reading', 'creating', 'finishing', 'cheerful', 'waving', 'curious', 'sleeping', 'paused', 'sad', 'surprised', 'nodding'];

function mood(target: HTMLElement): SVGSVGElement {
  return target.querySelector('svg') as SVGSVGElement;
}

describe('Mascot moods', () => {
  it('gives every mood its own accessible label', () => {
    const labels = MOODS.map((value) => mood(mountComponent(Mascot, { mood: value }).target).getAttribute('aria-label'));

    expect(new Set(labels).size).toBe(MOODS.length);
    expect(labels.every((label) => label?.startsWith('Z-Vision mascot'))).toBe(true);
  });

  it('is hidden from assistive technology when it only decorates', () => {
    const { target } = mountComponent(Mascot, { decorative: true });

    expect(mood(target).getAttribute('aria-hidden')).toBe('true');
    expect(mood(target).hasAttribute('aria-label')).toBe(false);
  });

  it('carries both tools and shows the one it is given', () => {
    const { target } = mountComponent(Mascot, { tool: 'clapper' });

    expect(mood(target).dataset.tool).toBe('clapper');
    expect(target.querySelector('.tool-brush')).not.toBeNull();
    expect(target.querySelector('.tool-clapper')).not.toBeNull();
  });

  it('draws a simpler form below 48 px', () => {
    expect(mood(mountComponent(Mascot, { size: 24 }).target).hasAttribute('data-simple')).toBe(true);
    expect(mood(mountComponent(Mascot, { size: 120 }).target).hasAttribute('data-simple')).toBe(false);
  });
});

describe('Mascot pointer tracking', () => {
  afterEach(() => vi.restoreAllMocks());

  const listens = (spy: { mock: { calls: unknown[][] } }): boolean => spy.mock.calls.some(([type]) => type === 'pointermove');

  it('follows the pointer while idle', () => {
    const spy = vi.spyOn(window, 'addEventListener');
    mountComponent(Mascot, { mood: 'idle', size: 120 });
    expect(listens(spy)).toBe(true);
  });

  it('does not listen in the simple form, which has no eyes to move', () => {
    const spy = vi.spyOn(window, 'addEventListener');
    mountComponent(Mascot, { mood: 'idle', size: 24 });
    expect(listens(spy)).toBe(false);
  });
});

describe('Mascot creating canvas', () => {
  type Props = { mood: MascotMood; progress: number | undefined };
  /** What the mascot draws right now: while creating, only his canvas changes with the job's steps. */
  const drawing = (target: HTMLElement): string => mood(target).innerHTML;

  it('follows the job\'s steps and holds its last step while the job reports none', () => {
    const props = reactiveProps<Props>({ mood: 'creating', progress: 0 });
    const { target } = mountComponent(Mascot, props);
    const start = drawing(target);

    props.progress = 0.4;
    flushSync();
    const midway = drawing(target);
    expect(midway).not.toBe(start);

    props.progress = 0.8;
    flushSync();
    const late = drawing(target);
    expect(late).not.toBe(midway);

    // Between the images of a batch the job has no steps for a moment: the canvas keeps where it was.
    props.progress = undefined;
    flushSync();
    expect(drawing(target)).toBe(late);

    // The next image starts over from its own steps.
    props.progress = 0.4;
    flushSync();
    expect(drawing(target)).toBe(midway);
  });

  it('starts at the first step when the job has none yet, and again for the next job', () => {
    const props = reactiveProps<Props>({ mood: 'creating', progress: undefined });
    const { target } = mountComponent(Mascot, props);
    const first = drawing(target);

    props.progress = 0.8;
    flushSync();
    expect(drawing(target)).not.toBe(first);

    props.mood = 'thinking';
    flushSync();
    props.mood = 'creating';
    props.progress = undefined;
    flushSync();
    expect(drawing(target)).toBe(first);
  });

  it('draws the finished canvas while finishing, whatever the progress', () => {
    const early = mountComponent(Mascot, { mood: 'finishing', progress: 0.1 }).target;
    const late = mountComponent(Mascot, { mood: 'finishing', progress: 0.9 }).target;

    // Each mascot has its own ids; the finished canvas is the same picture.
    const strip = (html: string): string => html.replace(/ id="[^"]*"|url\(#[^)]*\)/g, '');
    expect(strip(drawing(early))).toBe(strip(drawing(late)));
  });
});
