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

  it('takes the audio tool', () => {
    const { target } = mountComponent(Mascot, { tool: 'audio' });

    expect(mood(target).dataset.tool).toBe('audio');
  });

  it('does not say it is creating an image while it makes audio', () => {
    const image = mood(mountComponent(Mascot, { mood: 'creating', tool: 'brush' }).target).getAttribute('aria-label');
    const audio = mood(mountComponent(Mascot, { mood: 'creating', tool: 'audio' }).target).getAttribute('aria-label');

    expect(audio).not.toBe(image);
    expect(audio?.startsWith('Z-Vision mascot')).toBe(true);
    expect(mood(mountComponent(Mascot, { mood: 'sad', tool: 'audio' }).target).getAttribute('aria-label')).toBe(
      mood(mountComponent(Mascot, { mood: 'sad', tool: 'brush' }).target).getAttribute('aria-label'),
    );
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

describe('Mascot gaze', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    delete (window as { matchMedia?: unknown }).matchMedia;
  });

  const look = (svg: SVGSVGElement): [number, number] => [parseFloat(svg.style.getPropertyValue('--look-x')), parseFloat(svg.style.getPropertyValue('--look-y'))];

  it('aims the irises along the angle, clockwise from straight up', () => {
    const cases: [number, number, number][] = [
      [0, 0, -2],
      [90, 5, 0],
      [180, 0, 4],
      [270, -5, 0]
    ];
    for (const [gaze, x, y] of cases) {
      const svg = mood(mountComponent(Mascot, { mood: 'creating', size: 120, gaze }).target);
      expect(svg.hasAttribute('data-gaze'), `${gaze}`).toBe(true);
      const [lookX, lookY] = look(svg);
      expect(lookX, `${gaze} x`).toBeCloseTo(x, 1);
      expect(lookY, `${gaze} y`).toBeCloseTo(y, 1);
    }
  });

  it('looks half as far up as it looks down', () => {
    const up = look(mood(mountComponent(Mascot, { mood: 'creating', size: 120, gaze: 0 }).target))[1];
    const down = look(mood(mountComponent(Mascot, { mood: 'creating', size: 120, gaze: 180 }).target))[1];
    expect(up).toBeCloseTo(-down / 2, 1);
  });

  it('follows a changing angle and lets go when the gaze is removed', () => {
    const props = reactiveProps<{ mood: MascotMood; size: number; gaze: number | undefined }>({ mood: 'creating', size: 120, gaze: 90 });
    const { target } = mountComponent(Mascot, props);
    const svg = mood(target);
    expect(look(svg)[0]).toBeCloseTo(5, 1);

    props.gaze = 270;
    flushSync();
    expect(look(svg)[0]).toBeCloseTo(-5, 1);

    props.gaze = undefined;
    flushSync();
    expect(svg.hasAttribute('data-gaze')).toBe(false);
    expect(svg.style.getPropertyValue('--look-x')).toBe('');
  });

  it('leaves the irises to the mood and the pointer when no gaze is given', () => {
    const svg = mood(mountComponent(Mascot, { mood: 'creating', size: 120 }).target);
    expect(svg.hasAttribute('data-gaze')).toBe(false);
    expect(svg.style.getPropertyValue('--look-x')).toBe('');
  });

  it('does not follow the pointer while it gazes', () => {
    const spy = vi.spyOn(window, 'addEventListener');
    mountComponent(Mascot, { mood: 'idle', size: 120, gaze: 90 });
    expect(spy.mock.calls.some(([type]) => type === 'pointermove')).toBe(false);
  });

  it('does nothing in the simple form, which has no irises', () => {
    const svg = mood(mountComponent(Mascot, { mood: 'creating', size: 24, gaze: 90 }).target);
    expect(svg.hasAttribute('data-gaze')).toBe(false);
  });

  it('does nothing when the user asked for reduced motion', () => {
    window.matchMedia = vi.fn().mockReturnValue({ matches: true }) as unknown as typeof window.matchMedia;
    const svg = mood(mountComponent(Mascot, { mood: 'creating', size: 120, gaze: 90 }).target);
    expect(svg.hasAttribute('data-gaze')).toBe(false);
    expect(svg.style.getPropertyValue('--look-x')).toBe('');
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
