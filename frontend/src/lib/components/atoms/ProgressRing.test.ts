import { describe, expect, it } from 'vitest';

import type { RingArc } from '$lib/state/jobRing';
import ProgressRing from './ProgressRing.svelte';
import { mountComponent } from '../../../test-utils/mount';

const arcs = (fills: number[]): RingArc[] => fills.map((fill) => ({ fill, ink: 'work' }));
const ring = (target: HTMLElement) => target.querySelector<SVGElement>('[data-testid="progress-ring"]')!;
const dash = (arc: Element) => Number.parseFloat((arc as SVGElement).style.strokeDasharray);

describe('ProgressRing', () => {
  it('is a progress bar with the job\'s percentage while it shows arcs', () => {
    const { target } = mountComponent(ProgressRing, { size: 144, kind: 'arc', arcs: arcs([1, 0.5]), percent: 75, valuetext: 'Image 2 of 2' });

    expect(ring(target).getAttribute('role')).toBe('progressbar');
    expect(ring(target).getAttribute('aria-valuemin')).toBe('0');
    expect(ring(target).getAttribute('aria-valuemax')).toBe('100');
    expect(ring(target).getAttribute('aria-valuenow')).toBe('75');
    expect(ring(target).getAttribute('aria-valuetext')).toBe('Image 2 of 2');
  });

  it('says nothing to assistive technology when it is at rest or decorative', () => {
    for (const props of [{ kind: 'rest' }, { kind: 'arc', arcs: arcs([1]), decorative: true }]) {
      const { target } = mountComponent(ProgressRing, { size: 144, ...props });
      expect(ring(target).getAttribute('aria-hidden')).toBe('true');
      expect(ring(target).getAttribute('role')).toBeNull();
    }
  });

  it('does not sweep a busy ring while no work moves: it shows the dotted track alone', () => {
    const { target } = mountComponent(ProgressRing, { size: 144, kind: 'busy', live: false });

    expect(target.querySelector('[data-testid="ring-sweep"]')).toBeNull();
    expect(target.querySelector('[data-testid="ring-spark"]')).toBeNull();
    expect(target.querySelectorAll('[data-testid="ring-track"]')).toHaveLength(1);
  });

  it('is busy, with no value, while the length is unknown', () => {
    const { target } = mountComponent(ProgressRing, { size: 144, kind: 'busy', live: true });

    expect(ring(target).getAttribute('aria-busy')).toBe('true');
    expect(ring(target).hasAttribute('aria-valuenow')).toBe(false);
    expect(target.querySelector('[data-testid="ring-sweep"]')).not.toBeNull();
  });

  it('draws one arc per image, and the fill of each by its share', () => {
    const { target } = mountComponent(ProgressRing, { size: 144, kind: 'arc', arcs: arcs([1, 0.5, 0, 0]) });
    const drawn = Array.from(target.querySelectorAll('[data-testid="ring-arc"]'));

    expect(drawn).toHaveLength(4);
    expect(dash(drawn[0])).toBeGreaterThan(dash(drawn[1]));
    expect(dash(drawn[1])).toBeCloseTo(dash(drawn[0]) / 2, 5);
    expect(dash(drawn[2])).toBe(0);
  });

  it('colours the arcs by their role', () => {
    const { target } = mountComponent(ProgressRing, { size: 144, kind: 'arc', arcs: [{ fill: 1, ink: 'done' }, { fill: 0.4, ink: 'failed' }] });

    expect(Array.from(target.querySelectorAll('[data-testid="ring-arc"]')).map((arc) => arc.getAttribute('data-ink'))).toEqual(['done', 'failed']);
  });

  it('shows one arc over a single track, and a track under each arc of several', () => {
    const one = mountComponent(ProgressRing, { size: 144, kind: 'arc', arcs: arcs([0.3]) });
    expect(one.target.querySelectorAll('[data-testid="ring-track"]')).toHaveLength(1);

    const four = mountComponent(ProgressRing, { size: 144, kind: 'arc', arcs: arcs([0, 0, 0, 0]) });
    expect(four.target.querySelectorAll('[data-testid="ring-track"]')).toHaveLength(4);
  });

  it('draws the ghost only with a step time, and the spark at the working head while it runs', () => {
    const base = { size: 144, kind: 'arc', arcs: arcs([1, 0.5]), live: true, head: 1, ghost: { arc: 1, from: 0.5, share: 0.1 }, stepKey: 'a' };

    const without = mountComponent(ProgressRing, base);
    expect(without.target.querySelector('[data-testid="ring-ghost"]')).toBeNull();
    expect(without.target.querySelector('[data-testid="ring-spark"]')).not.toBeNull();
    expect(without.target.querySelector('[data-testid="ring-spark"]')?.hasAttribute('data-riding')).toBe(false);

    const timed = mountComponent(ProgressRing, { ...base, stepMs: 2000 });
    expect(timed.target.querySelector('[data-testid="ring-ghost"]')).not.toBeNull();
    expect(timed.target.querySelector('[data-testid="ring-spark"]')?.hasAttribute('data-riding')).toBe(true);
  });

  it('shows no spark when work does not run', () => {
    const { target } = mountComponent(ProgressRing, { size: 144, kind: 'arc', arcs: arcs([0.5]), live: false, head: null });
    expect(target.querySelector('[data-testid="ring-spark"]')).toBeNull();
  });

  it('stands still when asked: no ghost, no spark, no sweep', () => {
    const arc = mountComponent(ProgressRing, { size: 14, kind: 'arc', arcs: arcs([0.5]), live: true, head: 0, ghost: { arc: 0, from: 0.5, share: 0.1 }, stepMs: 2000, still: true });
    expect(arc.target.querySelector('[data-testid="ring-ghost"]')).toBeNull();
    expect(arc.target.querySelector('[data-testid="ring-spark"]')).toBeNull();

    const busy = mountComponent(ProgressRing, { size: 14, kind: 'busy', live: true, still: true });
    expect(busy.target.querySelector('[data-testid="ring-sweep"]')).toBeNull();
    expect(busy.target.querySelector('[data-testid="ring-spark"]')).toBeNull();
    // An unknown length is only the track, drawn dotted by the ring itself.
    expect(busy.target.querySelectorAll('[data-testid="ring-track"]')).toHaveLength(1);
    expect(busy.target.querySelector('[data-testid="ring-arc"]')).toBeNull();
  });
});
