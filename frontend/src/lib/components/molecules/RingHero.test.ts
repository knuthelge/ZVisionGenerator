import { createRawSnippet } from 'svelte';
import { describe, expect, it } from 'vitest';

import { ringView } from '$lib/state/jobRing';
import type { ActiveJobState } from '$lib/types';
import RingHero from './RingHero.svelte';
import { mountComponent } from '../../../test-utils/mount';

function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'j', workflow: 'txt2img', prompt: 'p', model: 'zit', runs: 1, created_at: '', status: 'running', phase: 'generate', currentStep: 5, totalSteps: 8,
    elapsed: 0, remaining: 0, stageName: '', stageIndex: 0, batchLabel: '', batchIndex: 0, paused: false, message: '', outputs: [], previewUrl: null, supported_controls: [], ...overrides,
  };
}

const hero = (job: ActiveJobState, props: Record<string, unknown> = {}) => mountComponent(RingHero, { size: 'L', view: ringView(job, props.stepMs as number | null ?? null), ...props }).target;

describe('RingHero', () => {
  it('shows the job\'s percentage as the stamp while work moves', () => {
    expect(hero(makeJob()).querySelector('[data-testid="ring-stamp"]')?.textContent).toBe('62%');
  });

  it('replaces the stamp with the status mark when work stops, never both', () => {
    const paused = hero(makeJob({ status: 'paused', paused: true }));
    expect(paused.querySelector('[data-testid="ring-stamp"]')).toBeNull();
    expect(paused.querySelector('[data-testid="ring-mark"]')?.getAttribute('aria-label')).toBe('Paused');
    expect(paused.querySelector('[data-testid="ring-mark"]')?.getAttribute('role')).toBe('img');
  });

  it.each([
    ['completed', 'Completed'],
    ['failed', 'Failed'],
    ['cancelled', 'Stopped'],
    ['unknown', 'Outcome unknown'],
  ] as const)('names the %s outcome', (status, label) => {
    expect(hero(makeJob({ status })).querySelector('[data-testid="ring-mark"]')?.getAttribute('aria-label')).toBe(label);
  });

  it('draws Blob through the snippet at the size of the ring', () => {
    const mascot = createRawSnippet((size: () => number) => ({ render: () => `<i data-testid="blob">${size()}</i>` }));
    expect(hero(makeJob(), { mascot }).querySelector('[data-testid="blob"]')?.textContent).toBe('112');
    expect(hero(makeJob(), { mascot, size: 'M' }).querySelector('[data-testid="blob"]')?.textContent).toBe('100');
    expect(hero(makeJob(), { mascot, size: 'S' }).querySelector('[data-testid="blob"]')?.textContent).toBe('56');
  });

  it('has the launcher\'s light behind the large ring, and none on the phone bar\'s', () => {
    expect(hero(makeJob()).querySelector('[data-pool]')?.getAttribute('data-pool')).toBe('work');
    expect(hero(makeJob(), { size: 'S' }).querySelector('[data-pool]')).toBeNull();
    expect(hero(makeJob({ status: 'cancelled' })).querySelector('[data-pool]')).toBeNull();
    expect(hero(makeJob({ status: 'failed' })).querySelector('[data-pool]')?.getAttribute('data-pool')).toBe('dim');
  });

  it('tells Blob where to look while work runs, and nothing when it does not', () => {
    const seen: (number | undefined)[] = [];
    const mascot = createRawSnippet((_size: () => number, gaze: () => number | undefined) => ({ render: () => { seen.push(gaze()); return '<i></i>'; } }));

    hero(makeJob(), { mascot });
    hero(makeJob({ status: 'paused', paused: true }), { mascot });

    expect(seen[0]).toBeCloseTo(225, -1);
    expect(seen[1]).toBeUndefined();
  });

  describe('where the step ghost shows', () => {
    it('on the ring while a step is long enough to see', () => {
      const target = hero(makeJob({ runs: 4, batchIndex: 2 }), { stepMs: 1500, stepKey: 'k' });
      expect(target.querySelector('[data-testid="ring-ghost"]')).not.toBeNull();
      expect(target.querySelector('[data-testid="stamp-ghost"]')).toBeNull();
    });

    it('on the stamp\'s foot line when a hundred images make a step too short to see', () => {
      const target = hero(makeJob({ runs: 100, batchIndex: 32 }), { stepMs: 1500, stepKey: 'k' });
      expect(target.querySelector('[data-testid="ring-ghost"]')).toBeNull();
      expect(target.querySelector('[data-testid="stamp-ghost"]')).not.toBeNull();
    });

    it('nowhere before a step has been timed', () => {
      const target = hero(makeJob({ runs: 100, batchIndex: 32 }));
      expect(target.querySelector('[data-testid="ring-ghost"]')).toBeNull();
      expect(target.querySelector('[data-testid="stamp-ghost"]')).toBeNull();
    });
  });
});
