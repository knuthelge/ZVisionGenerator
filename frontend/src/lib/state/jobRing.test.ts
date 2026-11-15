import { describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import { jobPercent, ringGaze, ringView, stepKey } from './jobRing';
import { headAngles, RING_SPECS } from './ringGeom';

const IMAGE_CONTROLS = ['next', 'pause', 'resume', 'repeat', 'quit'];

function job(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'j', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '', status: 'running', phase: 'generate',
    currentStep: 0, totalSteps: 8, elapsed: 0, remaining: 0, stageName: '', stageIndex: 0, batchLabel: '', batchIndex: 0,
    paused: false, message: '', outputs: [], previewUrl: null, supported_controls: IMAGE_CONTROLS, ...overrides,
  };
}

/** The outputs of *count* saved images. */
const saved = (count: number) => Array.from({ length: count }, (_, k) => ({ id: `saved-${k}` }) as never);

/** Image 3 of 4, step 5 of 8: two images done, the third five eighths in. */
const batch4 = (overrides: Partial<ActiveJobState> = {}) => job({ runs: 4, batchIndex: 2, currentStep: 5, ...overrides });

describe('jobPercent', () => {
  it('floors, so the number reads 100 only when the job is done', () => {
    expect(jobPercent(0.629)).toBe(62);
    expect(jobPercent(0.999)).toBe(99);
    expect(jobPercent(1)).toBe(99);
    expect(jobPercent(0)).toBe(0);
  });
});

describe('ringView: what the ring draws', () => {
  it('rests with only its track while the job waits', () => {
    const view = ringView(job({ status: 'queued', phase: 'queue' }));

    expect(view).toMatchObject({ kind: 'rest', arcs: [], stamp: null, mark: null, pool: null, live: false });
    expect(view.decorative).toBe(true);
  });

  it('sweeps while setup has no fraction, and never shows a stamp', () => {
    for (const phase of ['load', 'enhance'] as const) {
      const view = ringView(job({ phase, stageName: phase === 'enhance' ? 'enhancing_prompts' : '', totalSteps: 0 }));
      expect(view).toMatchObject({ kind: 'busy', stamp: null, live: true, pool: 'work' });
    }
  });

  it('is the download while a model downloads, and sweeps when its size is unknown', () => {
    const download = { role: 'model' as const, model: 'zit', repo: 'org/zit', part: 1, parts: 2, bytes_done: 360, bytes_total: 1000, bytes_per_sec: null, download_eta_secs: null, shared: false };

    expect(ringView(job({ phase: 'load', download }))).toMatchObject({ kind: 'arc', arcs: [{ fill: 0.36, ink: 'work' }], stamp: 36, head: 0 });
    expect(ringView(job({ phase: 'load', download: { ...download, bytes_total: null } }))).toMatchObject({ kind: 'busy', stamp: null, live: true });
  });

  it('draws one arc at the step fraction for a single image, with the job\'s percentage as the stamp', () => {
    const view = ringView(job({ currentStep: 5 }));

    expect(view.arcs).toEqual([{ fill: 0.625, ink: 'work' }]);
    expect(view.stamp).toBe(62);
    expect(view.percent).toBe(62);
    expect(view).toMatchObject({ kind: 'arc', mark: null, pool: 'work', live: true, head: 0, decorative: false });
  });

  it('draws one arc per image of a batch: done ones full, the current one at its step fraction', () => {
    const view = ringView(batch4());

    expect(view.arcs.map((arc) => arc.fill)).toEqual([1, 1, 0.625, 0]);
    expect(view.stamp).toBe(65);
    expect(view.head).toBe(2);
    expect(view.valuetext).toBe('Image 3 of 4, step 5 of 8, 65 percent of the job');
  });

  it.each([2, 4, 24, 100, 500])('draws one arc per image at every count (%d)', (runs) => {
    expect(ringView(job({ runs, batchIndex: 1, currentStep: 3 })).arcs).toHaveLength(runs);
  });

  it('counts prompts times runs as the arcs of a prompts-file batch', () => {
    expect(ringView(job({ runs: 10, promptCount: 10, promptNumber: 3, batchIndex: 3 })).arcs).toHaveLength(100);
  });

  it('draws the two passes of one item as two arcs', () => {
    const view = ringView(job({ stageNames: ['text_to_image', 'upscale'], stageName: 'upscale', stageSteps: { text_to_image: 8, upscale: 6 }, currentStep: 3, totalSteps: 6 }));

    expect(view.arcs.map((arc) => arc.fill)).toEqual([1, 0.5]);
    expect(view.stamp).toBe(75);
  });

  it('draws a song at Aurora\'s estimate and shows no stamp', () => {
    const view = ringView(job({ workflow: 'music', media_type: 'audio', currentStep: 0, totalSteps: 0, audio: { phase: 'Song tokens', step: 900, total: null, part: 1, parts: 1 } }));

    expect(view.kind).toBe('arc');
    expect(view.arcs[0].fill).toBeGreaterThan(0);
    expect(view.stamp).toBeNull();
    expect(view.live).toBe(true);
  });

  it('draws one arc per voice of a design job, filled by the voices made', () => {
    const view = ringView(job({ workflow: 'voice_design', media_type: 'audio', runs: 3, audio: { phase: 'Design', step: 1, total: 3, part: 1, parts: 1 } }));

    expect(view.arcs).toHaveLength(3);
    expect(view.stamp).toBe(33);
  });

  it('draws one arc per rewrite while prompts are enhanced, with the ghost on the current one', () => {
    const rewriting = job({ phase: 'enhance', stageName: 'enhancing_prompts', currentStep: 1, totalSteps: 4 });

    expect(ringView(rewriting).arcs.map((arc) => arc.fill)).toEqual([1, 0, 0, 0]);
    expect(ringView(rewriting).stamp).toBe(25);
    expect(ringView(rewriting).ghost).toBeNull();
    expect(ringView(rewriting, 2000).ghost).toEqual({ arc: 1, from: 0, share: 1 });
  });
});

describe('ringView: the step ghost', () => {
  it('fills the current step once a step has been timed, on the head arc, starting at its solid fill', () => {
    expect(ringView(batch4()).ghost).toBeNull();

    const ghost = ringView(batch4(), 1500).ghost!;
    expect(ghost.arc).toBe(2);
    expect(ghost.from).toBe(0.625);
    expect(ghost.share).toBeCloseTo(1 / 8, 6);
  });

  it('is one step of the passes of a two-pass item over its own arc', () => {
    const view = ringView(job({ stageNames: ['text_to_image', 'upscale'], stageName: 'upscale', stageSteps: { text_to_image: 8, upscale: 6 }, currentStep: 3, totalSteps: 6 }), 1500);

    expect(view.ghost).toEqual({ arc: 1, from: 0.5, share: 1 / 6 });
  });

  it('does not show while paused, reconnecting, between steps, or for a song or a download', () => {
    expect(ringView(batch4({ status: 'paused', paused: true }), 1500).ghost).toBeNull();
    expect(ringView(batch4({ connection: 'reconnecting' }), 1500).ghost).toBeNull();
    expect(ringView(batch4({ currentStep: 8 }), 1500).ghost).toBeNull();
    expect(ringView(job({ workflow: 'music', media_type: 'audio', totalSteps: 0, audio: { phase: 'Rendering', step: 1, total: 4, part: 1, parts: 1 } }), 1500).ghost).toBeNull();
  });
});

describe('ringView: the spark', () => {
  it('shows whenever work runs, also between steps and while it sweeps', () => {
    expect(ringView(batch4()).live).toBe(true);
    expect(ringView(batch4({ currentStep: 8 })).live).toBe(true);
    expect(ringView(job({ phase: 'load' })).live).toBe(true);
  });

  it('stops while paused, reconnecting, in review and once the job ended', () => {
    expect(ringView(batch4({ status: 'paused', paused: true })).live).toBe(false);
    expect(ringView(batch4({ connection: 'reconnecting' })).live).toBe(false);
    expect(ringView(batch4({ status: 'paused', paused: true, review: { iteration: 1 } })).live).toBe(false);
    expect(ringView(batch4({ status: 'failed' })).live).toBe(false);
  });
});

describe('ringView: the status mark at 6 o\'clock', () => {
  it('replaces the stamp when work stops', () => {
    expect(ringView(batch4({ status: 'paused', paused: true }))).toMatchObject({ mark: 'paused', stamp: null });
    expect(ringView(batch4({ status: 'paused', paused: true, review: { iteration: 1 } }))).toMatchObject({ mark: 'review', stamp: null });
    expect(ringView(batch4({ connection: 'reconnecting' }))).toMatchObject({ mark: 'stale', stamp: null, faded: true });
  });

  it('freezes the arcs in amber while paused', () => {
    expect(ringView(batch4({ status: 'paused', paused: true })).arcs.filter((arc) => arc.fill > 0).map((arc) => arc.ink)).toEqual(['paused', 'paused', 'paused']);
  });

  it('closes a finished job\'s ring in teal with the completed mark', () => {
    const view = ringView(batch4({ status: 'completed', itemFinished: true }));

    expect(view.arcs.every((arc) => arc.fill === 1 && arc.ink === 'done')).toBe(true);
    expect(view).toMatchObject({ mark: 'completed', stamp: null, percent: 100, decorative: true });
  });

  it('keeps saved arcs teal, freezes the failed arc in red where it stopped, and leaves the rest as track', () => {
    const view = ringView(batch4({ status: 'failed', currentStep: 3, outputs: saved(2) }));

    expect(view.arcs).toEqual([
      { fill: 1, ink: 'done' }, { fill: 1, ink: 'done' }, { fill: 3 / 8, ink: 'failed' }, { fill: 0, ink: 'done' },
    ]);
    expect(view).toMatchObject({ mark: 'failed', stamp: null, pool: 'dim' });
  });

  it('always draws a sliver of the failed arc, so a failure at step 0 is seen', () => {
    expect(ringView(batch4({ status: 'failed', currentStep: 0 })).arcs[2].fill).toBe(0.04);
  });

  it('shows only the mark for a job that failed before generating', () => {
    expect(ringView(job({ status: 'failed', phase: 'load' }))).toMatchObject({ kind: 'rest', arcs: [], mark: 'failed', pool: 'dim' });
  });

  it('keeps saved arcs teal and greys the unfinished part of a stopped job, and of a lost one', () => {
    for (const [status, mark] of [['cancelled', 'stopped'], ['unknown', 'unknown']] as const) {
      const view = ringView(batch4({ status, currentStep: 4, outputs: saved(2) }));
      expect(view.arcs).toEqual([{ fill: 1, ink: 'done' }, { fill: 1, ink: 'done' }, { fill: 0.5, ink: 'stopped' }, { fill: 0, ink: 'done' }]);
      expect(view.mark).toBe(mark);
    }
  });

  it('counts only what was saved: an image stopped or skipped inside is grey at the step it reached, never teal', () => {
    const view = ringView(batch4({ status: 'cancelled', currentStep: 3, itemFinished: true, itemSkipped: true, outputs: saved(2) }));

    expect(view.arcs).toEqual([{ fill: 1, ink: 'done' }, { fill: 1, ink: 'done' }, { fill: 3 / 8, ink: 'stopped' }, { fill: 0, ink: 'done' }]);
  });

  it('does not paint an arc teal for an earlier image that was skipped and saved nothing', () => {
    // Three images reached, one skipped before the stop: only two outputs exist.
    const view = ringView(batch4({ status: 'cancelled', currentStep: 3, outputs: saved(1) }));

    expect(view.arcs.map((arc) => arc.fill)).toEqual([1, 0, 3 / 8, 0]);
  });

  it('marks a job stopped before it began with only the mark', () => {
    expect(ringView(job({ status: 'cancelled', phase: 'enhance' }))).toMatchObject({ kind: 'rest', mark: 'stopped', pool: null });
  });
});

describe('ringView: accessibility', () => {
  it('is a progress value only while work moves; rest and ended rings say nothing', () => {
    expect(ringView(batch4()).decorative).toBe(false);
    expect(ringView(job({ status: 'queued', phase: 'queue' })).decorative).toBe(true);
    expect(ringView(batch4({ status: 'cancelled' })).decorative).toBe(true);
  });
});

describe('stepKey', () => {
  it('changes with every step and every item, so the ghost restarts', () => {
    const keys = [batch4(), batch4({ currentStep: 6 }), batch4({ batchIndex: 3 }), batch4({ promptNumber: 2 })].map(stepKey);

    expect(new Set(keys).size).toBe(4);
  });
});

describe('ringGaze: where Blob looks', () => {
  it('is the angle of the spark: the end of the working arc\'s fill', () => {
    const view = ringView(job({ currentStep: 4, totalSteps: 8 }));
    const { box, stroke } = RING_SPECS.L;

    expect(ringGaze(view, 'L', null)).toBeCloseTo(180, 5);
    expect(ringGaze(view, 'L', null)).toBeCloseTo(headAngles(box, stroke, view.arcs, view.head, null)!.from, 8);
  });

  it('is the front of the ghost while a step fills in, the point the spark rides to', () => {
    const view = ringView(batch4(), 1500);
    const { box, stroke } = RING_SPECS.L;
    const front = headAngles(box, stroke, view.arcs, view.head, view.ghost)!.to;

    expect(ringGaze(view, 'L', view.ghost)).toBeCloseTo(front, 8);
    expect(ringGaze(view, 'L', view.ghost)!).toBeGreaterThan(ringGaze(view, 'L', null)!);
  });

  it('follows the ring\'s size', () => {
    const view = ringView(batch4());

    expect(ringGaze(view, 'S', null)).not.toBeCloseTo(ringGaze(view, 'L', null)!, 3);
  });

  it('looks at the head of a batch, in the arc of the image being made', () => {
    const gaze = ringGaze(ringView(batch4()), 'L', null)!;

    expect(gaze).toBeGreaterThan(180);
    expect(gaze).toBeLessThan(270);
  });

  it.each([
    ['queued', job({ status: 'queued', phase: 'queue' })],
    ['sweeping with no fraction', job({ phase: 'load' })],
    ['paused', batch4({ status: 'paused', paused: true })],
    ['reconnecting', batch4({ connection: 'reconnecting' })],
    ['waiting for review', batch4({ status: 'paused', paused: true, review: { iteration: 1 } })],
    ['ended', batch4({ status: 'failed' })],
    ['done', batch4({ status: 'completed' })],
  ])('is undefined while %s, so Blob behaves as before', (_name, state) => {
    expect(ringGaze(ringView(state, 1500), 'L', null)).toBeUndefined();
  });
});

describe('ringView: reconnecting looks the same in every phase', () => {
  const download = { role: 'model' as const, model: 'zit', repo: 'org/zit', part: 1, parts: 2, bytes_done: 360, bytes_total: 1000, bytes_per_sec: null, download_eta_secs: null, shared: false };
  const reconnecting = { connection: 'reconnecting' as const };

  it.each([
    ['a download with a size', job({ phase: 'load', download, ...reconnecting })],
    ['a download of unknown size', job({ phase: 'load', download: { ...download, bytes_total: null }, ...reconnecting })],
    ['the prompt enhancer loading', job({ phase: 'enhance', stageName: 'enhancing_prompts', totalSteps: 0, ...reconnecting })],
    ['prompt rewrites', job({ phase: 'enhance', stageName: 'enhancing_prompts', currentStep: 1, totalSteps: 4, ...reconnecting })],
    ['the model loading', job({ phase: 'load', ...reconnecting })],
    ['generating', batch4(reconnecting)],
  ])('shows the offline mark, muted, with no glow, stamp, spark or ghost during %s', (_name, state) => {
    expect(ringView(state, 1500)).toMatchObject({ mark: 'stale', faded: true, pool: null, live: false, head: null, ghost: null, stamp: null });
  });

  it('keeps the arcs it had, so the ring shows how far the job came', () => {
    expect(ringView(batch4(reconnecting)).arcs.map((arc) => arc.fill)).toEqual([1, 1, 0.625, 0]);
  });

  it('does the same for a job that waits in the queue, and never sweeps', () => {
    expect(ringView(job({ status: 'queued', phase: 'queue', ...reconnecting }))).toMatchObject({ kind: 'rest', mark: 'stale', live: false });
  });
});

describe('ringView: every phase shows the same for paused and for review', () => {
  it.each([
    ['a download', job({ phase: 'load', download: { role: 'model', model: 'zit', repo: 'org/zit', part: 1, parts: 1, bytes_done: 5, bytes_total: 10, bytes_per_sec: null, download_eta_secs: null, shared: false } })],
    ['prompt rewrites', job({ phase: 'enhance', stageName: 'enhancing_prompts', currentStep: 1, totalSteps: 4 })],
    ['generating', batch4()],
  ])('shows the pause mark and stops the spark and stamp during %s', (_name, state) => {
    expect(ringView({ ...state, status: 'paused', paused: true }, 1500)).toMatchObject({ mark: 'paused', live: false, head: null, ghost: null, stamp: null, pool: null });
  });
});
