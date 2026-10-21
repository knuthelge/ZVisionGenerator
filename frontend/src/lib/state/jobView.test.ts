import { describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import { downloadFromEvent } from './jobPhases';
import {
  activeDownload, documentTitle, downloadFraction, downloadRowKey, downloadView, etaLabel, failureToastText, finishingLabel, formatClock, GENERIC_FAILURE_MESSAGE, headline, isReviewing, isSteppedStage, isVideo, itemActivity, itemFraction, iterations,
  jobBar, jobControls, jobTone, loadBody, outcome, phaseRows, positionLabel, readySummary, reviewDecisions, reviewFactor, reviewHasSound, reviewLine,
  reviewVideoUrl, stageBars, timeLeft
} from './jobView';

const IMAGE_CONTROLS = ['next', 'pause', 'resume', 'repeat', 'quit'];

function job(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'j', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '', status: 'running', phase: 'generate',
    currentStep: 0, totalSteps: 28, elapsed: 0, remaining: 0, stageName: '', stageIndex: 0, batchLabel: '', batchIndex: 0,
    paused: false, message: '', outputs: [], previewUrl: null, supported_controls: IMAGE_CONTROLS, ...overrides,
  };
}

const batch = (overrides: Partial<ActiveJobState> = {}) => job({ runs: 8, promptCount: 3, promptNumber: 2, batchIndex: 2, ...overrides });

describe('jobView', () => {
  it('places the current item in the batch from its run and prompt', () => {
    expect(iterations(batch())).toEqual({ total: 24, index: 7, done: 7 });
    expect(iterations(batch({ itemFinished: true })).done).toBe(8);
    expect(iterations(job({ phase: 'enhance', runs: 4 })).index).toBe(0);
  });

  describe('model name', () => {
    it('shows the model label in the load headline, body and row, and the id when there is none', () => {
      const named = { model: 'klein9b', model_label: 'FLUX.2 Klein 9B' };
      const loadDetail = (state: ActiveJobState) => phaseRows(state).find((row) => row.key === 'load')!.detail;

      expect(headline(job({ phase: 'load', loadPhase: 'downloading', ...named }))).toBe('Downloading FLUX.2 Klein 9B');
      expect(headline(job({ phase: 'load', loadPhase: 'building', ...named }))).toBe('Building FLUX.2 Klein 9B');
      expect(loadBody(job(named))).toContain('FLUX.2 Klein 9B');
      expect(loadDetail(job({ phase: 'load', ...named }))).toBe('FLUX.2 Klein 9B');
      expect(headline(job({ phase: 'load', loadPhase: 'downloading', model: 'klein9b' }))).toBe('Downloading klein9b');
      expect(loadDetail(job({ phase: 'load', model: 'klein9b' }))).toBe('klein9b');
    });
  });

  describe('headline', () => {
    it.each([
      ['a waiting job', job({ status: 'queued', phase: 'queue' }), 'Waiting to start'],
      ['a rewrite', job({ phase: 'enhance', stageName: 'enhancing_prompts', currentStep: 1, totalSteps: 4 }), 'Enhancing prompt 2 of 4'],
      ['a first-use model prep', job({ phase: 'load', loadQuant: 8 }), 'Preparing the model'],
      ['a batch item', batch(), 'Generating image 8 of 24'],
      ['the upscale pass', batch({ stageNames: ['text_to_image', 'upscale'], stageName: 'upscale' }), 'Upscaling image 8 of 24'],
      ['a pause between images', batch({ status: 'paused', paused: true, itemFinished: true }), 'Paused after image 8 of 24'],
      ['a pause before the first image', batch({ status: 'paused', paused: true, batchIndex: 0, promptNumber: 1 }), 'Paused before image 1 of 24'],
      ['a failure while loading', job({ status: 'failed', phase: 'load' }), 'Failed while loading the model'],
      ['a failure on an item', batch({ status: 'failed' }), 'Failed on image 8 of 24'],
      ['a stop during enhancement', job({ status: 'cancelled', phase: 'enhance' }), 'Stopped while enhancing prompts'],
      ['a stop mid-batch', batch({ status: 'cancelled', itemFinished: true }), 'Stopped after 8 of 24 images'],
      ['a lost connection', batch({ connection: 'reconnecting' }), 'Reconnecting…'],
      ['a video', job({ workflow: 'txt2vid' }), 'Generating video'],
    ])('describes %s', (_, state, text) => {
      expect(headline(state)).toBe(text);
    });

    it('tells a model download, the copy saved after it and a plain load apart', () => {
      const downloading = headline(job({ phase: 'load', model: 'SeedVR2 7B q8', loadPhase: 'downloading' }));
      const saving = headline(job({ phase: 'load', loadPhase: 'saving_quant', loadQuant: 8, loadReplacesDownload: true }));
      const loading = headline(job({ phase: 'load', loadPhase: 'loading' }));

      expect(new Set([downloading, saving, loading]).size).toBe(3);
      expect(downloading).toContain('SeedVR2 7B q8');
      expect(saving).toBe(headline(job({ phase: 'load', loadQuant: 8 })));
    });
  });

  describe('controls', () => {
    it('splits job and item controls while generating a batch', () => {
      expect(jobControls(batch())).toEqual({ job: ['pause', 'quit'], item: ['next', 'repeat'] });
    });

    it('offers no Skip for a single item', () => {
      expect(jobControls(job()).item).toEqual(['repeat']);
    });

    it('follows what the server honours in each phase', () => {
      expect(jobControls(job({ status: 'queued', phase: 'queue' }))).toEqual({ job: ['quit'], item: [] });
      expect(jobControls(job({ phase: 'enhance', stageName: 'enhancing_prompts', currentStep: 0, totalSteps: 3 }))).toEqual({ job: ['pause', 'quit'], item: ['next'] });
      expect(jobControls(job({ phase: 'load' }))).toEqual({ job: ['pause', 'quit'], item: [] });
      expect(jobControls(batch({ status: 'paused', paused: true }))).toEqual({ job: ['resume', 'quit'], item: [] });
      expect(jobControls(job({ workflow: 'upscale', supported_controls: ['quit'] }))).toEqual({ job: ['quit'], item: [] });
      expect(jobControls(batch({ workflow: 'txt2vid', supported_controls: ['quit'] }))).toEqual({ job: ['quit'], item: [] });
      expect(jobControls(batch({ status: 'completed' }))).toEqual({ job: [], item: [] });
    });
  });

  describe('job bar', () => {
    it('gives each item a segment, marks run boundaries and fills the current one by step', () => {
      const bar = jobBar(job({ runs: 2, promptCount: 2, promptNumber: 1, batchIndex: 1, currentStep: 7, totalSteps: 28 }));
      expect(bar.kind === 'segments' && bar.segments.map((s) => [s.state, s.runBreak, s.fill])).toEqual([
        ['done', false, 1], ['done', false, 1], ['now', true, 0.25], ['pending', false, 0],
      ]);
    });

    it('marks the failed item and leaves no current item once the job ended', () => {
      const bar = jobBar(job({ runs: 3, batchIndex: 1, status: 'failed' }));
      expect(bar.kind === 'segments' && bar.segments.map((s) => s.state)).toEqual(['done', 'failed', 'pending']);
    });

    it('becomes one bar above 48 items', () => {
      expect(jobBar(job({ runs: 200, batchIndex: 50, currentStep: 14 }))).toEqual({ kind: 'bar', fraction: 50.5 / 200 });
    });

    it('fills a single item by its progress', () => {
      expect(jobBar(job({ currentStep: 7 }))).toEqual({ kind: 'bar', fraction: 0.25 });
    });
  });

  describe('stages', () => {
    const upscaling = { stageNames: ['resolve_prompt', 'text_to_image', 'upscale', 'save_image'], stageSteps: { text_to_image: 28, upscale: 12 } };

    it('shows a bar per stepped stage only when an item has more than one', () => {
      expect(stageBars(job({ stageNames: ['resolve_prompt', 'text_to_image', 'save_image'] }))).toEqual([]);
      expect(stageBars(job({ ...upscaling, stageName: 'text_to_image', currentStep: 14, totalSteps: 28 })).map((bar) => bar.state)).toEqual(['current', 'next']);
      expect(stageBars(job({ ...upscaling, stageName: 'upscale', currentStep: 6, totalSteps: 12 })).map((bar) => [bar.label, bar.state, bar.total])).toEqual([
        ['Generate', 'done', 28], ['Upscale', 'current', 12],
      ]);
    });

    it('counts each stage of the item equally', () => {
      expect(itemFraction(job({ ...upscaling, stageName: 'text_to_image', currentStep: 14, totalSteps: 28 }))).toBe(0.25);
      expect(itemFraction(job({ ...upscaling, stageName: 'upscale', currentStep: 6, totalSteps: 12 }))).toBe(0.75);
    });
  });

  it('lists the run log with the ended phase marked', () => {
    const rows = phaseRows(job({ status: 'failed', phase: 'load', enhanceTotal: 0, phaseMarks: { queue: { start: 0, end: 2 }, load: { start: 2, end: 9 } } }));
    expect(rows.map((row) => [row.key, row.status, row.secs ?? null, row.detail])).toEqual([
      ['queue', 'done', 2, ''], ['load', 'failed', 7, 'zit'], ['generate', 'pending', null, 'Not started'],
    ]);
  });

  it('tells a model build apart from a download and a plain load', () => {
    const loadRow = (overrides: Partial<ActiveJobState>) => phaseRows(job({ phase: 'load', model: 'ideo', ...overrides })).find((row) => row.key === 'load')!;
    const building = job({ phase: 'load', model: 'ideo', loadPhase: 'building' });

    expect(headline(building)).toContain('ideo');
    expect(headline(building)).not.toBe(headline(job({ phase: 'load', model: 'ideo', loadPhase: 'downloading' })));
    expect(loadRow({ loadPhase: 'building' }).label).not.toBe(loadRow({ loadPhase: 'loading' }).label);
    expect(loadBody(building)).not.toBe(loadBody(job({ phase: 'load', model: 'ideo', loadPhase: 'downloading' })));
  });

  it('labels the load row for a download, a copy being saved and a plain load', () => {
    const loadRow = (overrides: Partial<ActiveJobState>) => phaseRows(job({ phase: 'load', model: 'SeedVR2 7B q8', ...overrides })).find((row) => row.key === 'load')!;

    const downloading = loadRow({ loadPhase: 'downloading' });
    const saving = loadRow({ loadPhase: 'saving_quant', loadQuant: 8 });
    const loading = loadRow({ loadPhase: 'loading' });

    expect(new Set([downloading.label, saving.label, loading.label]).size).toBe(3);
    expect(downloading.detail).toBe('SeedVR2 7B q8');
    expect(loading.detail).toBe('SeedVR2 7B q8');
    expect(saving.detail).toContain('q8');
  });

  it('explains the load row differently for a download, a replaced download, a saved copy and a plain load', () => {
    const downloading = loadBody(job({ loadPhase: 'downloading' }));
    const replacing = loadBody(job({ loadPhase: 'saving_quant', loadQuant: 8, loadReplacesDownload: true }));
    const quantizing = loadBody(job({ loadPhase: 'saving_quant', loadQuant: 8 }));
    const loading = loadBody(job({ model: 'zit' }));

    expect(new Set([downloading, replacing, quantizing, loading]).size).toBe(4);
    expect(loading).toContain('zit');
  });

  it('lists prompt enhancement only for a job that rewrites prompts', () => {
    expect(phaseRows(job({ phase: 'load', enhanceTotal: 0 })).map((row) => row.key)).toEqual(['queue', 'load', 'generate']);
    expect(phaseRows(job({ phase: 'load', enhanceTotal: 2 })).map((row) => [row.key, row.status])).toEqual([
      ['queue', 'done'], ['enhance', 'done'], ['load', 'current'], ['generate', 'pending'],
    ]);
  });

  describe('item activity', () => {
    const upscaled = { stageNames: ['resolve_prompt', 'text_to_image', 'upscale', 'save_image'] };

    it.each([
      ['before the first step', job({ totalSteps: 0 }), 'starting'],
      ['while steps run', job({ currentStep: 3 }), 'stepping'],
      ['after the last step, while the file is written', job({ currentStep: 28 }), 'finishing'],
      ['in a stage after the steps (saving)', job({ ...upscaled, stageName: 'save_image', stageSteps: { text_to_image: 28, upscale: 12 }, currentStep: 12, totalSteps: 12 }), 'finishing'],
      ['after the generate pass, before the upscale pass', job({ ...upscaled, stageName: 'text_to_image', currentStep: 28 }), 'starting'],
      ['between the two passes of an upscaled video', job({ workflow: 'txt2vid', stepPhase: 'video_upscale_stage_1', currentStep: 30, totalSteps: 30 }), 'starting'],
    ] as const)('is %s', (_, state, activity) => {
      expect(itemActivity(state)).toBe(activity);
    });

    it('counts a finishing item as complete in the job bar', () => {
      expect(itemFraction(job({ currentStep: 28 }))).toBe(1);
    });
  });

  it('leaves out prompt enhancement for gallery upscales', () => {
    expect(phaseRows(job({ workflow: 'upscale' })).map((row) => row.key)).toEqual(['queue', 'load', 'generate']);
  });

  it('folds the setup phases into one Ready line, without durations it does not know', () => {
    const marks = { queue: { start: 0, end: 1 }, enhance: { start: 1, end: 23 }, load: { start: 23, end: 37 } };
    expect(readySummary(job({ enhanceTotal: 3, phaseMarks: marks }))).toEqual({ parts: ['queued 0:01', '3 prompts enhanced 0:22', 'model loaded 0:14'], secs: 37, warn: false });
    expect(readySummary(job({ enhanceStatus: 'failed' }))).toEqual({ parts: ['queued', 'model loaded'], secs: undefined, warn: true });
  });

  it('says where the item sits, or what runs next when paused between items', () => {
    expect(positionLabel(batch())).toBe('Prompt 2 of 3 · run 3 of 8');
    expect(positionLabel(batch({ status: 'paused', paused: true, itemFinished: true }))).toBe('Next: prompt 3 of 3 · run 3 of 8');
    expect(positionLabel(job({ runs: 4 }))).toBe('Run 1 of 4');
  });

  it('gives an estimate only for running multi-item jobs', () => {
    expect(etaLabel(batch({ remaining: 1104 }))).toBe('About 18 min left');
    expect(etaLabel(batch())).toBe('Estimate after the first image');
    expect(etaLabel(job({ remaining: 300 }))).toBe('');
    expect(etaLabel(batch({ remaining: 600, status: 'paused', paused: true }))).toBe('');
    expect(timeLeft(30)).toBe('Under a minute left');
    expect(timeLeft(7210)).toBe('About 2 h 0 min left');
  });

  it('counts what an ended job saved and what never ran', () => {
    const asset = { id: 'a' } as ActiveJobState['outputs'][number];
    expect(outcome(batch({ status: 'failed', message: 'CUDA out of memory.', outputs: Array(7).fill(asset) }))).toEqual({
      text: 'CUDA out of memory.', error: true, saved: '7 of 24 saved to History · 16 not run',
    });
    expect(outcome(job({ status: 'cancelled', message: 'Job stopped.' }))).toEqual({ text: 'You stopped the job.', error: false, saved: 'Nothing saved' });
    expect(outcome(batch())).toBeNull();
  });

  it('tones the job by status, connection and phase', () => {
    expect(jobTone(job({ phase: 'load' }))).toBe('preparing');
    expect(jobTone(batch({ connection: 'reconnecting' }))).toBe('stale');
    expect(jobTone(batch({ status: 'cancelled' }))).toBe('stopped');
  });

  it('formats clocks from seconds', () => {
    expect(formatClock(75)).toBe('1:15');
    expect(formatClock(3725)).toBe('1:02:05');
    expect(formatClock(Number.NaN)).toBe('0:00');
  });

  describe('extend', () => {
    const extending = (overrides: Partial<ActiveJobState> = {}) => job({ workflow: 'extend', stageNames: ['extend_video', 'join_clips'], stageName: 'extend_video', ...overrides });

    it('treats an extension as a video job', () => {
      expect(isVideo(extending())).toBe(true);
      expect(headline(extending())).toBe('Generating video');
    });

    it('steps through the extension stage and not through the join', () => {
      expect(isSteppedStage('extend_video')).toBe(true);
      expect(isSteppedStage('join_clips')).toBe(false);
      expect(itemActivity(extending({ currentStep: 3, totalSteps: 8, stageSteps: { extend_video: 8 } }))).toBe('stepping');
    });

    it('is finishing while the clips are joined, after the last step', () => {
      const joining = extending({ stageName: 'join_clips', currentStep: 8, totalSteps: 8, stageSteps: { extend_video: 8 } });
      expect(itemActivity(joining)).toBe('finishing');
      expect(finishingLabel(joining)).toBe('Joining clips…');
    });

    it('labels the finishing stretch of other stages as finishing up', () => {
      expect(finishingLabel(job({ stageName: 'save' }))).toBe('Finishing up…');
      expect(finishingLabel(extending())).toBe('Finishing up…');
    });

    it('labels the extension bar Generate when the item has more than one stepped stage', () => {
      const bars = stageBars(extending({ stageNames: ['extend_video', 'upscale'], currentStep: 2, totalSteps: 8 }));
      expect(bars.map((bar) => bar.label)).toEqual(['Generate', 'Upscale']);
    });
  });
});

describe('review before upscaling', () => {
  const REVIEW_CONTROLS = [...IMAGE_CONTROLS.slice(0, -1), 'approve', 'discard', 'quit'];
  const reviewing = (overrides: Partial<ActiveJobState> = {}) => batch({
    status: 'paused', paused: true, review: { iteration: 3 }, supported_controls: REVIEW_CONTROLS, stageName: 'review',
    stageNames: ['text_to_image', 'review', 'upscale'], stageSteps: { text_to_image: 8 }, currentStep: 8, totalSteps: 8,
    settings: { upscale: '2' }, ...overrides,
  });

  it('offers only Stop with the job while a review waits', () => {
    expect(jobControls(reviewing())).toEqual({ job: ['quit'], item: [] });
    expect(jobControls(reviewing({ supported_controls: ['approve', 'discard'] }))).toEqual({ job: [], item: [] });
  });

  it('offers the decisions the job lists, only while a review waits', () => {
    expect(reviewDecisions(reviewing())).toEqual(['approve', 'discard']);
    expect(reviewDecisions(reviewing({ supported_controls: ['quit'] }))).toEqual([]);
    expect(reviewDecisions(reviewing({ review: null }))).toEqual([]);
    expect(isReviewing(reviewing({ status: 'cancelled' }))).toBe(false);
    expect(isReviewing(null)).toBe(false);
  });

  it('puts the review ahead of finishing, which the review stage would otherwise give', () => {
    const waiting = reviewing();
    expect(itemActivity({ ...waiting, review: null, status: 'running', paused: false })).toBe('finishing');
    expect(itemActivity(waiting)).toBe('reviewing');
    expect(itemFraction(waiting)).toBeLessThan(1);
    expect(headline(waiting)).toBe('Waiting for your review');
    expect(jobTone(waiting)).toBe('paused');
  });

  it('describes the render by its position and its sizes before and after the upscale', () => {
    const waiting = reviewing();
    expect(reviewLine(waiting, { width: 832, height: 1216 })).toBe('Review 3 of 24 · 832×1216 → 1664×2432');
    expect(reviewLine(reviewing({ settings: { upscale: '4' } }), { width: 100, height: 50 })).toBe('Review 3 of 24 · 100×50 → 400×200');
  });

  it('leaves out what is not known', () => {
    expect(reviewLine(reviewing(), null)).toBe('Review 3 of 24');
    expect(reviewLine(reviewing({ settings: undefined }), { width: 832, height: 1216 })).toBe('Review 3 of 24 · 832×1216');
    expect(reviewFactor(reviewing({ settings: { upscale: ['2', '2'] } }))).toBe(2);
    expect(reviewFactor(reviewing({ settings: { upscale: 'x' } }))).toBeNull();
  });

  describe('for a video', () => {
    const video = (overrides: Partial<ActiveJobState> = {}) => reviewing({ workflow: 'txt2vid', supported_controls: ['approve', 'discard', 'quit'], settings: { upscale: '2', audio: ['on', 'false'] }, ...overrides });

    it('serves the render from the review route, one URL per video of the batch', () => {
      expect(reviewVideoUrl(video())).toBe('/jobs/j/review?v=3');
      expect(reviewVideoUrl(video({ review: { iteration: 4 } }))).toBe('/jobs/j/review?v=4');
    });

    it('has no video URL for an image, outside a review, or after the job ended', () => {
      expect(reviewVideoUrl(reviewing())).toBeNull();
      expect(reviewVideoUrl(video({ review: null }))).toBeNull();
      expect(reviewVideoUrl(video({ status: 'cancelled' }))).toBeNull();
    });

    it('offers Approve and Discard, and only Stop with the job', () => {
      expect(reviewDecisions(video())).toEqual(['approve', 'discard']);
      expect(jobControls(video())).toEqual({ job: ['quit'], item: [] });
      expect(itemActivity(video())).toBe('reviewing');
      expect(headline(video())).toBe('Waiting for your review');
    });

    it('keeps the sound only when the job was submitted with Audio on', () => {
      expect(reviewHasSound(video())).toBe(true);
      expect(reviewHasSound(video({ settings: { upscale: '2', audio: 'false' } }))).toBe(false);
      expect(reviewHasSound(video({ settings: { upscale: '2' } }))).toBe(false);
      expect(reviewHasSound(video({ settings: undefined }))).toBe(false);
    });

    it('describes the video by its position and its sizes before and after the upscale', () => {
      expect(reviewLine(video(), { width: 704, height: 448 })).toBe('Review 3 of 24 · 704×448 → 1408×896');
    });
  });

  it('marks the tab title only while a review waits', () => {
    expect(documentTitle('ZVision Generator', true)).toBe('⏸ Review · ZVision Generator');
    expect(documentTitle('ZVision Generator', false)).toBe('ZVision Generator');
  });

  describe('model downloads', () => {
    const base = { bytes_done: 250, bytes_total: 1000, bytes_per_sec: null, download_eta_secs: null, repo: 'org/repo', part: 1, parts: 1, shared: false };
    const download = (overrides: Record<string, unknown> = {}) => downloadFromEvent({ role: 'model', model: 'klein9b', ...base, ...overrides })!;
    const downloading = (overrides: Partial<ActiveJobState> = {}, event: Record<string, unknown> = {}) =>
      job({ phase: 'load', model: 'klein9b', download: download(event), ...overrides });

    it('puts the download on the load row for a model and the enhance row for an enhancer', () => {
      expect(downloadRowKey(download())).toBe('load');
      expect(downloadRowKey(download({ role: 'enhancer' }))).toBe('enhance');
    });

    it('shows the download only while the job waits on it', () => {
      expect(activeDownload(downloading())).not.toBeNull();
      expect(activeDownload(downloading({ phase: 'generate' }))).toBeNull();
      expect(activeDownload(downloading({ status: 'cancelled' }))).toBeNull();
      expect(activeDownload(job())).toBeNull();
    });

    it('derives the fraction from the bytes, or none when the total is unknown', () => {
      expect(downloadFraction({ bytes_done: 250, bytes_total: 1000 })).toBe(0.25);
      expect(downloadFraction({ bytes_done: 2000, bytes_total: 1000 })).toBe(1);
      expect(downloadFraction({ bytes_done: 250, bytes_total: null })).toBeNull();
      expect(downloadFraction({ bytes_done: 0, bytes_total: 0 })).toBeNull();
    });

    it('drives the job bar from the download, then back to the steps once generating', () => {
      expect(jobBar(downloading())).toEqual({ kind: 'download', fraction: 0.25, role: 'model' });
      expect(jobBar(downloading({}, { bytes_total: null }))).toEqual({ kind: 'download', fraction: null, role: 'model' });
      expect(jobBar(downloading({ phase: 'enhance' }, { role: 'enhancer' }))).toMatchObject({ kind: 'download', role: 'enhancer' });
      expect(jobBar(downloading({ phase: 'generate', currentStep: 14 })).kind).toBe('bar');
      expect(jobBar(downloading({ status: 'cancelled' })).kind).toBe('bar');
    });

    it('describes the block: percent, parts, accessible values and the help for each kind of download', () => {
      const own = downloadView(download({ parts: 2, part: 2 }));
      expect(own.percent).toBe(25);
      expect(own.segments).toHaveLength(2);
      expect(own.valuetext).not.toBe(downloadView(download()).valuetext);

      const unknown = downloadView(download({ bytes_total: null }));
      expect(unknown.percent).toBeNull();
      expect(unknown.valuetext).not.toBe(own.valuetext);

      const shared = downloadView(download({ shared: true }));
      const enhancer = downloadView(download({ role: 'enhancer' }));
      expect(new Set([own.help, shared.help, enhancer.help]).size).toBe(3);
      expect(enhancer.help).toContain('org/repo');
      expect(own.label).not.toBe(enhancer.label);
    });

    it('names the download in the headline, the row and the way a stop ended', () => {
      expect(headline(downloading())).toContain('klein9b');
      expect(headline(downloading())).not.toBe(headline(job({ phase: 'load' })));
      const enhancing = downloading({ phase: 'enhance', enhanceTotal: 2 }, { role: 'enhancer' });
      expect(headline(enhancing)).not.toBe(headline(job({ phase: 'enhance', enhanceTotal: 2 })));
      expect(headline(enhancing)).not.toContain('klein9b');

      const rows = (state: ActiveJobState) => phaseRows(state);
      expect(rows(downloading()).find((row) => row.key === 'load')!.label).not.toBe(rows(job({ phase: 'load' })).find((row) => row.key === 'load')!.label);

      const stopped = downloading({ status: 'cancelled', downloadStop: 'stopped' });
      const detached = downloading({ status: 'cancelled', downloadStop: 'detached' });
      expect(headline(stopped)).toBe(headline(detached));
      expect(headline(stopped)).not.toBe(headline(job({ status: 'cancelled', phase: 'load' })));
      expect(new Set([outcome(stopped)!.text, outcome(detached)!.text, outcome(job({ status: 'cancelled', message: 'Job stopped.' }))!.text]).size).toBe(3);
    });
  });
});

describe('failureToastText', () => {
  it('includes the reason the job failed with', () => {
    expect(failureToastText('CUDA out of memory', false)).toContain('CUDA out of memory');
    expect(failureToastText('CUDA out of memory', true)).toContain('CUDA out of memory');
  });

  it('never appends the generic job-card fallback to the toast', () => {
    expect(failureToastText(GENERIC_FAILURE_MESSAGE, false)).not.toContain(GENERIC_FAILURE_MESSAGE);
    expect(failureToastText(GENERIC_FAILURE_MESSAGE, false)).toBe('Generation failed');
  });

  it('falls back to the plain texts without a reason', () => {
    for (const reason of ['', '   ', null, undefined]) {
      expect(failureToastText(reason, false)).toBe('Generation failed');
      expect(failureToastText(reason, true)).toBe('Generation failed. Starting the next queued job.');
    }
  });

  it('still says the next queued job starts when a reason is shown', () => {
    expect(failureToastText('Out of memory', true)).toMatch(/next queued job/);
    expect(failureToastText('Out of memory.', true)).not.toContain('..');
  });
});
