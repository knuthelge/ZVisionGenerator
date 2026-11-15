import { describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import { barSubline, caption, cardHeadline, pendingNotice, promptText, shortHeadline, splitModelName, timeParts } from './jobCaption';

function job(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'j', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '', status: 'running', phase: 'generate',
    currentStep: 5, totalSteps: 8, elapsed: 0, remaining: 0, stageName: '', stageIndex: 0, batchLabel: '', batchIndex: 0,
    paused: false, message: '', outputs: [], previewUrl: null, supported_controls: [], ...overrides,
  };
}

describe('caption', () => {
  it.each([
    ['a waiting job', job({ status: 'queued', phase: 'queue' }), { lead: ['The worker picks it up as soon as it’s free.'], line: [], wrap: true }],
    ['the step of an image', job(), { lead: ['Step 5 of 8'], line: [] }],
    ['the step of a batch, with no raw line', job({ runs: 4, batchIndex: 2 }), { lead: ['Step 5 of 8'], line: [] }],
    ['a rewrite, with no model named as the enhancer', job({ phase: 'enhance', stageName: 'enhancing_prompts', currentStep: 1, totalSteps: 4, model_label: 'Z-Image Turbo' }), { lead: ['1 of 4 rewritten'], line: [] }],
    ['the end of an image', job({ currentStep: 8 }), { lead: ['Finishing up…'], line: [] }],
    ['a pause between images', job({ runs: 3, status: 'paused', paused: true, itemFinished: true }), { lead: ['Resumes with the next image.'], line: [], wrap: true }],
    ['a pause between takes', job({ workflow: 'speech', media_type: 'audio', runs: 3, status: 'paused', paused: true, itemFinished: true, totalSteps: 0 }), { lead: ['Resumes with the next take.'], line: [], wrap: true }],
    ['a pause between videos', job({ workflow: 'txt2vid', runs: 3, status: 'paused', paused: true, itemFinished: true }), { lead: ['Resumes with the next video.'], line: [], wrap: true }],
    ['a review', job({ status: 'paused', paused: true, review: { iteration: 1 } }), { lead: ['Approve or discard the small render'], line: [], wrap: true }],
  ])('says what happens in %s', (_name, state, expected) => {
    expect(caption(state)).toEqual(expected);
  });

  it('splits a download into what a person waits for and the raw bytes', () => {
    const download = { role: 'model' as const, model: 'zit', repo: 'org/zit', part: 1, parts: 2, bytes_done: 7.4e9, bytes_total: 20.6e9, bytes_per_sec: 52e6, download_eta_secs: 240, shared: false };
    const lines = caption(job({ phase: 'load', download }))!;

    expect(lines).toEqual({ lead: ['Part 1 of 2', 'about 4 min'], line: ['7.4 / 20.6 GB', '52 MB/s'] });
  });

  it('keeps "less than a minute" in the lead with the part, and leaves out what is not known', () => {
    const base = { role: 'model' as const, model: 'zit', repo: 'org/zit', part: 1, parts: 1, bytes_done: 9.5e9, bytes_total: 10e9, shared: false };

    expect(caption(job({ phase: 'load', download: { ...base, bytes_per_sec: 52e6, download_eta_secs: 10 } }))).toEqual({ lead: ['Downloading', 'less than a minute'], line: ['9.5 / 10.0 GB', '52 MB/s'] });
    expect(caption(job({ phase: 'load', download: { ...base, bytes_total: null, bytes_per_sec: null, download_eta_secs: null } }))).toEqual({ lead: ['Downloading'], line: ['9.5 GB'] });
  });

  it('names both passes of a two-pass item, the current one in the lead', () => {
    const two = job({ stageNames: ['text_to_image', 'upscale'], stageName: 'upscale', stageSteps: { text_to_image: 8, upscale: 6 }, currentStep: 3, totalSteps: 6 });

    expect(caption(two)).toEqual({ lead: ['Upscale · step 3 of 6'], line: ['Generate · 8 steps ✓'] });
  });

  it('says a pass that has not reported its steps is starting, and that the job is starting before any pass is current', () => {
    const passes = { stageNames: ['text_to_image', 'upscale'], stageSteps: { text_to_image: 8 } };

    expect(caption(job({ ...passes, stageName: 'upscale', currentStep: 0, totalSteps: 0 }))).toEqual({ lead: ['Upscale · starting'], line: ['Generate · 8 steps ✓'] });
    expect(caption(job({ ...passes, stageName: 'text_to_image', currentStep: 0, totalSteps: 0 }))).toEqual({ lead: ['Generate · starting'], line: ['Upscale next'] });
    expect(caption(job({ stageNames: passes.stageNames, stageName: '', stageSteps: {}, currentStep: 0, totalSteps: 0 }))).toEqual({ lead: ['Starting…'], line: ['Generate next', 'Upscale next'] });
  });

  it('lists a song\'s phases in the raw line', () => {
    const song = job({ workflow: 'music', media_type: 'audio', totalSteps: 0, audio: { phase: 'Song tokens', step: 900, total: null, part: 1, parts: 1 } });
    const lines = caption(song)!;

    expect(lines.lead[0]).toMatch(/^Writing the song/);
    expect(lines.line).toContain('Planning ✓');
  });

  it('counts the voices made for voice design', () => {
    const design = job({ workflow: 'voice_design', media_type: 'audio', runs: 3, audio: { phase: 'Sampling', step: 1, total: 3, part: 1, parts: 1 } });

    expect(caption(design)).toEqual({ lead: ['Sampling'], line: ['1 of 3 voices made'] });
  });

  describe('once the job ended', () => {
    it('says nothing for Done: the headline and the check say it', () => {
      expect(caption(job({ status: 'completed' }))).toBeNull();
    });

    it('says how far a failed job got', () => {
      expect(caption(job({ status: 'failed', currentStep: 3 }))).toEqual({ lead: ['Stopped at step 3 of 8'], line: [] });
      expect(caption(job({ status: 'failed', phase: 'load' }))).toBeNull();
    });

    it('gives the outcome of a stopped or lost job as a sentence', () => {
      expect(caption(job({ status: 'cancelled' }))).toMatchObject({ lead: ['You stopped the job.'], wrap: true });
      expect(caption(job({ status: 'unknown' }))).toMatchObject({ lead: ['The server restarted and no longer has this job.'], wrap: true });
    });
  });
});

describe('headline text', () => {
  it('drops "Generating" from a count of image or video', () => {
    expect(shortHeadline('Generating image 33 of 100')).toBe('Image 33 of 100');
    expect(shortHeadline('Generating video 2 of 120')).toBe('Video 2 of 120');
    expect(shortHeadline('Upscaling image 1 of 2')).toBe('Upscaling image 1 of 2');
    expect(shortHeadline('Generating image')).toBe('Generating image');
  });

  it('keeps the full headline below 100 items and shortens it from 100', () => {
    expect(cardHeadline(job({ runs: 99, batchIndex: 32 }))).toBe('Generating image 33 of 99');
    expect(cardHeadline(job({ runs: 100, batchIndex: 32 }))).toBe('Image 33 of 100');
  });

  it('keeps the model\'s name as one run of its own', () => {
    expect(splitModelName('Downloading Z-Image Turbo', 'Z-Image Turbo')).toEqual([{ text: 'Downloading ', model: false }, { text: 'Z-Image Turbo', model: true }]);
    expect(splitModelName('Loading the model', 'Z-Image Turbo')).toEqual([{ text: 'Loading the model', model: false }]);
    expect(splitModelName('Loading the model', undefined)).toEqual([{ text: 'Loading the model', model: false }]);
  });
});

describe('timeParts', () => {
  const base = { active: true, offline: false, eta: '', elapsed: '1:14', sinceUpdate: 0 };

  it('puts the ETA first, then the elapsed time', () => {
    expect(timeParts({ ...base, eta: 'About 2 min left' })).toEqual([{ text: 'About 2 min left', kind: 'eta' }, { text: '1:14 elapsed', kind: 'plain' }]);
  });

  it('marks a placeholder ETA as an estimate', () => {
    expect(timeParts({ ...base, eta: 'Estimate after the first take' })[0].kind).toBe('estimate');
  });

  it('shows the elapsed time alone without an ETA', () => {
    expect(timeParts(base)).toEqual([{ text: '1:14 elapsed', kind: 'plain' }]);
  });

  it('shows the total once the job ended and the age of the last update while reconnecting', () => {
    expect(timeParts({ ...base, active: false })).toEqual([{ text: '1:14 total', kind: 'plain' }]);
    expect(timeParts({ ...base, offline: true, eta: 'About 2 min left', sinceUpdate: 14 })).toEqual([{ text: 'Last update 14 s ago', kind: 'plain' }]);
  });
});

describe('barSubline', () => {
  it('leads with the ETA and says what is moving', () => {
    expect(barSubline(job({ runs: 4, batchIndex: 2 }), 'About 2 min left', '1:14')).toEqual({ eta: 'About 2 min left', text: 'Step 5 of 8' });
  });

  it('reads the job\'s total once it ended, and the lost connection while reconnecting', () => {
    expect(barSubline(job({ status: 'completed' }), '', '2:42')).toEqual({ eta: null, text: '2:42 total' });
    expect(barSubline(job({ connection: 'reconnecting' }), 'About 2 min left', '1:14').eta).toBeNull();
  });

  it('gives a sentence caption\'s place to the status line', () => {
    expect(barSubline(job({ phase: 'load' }), '', '0:03').text).toBe('Loading model…');
  });
});

describe('promptText', () => {
  it('prefers the enhanced prompt and names what stands in for an empty one', () => {
    expect(promptText(job({ prompt: 'a fox', enhancedPrompt: 'a red fox at dusk' }))).toBe('a red fox at dusk');
    expect(promptText(job({ prompt: 'a fox' }))).toBe('a fox');
    expect(promptText(job({ prompt: '' }))).toBe('No prompt');
    expect(promptText(job({ prompt: '', settings: { prompt_source: 'inline' } }))).toBe('No prompt');
    expect(promptText(job({ prompt: '', settings: { prompt_source: 'file' } }))).toBe('Prompt file');
  });

  it('leads an upscale with the name of its source file, whatever prompt was recorded', () => {
    const source = [{ role: 'source' as const, name: 'portrait.png', kind: 'image' as const }];

    expect(promptText(job({ workflow: 'upscale', prompt: '', inputs: source }))).toBe('portrait.png');
    expect(promptText(job({ workflow: 'upscale', prompt: 'a fox', inputs: source }))).toBe('portrait.png');
    expect(promptText(job({ workflow: 'txt2img', prompt: 'a fox', inputs: source }))).toBe('a fox');
  });
});

describe('pendingNotice', () => {
  it.each([
    ['pause', {}, 'Pausing after this image', 'warning'],
    ['pause', { phase: 'load' as const }, 'Pauses once the model is loaded', 'warning'],
    ['pause', { phase: 'enhance' as const }, 'Pausing after this rewrite', 'warning'],
    ['pause', { workflow: 'speech' as const, media_type: 'audio' as const }, 'Pausing after this take', 'warning'],
    ['next', {}, 'Skipping after this step', 'info'],
    ['next', { phase: 'enhance' as const }, 'Skipping this rewrite', 'info'],
    ['repeat', { workflow: 'txt2vid' as const }, 'Repeating after this video', 'info'],
    ['quit', {}, 'Stopping at the next step', 'warning'],
  ] as const)('says what a waiting %s does (%j)', (action, overrides, text, tone) => {
    expect(pendingNotice(job(overrides), action)).toEqual({ text, tone });
  });

  it('says nothing for a control that does not wait, or for none', () => {
    expect(pendingNotice(job(), 'resume')).toBeNull();
    expect(pendingNotice(job(), null)).toBeNull();
  });
});
