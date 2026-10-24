import { describe, expect, it } from 'vitest';

import type { DraftState } from '$lib/types';

import { clipMemoryRequest, clipTimeNote, memoryFitNote, type ClipMemoryControls } from './memoryFit';

function draftState(overrides: Partial<DraftState> = {}): DraftState {
  return {
    workflow: 'txt2vid', model: 'ltx-8', dimensionMode: 'ratio', ratio: '16:9', size: 'm', width: 704, height: 448, frameCount: 121, lowMemory: true,
    videoUpscaleEnabled: false, videoUpscaleFactor: 2, ...overrides,
  } as DraftState;
}

const CONTROLS: ClipMemoryControls = { frameRule: { alignment: 8, offset: 1, round: 'nearest' }, lowMemory: true, upscale: true };

describe('memoryFitNote', () => {
  it.each(['clip', 'extension'] as const)('carries the status and the memory figures of a tight and a too-large %s', (context) => {
    const tight = memoryFitNote({ status: 'tight', required_gb: 21, budget_gb: 24 }, context);
    const tooLarge = memoryFitNote({ status: 'too_large', required_gb: 40, budget_gb: 24 }, context);

    expect(tight?.status).toBe('tight');
    expect(tooLarge?.status).toBe('too_large');
    expect(tight?.text).toMatch(/\b21\b.*\b24\b/);
    expect(tooLarge?.text).toMatch(/\b40\b.*\b24\b/);
  });

  it.each(['clip', 'extension'] as const)('names the memory a discrete GPU runs short of in a %s', (context) => {
    const gpu = memoryFitNote({ status: 'too_large', required_gb: 11.2, budget_gb: 10, kind: 'discrete', resource: 'gpu' }, context);
    const system = memoryFitNote({ status: 'tight', required_gb: 26, budget_gb: 30.3, kind: 'discrete', resource: 'system' }, context);

    expect(gpu?.text).toMatch(/\b11\.2\b.*\b10\b.*GPU/);
    expect(system?.text).toMatch(/\b26\b.*\b30\.3\b.*system/);
    expect(system?.status).toBe('tight');
  });

  it.each(['clip', 'extension'] as const)('keeps the LTX advice for LTX and drops it for MiniMax H3 when system memory is too small in a %s', (context) => {
    const ltx = memoryFitNote({ status: 'too_large', required_gb: 38, budget_gb: 30, kind: 'discrete', resource: 'system' }, context);
    const h3 = memoryFitNote({ status: 'too_large', required_gb: 38, budget_gb: 30, kind: 'discrete', resource: 'system', family: 'h3' }, context);

    expect(ltx?.text).toContain('ltx-4 needs less than ltx-8');
    expect(h3?.text).not.toMatch(/ltx|Upscale adds/i);
    expect(h3?.text).toMatch(/\b38\b.*\b30\b.*system/);
  });

  it.each(['clip', 'extension'] as const)('has nothing to say about a %s that fits or is unknown', (context) => {
    expect(memoryFitNote({ status: 'fits', required_gb: 10, budget_gb: 24 }, context)).toBeNull();
    expect(memoryFitNote(null, context)).toBeNull();
  });
});

describe('clipTimeNote', () => {
  it('rounds the estimate to whole minutes and labels it as estimated', () => {
    expect(clipTimeNote({ status: 'fits', required_gb: 9, budget_gb: 10, expected_seconds: 277 })).toBe('About 5 min (estimated).');
    expect(clipTimeNote({ status: 'tight', required_gb: 9, budget_gb: 10, kind: 'discrete', expected_seconds: 1380 })).toBe('About 23 min (estimated).');
  });

  it('never says less than a minute', () => {
    expect(clipTimeNote({ status: 'fits', required_gb: 9, budget_gb: 10, expected_seconds: 20 })).toBe('About 1 min (estimated).');
  });

  it('has nothing to say without an estimate', () => {
    expect(clipTimeNote({ status: 'fits', required_gb: 9, budget_gb: 10 })).toBeNull();
    expect(clipTimeNote(null)).toBeNull();
  });
});

describe('clipMemoryRequest', () => {
  it('tells an Image to Video clip from a Text to Video clip and sends the steps only for a model that offers them', () => {
    expect(clipMemoryRequest(draftState({ workflow: 'img2vid', steps: 4 }), { ...CONTROLS, steps: true })).toMatchObject({ image: true, steps: 4 });
    const text = clipMemoryRequest(draftState({ steps: 4 }), CONTROLS);
    expect(text).not.toHaveProperty('image');
    expect(text).not.toHaveProperty('steps');
  });

  it('asks for what the job form submits: the model, a preset size, the frames, low memory and upscale', () => {
    expect(clipMemoryRequest(draftState(), CONTROLS)).toEqual({ model: 'ltx-8', ratio: '16:9', size: 'm', frames: 121, low_memory: true, upscale: null });
  });

  it('asks for the width and height of a custom size, and not for its ratio and size', () => {
    const request = clipMemoryRequest(draftState({ dimensionMode: 'custom', width: 640, height: 384 }), CONTROLS);

    expect(request).toEqual({ model: 'ltx-8', width: 640, height: 384, frames: 121, low_memory: true, upscale: null });
    expect(request).not.toHaveProperty('ratio');
  });

  it('has nothing to ask for a custom size that is not complete', () => {
    expect(clipMemoryRequest(draftState({ dimensionMode: 'custom', width: 640, height: 0 }), CONTROLS)).toBeNull();
    expect(clipMemoryRequest(draftState({ dimensionMode: 'custom', width: 0, height: 384 }), CONTROLS)).toBeNull();
  });

  it('sends the frame count the job would use', () => {
    expect(clipMemoryRequest(draftState({ frameCount: 100 }), CONTROLS)?.frames).toBe(97);
    expect(clipMemoryRequest(draftState({ frameCount: 100 }), { ...CONTROLS, frameRule: { alignment: 1, offset: 1, round: 'nearest' } })?.frames).toBe(100);
    expect(clipMemoryRequest(draftState({ frameCount: 125 }), { ...CONTROLS, frameRule: { alignment: 17, offset: 5, round: 'up' } })?.frames).toBe(141);
  });

  it('sends the upscale factor only while the toggle is on and offered', () => {
    expect(clipMemoryRequest(draftState({ videoUpscaleEnabled: true }), CONTROLS)?.upscale).toBe(2);
    expect(clipMemoryRequest(draftState({ videoUpscaleEnabled: true }), { ...CONTROLS, upscale: false })?.upscale).toBeNull();
  });

  it('leaves Low memory on when the model does not offer the toggle', () => {
    expect(clipMemoryRequest(draftState({ lowMemory: false }), CONTROLS)?.low_memory).toBe(false);
    expect(clipMemoryRequest(draftState({ lowMemory: false }), { ...CONTROLS, lowMemory: false })?.low_memory).toBe(true);
  });

  it('checks Image to Video like Text to Video', () => {
    expect(clipMemoryRequest(draftState({ workflow: 'img2vid' }), CONTROLS)).not.toBeNull();
  });

  it.each(['extend', 'txt2img', 'img2img'] as const)('does not check %s', (workflow) => {
    expect(clipMemoryRequest(draftState({ workflow }), CONTROLS)).toBeNull();
  });

  it('has nothing to ask without a model or a length', () => {
    expect(clipMemoryRequest(draftState({ model: '' }), CONTROLS)).toBeNull();
    expect(clipMemoryRequest(draftState({ frameCount: 0 }), CONTROLS)).toBeNull();
  });
});
