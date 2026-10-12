import { describe, expect, it } from 'vitest';

import type { DraftState } from '$lib/types';

import { clipMemoryRequest, memoryFitNote, type ClipMemoryControls } from './memoryFit';

function draftState(overrides: Partial<DraftState> = {}): DraftState {
  return {
    workflow: 'txt2vid', model: 'ltx-8', dimensionMode: 'ratio', ratio: '16:9', size: 'm', width: 704, height: 448, frameCount: 121, lowMemory: true,
    videoUpscaleEnabled: false, videoUpscaleFactor: 2, ...overrides,
  } as DraftState;
}

const CONTROLS: ClipMemoryControls = { frameAlignment: 8, lowMemory: true, upscale: true };

describe('memoryFitNote', () => {
  it.each(['clip', 'extension'] as const)('carries the status and the memory figures of a tight and a too-large %s', (context) => {
    const tight = memoryFitNote({ status: 'tight', required_gb: 21, budget_gb: 24 }, context);
    const tooLarge = memoryFitNote({ status: 'too_large', required_gb: 40, budget_gb: 24 }, context);

    expect(tight?.status).toBe('tight');
    expect(tooLarge?.status).toBe('too_large');
    expect(tight?.text).toMatch(/\b21\b.*\b24\b/);
    expect(tooLarge?.text).toMatch(/\b40\b.*\b24\b/);
  });

  it.each(['clip', 'extension'] as const)('has nothing to say about a %s that fits or is unknown', (context) => {
    expect(memoryFitNote({ status: 'fits', required_gb: 10, budget_gb: 24 }, context)).toBeNull();
    expect(memoryFitNote(null, context)).toBeNull();
  });
});

describe('clipMemoryRequest', () => {
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
    expect(clipMemoryRequest(draftState({ frameCount: 100 }), { ...CONTROLS, frameAlignment: 1 })?.frames).toBe(100);
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
