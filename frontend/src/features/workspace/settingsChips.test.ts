import { describe, expect, it } from 'vitest';

import type { DraftState } from '$lib/types';
import type { WorkspaceCapabilities } from './capabilities';
import { pendingChips, settingsChips, type ChipFacts } from './settingsChips';

const imageCaps = {
  isImageMode: true, showFrameCount: false, showExtendLength: false, showSteps: true, showGuidance: true, showI2IStrength: false, showSeed: true,
  showRuns: true, showRefImage: false, showSourceVideo: false, showDimensions: true, showAudio: false, showLowMemory: false, showVideoUpscale: false, showImageUpscale: true,
} as WorkspaceCapabilities;

const state = {
  steps: 32, guidance: 4, referenceImageStrength: 0.75, seed: null, runs: 1, audio: true, lowMemory: false, videoUpscaleEnabled: false,
  upscaleEnabled: false, upscaleFactor: 2, missingSourceVideo: false,
} as DraftState;

function facts(overrides: Partial<ChipFacts> = {}): ChipFacts {
  return {
    caps: imageCaps, state, changed: () => false, width: 1024, height: 1024, seconds: null, extendSeconds: null, extendChanged: false,
    loraNames: [], modelLabel: 'Z-Image Turbo', modelDownloaded: true, referenceName: null, sourceName: null, memoryWarn: false,
    presetName: null, presetSlug: null, presetChanged: false,
    ...overrides,
  };
}

describe('settingsChips', () => {
  it('offers placeholder chips in the workflow order, with no value, while the context loads', () => {
    const image = pendingChips('txt2img');
    expect(image.map((chip) => chip.key)).toEqual(['preset', 'steps', 'guidance', 'seed', 'runs', 'size', 'loras', 'upscale', 'model']);
    expect(image.every((chip) => chip.value === '—' && !chip.changed)).toBe(true);
    expect(pendingChips('txt2vid').map((chip) => chip.key)).toEqual(['preset', 'seconds', 'steps', 'guidance', 'seed', 'size', 'audio', 'loras', 'model']);
    expect(pendingChips('extend')[1]?.key).toBe('extendSeconds');
  });

  it('orders an image workflow by how often each setting changes, model last', () => {
    expect(settingsChips(facts()).map((chip) => chip.key)).toEqual(['preset', 'steps', 'guidance', 'seed', 'runs', 'size', 'loras', 'upscale', 'model']);
  });

  it('puts the clip length first and audio after size for video, without the image upscale', () => {
    const caps = { ...imageCaps, isImageMode: false, showFrameCount: true, showGuidance: false, showImageUpscale: false, showAudio: true, showLowMemory: true, showVideoUpscale: true } as WorkspaceCapabilities;
    expect(settingsChips(facts({ caps, seconds: 5 })).map((chip) => chip.key)).toEqual(['preset', 'seconds', 'steps', 'seed', 'runs', 'size', 'audio', 'loras', 'model']);
  });

  it('adds Strength and Reference for image to image, and Source and Extend for an extension', () => {
    const i2i = { ...imageCaps, showI2IStrength: true, showRefImage: true } as WorkspaceCapabilities;
    expect(settingsChips(facts({ caps: i2i })).map((chip) => chip.key)).toContain('strength');
    expect(settingsChips(facts({ caps: i2i })).map((chip) => chip.key)).toContain('reference');
    const extend = { ...imageCaps, isImageMode: false, showExtendLength: true, showSourceVideo: true, showGuidance: false, showImageUpscale: false, showDimensions: false } as WorkspaceCapabilities;
    const keys = settingsChips(facts({ caps: extend, extendSeconds: 3, sourceName: 'harbour.mp4' })).map((chip) => chip.key);
    expect(keys.slice(0, 2)).toEqual(['preset', 'extendSeconds']);
    expect(keys).toContain('source');
  });

  it('shows values, the changed flag and the on state', () => {
    const chips = settingsChips(facts({ state: { ...state, seed: 4242, upscaleEnabled: true, upscaleFactor: 4 }, changed: (...keys) => keys.includes('seed'), loraNames: ['detail-tweaker'] }));
    const byKey = Object.fromEntries(chips.map((chip) => [chip.key, chip]));
    expect(byKey.seed).toMatchObject({ value: '4242', changed: true, on: true });
    expect(byKey.upscale).toMatchObject({ value: '4×', on: true });
    expect(byKey.loras).toMatchObject({ value: 'detail-tweaker', on: true });
    expect(byKey.size.value).toBe('1024×1024');
    expect(byKey.runs.value).toBe('×1');
  });

  it('puts the preset first with its name, or none, and the changed flag', () => {
    expect(settingsChips(facts())[0]).toMatchObject({ key: 'preset', applied: '', changed: false, on: false });
    const applied = settingsChips(facts({ presetName: 'Soft portrait', presetSlug: 'soft-portrait', presetChanged: true }))[0];
    expect(applied).toMatchObject({ key: 'preset', applied: 'soft-portrait', changed: true, on: true });
  });

  it('warns on a model that is not downloaded and a clip that may not fit', () => {
    const chips = settingsChips(facts({ modelDownloaded: false, caps: { ...imageCaps, showFrameCount: true } as WorkspaceCapabilities, seconds: 8, memoryWarn: true }));
    expect(chips.find((chip) => chip.key === 'model')?.warn).toBe(true);
    expect(chips.find((chip) => chip.key === 'seconds')?.warn).toBe(true);
  });
});
