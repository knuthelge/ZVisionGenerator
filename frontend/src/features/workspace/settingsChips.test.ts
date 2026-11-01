import { describe, expect, it } from 'vitest';

import type { DraftState } from '$lib/types';
import type { WorkspaceCapabilities } from './capabilities';
import type { ImagesSummary } from './imagesSummary';
import { pendingChips, settingsChips, type ChipFacts } from './settingsChips';

const imageCaps = {
  isImageMode: true, showFrameCount: false, showExtendLength: false, showSteps: true, showGuidance: true, showI2IStrength: false, showSeed: true,
  showRuns: true, showRefImage: false, showSourceVideo: false, showDimensions: true, showAudio: false, showLowMemory: false, showVideoUpscale: false, showImageUpscale: true,
} as WorkspaceCapabilities;

const state = {
  steps: 32, guidance: 4, referenceImageStrength: 0.75, seed: null, runs: 1, audio: true, lowMemory: false, videoUpscaleEnabled: false,
  upscaleEnabled: false, upscaleFactor: 2, missingSourceVideo: false, postprocessTransparent: false,
} as DraftState;

function facts(overrides: Partial<ChipFacts> = {}): ChipFacts {
  return {
    caps: imageCaps, state, changed: () => false, width: 1024, height: 1024, seconds: null, extendSeconds: null, extendChanged: false,
    loraNames: [], modelLabel: 'Z-Image Turbo', modelDownloaded: true, sourceName: null, images: null, memoryWarn: false,
    presetName: null, presetSlug: null, presetChanged: false,
    ...overrides,
  };
}

describe('settingsChips', () => {
  it('offers placeholder chips in the workflow order, with no value, while the context loads', () => {
    const image = pendingChips('txt2img');
    expect(image.map((chip) => chip.key)).toEqual(['preset', 'steps', 'guidance', 'seed', 'runs', 'images', 'size', 'loras', 'upscale', 'model']);
    expect(image.every((chip) => chip.value === '—' && !chip.changed)).toBe(true);
    expect(pendingChips('txt2vid').map((chip) => chip.key)).toEqual(['preset', 'seconds', 'steps', 'guidance', 'seed', 'images', 'size', 'audio', 'loras', 'model']);
    expect(pendingChips('extend')[1]?.key).toBe('extendSeconds');
    expect(pendingChips('extend').map((chip) => chip.key)).not.toContain('images');
  });

  it('carries what the LoRA files are doing on the LoRA chip only', () => {
    const chips = settingsChips(facts({ loraNames: ['style'], loraDownload: { kind: 'downloading', percent: 40 } }));

    expect(chips.find((chip) => chip.key === 'loras')?.download).toEqual({ kind: 'downloading', percent: 40 });
    expect(chips.filter((chip) => chip.download).map((chip) => chip.key)).toEqual(['loras']);
  });

  it('orders an image workflow by how often each setting changes, model last', () => {
    expect(settingsChips(facts()).map((chip) => chip.key)).toEqual(['preset', 'steps', 'guidance', 'seed', 'runs', 'size', 'loras', 'upscale', 'model']);
  });

  it('puts the clip length first and audio after size for video, without the image upscale', () => {
    const caps = { ...imageCaps, isImageMode: false, showFrameCount: true, showGuidance: false, showImageUpscale: false, showAudio: true, showLowMemory: true, showVideoUpscale: true } as WorkspaceCapabilities;
    expect(settingsChips(facts({ caps, seconds: 5 })).map((chip) => chip.key)).toEqual(['preset', 'seconds', 'steps', 'seed', 'runs', 'size', 'audio', 'loras', 'model']);
  });

  describe('the Images chip', () => {
    const summary = (overrides: Partial<ImagesSummary> = {}): ImagesSummary => ({ text: 'pond.png · 2 references', tone: 'active', ...overrides });
    const images = (f: Partial<ChipFacts>) => settingsChips(facts(f)).find((chip) => chip.key === 'images');

    it('is one chip for every image, between Batch and Size, with the section summary as its value', () => {
      const keys = settingsChips(facts({ images: summary() })).map((chip) => chip.key);

      expect(keys).toEqual(['preset', 'steps', 'guidance', 'seed', 'runs', 'images', 'size', 'loras', 'upscale', 'model']);
      expect(images({ images: summary() })).toMatchObject({ label: 'Images', value: 'pond.png · 2 references', on: true });
    });

    it('is left out where the section is not offered', () => {
      expect(images({ images: null })).toBeUndefined();
    });

    it('reads None unlit, and warns when an image the run uses is gone', () => {
      expect(images({ images: summary({ text: 'None', tone: 'muted' }) })).toMatchObject({ value: 'None', on: false, warn: false });
      expect(images({ images: summary({ tone: 'error' }) })).toMatchObject({ on: false, warn: true });
    });

    it('carries the changed flag of Strength while an image-to-image run offers it', () => {
      const caps = { ...imageCaps, showI2IStrength: true } as WorkspaceCapabilities;
      const changed = (...keys: string[]) => keys.includes('referenceImageStrength');

      expect(images({ images: summary(), caps, state: { ...state, workflow: 'img2img' } as DraftState, changed })?.changed).toBe(true);
      expect(images({ images: summary(), caps, state: { ...state, workflow: 'txt2img' } as DraftState, changed })?.changed).toBe(false);
      expect(images({ images: summary(), caps: imageCaps, state: { ...state, workflow: 'img2img' } as DraftState, changed })?.changed).toBe(false);
    });

    it('replaces the old per-image chips', () => {
      const everything = { ...imageCaps, showRefImage: true, takesLastFrame: true, maxReferences: 4, showI2IStrength: true } as WorkspaceCapabilities;
      const keys = settingsChips(facts({ caps: everything, images: summary() })).map((chip) => chip.key as string);

      for (const key of ['strength', 'reference', 'references', 'endReference']) expect(keys).not.toContain(key);
    });
  });

  it("a tab's two workflows share placeholders", () => {
    const keys = (workflow: Parameters<typeof pendingChips>[0]) => pendingChips(workflow).map((chip) => chip.key);

    expect(keys('img2img')).toEqual(keys('txt2img'));
    expect(keys('img2vid')).toEqual(keys('txt2vid'));
  });

  it('adds Source and Extend for an extension', () => {
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

  describe('the Transparent chip', () => {
    const caps = { ...imageCaps, showPostprocessTransparent: true } as WorkspaceCapabilities;
    const transparent = (f: Partial<ChipFacts>) => settingsChips(facts({ caps, ...f })).find((chip) => chip.key === 'transparent');

    it('shows only while the switch is on, between Upscale and Model', () => {
      expect(transparent({})).toBeUndefined();
      const on = { ...state, postprocessTransparent: true } as DraftState;
      expect(transparent({ state: on })).toMatchObject({ label: 'Transparent', value: 'on', on: true });
      expect(settingsChips(facts({ caps, state: on })).map((chip) => chip.key).slice(-3)).toEqual(['upscale', 'transparent', 'model']);
    });

    it('stays, reading off, while its popover is open', () => {
      expect(transparent({ openKey: 'transparent' })).toMatchObject({ value: 'off', on: false });
      expect(transparent({ openKey: 'steps' })).toBeUndefined();
    });

    it('carries the changed flag', () => {
      const on = { ...state, postprocessTransparent: true } as DraftState;
      expect(transparent({ state: on, changed: (...keys) => keys.includes('postprocessTransparent') })?.changed).toBe(true);
    });

    it('is never offered where the workflow has no Transparent control, even when the draft has it on', () => {
      const on = { ...state, postprocessTransparent: true } as DraftState;
      expect(settingsChips(facts({ state: on, openKey: 'transparent' })).map((chip) => chip.key)).not.toContain('transparent');
    });
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
