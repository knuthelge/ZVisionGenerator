import { describe, expect, it } from 'vitest';

import type { AssetUpscaleFactor, GalleryAsset } from '$lib/types';

import { assetAspect, canExtend, canUseAsReference, describeFallbackReason, extendParams, extendReason, plainUpscaleMethod, referenceParams, reuseParams, upscaleOption, workflowLabel } from './assetActions';

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/a.png',
    url: '/media/out/a.png',
    thumbnail_url: '/media/out/a.png',
    filename: 'a.png',
    created_at: '2026-10-03T12:00:00Z',
    workflow: 'txt2img',
    prompt: 'a fox',
    model: 'zit',
    media_type: 'image',
    file_path: '/outputs/out/a.png',
    reuse_workspace_url: '#/workspace?workflow=txt2img&prompt=a+fox&seed=42',
    ...overrides,
  };
}

describe('asset actions', () => {
  it('parses reuse params from the reuse URL', () => {
    expect(reuseParams(makeAsset())).toEqual({ workflow: 'txt2img', prompt: 'a fox', seed: '42' });
    expect(reuseParams(makeAsset({ reuse_workspace_url: '' }))).toEqual({});
  });

  it('offers images with a host path as references, never videos', () => {
    expect(canUseAsReference(makeAsset())).toBe(true);
    expect(canUseAsReference(makeAsset({ file_path: undefined }))).toBe(false);
    expect(canUseAsReference(makeAsset({ media_type: 'video' }))).toBe(false);
  });

  it('maps the reference target to its workflow, the asset path and its prompt', () => {
    expect(referenceParams(makeAsset(), 'image')).toEqual({ workflow: 'img2img', image_path: '/outputs/out/a.png', prompt: 'a fox' });
    expect(referenceParams(makeAsset(), 'video')).toEqual({ workflow: 'img2vid', image_path: '/outputs/out/a.png', prompt: 'a fox' });
  });

  it('leaves the prompt out when the asset has none', () => {
    expect(referenceParams(makeAsset({ prompt: '' }), 'image')).toEqual({ workflow: 'img2img', image_path: '/outputs/out/a.png' });
  });

  it('clamps the aspect ratio and defaults to square without dimensions', () => {
    expect(assetAspect(makeAsset({ width: 1024, height: 512 }))).toBe(2);
    expect(assetAspect(makeAsset({ width: 4000, height: 500 }))).toBe(2);
    expect(assetAspect(makeAsset({ width: 512, height: 768 }))).toBeCloseTo(0.667, 3);
    expect(assetAspect(makeAsset())).toBe(1);
  });

  describe('upscale methods', () => {
    const two: AssetUpscaleFactor = { factor: 2, width: 200, height: 200, allowed: true, reason: null };
    const plainTwo: AssetUpscaleFactor = { ...two, warning: null };
    const withSeedvr2 = makeAsset({
      upscale: { factors: [two], seedvr2: { options: [plainTwo], model: 'SeedVR2 7B q8', downloaded: true, download_size_label: null } },
    });
    const creativeOnly = makeAsset({ upscale: { factors: [two], seedvr2: null } });

    it('runs the plain upscale with SeedVR2 where it is offered, otherwise creative', () => {
      expect(plainUpscaleMethod(withSeedvr2)).toBe('seedvr2');
      expect(plainUpscaleMethod(creativeOnly)).toBe('creative');
      expect(plainUpscaleMethod(makeAsset())).toBe('creative');
    });

    it('finds the option of a factor within its method', () => {
      expect(upscaleOption(withSeedvr2, 2, 'seedvr2')).toBe(plainTwo);
      expect(upscaleOption(withSeedvr2, 2, 'creative')).toBe(two);
      expect(upscaleOption(creativeOnly, 2, 'seedvr2')).toBeUndefined();
      expect(upscaleOption(creativeOnly, 4, 'creative')).toBeUndefined();
    });
  });
});

describe('extend actions', () => {
  const video = (overrides: Partial<GalleryAsset> = {}): GalleryAsset => makeAsset({
    id: 'out/clip.mp4',
    filename: 'clip.mp4',
    workflow: 'txt2vid',
    media_type: 'video',
    file_path: '/outputs/out/clip.mp4',
    extend: { available: true, reason: null },
    ...overrides,
  });

  it('offers extending only when the server says the video can be', () => {
    expect(canExtend(video())).toBe(true);
    expect(canExtend(video({ extend: { available: false, reason: "Upscaled videos can't be extended yet." } }))).toBe(false);
    expect(canExtend(video({ extend: null }))).toBe(false);
    expect(canExtend(makeAsset())).toBe(false);
  });

  it('explains why a video cannot be extended', () => {
    const reason = "Upscaled videos can't be extended yet.";
    expect(extendReason(video({ extend: { available: false, reason } }))).toBe(reason);
    expect(extendReason(video())).toBeNull();
    expect(extendReason(video({ extend: null }))).toBeNull();
  });

  it('sends the source, model and LoRAs and no text, which the server fills in once it has read the source', () => {
    const asset = video({ scene: 'A woman on a rainy street.', prompt: 'A woman on a rainy street.', lora: 'style:0.7', reuse_state: { resolved_model: 'ltx-2', requested_model: 'ltx-2' } as GalleryAsset['reuse_state'] });
    expect(extendParams(asset)).toEqual({
      workflow: 'extend',
      source_video: '/outputs/out/clip.mp4',
      model: 'ltx-2',
      lora: 'style:0.7',
    });
  });

  it('never sends the display prompt, which is the file name for a clip with no recorded text', () => {
    expect(extendParams(video({ scene: null, prompt: 'clip mp4' }))).toEqual({ workflow: 'extend', source_video: '/outputs/out/clip.mp4', lora: '' });
  });

  it('labels the workflow and explains the extend fallbacks', () => {
    expect(workflowLabel('extend')).toBe('Extend video');
    expect(describeFallbackReason('missing_source_video')).not.toBe('missing_source_video');
    expect(describeFallbackReason('extend_unavailable')).not.toBe('extend_unavailable');
  });
});

