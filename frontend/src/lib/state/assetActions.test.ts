import { describe, expect, it } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import type { AssetUpscaleFactor, DraftState, GalleryAsset } from '$lib/types';

import { makeContext, makeIdeogramDefaults, makeImageDefaults, makeVideoDefaults } from '../../test-utils/workspace';
import { assetAspect, canExtend, canUseAsReference, describeFallbackReason, extendParams, extendReason, ltxOptions, offeredUpscaleFactors, plainUpscaleMethod, referenceParams, reuseParams, seedvr2Options, upscaleFactors, upscaleOption, useAsState, workflowLabel } from './assetActions';

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
    expect(reuseParams(makeAsset())).toEqual({ workflow: 'txt2img', prompt: 'a fox', seed: '42', reference_path: '' });
    expect(reuseParams(makeAsset({ reuse_workspace_url: '' }))).toEqual({});
  });

  it('reads a repeated parameter as a list in order', () => {
    const url = '#/workspace?workflow=img2img&reference_path=%2Fin%2Fa.png&reference_path=%2Fin%2Fb.png';

    expect(reuseParams(makeAsset({ reuse_workspace_url: url }))).toEqual({ workflow: 'img2img', reference_path: ['/in/a.png', '/in/b.png'] });
  });

  it('offers images with a host path as references, never videos', () => {
    expect(canUseAsReference(makeAsset())).toBe(true);
    expect(canUseAsReference(makeAsset({ file_path: undefined }))).toBe(false);
    expect(canUseAsReference(makeAsset({ media_type: 'video' }))).toBe(false);
  });

  it('maps the reference target to its workflow and the asset path, with the prompt as an image prompt or a video scene', () => {
    expect(referenceParams(makeAsset(), 'image')).toEqual({ workflow: 'img2img', image_path: '/outputs/out/a.png', use_as: 'start', prompt: 'a fox' });
    expect(referenceParams(makeAsset(), 'video')).toEqual({ workflow: 'img2vid', image_path: '/outputs/out/a.png', use_as: 'start', scene: 'a fox' });
  });

  it('leaves the prompt out when the asset has none', () => {
    expect(referenceParams(makeAsset({ prompt: '' }), 'image')).toEqual({ workflow: 'img2img', image_path: '/outputs/out/a.png', use_as: 'start' });
    expect(referenceParams(makeAsset({ prompt: '' }), 'video')).toEqual({ workflow: 'img2vid', image_path: '/outputs/out/a.png', use_as: 'start' });
  });

  it('makes an asset the last frame with its path alone: no text, no first frame', () => {
    expect(referenceParams(makeAsset(), 'last')).toEqual({ workflow: 'img2vid', end_image_path: '/outputs/out/a.png', use_as: 'last' });
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

    it('reads the SeedVR2 options of a video but never its creative factors', () => {
      const video = makeAsset({ media_type: 'video', upscale: { factors: [two], seedvr2: { options: [plainTwo], model: 'SeedVR2 7B fp8', downloaded: true, download_size_label: null } } });

      expect(seedvr2Options(video)).toEqual([plainTwo]);
      expect(upscaleFactors(video)).toEqual([]);
      expect(upscaleOption(video, 2, 'seedvr2')).toBe(plainTwo);
      expect(plainUpscaleMethod(video)).toBe('seedvr2');
      expect(seedvr2Options(makeAsset({ media_type: 'video', upscale: null }))).toEqual([]);
    });

    it('offers a video on macOS its one LTX option, which the plain shortcut runs with the ltx method', () => {
      const video = makeAsset({ media_type: 'video', upscale: { factors: [], seedvr2: null, ltx: { options: [plainTwo], model: 'LTX-2.3 4-bit', downloaded: true, download_size_label: null } } });

      expect(ltxOptions(video)).toEqual([plainTwo]);
      expect(offeredUpscaleFactors(video)).toEqual([2]);
      expect(plainUpscaleMethod(video)).toBe('ltx');
      expect(upscaleOption(video, 2, 'ltx')).toBe(plainTwo);
      expect(upscaleOption(video, 2, 'seedvr2')).toBeUndefined();
      expect(ltxOptions(makeAsset({ media_type: 'video', upscale: null }))).toEqual([]);
    });

    it('lists the factors an asset offers by any method, smallest first', () => {
      const four: AssetUpscaleFactor = { ...two, factor: 4 };
      const both = makeAsset({ upscale: { factors: [two, four], seedvr2: { options: [{ ...plainTwo }, { ...four, warning: null }], model: 'm', downloaded: true, download_size_label: null } } });
      const video = makeAsset({ media_type: 'video', upscale: { factors: [], seedvr2: { options: [plainTwo], model: 'm', downloaded: true, download_size_label: null } } });

      expect(offeredUpscaleFactors(both)).toEqual([2, 4]);
      expect(offeredUpscaleFactors(video)).toEqual([2]);
      expect(offeredUpscaleFactors(makeAsset())).toEqual([]);
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

describe('useAsState', () => {
  const state = (overrides: Partial<DraftState> = {}): DraftState => ({ ...draft.state, lastImageModel: '', lastVideoModel: '', referenceImagePaths: [], ...overrides });
  const context = () => makeContext({
    image_models: [
      { id: 'klein', label: 'Klein 9B', type: 'image' },
      { id: 'ideogram', label: 'Ideogram 4', type: 'image' },
    ],
    video_models: [{ id: 'ltx-8', label: 'LTX 2.3', type: 'video' }, { id: 'h3', label: 'MiniMax H3', type: 'video' }],
    image_model_defaults: { klein: makeImageDefaults({ max_reference_images: 4 }), ideogram: makeIdeogramDefaults() },
    video_model_defaults: {
      'ltx-8': makeVideoDefaults({ hidden_controls: ['end_image_path'] }),
      h3: makeVideoDefaults(),
    },
    current_image_model: 'klein',
    current_video_model: 'ltx-8',
  });

  it('shows nothing without a context', () => {
    expect(useAsState(null, state())).toEqual({});
  });

  it('reads the model each tab opens on', () => {
    // Image tab open on Ideogram: the start image is unavailable; the video tab opens on the video model last used.
    const onImage = useAsState(context(), state({ workflow: 'txt2img', model: 'ideogram', lastVideoModel: 'h3' }));
    expect(onImage.image).toEqual({ disabled: true, hint: "Ideogram 4 can't use a start image" });
    expect(onImage.reference).toEqual({ disabled: true, hint: 'Ideogram 4 takes none' });
    expect(onImage.video).toEqual({ disabled: false });
    expect(onImage.last).toEqual({ disabled: false });

    // Video tab open on LTX: the image tab opens on the image model last used, Klein.
    const onVideo = useAsState(context(), state({ workflow: 'txt2vid', model: 'ltx-8', lastImageModel: 'klein' }));
    expect(onVideo.image).toEqual({ disabled: false });
    expect(onVideo.reference).toMatchObject({ disabled: false });
    expect(onVideo.last).toEqual({ disabled: true, hint: 'LTX 2.3 has no last frame' });

    // Neither mode used yet: the context's current models.
    const fresh = useAsState(context(), state({ workflow: 'txt2img', model: '' }));
    expect(fresh.reference).toMatchObject({ disabled: false });
    expect(fresh.last).toEqual({ disabled: true, hint: 'LTX 2.3 has no last frame' });
  });

  it('names a video model that cannot start from an image', () => {
    const ctx = context();
    ctx.video_model_defaults['h3'] = makeVideoDefaults({ supports_i2v: false });
    const result = useAsState(ctx, state({ workflow: 'txt2vid', model: 'h3' }));
    expect(result.video).toEqual({ disabled: true, hint: "MiniMax H3 can't start from an image" });
    expect(result.last?.disabled).toBe(true);
  });

  it('offers a reference while there is room, and says why not once the list is full or the model takes none', () => {
    const open = useAsState(context(), state({ workflow: 'img2img', model: 'klein', referenceImagePaths: ['/a.png', '/b.png'] }));
    expect(open.reference).toMatchObject({ disabled: false });

    const full = useAsState(context(), state({ workflow: 'img2img', model: 'klein', referenceImagePaths: ['/a.png', '/b.png', '/c.png', '/d.png'] }));
    expect(full.reference?.disabled).toBe(true);
    expect(full.reference?.hint).toBeTruthy();

    const none = useAsState(context(), state({ workflow: 'txt2img', model: 'ideogram' }));
    expect(none.reference).toEqual({ disabled: true, hint: 'Ideogram 4 takes none' });
  });
});

