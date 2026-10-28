import { describe, expect, it } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import { assetDetailSections, describeUpscale } from './assetDetails';

describe('describeUpscale', () => {
  it('leads with the factor and names the method, model and softness of a SeedVR2 upscale', () => {
    const text = describeUpscale({ factor: 2, method: 'seedvr2', model: 'seedvr2-7b', quantize: 8, softness: 0.2 });
    expect(text).toBe('2× · SeedVR2 · seedvr2-7b q8 · softness 0.2');
  });

  it('names a creative upscale and keeps its settings', () => {
    expect(describeUpscale({ factor: 4, method: 'creative', denoise: 0.4, steps: 3 })).toBe('4× · creative · denoise 0.4 · 3 steps');
  });

  it('describes files recorded without a method by the factor and settings alone', () => {
    expect(describeUpscale({ factor: 2, denoise: 0.4 })).toBe('2× · denoise 0.4');
    expect(describeUpscale(undefined)).toBe('');
  });
});

describe('assetDetailSections', () => {
  const clip = (overrides: Partial<GalleryAsset> = {}): GalleryAsset => ({
    id: 'out/c.mp4', url: '/media/out/c.mp4', thumbnail_url: '/media/out/c.mp4', filename: 'c.mp4', created_at: '2026-10-03T12:00:00Z',
    workflow: 'img2vid', prompt: 'a fox', model: 'h3', media_type: 'video', reuse_workspace_url: '', ...overrides,
  });
  const labels = (asset: GalleryAsset) => assetDetailSections(asset).generation.map((fact) => fact.label);

  it('lists the last frame after the first frame, by file name', () => {
    const asset = clip({ image_path: '/in/start.png', end_image_path: '/in/end.png' });
    const generation = assetDetailSections(asset).generation;
    expect(labels(asset).indexOf('Last frame')).toBe(labels(asset).indexOf('First frame') + 1);
    expect(generation.find((fact) => fact.label === 'Last frame')?.value).toBe('end.png');
  });

  it('lists the references of an image by file name, in order, apart from its start image', () => {
    const asset = clip({ media_type: 'image', workflow: 'img2img', image_path: '/in/start.png', reference_images: ['/in/woman.png', '/in/jacket.png'] });
    const generation = assetDetailSections(asset).generation;

    expect(generation.find((fact) => fact.label === 'Start image')?.value).toBe('start.png');
    expect(generation.find((fact) => fact.label === 'References')?.value).toBe('woman.png, jacket.png');
    expect(labels(clip({ media_type: 'image' }))).not.toContain('References');
  });

  it('leaves the last frame out of a clip without one', () => {
    expect(labels(clip({ image_path: '/in/start.png', end_image_path: null }))).not.toContain('Last frame');
  });
});
