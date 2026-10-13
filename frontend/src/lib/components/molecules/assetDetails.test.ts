import { describe, expect, it } from 'vitest';

import { describeUpscale } from './assetDetails';

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
