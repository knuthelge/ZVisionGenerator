import { describe, expect, it } from 'vitest';

import type { DraftState } from '$lib/types';
import { editFormData, type EditFormInput } from './editForm';

const state = (overrides: Partial<DraftState> = {}): DraftState => ({
  editSourcePath: '/out/a.png', editSourceUpload: null, editMaskUpload: null, editPaintedShare: 0, editExtendSide: null, editExtendAmount: 256, editRepaint: false,
  model: 'klein', quantize: null, steps: 4, seed: null, runs: 1,
  ...overrides,
} as DraftState);

const input = (overrides: Partial<EditFormInput> = {}): EditFormInput => ({ sourceFile: null, maskFile: null, prompt: 'a red door', lora: '', quantize: true, ...overrides });
const fields = (form: FormData): string[] => [...new Set(form.keys())].sort();

describe('the fields an Edit Generate sends', () => {
  it('sends only the contract\'s fields for a whole-picture edit of a picture on the host', () => {
    const form = editFormData(state(), input());
    expect(fields(form)).toEqual(['edit_source_path', 'lora', 'mode', 'model', 'prompt', 'runs', 'steps', 'workflow']);
    expect(Object.fromEntries(form)).toEqual({ mode: 'image', workflow: 'edit', edit_source_path: '/out/a.png', prompt: 'a red door', model: 'klein', lora: '', steps: '4', runs: '1' });
  });

  it('sends the painted mask with its flag when one is held and the draft marks it', () => {
    const mask = new File(['png'], 'mask.png', { type: 'image/png' });
    const form = editFormData(state({ editMaskUpload: 'mask.png', editPaintedShare: 0.2 }), input({ maskFile: mask }));
    expect(form.get('edit_mask_file')).toBe(mask);
    expect(form.get('edit_painted')).toBe('true');
    expect(form.has('edit_extend_side')).toBe(false);
  });

  it('sends no mask a file without the draft\'s marker, as no held upload is an input without its marker', () => {
    const form = editFormData(state(), input({ maskFile: new File(['png'], 'mask.png') }));
    expect(form.has('edit_mask_file')).toBe(false);
    expect(form.has('edit_painted')).toBe(false);
  });

  it('sends the extension as a side and an amount, together', () => {
    const form = editFormData(state({ editExtendSide: 'left', editExtendAmount: 192 }), input());
    expect([form.get('edit_extend_side'), form.get('edit_extend_amount')]).toEqual(['left', '192']);
    expect(form.has('edit_mask_file')).toBe(false);
  });

  it('sends paint and an extension both', () => {
    const form = editFormData(state({ editMaskUpload: 'mask.png', editExtendSide: 'bottom', editExtendAmount: 128 }), input({ maskFile: new File(['png'], 'mask.png') }));
    expect(fields(form)).toEqual(expect.arrayContaining(['edit_mask_file', 'edit_painted', 'edit_extend_side', 'edit_extend_amount']));
  });

  it('sends an uploaded picture as a file, which wins over a path', () => {
    const upload = new File(['x'], 'photo.png', { type: 'image/png' });
    const form = editFormData(state({ editSourcePath: '/out/a.png', editSourceUpload: 'photo.png' }), input({ sourceFile: upload }));
    expect(form.get('edit_source_file')).toBe(upload);
    expect(form.has('edit_source_path')).toBe(false);
  });

  it('sends the seed, the LoRAs and the quantization when they are set, and the quantization only for a model that takes one', () => {
    const set = state({ seed: 7, quantize: 8 });
    const form = editFormData(set, input({ lora: 'detail:0.8' }));
    expect([form.get('seed'), form.get('lora'), form.get('quantize')]).toEqual(['7', 'detail:0.8', '8']);
    expect(editFormData(set, input({ quantize: false })).has('quantize')).toBe(false);
  });

  it('carries nothing of New: no size, guidance, scheduler, post-processing, upscale, start image or references', () => {
    const form = editFormData(state({ referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/r.png'], upscaleEnabled: true, guidance: 3, scheduler: 'beta', ratio: '2:3', size: 'm' }), input());
    for (const key of ['ratio', 'size', 'width', 'height', 'guidance', 'scheduler', 'image_path', 'image_file', 'reference', 'upscale', 'sharpen_enabled', 'transparent', 'image_strength']) {
      expect(form.has(key), key).toBe(false);
    }
  });
});
