import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { makeAsset } from '../../test-utils/workspace';

const api = vi.hoisted(() => ({ saveAdjusted: vi.fn() }));
const toast = vi.hoisted(() => ({ addToast: vi.fn() }));
vi.mock('$lib/api/adjust', () => ({ saveAdjusted: api.saveAdjusted }));
vi.mock('$lib/state/toasts.svelte', () => toast);

import type { AdjustBody } from '$lib/api/adjust';
import { adjustAndReconcile, loadAdjustSection, rememberReplaceAsked, replaceNeedsAsking, saveAdjustSection } from './adjust';
import { draft } from './draft.svelte';
import { galleryStore } from './gallery.svelte';
import { jobStore } from './job.svelte';

const source = makeAsset({ id: 'Zoo/a.png', file_path: '/out/Zoo/a.png', filename: 'a.png' });
const MIX_NEUTRAL = { mix_red: 0, mix_orange: 0, mix_yellow: 0, mix_green: 0, mix_aqua: 0, mix_blue: 0, mix_purple: 0, mix_magenta: 0 };
const body = (save: 'copy' | 'replace'): AdjustBody => ({ asset_id: source.id, save, rotate: 90, flip: false, crop: null, exposure: 0, contrast: 0, highlights: 0, shadows: 0, saturation: 0, vibrance: 0, warmth: 0, ...MIX_NEUTRAL });

beforeEach(() => {
  localStorage.clear();
  api.saveAdjusted.mockReset();
  toast.addToast.mockReset();
  vi.restoreAllMocks();
});

describe('adjustAndReconcile', () => {
  it('adds a saved copy beside its source and touches nothing else', async () => {
    const copy = makeAsset({ id: 'Zoo/a_adjusted.png' });
    api.saveAdjusted.mockResolvedValue(copy);
    const addCopy = vi.spyOn(galleryStore, 'addCopy');
    const replaceAsset = vi.spyOn(galleryStore, 'replaceAsset');
    const replaceOutput = vi.spyOn(jobStore, 'replaceOutput');
    const refresh = vi.spyOn(draft, 'refreshSlotsHolding');

    expect(await adjustAndReconcile(source, body('copy'))).toBe(copy);

    expect(api.saveAdjusted).toHaveBeenCalledWith(body('copy'));
    expect(addCopy).toHaveBeenCalledWith(copy, 'Zoo/a.png');
    expect(replaceAsset).not.toHaveBeenCalled();
    expect(replaceOutput).not.toHaveBeenCalled();
    expect(refresh).not.toHaveBeenCalled();
    expect(toast.addToast).not.toHaveBeenCalled();
  });

  it('swaps a replaced file in the lists and the job, and refreshes what holds its path', async () => {
    const replaced = makeAsset({ id: 'Zoo/a.png', url: '/media/Zoo/a.png?v=2' });
    api.saveAdjusted.mockResolvedValue(replaced);
    const addCopy = vi.spyOn(galleryStore, 'addCopy');
    const replaceAsset = vi.spyOn(galleryStore, 'replaceAsset');
    const replaceOutput = vi.spyOn(jobStore, 'replaceOutput');
    const refresh = vi.spyOn(draft, 'refreshSlotsHolding').mockImplementation(() => {});

    expect(await adjustAndReconcile(source, body('replace'))).toBe(replaced);

    expect(replaceAsset).toHaveBeenCalledWith(replaced);
    expect(replaceOutput).toHaveBeenCalledWith(replaced);
    expect(refresh).toHaveBeenCalledWith('/out/Zoo/a.png');
    expect(addCopy).not.toHaveBeenCalled();
  });

  it('answers null and changes nothing when the request fails, with a toast that has no action', async () => {
    api.saveAdjusted.mockRejectedValueOnce(new ApiError('POST /api/adjust → 500', 500, 'Could not save a.png: disk full'));
    const addCopy = vi.spyOn(galleryStore, 'addCopy');

    expect(await adjustAndReconcile(source, body('copy'))).toBeNull();

    expect(addCopy).not.toHaveBeenCalled();
    expect(toast.addToast).toHaveBeenCalledTimes(1);
    expect(toast.addToast).toHaveBeenCalledWith('Adjust failed: Could not save a.png: disk full', 'error');
  });

  it('offers no action after a network failure either', async () => {
    api.saveAdjusted.mockRejectedValue(new TypeError('Failed to fetch'));
    await adjustAndReconcile(source, body('copy'));
    expect(toast.addToast).toHaveBeenCalledWith('Adjust failed: Failed to fetch', 'error');
  });

  it('offers no action when the server rejected the request', async () => {
    api.saveAdjusted.mockRejectedValue(new ApiError('POST /api/adjust → 422', 422, 'The crop is outside the picture.'));
    expect(await adjustAndReconcile(source, body('copy'))).toBeNull();
    expect(toast.addToast).toHaveBeenCalledWith('Adjust failed: The crop is outside the picture.', 'error');
  });

  it('ignores a second call for a file whose request is still pending, and accepts one after it settles', async () => {
    let finish: (asset: ReturnType<typeof makeAsset>) => void = () => {};
    api.saveAdjusted.mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
    const first = adjustAndReconcile(source, body('copy'));

    expect(await adjustAndReconcile(source, body('copy'))).toBeNull();
    expect(api.saveAdjusted).toHaveBeenCalledTimes(1);
    expect(toast.addToast).not.toHaveBeenCalled();

    finish(makeAsset({ id: 'Zoo/a_adjusted.png' }));
    await first;
    api.saveAdjusted.mockResolvedValueOnce(makeAsset({ id: 'Zoo/a_adjusted_2.png' }));
    expect(await adjustAndReconcile(source, body('copy'))).not.toBeNull();
  });

  it('lets two different files be adjusted at once', async () => {
    api.saveAdjusted.mockReturnValue(new Promise(() => {}));
    void adjustAndReconcile(source, body('copy'));
    void adjustAndReconcile(makeAsset({ id: 'Zoo/b.png' }), { ...body('copy'), asset_id: 'Zoo/b.png' });
    expect(api.saveAdjusted).toHaveBeenCalledTimes(2);
  });
});

describe('the Replace confirmation', () => {
  it('asks until it was answered once, then never again', () => {
    expect(replaceNeedsAsking()).toBe(true);
    rememberReplaceAsked();
    expect(replaceNeedsAsking()).toBe(false);
    expect(localStorage.getItem('ziv-adjust-replace-asked-v1')).toBe('1');
  });
});

describe('the open section', () => {
  beforeEach(() => localStorage.clear());

  it('is Light & colour on a first visit', () => {
    expect(loadAdjustSection()).toBe('light');
  });

  it('gives back each section, and none, as saved under ziv-adjust-section-v1', () => {
    for (const section of ['crop', 'light', 'mixer', 'none'] as const) {
      saveAdjustSection(section);
      expect(localStorage.getItem('ziv-adjust-section-v1')).toBe(section);
      expect(loadAdjustSection()).toBe(section);
    }
  });

  it('falls back to Light & colour for a value that is not one of the four', () => {
    for (const bad of ['', 'colour', 'Crop', '1', 'null']) {
      localStorage.setItem('ziv-adjust-section-v1', bad);
      expect(loadAdjustSection()).toBe('light');
    }
  });
});
