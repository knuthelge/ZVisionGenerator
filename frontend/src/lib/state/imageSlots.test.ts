import { describe, expect, it } from 'vitest';
import { imageSupport, lastSlotPatch, slotSet, startImageSet, startSlotPatch } from './imageSlots';
import { makeContext } from '../../test-utils/workspace';
import type { DraftState, VideoModelDefaults, Workflow } from '$lib/types';

type Slots = Pick<DraftState, 'workflow' | 'referenceImagePath' | 'referenceImageUpload' | 'firstFramePath' | 'firstFrameUpload' | 'endImagePath' | 'endImageUpload' | 'referenceImageStrength' | 'missingReferencePaths'>;

function slots(workflow: Workflow, overrides: Partial<Slots> = {}): Slots {
  return {
    workflow,
    referenceImagePath: null, referenceImageUpload: null, firstFramePath: null, firstFrameUpload: null, endImagePath: null, endImageUpload: null,
    referenceImageStrength: 0.65, missingReferencePaths: [],
    ...overrides,
  };
}

const file = (name: string): File => new File(['x'], name, { type: 'image/png' });

describe('slotSet', () => {
  it('is set by a path or an upload marker, and by nothing else', () => {
    expect(slotSet('/in/a.png', null)).toBe(true);
    expect(slotSet(null, 'a.png')).toBe(true);
    expect(slotSet(null, null)).toBe(false);
    expect(slotSet('', null)).toBe(false);
  });
});

describe('startImageSet', () => {
  it('reads the start image in Image and the first frame in Video, as a path or an upload marker', () => {
    expect(startImageSet(slots('txt2img'), 'image')).toBe(false);
    expect(startImageSet(slots('txt2img', { referenceImagePath: '/in/a.png' }), 'image')).toBe(true);
    expect(startImageSet(slots('txt2img', { referenceImageUpload: 'a.png' }), 'image')).toBe(true);
    expect(startImageSet(slots('txt2img', { firstFramePath: '/in/a.png' }), 'image')).toBe(false);
    expect(startImageSet(slots('txt2vid', { firstFramePath: '/in/a.png' }), 'video')).toBe(true);
    expect(startImageSet(slots('txt2vid', { firstFrameUpload: 'a.png' }), 'video')).toBe(true);
    expect(startImageSet(slots('txt2vid', { referenceImagePath: '/in/a.png' }), 'video')).toBe(false);
  });

  it('does not count an empty path', () => {
    expect(startImageSet(slots('txt2img', { referenceImagePath: '' }), 'image')).toBe(false);
  });
});

describe('slot patches', () => {
  it('startSlotPatch sets the path or the marker, never both', () => {
    expect(startSlotPatch('image', { path: '/in/a.png' })).toEqual({ referenceImagePath: '/in/a.png', referenceImageUpload: null });
    expect(startSlotPatch('image', { file: file('a.png') })).toEqual({ referenceImagePath: null, referenceImageUpload: 'a.png' });
    expect(startSlotPatch('image', null)).toEqual({ referenceImagePath: null, referenceImageUpload: null });
    expect(startSlotPatch('video', { path: '/in/a.png' })).toEqual({ firstFramePath: '/in/a.png', firstFrameUpload: null });
    expect(startSlotPatch('video', { file: file('a.png') })).toEqual({ firstFramePath: null, firstFrameUpload: 'a.png' });
    expect(startSlotPatch('video', null)).toEqual({ firstFramePath: null, firstFrameUpload: null });
  });

  it('lastSlotPatch sets the path or the marker, never both', () => {
    expect(lastSlotPatch({ path: '/in/z.png' })).toEqual({ endImagePath: '/in/z.png', endImageUpload: null });
    expect(lastSlotPatch({ file: file('z.png') })).toEqual({ endImagePath: null, endImageUpload: 'z.png' });
    expect(lastSlotPatch(null)).toEqual({ endImagePath: null, endImageUpload: null });
  });
});

describe('imageSupport', () => {
  const withVideo = (id: string, patch: Partial<VideoModelDefaults>) => {
    const ctx = makeContext();
    ctx.video_model_defaults[id] = { ...ctx.video_model_defaults[id], ...patch } as VideoModelDefaults;
    return ctx;
  };

  it('an image model takes a start image unless it has no image-to-image, and the references it declares', () => {
    const ctx = makeContext();
    ctx.image_model_defaults['flux-lite'] = { ...ctx.image_model_defaults['flux-lite'], supports_img2img: false, max_reference_images: 3 };
    expect(imageSupport(ctx, 'zit', 'image').start).toBe(true);
    expect(imageSupport(ctx, 'flux-lite', 'image')).toEqual({ start: false, last: false, references: 3 });
  });

  it('a video model takes a start image only when it starts from an image (supports_i2v)', () => {
    const ctx = withVideo('ltx-8', { supports_i2v: false });
    expect(imageSupport(ctx, 'ltx-8', 'video')).toEqual({ start: false, last: false, references: 0 });
    expect(imageSupport(makeContext(), 'ltx-8', 'video').start).toBe(true);
  });

  it('a video model takes a last frame unless it hides the last-frame control', () => {
    expect(imageSupport(makeContext(), 'ltx-8', 'video').last).toBe(true);
    const ctx = withVideo('ltx-8', { hidden_controls: ['low_memory', 'end_image_path'] });
    expect(imageSupport(ctx, 'ltx-8', 'video')).toEqual({ start: true, last: false, references: 0 });
  });

  it('a model the context lists no defaults for follows the default video model', () => {
    const ctx = makeContext();
    ctx.video_defaults = { ...ctx.video_defaults, hidden_controls: ['end_image_path'] } as VideoModelDefaults;
    expect(imageSupport(ctx, 'unlisted', 'video')).toEqual({ start: true, last: false, references: 0 });
  });

  it('assumes a start image without a context, and a last frame only in Video', () => {
    expect(imageSupport(null, 'any', 'image')).toEqual({ start: true, last: false, references: 0 });
    expect(imageSupport(null, 'any', 'video')).toEqual({ start: true, last: true, references: 0 });
  });

  it('takes a last frame only with a start image, and a last frame or references only in their own tab', () => {
    expect(imageSupport(withVideo('ltx-8', { supports_i2v: false }), 'ltx-8', 'video')).toMatchObject({ start: false, last: false });
    expect(imageSupport(makeContext(), 'zit', 'image').last).toBe(false);
    expect(imageSupport(makeContext(), 'ltx-8', 'video').references).toBe(0);
  });
});
