import { describe, expect, it } from 'vitest';

import { startImageSet, type ImageTab, type SlotState } from '$lib/state/imageSlots';
import { workflowFor, workspaceTab } from '$lib/state/workspaceTab';
import type { Workflow } from '$lib/types';
import type { WorkspaceCapabilities } from './capabilities';
import { imagesView, type ImagesViewInput } from './imagesView';

const file = (name: string): File => new File(['x'], name, { type: 'image/png' });

// The capabilities of a model that takes every image of its tab: `workspaceCapabilities` gives references and Strength only to
// Image and a last frame only to Video, and the view follows them as given.
const IMAGE_CAPS = { showRefImage: true, takesLastFrame: false, maxReferences: 3, showI2IStrength: true } as WorkspaceCapabilities;
const VIDEO_CAPS = { showRefImage: true, takesLastFrame: true, maxReferences: 0, showI2IStrength: false } as WorkspaceCapabilities;

type Held = ImagesViewInput['files'];
const NO_FILES: Held = { startImage: null, firstFrame: null, lastFrame: null, references: [] };
type Overrides = Partial<SlotState> & { referenceImagePaths?: string[] };

function state(workflow: Workflow, overrides: Overrides = {}): ImagesViewInput['state'] {
  return {
    workflow,
    referenceImagePath: null, referenceImageUpload: null, firstFramePath: null, firstFrameUpload: null, endImagePath: null, endImageUpload: null,
    referenceImageStrength: 0.65, missingReferencePaths: [], referenceImagePaths: [],
    ...overrides,
  };
}

function view(workflow: Workflow, overrides: Overrides = {}, options: { caps?: Partial<WorkspaceCapabilities>; files?: Partial<Held>; restored?: boolean } = {}) {
  const caps = workspaceTab(workflow) === 'video' ? VIDEO_CAPS : IMAGE_CAPS;
  return imagesView({ state: state(workflow, overrides), caps: { ...caps, ...options.caps }, files: { ...NO_FILES, ...options.files }, restored: options.restored ?? true });
}

describe('imagesView in Image', () => {
  it('shows an empty start slot and an empty reference list on a model that takes both', () => {
    const v = view('txt2img');

    expect(v.tab).toBe('image');
    expect(v.visible).toBe(true);
    expect(v.start).toMatchObject({ set: false, gone: false, name: '' });
    expect(v.last).toBeNull();
    expect(v.strength).toBe(false);
    expect(v.references).toMatchObject({ rows: [], max: 3, hasRoom: true });
  });

  it('leaves out the start slot on a model without image-to-image, and the list on one without references', () => {
    expect(view('txt2img', {}, { caps: { showRefImage: false } }).start).toBeNull();
    expect(view('txt2img', {}, { caps: { maxReferences: 0 } }).references).toBeNull();
  });

  it('is not visible when the model takes neither, nor in Extend Video', () => {
    expect(view('txt2img', {}, { caps: { showRefImage: false, maxReferences: 0 } }).visible).toBe(false);
    expect(view('extend', {}, { caps: { showRefImage: false, takesLastFrame: false, maxReferences: 0 } }).visible).toBe(false);
  });

  it('offers Strength once the start image is set, where the workflow has it', () => {
    expect(view('txt2img').strength).toBe(false);
    expect(view('img2img', { referenceImagePath: '/in/a.png' }).strength).toBe(true);
    expect(view('img2img', { referenceImagePath: '/in/a.png' }, { caps: { showI2IStrength: false } }).strength).toBe(false);
  });

  it('names the start slot by its file, its path or its upload marker', () => {
    expect(view('img2img', { referenceImagePath: '/in/a.png' }).start?.name).toBe('a.png');
    expect(view('img2img', { referenceImageUpload: 'up.png' }).start?.name).toBe('up.png');
    expect(view('img2img', { referenceImageUpload: 'up.png' }, { files: { startImage: file('held.png') } }).start?.name).toBe('held.png');
  });

  it('marks a start image gone: a flagged path, or an upload without its file once restored', () => {
    expect(view('img2img', { referenceImagePath: '/in/a.png', missingReferencePaths: ['/in/a.png'] }).start?.gone).toBe(true);
    expect(view('img2img', { referenceImageUpload: 'up.png' }, { restored: false }).start?.gone).toBe(false);
    expect(view('img2img', { referenceImageUpload: 'up.png' }, { restored: true }).start?.gone).toBe(true);
    expect(view('img2img', { referenceImageUpload: 'up.png' }, { files: { startImage: file('up.png') } }).start?.gone).toBe(false);
  });

  it('gives a gone start image its reason, and none while it is there', () => {
    expect(view('img2img', { referenceImagePath: '/in/a.png', missingReferencePaths: ['/in/a.png'] }).start?.error).toEqual(expect.any(String));
    expect(view('img2img', { referenceImageUpload: 'up.png' }).start?.error).toEqual(expect.any(String));
    expect(view('img2img', { referenceImagePath: '/in/a.png' }).start?.error).toBeNull();
  });

  it('shows the first `max` references and keeps the rest out of sight', () => {
    const v = view('txt2img', { referenceImagePaths: ['/in/1.png', '/in/2.png', '/in/3.png', '/in/4.png'] });

    expect(v.references?.rows.map((row) => row.path)).toEqual(['/in/1.png', '/in/2.png', '/in/3.png']);
    expect(v.references?.hasRoom).toBe(false);
  });

  it('pairs each reference with its held upload by row', () => {
    const v = view('txt2img', { referenceImagePaths: ['', '/in/2.png'] }, { files: { references: [file('a.png'), null] } });

    expect(v.references?.rows[0]!.file?.name).toBe('a.png');
    expect(v.references?.rows[1]!.file).toBeNull();
  });

  it('marks a reference gone: a flagged path, or an upload without its file once restored', () => {
    const rows = (restored: boolean, held: Partial<Held> = {}) => view('txt2img', { referenceImagePaths: ['/in/ok.png', '/in/moved.png', ''], missingReferencePaths: ['/in/moved.png'] }, { restored, files: held }).references?.rows.map((row) => row.gone);

    expect(rows(true)).toEqual([false, true, true]);
    expect(rows(false)).toEqual([false, true, false]);
    expect(rows(true, { references: [null, null, file('up.png')] })).toEqual([false, true, false]);
  });

  it('names each reference by its file or its path, and gives a gone one the reason to choose it again', () => {
    const paths = ['/in/ok.png', '/in/moved.png', ''];
    const lost = view('txt2img', { referenceImagePaths: paths, missingReferencePaths: ['/in/moved.png'] }).references!.rows;
    const held = view('txt2img', { referenceImagePaths: paths }, { files: { references: [null, null, file('up.png')] } }).references!.rows;

    expect(lost.map((row) => row.name)).toEqual(['ok.png', 'moved.png', '']);
    expect(lost.map((row) => row.error === null)).toEqual([true, false, false]);
    expect(lost[1]!.error).not.toBe(lost[2]!.error);
    expect([held[2]!.name, held[2]!.error]).toEqual(['up.png', null]);
  });
});

describe('imagesView in Video', () => {
  it('reads the first frame from its own fields, not the start image', () => {
    const v = view('img2vid', { firstFramePath: '/in/first.png', referenceImagePath: '/in/start.png' }, { files: { startImage: file('start.png'), firstFrame: file('first.png') } });

    expect(v.tab).toBe('video');
    expect(v.start).toMatchObject({ path: '/in/first.png', name: 'first.png' });
    expect(v.strength).toBe(false);
    expect(v.references).toBeNull();
  });

  it('shows the last frame on a model that takes one, used with a first frame', () => {
    const v = view('img2vid', { firstFramePath: '/in/a.png', endImagePath: '/in/z.png' });

    expect(v.last).toMatchObject({ set: true, used: true, name: 'z.png' });
    expect(view('img2vid', { firstFramePath: '/in/a.png' }, { caps: { takesLastFrame: false } }).last).toBeNull();
  });

  it('keeps a last frame without a first frame as not used, and never gone', () => {
    const v = view('txt2vid', { endImagePath: '/in/z.png', missingReferencePaths: ['/in/z.png'] });

    expect(v.last).toMatchObject({ set: true, used: false, gone: false });
  });

  it('marks a used last frame gone', () => {
    const v = view('img2vid', { firstFramePath: '/in/a.png', endImagePath: '/in/z.png', missingReferencePaths: ['/in/z.png'] });

    expect(v.last?.gone).toBe(true);
    expect(v.start?.gone).toBe(false);
  });

  it('has neither slot on a model that cannot start from an image', () => {
    const v = view('txt2vid', { firstFramePath: '/in/a.png', endImagePath: '/in/z.png' }, { caps: { showRefImage: false, takesLastFrame: false, maxReferences: 0 } });

    expect([v.start, v.last, v.visible]).toEqual([null, null, false]);
  });
});

describe('imagesView block', () => {
  it('names the first used image that is gone, then a missing reference', () => {
    const base = { firstFramePath: '/in/first.png', endImagePath: '/in/last.png' };
    const both = ['/in/first.png', '/in/last.png'];

    expect(view('img2vid', { ...base, missingReferencePaths: both }).block).toBe('The first frame is missing.');
    expect(view('img2vid', { ...base, missingReferencePaths: ['/in/last.png'] }).block).toBe('The last frame is missing.');
    expect(view('img2vid', base).block).toBeNull();
    expect(view('img2img', { referenceImagePath: '/in/start.png', missingReferencePaths: ['/in/start.png'] }).block).toBe('The start image is missing.');
    expect(view('txt2img', { referenceImagePaths: ['/in/a.png', '/in/gone.png'], missingReferencePaths: ['/in/gone.png'] }).block).toBe('Reference 2 is missing.');
    expect(view('img2img', { referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/gone.png'], missingReferencePaths: ['/in/start.png', '/in/gone.png'] }).block).toBe('The start image is missing.');
  });

  it('names an upload that did not come back, once restored', () => {
    expect(view('img2img', { referenceImageUpload: 'a.png' }).block).toBe('The start image is missing.');
    expect(view('img2vid', { firstFrameUpload: 'a.png' }).block).toBe('The first frame is missing.');
    expect(view('txt2img', { referenceImagePaths: ['/in/a.png', ''] }).block).toBe('Reference 2 is missing.');
    expect(view('img2vid', { firstFrameUpload: 'a.png' }, { files: { firstFrame: file('a.png') } }).block).toBeNull();
  });

  it('waits for the held uploads to be read back', () => {
    expect(view('img2img', { referenceImageUpload: 'a.png' }, { restored: false }).block).toBeNull();
  });

  it('a hidden or unused image never blocks', () => {
    const missing = ['/in/a.png', '/in/b.png'];

    // A last frame kept without a first frame, and one on a model that takes none.
    expect(view('txt2vid', { endImagePath: '/in/b.png', missingReferencePaths: missing }).block).toBeNull();
    expect(view('img2vid', { firstFramePath: '/in/a.png', endImagePath: '/in/b.png', missingReferencePaths: ['/in/b.png'] }, { caps: { takesLastFrame: false } }).block).toBeNull();
    // A start image the model cannot use, and the other tab's first frame.
    expect(view('txt2img', { referenceImagePath: '/in/a.png', missingReferencePaths: missing }, { caps: { showRefImage: false } }).block).toBeNull();
    expect(view('img2img', { firstFramePath: '/in/b.png', missingReferencePaths: missing }).block).toBeNull();
    // A reference beyond what the model takes.
    expect(view('txt2img', { referenceImagePaths: ['/in/1.png', '/in/2.png', '/in/a.png'], missingReferencePaths: ['/in/a.png'] }, { caps: { maxReferences: 2 } }).block).toBeNull();
  });
});

describe('the workflow and the view agree on a set start image', () => {
  const cases: { name: string; path: string | null; upload: string | null; file: File | null; set: boolean; sends: boolean }[] = [
    { name: 'nothing', path: null, upload: null, file: null, set: false, sends: false },
    { name: 'an empty path', path: '', upload: null, file: null, set: false, sends: false },
    { name: 'a path', path: '/in/a.png', upload: null, file: null, set: true, sends: true },
    { name: 'an upload with its file', path: null, upload: 'a.png', file: file('a.png'), set: true, sends: true },
    { name: 'an upload whose file did not come back', path: null, upload: 'a.png', file: null, set: true, sends: false },
    { name: 'an empty path beside an upload marker whose file did not come back', path: '', upload: 'a.png', file: null, set: true, sends: false },
    { name: 'a held file without a marker', path: null, upload: null, file: file('a.png'), set: false, sends: false },
  ];

  it.each(['image', 'video'] as ImageTab[])('in %s, one rule says whether the start image is set', (tab) => {
    for (const c of cases) {
      const slots: Overrides = tab === 'image' ? { referenceImagePath: c.path, referenceImageUpload: c.upload } : { firstFramePath: c.path, firstFrameUpload: c.upload };
      const files: Partial<Held> = tab === 'image' ? { startImage: c.file } : { firstFrame: c.file };
      const workflow = workflowFor(tab, state('txt2img', slots), true);
      const v = view(workflow, slots, { files });

      expect(workflow === 'img2img' || workflow === 'img2vid', c.name).toBe(c.set);
      expect(startImageSet(state(workflow, slots), tab), c.name).toBe(c.set);
      expect(v.start?.set, c.name).toBe(c.set);
      expect(v.sent.image !== null, c.name).toBe(c.sends);
    }
  });
});

describe('imagesView sent', () => {
  const none = { image: null, strength: null, end: null };

  it('sends the start image and its strength in Image, the first and last frame in Video', () => {
    expect(view('img2img', { referenceImagePath: '/in/start.png', firstFramePath: '/in/first.png', endImagePath: '/in/last.png' }).sent).toEqual({ image: '/in/start.png', strength: 0.65, end: null });
    expect(view('img2vid', { referenceImagePath: '/in/start.png', firstFramePath: '/in/first.png', endImagePath: '/in/last.png' }).sent).toEqual({ image: '/in/first.png', strength: null, end: '/in/last.png' });
  });

  it('sends nothing without a start image, whichever tab', () => {
    expect(view('txt2img').sent).toEqual(none);
    expect(view('txt2vid').sent).toEqual(none);
    expect(view('extend', {}, { caps: { showRefImage: false, takesLastFrame: false, maxReferences: 0 } }).sent).toEqual(none);
  });

  it('prefers a held upload to a path', () => {
    const start = file('start.png');
    const last = file('last.png');

    expect(view('img2img', { referenceImageUpload: 'start.png' }, { files: { startImage: start } }).sent).toEqual({ image: start, strength: 0.65, end: null });
    expect(view('img2vid', { firstFrameUpload: 'start.png', endImageUpload: 'last.png' }, { files: { firstFrame: start, lastFrame: last } }).sent).toEqual({ image: start, strength: null, end: last });
  });

  it('sends no last frame without a first frame, and sends the kept one again once a first frame is added', () => {
    expect(view('txt2vid', { endImagePath: '/in/last.png' }).sent).toEqual(none);
    expect(view('img2vid', { firstFramePath: '/in/first.png', endImagePath: '/in/last.png' }).sent.end).toBe('/in/last.png');
  });

  it('sends no last frame on a model that takes none, nor any start image on one that cannot use it', () => {
    const frames = { firstFramePath: '/in/first.png', endImagePath: '/in/last.png' };

    expect(view('img2vid', frames, { caps: { takesLastFrame: false } }).sent).toEqual({ image: '/in/first.png', strength: null, end: null });
    expect(view('txt2vid', frames, { caps: { showRefImage: false, takesLastFrame: false } }).sent).toEqual(none);
    expect(view('txt2img', { referenceImagePath: '/in/start.png' }, { caps: { showRefImage: false } }).sent).toEqual(none);
  });

  it('shows a last frame as used exactly when it is sent', () => {
    for (const takesLastFrame of [true, false]) {
      const v = view('img2vid', { firstFramePath: '/in/first.png', endImagePath: '/in/last.png' }, { caps: { takesLastFrame } });
      expect(v.last?.used === true && v.last.set).toBe(v.sent.end !== null);
    }
  });
});
