import { beforeEach, describe, expect, it } from 'vitest';

import { makeEditContext } from '../../test-utils/workspace';
import { draft } from './draft.svelte';
import { heldUploads } from './heldUploads.svelte';
import { memoryUploadStore } from './uploadStore';
import type { WorkspaceContext } from '$lib/types';

const png = (name: string): File => new File(['x'], name, { type: 'image/png' });

/** Give the draft an image to edit with paint, an extension and a held mask, as a Done in the painter leaves it. */
function paintedEdit(path = '/out/a.png'): void {
  draft.setEditSource({ path });
  draft.commitEditPaint({ mask: png('mask.png'), share: 0.25, extension: { side: 'top', amount: 320 }, amount: 320 });
}

const NO_PAINT = { editMaskUpload: null, editPaintedShare: 0, editExtendSide: null, editExtendAmount: 256, editRepaint: false };

describe('the draft\'s image to edit', () => {
  let ctx: WorkspaceContext;

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
    ctx = makeEditContext();
    draft.hydrateFromContext(ctx, 'klein');
  });

  it('starts with no image and no paint', () => {
    expect(draft.state).toMatchObject({ editSourcePath: null, editSourceUpload: null, ...NO_PAINT });
  });

  it('holds a path, or an upload by name with its file, never both', () => {
    draft.setEditSource({ path: '/out/a.png' });
    expect(draft.state).toMatchObject({ editSourcePath: '/out/a.png', editSourceUpload: null });
    const upload = png('photo.png');
    draft.setEditSource({ file: upload });
    expect(draft.state).toMatchObject({ editSourcePath: null, editSourceUpload: 'photo.png' });
    expect(heldUploads.editSource).toBe(upload);
    draft.setEditSource(null);
    expect([draft.state.editSourceUpload, heldUploads.editSource]).toEqual([null, null]);
  });

  it('forgets the mask, its share, the extension and the repaint flag with any change of the image', () => {
    paintedEdit();
    draft.patch({ editRepaint: true });
    expect(heldUploads.editMask).not.toBeNull();

    draft.setEditSource({ path: '/out/b.png' });

    expect(draft.state).toMatchObject({ editSourcePath: '/out/b.png', ...NO_PAINT });
    expect(heldUploads.editMask).toBeNull();
  });

  it('starts the painter\'s history over for a new image, a Reset and a Reuse, but not for a commit or a move', () => {
    const epoch = draft.editEpoch;
    draft.setEditSource({ path: '/out/a.png' });
    expect(draft.editEpoch).toBe(epoch + 1);
    draft.commitEditPaint({ mask: png('mask.png'), share: 0.1, extension: null, amount: 256 });
    draft.retargetPaths(new Map([['/out/a.png', '/out/Bus/a.png']]));
    expect(draft.editEpoch).toBe(epoch + 1);
    draft.resetSelections(ctx);
    expect(draft.editEpoch).toBe(epoch + 1);
    draft.setEditing(true, ctx);
    draft.resetSelections(ctx);
    expect(draft.editEpoch).toBe(epoch + 2);
  });

  it('keeps what the painter ended with: the mask by name with its share, and the extension', () => {
    paintedEdit();
    expect(draft.state).toMatchObject({ editMaskUpload: 'mask.png', editPaintedShare: 0.25, editExtendSide: 'top', editExtendAmount: 320 });
    expect(heldUploads.editMask?.name).toBe('mask.png');
  });

  it('keeps no mask, share or extension when nothing was painted or extended', () => {
    paintedEdit();
    draft.commitEditPaint({ mask: null, share: 0.3, extension: null, amount: 192 });
    expect(draft.state).toMatchObject({ editMaskUpload: null, editPaintedShare: 0, editExtendSide: null, editExtendAmount: 192 });
    expect(heldUploads.editMask).toBeNull();
  });

  it('answers a reused edit\'s repaint by added paint, or by an extension that was not there', () => {
    draft.setEditSource({ path: '/out/a.png' });
    draft.patch({ editRepaint: true });
    draft.commitEditPaint({ mask: null, share: 0, extension: null, amount: 256 });
    expect(draft.state.editRepaint).toBe(true);
    draft.commitEditPaint({ mask: png('mask.png'), share: 0.1, extension: null, amount: 256 });
    expect(draft.state.editRepaint).toBe(false);

    draft.patch({ editRepaint: true });
    draft.commitEditPaint({ mask: null, share: 0, extension: { side: 'left', amount: 128 }, amount: 128 });
    expect(draft.state.editRepaint).toBe(false);
  });

  it('keeps the repaint of an edit that already had its extension until paint is added', () => {
    draft.setEditSource({ path: '/out/a.png' });
    draft.patch({ editRepaint: true, editExtendSide: 'left', editExtendAmount: 128 });
    draft.commitEditPaint({ mask: null, share: 0, extension: { side: 'left', amount: 128 }, amount: 128 });
    expect(draft.state.editRepaint).toBe(true);
  });

  it('survives a reload with its paint summary and extension, as the draft is saved', () => {
    paintedEdit();
    draft.saveDraft();

    draft.loadDraft();

    expect(draft.state).toMatchObject({ editSourcePath: '/out/a.png', editMaskUpload: 'mask.png', editPaintedShare: 0.25, editExtendSide: 'top', editExtendAmount: 320 });
  });

  it('keeps the paint, the share and the extension when the image moves, and follows it to its new path', () => {
    paintedEdit();
    const mask = heldUploads.editMask;

    draft.retargetPaths(new Map([['/out/a.png', '/out/Bus/a.png']]));

    expect(draft.state).toMatchObject({ editSourcePath: '/out/Bus/a.png', editMaskUpload: 'mask.png', editPaintedShare: 0.25, editExtendSide: 'top', editExtendAmount: 320 });
    expect(heldUploads.editMask).toBe(mask);
  });

  it('is cleared by a Reset of Edit, with its held files, and left alone by a Reset of New or of the Video tab', () => {
    paintedEdit();
    draft.resetSelections(ctx);
    expect(draft.state).toMatchObject({ editSourcePath: '/out/a.png', editMaskUpload: 'mask.png' });
    draft.setTab('video');
    draft.onTabChange(ctx);
    draft.resetSelections(ctx);
    expect(draft.state).toMatchObject({ editSourcePath: '/out/a.png', editMaskUpload: 'mask.png' });
    expect(heldUploads.editMask).not.toBeNull();

    draft.setTab('image');
    draft.onTabChange(ctx);
    expect(draft.state.workflow).toBe('txt2img');
    draft.setEditing(true, ctx);
    draft.resetSelections(ctx);

    expect(draft.state).toMatchObject({ workflow: 'edit', editSourcePath: null, editSourceUpload: null, ...NO_PAINT });
    expect([heldUploads.editSource, heldUploads.editMask]).toEqual([null, null]);
  });

  it('is part of the slots a recheck looks at, so a moved or deleted file is flagged like any other', async () => {
    const { slotPaths } = await import('./draft.svelte');
    draft.setEditSource({ path: '/out/a.png' });
    expect(slotPaths(draft.state)).toContain('/out/a.png');
  });

  it('keeps New\'s prompt and Enhanced text and Edit\'s prompt apart across Kind switches, whichever way they go', () => {
    draft.patch({ prompt: 'a fox', enhancedPrompt: 'an enhanced fox' });
    draft.setEditing(true, ctx);
    expect(draft.state).toMatchObject({ workflow: 'edit', prompt: '', enhancedPrompt: '' });
    draft.patch({ prompt: 'make the door red' });
    draft.setEditing(false, ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a fox', enhancedPrompt: 'an enhanced fox' });
    draft.setEditing(true, ctx);
    expect(draft.state).toMatchObject({ workflow: 'edit', prompt: 'make the door red', enhancedPrompt: '' });
  });

  it('is a workflow of its own, and is remembered with the draft', () => {
    draft.setEditing(true, ctx);
    expect(draft.state.workflow).toBe('edit');
    draft.loadDraft();
    expect(draft.state.workflow).toBe('edit');
  });

  it('opens on the default model that can edit, with that model\'s defaults, when New is on a model that cannot', () => {
    draft.hydrateFromContext(ctx, 'flux-lite');
    draft.patch({ seed: 7, runs: 3 });
    draft.setEditing(true, ctx);
    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'zit', seed: null, runs: 1, steps: 28 });
  });

  it('opens on its own model, never New\'s, even when New\'s can edit', () => {
    draft.setEditing(true, ctx);
    expect(draft.state.model).toBe('zit');
    draft.patch({ model: 'klein' });
    draft.setEditing(false, ctx);
    draft.setEditing(true, ctx);
    expect(draft.state.model).toBe('klein');
  });

  it('opens on a model that can edit, whatever opens it', () => {
    draft.hydrateFromContext(ctx, 'flux-lite');
    draft.loadFromUrl({ workflow: 'edit' }, ctx);
    draft.hydrateFromContext(ctx, null);
    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'zit' });
  });
});

describe('the Image tab left on Edit', () => {
  let ctx: WorkspaceContext;

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
    ctx = makeEditContext();
    draft.hydrateFromContext(ctx, 'klein');
  });

  it('opens on Edit again, with Edit\'s text, after a visit to Video or Audio', () => {
    draft.setEditing(true, ctx);
    draft.patch({ prompt: 'make the door red' });
    draft.setTab('video');
    draft.patch({ prompt: 'a knight rides' });
    expect(draft.state).toMatchObject({ workflow: 'txt2vid', leftOnEdit: true, prompt: 'a knight rides' });

    draft.setTab('image');

    expect(draft.state).toMatchObject({ workflow: 'edit', leftOnEdit: false, prompt: 'make the door red' });
  });

  it('opens on New when it was left on New', () => {
    draft.patch({ prompt: 'a fox' });
    draft.setTab('video');
    draft.setTab('image');
    expect(draft.state).toMatchObject({ workflow: 'txt2img', leftOnEdit: false, prompt: 'a fox' });
  });

  it('is not changed by a video or an audio prefill', () => {
    draft.setEditing(true, ctx);
    draft.loadFromUrl({ workflow: 'txt2vid', scene: 'a castle' }, ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2vid', leftOnEdit: true });
    draft.loadFromUrl({ workflow: 'music' }, ctx);
    expect(draft.state.leftOnEdit).toBe(true);
    draft.setTab('image');
    expect(draft.state.workflow).toBe('edit');

    draft.setEditing(false, ctx);
    draft.loadFromUrl({ workflow: 'txt2vid', scene: 'a castle' }, ctx);
    draft.loadFromUrl({ workflow: 'speech' }, ctx);
    expect(draft.state.leftOnEdit).toBe(false);
    draft.setTab('image');
    expect(draft.state.workflow).toBe('txt2img');
  });

  it('is ended by an image prefill that opens New', () => {
    draft.setEditing(true, ctx);
    draft.setTab('video');
    draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a fox' }, ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', leftOnEdit: false, prompt: 'a fox' });
  });

  it('is told apart from the other tabs: Edit does not change the Video tab\'s model or LoRAs', () => {
    ctx.loras = [{ name: 'detail', path: '/loras/detail.safetensors' }];
    draft.setTab('video');
    draft.onTabChange(ctx);
    draft.patch({ loraString: 'detail:0.6' });
    const video = { model: draft.state.model, loraString: draft.state.loraString };

    draft.loadFromUrl({ workflow: 'edit' }, ctx);
    draft.hydrateFromContext(ctx, null);
    draft.setTab('video');
    draft.onTabChange(ctx);

    expect(draft.state).toMatchObject(video);
  });
});

describe('prefilling the image to edit', () => {
  let ctx: WorkspaceContext;

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
    ctx = makeEditContext();
    draft.hydrateFromContext(ctx, 'klein');
  });

  it('opens Edit, whatever the draft was on, and sets the image', () => {
    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/a.png' }, ctx);
    expect(draft.state).toMatchObject({ workflow: 'edit', editSourcePath: '/out/a.png' });
  });

  it('sets the image and the extension a Reuse names, asks for the paint again, and fills Edit\'s prompt, replacing the paint it had', () => {
    paintedEdit('/out/old.png');

    draft.loadFromUrl({
      workflow: 'edit', edit_source_path: '/out/a.png', edit_repaint: 'true', edit_extend_side: 'left', edit_extend_amount: '192', prompt: 'a door', model: 'klein', steps: '4',
    }, ctx, { replaceSlots: true, exact: true });

    expect(draft.state).toMatchObject({
      workflow: 'edit', editSourcePath: '/out/a.png', editSourceUpload: null, editMaskUpload: null, editPaintedShare: 0,
      editExtendSide: 'left', editExtendAmount: 192, editRepaint: true, prompt: 'a door', steps: 4,
    });
  });

  it('leaves New\'s prompt and Enhanced text as they were, from New and from the Video tab', () => {
    draft.patch({ prompt: 'a fox', enhancedPrompt: 'an enhanced fox', enhancedFrom: { prompt: 'a fox', scene: '', mode: 'image' } });
    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/a.png', prompt: 'a door' }, ctx, { replaceSlots: true, exact: true });
    expect(draft.state).toMatchObject({ workflow: 'edit', prompt: 'a door', enhancedPrompt: '', enhancedFrom: null });

    draft.setTab('video');
    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/b.png', prompt: 'a window' }, ctx, { replaceSlots: true, exact: true });
    draft.setEditing(false, ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a fox', enhancedPrompt: 'an enhanced fox' });
  });

  it('restores a saved Extend alone, with no repaint', () => {
    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/a.png', edit_extend_side: 'bottom', edit_extend_amount: '128' }, ctx, { replaceSlots: true, exact: true });
    expect(draft.state).toMatchObject({ editExtendSide: 'bottom', editExtendAmount: 128, editRepaint: false });
  });

  it('empties the image when the source is named blank (a picture that is gone), and keeps the rest of the edit', () => {
    paintedEdit();
    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '', edit_repaint: 'true', prompt: 'a door' }, ctx, { replaceSlots: true, exact: true });
    expect(draft.state).toMatchObject({ editSourcePath: null, editSourceUpload: null, editMaskUpload: null, editRepaint: true, prompt: 'a door' });
  });

  it('starts an Edit action with no paint and an empty prompt, and keeps New\'s images and text', () => {
    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    draft.patch({ prompt: 'a fox' });
    draft.setEditing(true, ctx);
    draft.patch({ prompt: 'an old edit' });
    paintedEdit('/out/old.png');

    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/a.png', use_as: 'edit' }, ctx, { replaceSlots: false });

    expect(draft.state).toMatchObject({ workflow: 'edit', editSourcePath: '/out/a.png', prompt: '', ...NO_PAINT, referenceImagePath: '/in/start.png' });
    draft.setEditing(false, ctx);
    expect(draft.state).toMatchObject({ workflow: 'img2img', prompt: 'a fox', referenceImagePath: '/in/start.png' });
  });

  it('never touches the image to edit, its paint or its held files for a Reuse of something else', () => {
    paintedEdit();
    const mask = heldUploads.editMask;
    const epoch = draft.editEpoch;

    draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a fox' }, ctx, { replaceSlots: true, exact: true });
    draft.loadFromUrl({ workflow: 'txt2img', image_path: '/in/start.png', use_as: 'start' }, ctx, { replaceSlots: false });
    draft.loadFromUrl({ workflow: 'txt2vid', scene: 'a castle' }, ctx, { replaceSlots: true, exact: true });

    expect(draft.state).toMatchObject({ editSourcePath: '/out/a.png', editMaskUpload: 'mask.png', editPaintedShare: 0.25, editExtendSide: 'top', editExtendAmount: 320 });
    expect(heldUploads.editMask).toBe(mask);
    expect(draft.editEpoch).toBe(epoch);
  });

  it('keeps its flag for a missing file when another tab\'s slots are replaced', () => {
    draft.setEditSource({ path: '/out/gone.png' });
    draft.patch({ missingReferencePaths: ['/out/gone.png'] });
    draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a fox' }, ctx, { replaceSlots: true, exact: true });
    expect(draft.state.missingReferencePaths).toEqual(['/out/gone.png']);
  });
});

/** Every setting Edit shows, apart from its prompt: its model, quantization, LoRAs, Steps, Seed and Batch. */
const EDIT_SETTINGS = ['model', 'quantize', 'loraString', 'steps', 'seed', 'runs'] as const;

describe('Edit\'s settings, apart from New\'s', () => {
  let ctx: WorkspaceContext;

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
    ctx = makeEditContext();
    ctx.loras = [{ name: 'detail', path: '/loras/detail.safetensors' }, { name: 'soft', path: '/loras/soft.safetensors' }];
    draft.hydrateFromContext(ctx, 'klein');
  });

  /** What New holds, whichever tab is open: the open fields while New is open, else what it saved. */
  const newKept = (): Record<string, unknown> => {
    const source = draft.state.workflow === 'edit' ? draft.state.savedSettings.image ?? {} : draft.state;
    return Object.fromEntries([...EDIT_SETTINGS, 'ratio', 'guidance', 'postprocessSharpenAmount', 'appliedPreset'].map((key) => [key, (source as Record<string, unknown>)[key]]));
  };

  const editKept = (): Record<string, unknown> => {
    const source = draft.state.workflow === 'edit' ? draft.state : draft.state.savedSettings.edit ?? {};
    return Object.fromEntries(EDIT_SETTINGS.map((key) => [key, (source as Record<string, unknown>)[key]]));
  };

  it('is changed in every setting without touching New\'s, and New\'s without touching Edit\'s', () => {
    draft.patch({ steps: 12, seed: 5, runs: 2, loraString: 'soft:0.4', quantize: 8, ratio: '16:9', size: 'l', guidance: 2.5, postprocessSharpenAmount: 1.1, appliedPreset: 'fast' });
    const before = newKept();

    draft.setEditing(true, ctx);
    draft.patch({ model: 'zit', steps: 6, seed: 99, runs: 4, quantize: 4, loraString: 'detail:0.7' });
    const edit = editKept();
    draft.setEditing(false, ctx);

    expect(newKept()).toEqual(before);
    expect(draft.state).toMatchObject({ model: 'klein', steps: 12, seed: 5, runs: 2, loraString: 'soft:0.4', quantize: 8, ratio: '16:9', size: 'l', guidance: 2.5, appliedPreset: 'fast' });

    draft.patch({ model: 'flux-lite', steps: 30, seed: 1, runs: 1, loraString: '', quantize: null });
    draft.setEditing(true, ctx);

    expect(draft.state).toMatchObject({ workflow: 'edit', ...edit });
  });

  it('keeps New on a model that cannot edit, with its LoRAs, across a visit to Edit', () => {
    draft.hydrateFromContext(ctx, 'flux-lite');
    draft.patch({ loraString: 'soft:0.5', steps: 14 });

    draft.setEditing(true, ctx);
    expect(draft.state.model).toBe('zit');
    expect(draft.state.loraString).toBe('');
    draft.setEditing(false, ctx);

    expect(draft.state).toMatchObject({ workflow: 'txt2img', model: 'flux-lite', loraString: 'soft:0.5', steps: 14 });
  });

  it('keeps the LoRAs each tab remembers for a model family apart', () => {
    draft.patch({ loraString: 'soft:0.5' });
    draft.setEditing(true, ctx);
    draft.hydrateFromContext(ctx, 'zit');
    draft.patch({ loraString: 'detail:0.7' });
    draft.hydrateFromContext(ctx, 'klein');
    draft.setEditing(false, ctx);
    draft.hydrateFromContext(ctx, 'zit');
    expect(draft.state.loraString).toBe('');
  });

  it('is put back by a Reset of Edit without touching New\'s ratio, guidance, sharpening, preset or model', () => {
    draft.patch({ ratio: '16:9', size: 'l', guidance: 2.5, postprocessSharpenAmount: 1.1, appliedPreset: 'fast', seed: 3 });
    const before = newKept();
    draft.setEditing(true, ctx);
    draft.patch({ steps: 6, seed: 99, runs: 4 });

    draft.resetSelections(ctx);

    expect(draft.state).toMatchObject({ workflow: 'edit', steps: 28, seed: null, runs: 1 });
    draft.setEditing(false, ctx);
    expect(newKept()).toEqual(before);
    expect(draft.state.appliedPreset).toBe('fast');
  });

  it('is set by a Reuse of an edit without touching New\'s ratio, guidance, sharpening, preset or model', () => {
    draft.patch({ ratio: '16:9', size: 'l', guidance: 2.5, postprocessSharpenAmount: 1.1, appliedPreset: 'fast', seed: 3 });
    const before = newKept();

    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/a.png', prompt: 'a door', model: 'zit', steps: '6', seed: '9', lora: 'detail:0.7' }, ctx, { replaceSlots: true, exact: true });

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'zit', steps: 6, seed: 9, loraString: 'detail:0.7' });
    draft.setEditing(false, ctx);
    expect(newKept()).toEqual(before);
    expect(draft.state.appliedPreset).toBe('fast');
  });

  it('leaves Edit alone when a normal image is reused', () => {
    draft.setEditing(true, ctx);
    draft.patch({ model: 'zit', steps: 6, seed: 99, runs: 4, loraString: 'detail:0.7' });
    const edit = editKept();

    draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a cat', model: 'klein', steps: '8', seed: '3' }, ctx, { replaceSlots: true, exact: true });

    expect(draft.state).toMatchObject({ workflow: 'txt2img', model: 'klein', steps: 8, seed: 3 });
    expect(editKept()).toEqual(edit);
  });

  it('shares Batch and Seed between New and Video, as before, and keeps Edit\'s own', () => {
    draft.patch({ seed: 5, runs: 2 });
    draft.setEditing(true, ctx);
    draft.patch({ seed: 77, runs: 6 });
    draft.setTab('video');
    expect(draft.state).toMatchObject({ seed: 5, runs: 2 });
    draft.patch({ seed: 8 });
    draft.setTab('image');
    expect(draft.state).toMatchObject({ workflow: 'edit', seed: 77, runs: 6 });
    draft.setEditing(false, ctx);
    expect(draft.state).toMatchObject({ seed: 8, runs: 2 });
  });

  it('leaves the prompt source New chose alone, as Edit offers none', () => {
    draft.patch({ promptSource: 'file', promptFilePath: '/prompts/a.yaml', promptFileOptionIds: ['a:0'] });
    draft.setEditing(true, ctx);
    expect(draft.state).toMatchObject({ promptSource: 'file', promptFilePath: '/prompts/a.yaml' });
    draft.setEditing(false, ctx);
    expect(draft.state).toMatchObject({ promptSource: 'file', promptFileOptionIds: ['a:0'] });
  });

  it('survives a reload, with New\'s kept beside it', () => {
    draft.patch({ steps: 12, seed: 5, loraString: 'soft:0.4' });
    draft.setEditing(true, ctx);
    draft.patch({ model: 'zit', steps: 6, seed: 99, loraString: 'detail:0.7' });
    draft.saveDraft();

    draft.loadDraft();

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'zit', steps: 6, seed: 99, loraString: 'detail:0.7' });
    draft.setEditing(false, null);
    expect(draft.state).toMatchObject({ model: 'klein', steps: 12, seed: 5, loraString: 'soft:0.4' });
  });

  it('forgets a removed LoRA in what each tab kept', () => {
    draft.patch({ loraString: 'soft:0.4' });
    draft.setEditing(true, ctx);
    draft.removeLora('soft', ctx.loras);
    expect(draft.state.savedSettings.image?.loraString).toBe('');
  });

  it('gives a model that can no longer edit its default when Edit is opened again', () => {
    draft.setEditing(true, ctx);
    draft.patch({ model: 'flux-lite' });
    draft.setEditing(false, ctx);
    draft.setEditing(true, ctx);
    expect(draft.state.model).toBe('zit');
  });
});

describe('a recorded model that Edit cannot use', () => {
  let ctx: WorkspaceContext;

  /** The steps of a prefill that opens a workflow on a recorded model and LoRAs, as the Workspace takes them. */
  function prefill(params: Record<string, string>, preferred: string): void {
    draft.loadFromUrl({ workflow: params.workflow }, ctx);
    draft.hydrateFromContext(ctx, preferred);
    draft.loadFromUrl(params, ctx, { replaceSlots: true });
  }

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
    ctx = makeEditContext();
    ctx.loras = [{ name: 'detail', path: '/loras/detail.safetensors' }, { name: 'own', path: '/loras/own.safetensors' }] as WorkspaceContext['loras'];
    draft.hydrateFromContext(ctx, 'klein');
    draft.setEditing(true, ctx);
    draft.patch({ model: 'klein', loraString: 'own:0.5' });
  });

  it('is not taken: Edit keeps its own model and LoRAs, and the rest of the prefill still applies', () => {
    prefill({ workflow: 'edit', model: 'flux-lite', lora: 'detail:0.8', steps: '9' }, 'flux-lite');

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein', loraString: 'own:0.5', steps: 9 });
  });

  it('is taken once the model can edit, with its LoRAs', () => {
    prefill({ workflow: 'edit', model: 'zit', lora: 'detail:0.8' }, 'zit');

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'zit', loraString: 'detail:0.8' });
  });

  it('falls to the default model that can edit when Edit\'s own model is not offered either', () => {
    draft.patch({ model: 'flux-lite' });

    prefill({ workflow: 'edit', model: 'flux-lite', lora: 'detail:0.8' }, 'flux-lite');

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'zit' });
    expect(draft.state.loraString).not.toBe('detail:0.8');
  });

  it('leaves New free to take the same model and LoRAs', () => {
    draft.setEditing(false, ctx);

    prefill({ workflow: 'txt2img', model: 'flux-lite', lora: 'detail:0.8' }, 'flux-lite');

    expect(draft.state).toMatchObject({ workflow: 'txt2img', model: 'flux-lite', loraString: 'detail:0.8' });
  });
});
