import { beforeEach, describe, expect, it } from 'vitest';

import { makeEditContext } from '../../test-utils/workspace';
import { draft, slotPaths } from './draft.svelte';
import { heldUploads } from './heldUploads.svelte';
import { memoryUploadStore } from './uploadStore';
import { modelsOf } from './workflowModels';
import type { WorkspaceContext } from '$lib/types';
import { openKind } from '../../test-utils/draftKind';

const png = (name: string): File => new File(['x'], name, { type: 'image/png' });

/** Give the draft an image to edit with paint and a held mask, as a Done in the painter in Paint leaves it. */
function paintedEdit(path = '/out/a.png'): void {
  draft.setEditSource({ path });
  draft.commitEdit({ mode: 'paint', mask: png('mask.png'), share: 0.25, extension: null, amount: 320 });
}

/** A clean draft on Klein with a context that can edit, and the held files in memory; `loras` are the ones installed. */
function freshEditDraft(loras?: WorkspaceContext['loras']): WorkspaceContext {
  localStorage.clear();
  draft.reset();
  heldUploads.useStore(memoryUploadStore());
  const context = makeEditContext();
  if (loras) context.loras = loras;
  draft.hydrateFromContext(context, 'klein');
  return context;
}

const NO_PAINT = { editMode: 'paint', editMaskUpload: null, editPaintedShare: 0, editExtendSide: null, editExtendAmount: 256, editRepaint: false };

describe('the draft\'s image to edit', () => {
  let ctx: WorkspaceContext;

  beforeEach(() => {
    ctx = freshEditDraft();
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
    draft.commitEdit({ mode: 'paint', mask: png('mask.png'), share: 0.1, extension: null, amount: 256 });
    draft.retargetPaths(new Map([['/out/a.png', '/out/Bus/a.png']]));
    expect(draft.editEpoch).toBe(epoch + 1);
    draft.resetSelections(ctx);
    expect(draft.editEpoch).toBe(epoch + 1);
    openKind('image', 'edit', ctx);
    draft.resetSelections(ctx);
    expect(draft.editEpoch).toBe(epoch + 2);
  });

  it('keeps what the painter ended with: the mode, and the mask by name with its share (Paint) or the side and amount (Extend)', () => {
    paintedEdit();
    expect(draft.state).toMatchObject({ editMode: 'paint', editMaskUpload: 'mask.png', editPaintedShare: 0.25, editExtendSide: null, editExtendAmount: 320 });
    expect(heldUploads.editMask?.name).toBe('mask.png');

    draft.commitEdit({ mode: 'extend', mask: png('mask.png'), share: 0.25, extension: { side: 'top', amount: 192 }, amount: 192 });
    expect(draft.state).toMatchObject({ editMode: 'extend', editMaskUpload: null, editPaintedShare: 0, editExtendSide: 'top', editExtendAmount: 192 });
    expect(heldUploads.editMask).toBeNull();

    draft.commitEdit({ mode: 'paint', mask: png('mask.png'), share: 0.1, extension: { side: 'top', amount: 192 }, amount: 192 });
    expect(draft.state).toMatchObject({ editMode: 'paint', editMaskUpload: 'mask.png', editExtendSide: null });
  });

  it('keeps no mask, share or extension when nothing was painted or extended', () => {
    paintedEdit();
    draft.commitEdit({ mode: 'paint', mask: null, share: 0.3, extension: null, amount: 192 });
    expect(draft.state).toMatchObject({ editMaskUpload: null, editPaintedShare: 0, editExtendSide: null, editExtendAmount: 192 });
    expect(heldUploads.editMask).toBeNull();
  });

  it('answers a reused edit\'s repaint by added paint, or by an extension that was not there', () => {
    draft.setEditSource({ path: '/out/a.png' });
    draft.patch({ editRepaint: true });
    draft.commitEdit({ mode: 'paint', mask: null, share: 0, extension: null, amount: 256 });
    expect(draft.state.editRepaint).toBe(true);
    draft.commitEdit({ mode: 'paint', mask: png('mask.png'), share: 0.1, extension: null, amount: 256 });
    expect(draft.state.editRepaint).toBe(false);

    draft.patch({ editRepaint: true });
    draft.commitEdit({ mode: 'extend', mask: null, share: 0, extension: { side: 'left', amount: 128 }, amount: 128 });
    expect(draft.state.editRepaint).toBe(false);
  });

  it('survives a reload with its mode and paint summary, as the draft is saved', () => {
    paintedEdit();
    draft.saveDraft();

    draft.loadDraft();

    expect(draft.state).toMatchObject({ editSourcePath: '/out/a.png', editMode: 'paint', editMaskUpload: 'mask.png', editPaintedShare: 0.25, editExtendSide: null, editExtendAmount: 320 });
  });

  it('keeps the paint and the share when the image moves, and follows it to its new path', () => {
    paintedEdit();
    const mask = heldUploads.editMask;

    draft.retargetPaths(new Map([['/out/a.png', '/out/Bus/a.png']]));

    expect(draft.state).toMatchObject({ editSourcePath: '/out/Bus/a.png', editMode: 'paint', editMaskUpload: 'mask.png', editPaintedShare: 0.25, editExtendSide: null, editExtendAmount: 320 });
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
    openKind('image', 'edit', ctx);
    draft.resetSelections(ctx);

    expect(draft.state).toMatchObject({ workflow: 'edit', editSourcePath: null, editSourceUpload: null, ...NO_PAINT });
    expect([heldUploads.editSource, heldUploads.editMask]).toEqual([null, null]);
  });

  it('is part of the slots a recheck looks at, so a moved or deleted file is flagged like any other', () => {
    draft.setEditSource({ path: '/out/a.png' });
    expect(slotPaths(draft.state)).toContain('/out/a.png');
  });

  it('keeps New\'s prompt and Enhanced text and Edit\'s prompt apart across Kind switches, whichever way they go', () => {
    draft.patch({ prompt: 'a fox', enhancedPrompt: 'an enhanced fox' });
    openKind('image', 'edit', ctx);
    expect(draft.state).toMatchObject({ workflow: 'edit', prompt: '', enhancedPrompt: '' });
    draft.patch({ prompt: 'make the door red' });
    openKind('image', 'new', ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a fox', enhancedPrompt: 'an enhanced fox' });
    openKind('image', 'edit', ctx);
    expect(draft.state).toMatchObject({ workflow: 'edit', prompt: 'make the door red', enhancedPrompt: '' });
  });

  it('opens on the default model that can edit, with that model\'s defaults, when New is on a model that cannot', () => {
    draft.hydrateFromContext(ctx, 'flux-lite');
    draft.patch({ seed: 7, runs: 3 });
    openKind('image', 'edit', ctx);
    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein4b', seed: null, runs: 1, steps: 28 });
  });

  it('offers only the models the context says can edit, and opens on the first of them when the default cannot', () => {
    // The list and the first-visit fallback follow the capability, not model names: with the first Klein out, the next one is offered.
    ctx.image_model_defaults!.klein4b = { ...ctx.image_model_defaults!.klein4b, supports_mask_edit: false };
    expect(modelsOf(ctx, 'edit').map((model) => model.id)).toEqual(['klein']);

    draft.hydrateFromContext(ctx, 'klein4b');
    openKind('image', 'edit', ctx);

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein' });
  });

  it('opens on its own model, never New\'s, even when New\'s can edit', () => {
    openKind('image', 'edit', ctx);
    expect(draft.state.model).toBe('klein4b');
    draft.patch({ model: 'klein' });
    openKind('image', 'new', ctx);
    openKind('image', 'edit', ctx);
    expect(draft.state.model).toBe('klein');
  });

  it('opens on a model that can edit, whatever opens it', () => {
    draft.hydrateFromContext(ctx, 'flux-lite');
    draft.loadFromUrl({ workflow: 'edit' }, ctx);
    draft.hydrateFromContext(ctx, null);
    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein4b' });
  });
});

describe('the Image tab left on Edit', () => {
  let ctx: WorkspaceContext;

  beforeEach(() => {
    ctx = freshEditDraft();
  });

  it('opens on Edit again, with Edit\'s text, after a visit to Video or Audio', () => {
    openKind('image', 'edit', ctx);
    draft.patch({ prompt: 'make the door red' });
    draft.setTab('video');
    draft.patch({ prompt: 'a knight rides' });
    expect(draft.state).toMatchObject({ workflow: 'txt2vid', imageKind: 'edit', prompt: 'a knight rides' });

    draft.setTab('image');

    expect(draft.state).toMatchObject({ workflow: 'edit', imageKind: 'edit', prompt: 'make the door red' });
  });

  it('opens on New when it was left on New', () => {
    draft.patch({ prompt: 'a fox' });
    draft.setTab('video');
    draft.setTab('image');
    expect(draft.state).toMatchObject({ workflow: 'txt2img', imageKind: 'new', prompt: 'a fox' });
  });

  it('is not changed by a video or an audio prefill', () => {
    openKind('image', 'edit', ctx);
    draft.loadFromUrl({ workflow: 'txt2vid', scene: 'a castle' }, ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2vid', imageKind: 'edit' });
    draft.loadFromUrl({ workflow: 'music' }, ctx);
    expect(draft.state.imageKind).toBe('edit');
    draft.setTab('image');
    expect(draft.state.workflow).toBe('edit');

    openKind('image', 'new', ctx);
    draft.loadFromUrl({ workflow: 'txt2vid', scene: 'a castle' }, ctx);
    draft.loadFromUrl({ workflow: 'speech' }, ctx);
    expect(draft.state.imageKind).toBe('new');
    draft.setTab('image');
    expect(draft.state.workflow).toBe('txt2img');
  });

  it('is ended by an image prefill that opens New', () => {
    openKind('image', 'edit', ctx);
    draft.setTab('video');
    draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a fox' }, ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', imageKind: 'new', prompt: 'a fox' });
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
    ctx = freshEditDraft();
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

  it('leaves New\'s prompt and Enhanced text as they were when the prefill comes while the Video tab is open', () => {
    draft.patch({ prompt: 'a fox', enhancedPrompt: 'an enhanced fox', enhancedFrom: { prompt: 'a fox', scene: '', mode: 'image' } });

    draft.setTab('video');
    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/b.png', prompt: 'a window' }, ctx, { replaceSlots: true, exact: true });
    openKind('image', 'new', ctx);

    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a fox', enhancedPrompt: 'an enhanced fox' });
  });

  it('starts an Edit action with no paint and an empty prompt, and keeps New\'s images and text', () => {
    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    draft.patch({ prompt: 'a fox' });
    openKind('image', 'edit', ctx);
    draft.patch({ prompt: 'an old edit' });
    paintedEdit('/out/old.png');

    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/a.png', use_as: 'edit' }, ctx, { replaceSlots: false });

    expect(draft.state).toMatchObject({ workflow: 'edit', editSourcePath: '/out/a.png', prompt: '', ...NO_PAINT, referenceImagePath: '/in/start.png' });
    openKind('image', 'new', ctx);
    expect(draft.state).toMatchObject({ workflow: 'img2img', prompt: 'a fox', referenceImagePath: '/in/start.png' });
  });

  it('never touches the image to edit, its paint or its held files for a Reuse of something else', () => {
    paintedEdit();
    const mask = heldUploads.editMask;
    const epoch = draft.editEpoch;

    draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a fox' }, ctx, { replaceSlots: true, exact: true });
    draft.loadFromUrl({ workflow: 'txt2img', image_path: '/in/start.png', use_as: 'start' }, ctx, { replaceSlots: false });
    draft.loadFromUrl({ workflow: 'txt2vid', scene: 'a castle' }, ctx, { replaceSlots: true, exact: true });

    expect(draft.state).toMatchObject({ editSourcePath: '/out/a.png', editMode: 'paint', editMaskUpload: 'mask.png', editPaintedShare: 0.25, editExtendSide: null, editExtendAmount: 320 });
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
    ctx = freshEditDraft([{ name: 'detail', path: '/loras/detail.safetensors' }, { name: 'soft', path: '/loras/soft.safetensors' }]);
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

    openKind('image', 'edit', ctx);
    draft.patch({ model: 'klein4b', steps: 6, seed: 99, runs: 4, quantize: 4, loraString: 'detail:0.7' });
    const edit = editKept();
    openKind('image', 'new', ctx);

    expect(newKept()).toEqual(before);
    expect(draft.state).toMatchObject({ model: 'klein', steps: 12, seed: 5, runs: 2, loraString: 'soft:0.4', quantize: 8, ratio: '16:9', size: 'l', guidance: 2.5, appliedPreset: 'fast' });

    draft.patch({ model: 'flux-lite', steps: 30, seed: 1, runs: 1, loraString: '', quantize: null });
    openKind('image', 'edit', ctx);

    expect(draft.state).toMatchObject({ workflow: 'edit', ...edit });
  });

  it('keeps New on a model that cannot edit, with its LoRAs, across a visit to Edit', () => {
    draft.hydrateFromContext(ctx, 'flux-lite');
    draft.patch({ loraString: 'soft:0.5', steps: 14 });

    openKind('image', 'edit', ctx);
    expect(draft.state.model).toBe('klein4b');
    expect(draft.state.loraString).toBe('');
    openKind('image', 'new', ctx);

    expect(draft.state).toMatchObject({ workflow: 'txt2img', model: 'flux-lite', loraString: 'soft:0.5', steps: 14 });
  });

  it('keeps the LoRAs each tab remembers for a model family apart', () => {
    draft.hydrateFromContext(ctx, 'zit');
    draft.patch({ loraString: 'soft:0.5' });
    openKind('image', 'edit', ctx);
    draft.hydrateFromContext(ctx, 'klein4b');
    draft.patch({ loraString: 'detail:0.7' });
    draft.hydrateFromContext(ctx, 'klein');
    openKind('image', 'new', ctx);
    // New comes back on its own model with its own LoRAs, and nothing of Edit's family memory reaches it.
    expect(draft.state).toMatchObject({ model: 'zit', loraString: 'soft:0.5' });
    draft.hydrateFromContext(ctx, 'klein4b');
    expect(draft.state.loraString).toBe('');
  });

  it('is put back by a Reset of Edit without touching New\'s ratio, guidance, sharpening, preset or model', () => {
    draft.patch({ ratio: '16:9', size: 'l', guidance: 2.5, postprocessSharpenAmount: 1.1, appliedPreset: 'fast', seed: 3 });
    const before = newKept();
    openKind('image', 'edit', ctx);
    draft.patch({ steps: 6, seed: 99, runs: 4 });

    draft.resetSelections(ctx);

    expect(draft.state).toMatchObject({ workflow: 'edit', steps: 28, seed: null, runs: 1 });
    openKind('image', 'new', ctx);
    expect(newKept()).toEqual(before);
    expect(draft.state.appliedPreset).toBe('fast');
  });

  it('is set by a Reuse of an edit without touching New\'s ratio, guidance, sharpening, preset or model', () => {
    draft.patch({ ratio: '16:9', size: 'l', guidance: 2.5, postprocessSharpenAmount: 1.1, appliedPreset: 'fast', seed: 3 });
    const before = newKept();

    draft.loadFromUrl({ workflow: 'edit', edit_source_path: '/out/a.png', prompt: 'a door', model: 'klein4b', steps: '6', seed: '9', lora: 'detail:0.7' }, ctx, { replaceSlots: true, exact: true });

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein4b', steps: 6, seed: 9, loraString: 'detail:0.7' });
    openKind('image', 'new', ctx);
    expect(newKept()).toEqual(before);
    expect(draft.state.appliedPreset).toBe('fast');
  });

  it('leaves Edit alone when a normal image is reused', () => {
    openKind('image', 'edit', ctx);
    draft.patch({ model: 'klein4b', steps: 6, seed: 99, runs: 4, loraString: 'detail:0.7' });
    const edit = editKept();

    draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a cat', model: 'klein', steps: '8', seed: '3' }, ctx, { replaceSlots: true, exact: true });

    expect(draft.state).toMatchObject({ workflow: 'txt2img', model: 'klein', steps: 8, seed: 3 });
    expect(editKept()).toEqual(edit);
  });

  it('survives a reload, with New\'s kept beside it', () => {
    draft.patch({ steps: 12, seed: 5, loraString: 'soft:0.4' });
    openKind('image', 'edit', ctx);
    draft.patch({ model: 'klein4b', steps: 6, seed: 99, loraString: 'detail:0.7' });
    draft.saveDraft();

    draft.loadDraft();

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein4b', steps: 6, seed: 99, loraString: 'detail:0.7' });
    openKind('image', 'new', null);
    expect(draft.state).toMatchObject({ model: 'klein', steps: 12, seed: 5, loraString: 'soft:0.4' });
  });

  it('forgets a removed LoRA in what each tab kept', () => {
    draft.patch({ loraString: 'soft:0.4' });
    openKind('image', 'edit', ctx);
    draft.removeLora('soft', ctx.loras);
    expect(draft.state.savedSettings.image?.loraString).toBe('');
  });

  it('gives a model that can no longer edit its default when Edit is opened again', () => {
    openKind('image', 'edit', ctx);
    draft.patch({ model: 'flux-lite' });
    openKind('image', 'new', ctx);
    openKind('image', 'edit', ctx);
    expect(draft.state.model).toBe('klein4b');
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
    ctx = freshEditDraft([{ name: 'detail', path: '/loras/detail.safetensors' }, { name: 'own', path: '/loras/own.safetensors' }] as WorkspaceContext['loras']);
    openKind('image', 'edit', ctx);
    draft.patch({ model: 'klein', loraString: 'own:0.5' });
  });

  it('is not taken: Edit keeps its own model and LoRAs, and the rest of the prefill still applies', () => {
    prefill({ workflow: 'edit', model: 'flux-lite', lora: 'detail:0.8', steps: '9' }, 'flux-lite');

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein', loraString: 'own:0.5', steps: 9 });
  });

  it('is taken once the model can edit, with its LoRAs', () => {
    prefill({ workflow: 'edit', model: 'klein4b', lora: 'detail:0.8' }, 'klein4b');

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein4b', loraString: 'detail:0.8' });
  });

  it('falls to the default model that can edit when Edit\'s own model is not offered either', () => {
    draft.patch({ model: 'flux-lite' });

    prefill({ workflow: 'edit', model: 'flux-lite', lora: 'detail:0.8' }, 'flux-lite');

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein4b' });
    expect(draft.state.loraString).not.toBe('detail:0.8');
  });

  it('leaves New free to take the same model and LoRAs', () => {
    openKind('image', 'new', ctx);

    prefill({ workflow: 'txt2img', model: 'flux-lite', lora: 'detail:0.8' }, 'flux-lite');

    expect(draft.state).toMatchObject({ workflow: 'txt2img', model: 'flux-lite', loraString: 'detail:0.8' });
  });
});
