import { beforeEach, describe, expect, it } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { canEdit, editParams } from '$lib/state/assetActions';
import { assetMenuEntries } from '$lib/components/molecules/assetMenu';
import { viewerActionFor, viewerShortcutsFor } from '$lib/components/molecules/viewerShortcuts';
import type { GalleryAsset, WorkspaceContext } from '$lib/types';

import { makeContext, makeEditContext } from '../../../test-utils/workspace';
import { workspaceCapabilities } from '../capabilities';
import { imagesView } from '../imagesView';
import { settingsChips } from '../settingsChips';
import { canEditWith, editableModels, editsWholePicture } from '$lib/state/editModels';

const editContext = makeEditContext;

describe('the Image tab\'s kind', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('is New until Edit is chosen, and Edit is a workflow of its own', () => {
    expect(draft.state.workflow).toBe('txt2img');
    draft.setEditing(true, editContext());
    expect(draft.state.workflow).toBe('edit');
    draft.setEditing(false, editContext());
    expect(draft.state.workflow).toBe('txt2img');
  });

  it('is not chosen off the Image tab', () => {
    const context = editContext();
    draft.setTab('video');
    draft.setEditing(true, context);
    expect(draft.state.workflow).toBe('txt2vid');
  });

  it('shows Kind as the first chip on a phone, and names the open one', () => {
    const context = editContext();
    draft.setEditing(true, context);
    const facts = (caps: ReturnType<typeof workspaceCapabilities>) => ({
      caps, state: draft.state, videoKind: null, changed: () => false, width: 832, height: 1216, seconds: null, extendSeconds: null,
      loraNames: [], modelLabel: 'Z-Image', modelDownloaded: true, sourceName: null, images: null, track: null, memoryWarn: false, presetName: null, presetSlug: null, presetChanged: false,
    });
    const chips = settingsChips(facts(workspaceCapabilities(context, draft.state)));
    expect(chips[0]).toMatchObject({ key: 'kind', value: 'Edit', icon: 'edit' });
    expect(chips.map((chip) => chip.key)).not.toContain('preset');
    expect(chips.map((chip) => chip.key)).not.toContain('size');

    draft.setEditing(false, context);
    expect(settingsChips(facts(workspaceCapabilities(context, draft.state)))[0]).toMatchObject({ key: 'kind', value: 'New' });
  });
});

describe('what Edit shows', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('keeps only the prompt, model, Quant, LoRAs, Batch, Steps and Seed', () => {
    const context = editContext();
    draft.hydrateFromContext(context, 'zit');
    const before = workspaceCapabilities(context, draft.state);
    expect([before.isEdit, before.showDimensions, before.showImageUpscale, before.showGuidance, before.showScheduler, before.showPreset, before.showEnhance]).toEqual([false, true, true, true, true, true, false]);
    draft.setEditing(true, context);
    const caps = workspaceCapabilities(context, draft.state);
    expect(caps.isEdit).toBe(true);
    expect([caps.showDimensions, caps.showI2IStrength, caps.showImageUpscale, caps.showUpscaleReview, caps.showPostprocessSharpen, caps.showPostprocessTransparent]).toEqual([false, false, false, false, false, false]);
    expect([caps.showGuidance, caps.showScheduler, caps.showPromptFileControls, caps.showPromptSource, caps.showEnhance, caps.showNegativePrompt, caps.showPreset]).toEqual([false, false, false, false, false, false, false]);
    expect([caps.showRefImage, caps.maxReferences]).toEqual([false, 0]);
    expect([caps.showRuns, caps.showSteps, caps.showSeed, caps.showPromptInline]).toEqual([true, true, true, true]);
  });

  it('takes no prompt file, whatever New last chose, and gives New its file back', () => {
    const context = editContext();
    context.prompt_sources = ['inline', 'file'];
    draft.hydrateFromContext(context, 'zit');
    draft.patch({ promptSource: 'file' });
    expect(workspaceCapabilities(context, draft.state).promptFileMode).toBe(true);
    draft.setEditing(true, context);
    expect(workspaceCapabilities(context, draft.state).promptFileMode).toBe(false);
    draft.setEditing(false, context);
    expect(workspaceCapabilities(context, draft.state).promptFileMode).toBe(true);
  });

  it('is New again, unchanged, when the kind goes back', () => {
    const context = editContext();
    draft.hydrateFromContext(context, 'zit');
    const before = workspaceCapabilities(context, draft.state);
    draft.setEditing(true, context);
    draft.setEditing(false, context);
    expect(workspaceCapabilities(context, draft.state)).toEqual(before);
  });

  it('does not touch a video workflow', () => {
    const context = editContext();
    draft.setEditing(true, context);
    draft.setTab('video');
    expect(workspaceCapabilities(context, draft.state).isEdit).toBe(false);
  });

  it('takes the reason Generate is off from the image to edit', () => {
    const state = { ...draft.state, referenceImagePath: null, referenceImageUpload: null, referenceImagePaths: [] };
    const caps = { showRefImage: false, takesLastFrame: false, maxReferences: 0, showI2IStrength: false };
    const files = { startImage: null, firstFrame: null, lastFrame: null, references: [] };
    const view = imagesView({ state, caps, files, restored: true, edit: { block: 'Choose an image to edit' } });
    expect([view.visible, view.block, view.sent.image]).toEqual([false, 'Choose an image to edit', null]);
    expect(imagesView({ state, caps, files, restored: true }).block).toBeNull();
  });
});

describe('which models can edit', () => {
  it('are the models whose metadata says so (`supports_mask_edit`), in the order the context lists them', () => {
    const context = editContext();
    expect(editableModels(context, context.image_models).map((model) => model.id)).toEqual(['zit', 'klein']);
    expect([canEditWith(context, 'flux-lite'), canEditWith(context, 'zit'), canEditWith(context, 'nope'), canEditWith(null, 'zit')]).toEqual([false, true, false, false]);
  });

  it('say whether they can change the whole picture from the prompt alone: a model that takes reference images can, Z-Image cannot', () => {
    const context = editContext();
    expect([editsWholePicture(context, 'klein'), editsWholePicture(context, 'zit'), editsWholePicture(context, 'flux-lite')]).toEqual([true, false, false]);
  });
});

describe('the viewer and menu action', () => {
  const image = { id: 'a', media_type: 'image', file_path: '/out/a.png', lora: 'detail:0.8', reuse_state: { resolved_model: 'klein' }, filename: 'a.png', url: '/media/a.png' } as unknown as GalleryAsset;
  const video = { ...image, media_type: 'video' } as GalleryAsset;

  it('is offered for an image with a file, and for nothing else', () => {
    expect([canEdit(image), canEdit(video), canEdit({ ...image, file_path: undefined } as GalleryAsset)]).toEqual([true, false, false]);
  });

  it('carries the image as a Use as with the model and LoRAs it was made with, and never the prompt', () => {
    expect(editParams(image)).toEqual({ workflow: 'edit', edit_source_path: '/out/a.png', use_as: 'edit', model: 'klein', lora: 'detail:0.8' });
    expect(Object.keys(editParams(image))).not.toContain('prompt');
  });

  it('lists Edit in the menu of an image, and only when there is a handler', () => {
    const ids = (asset: GalleryAsset, onedit?: () => void) => assetMenuEntries(asset, { onedit }).flatMap((entry) => (entry.kind === 'item' ? [entry.id] : []));
    expect(ids(image, () => {})).toContain('edit');
    expect(ids(image)).not.toContain('edit');
    expect(ids(video, () => {})).not.toContain('edit');
  });

  it('answers M with the Edit action, in the help list too, but not for a take', () => {
    expect(viewerActionFor({ key: 'm', shiftKey: false, metaKey: false, ctrlKey: false, altKey: false })).toBe('edit');
    expect(viewerActionFor({ key: 'm', shiftKey: false, metaKey: true, ctrlKey: false, altKey: false })).toBeNull();
    expect(viewerShortcutsFor('image').some((entry) => entry.keys[0] === 'M')).toBe(true);
    expect(viewerShortcutsFor('audio').some((entry) => entry.keys[0] === 'M')).toBe(false);
  });
});
