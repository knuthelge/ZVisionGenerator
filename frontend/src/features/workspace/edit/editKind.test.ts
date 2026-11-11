import { beforeEach, describe, expect, it } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { modelsOf } from '$lib/state/workflowModels';
import { assetMenuEntries } from '$lib/components/molecules/assetMenu';
import { viewerActionFor, viewerShortcutsFor } from '$lib/components/molecules/viewerShortcuts';
import type { GalleryAsset, WorkspaceContext } from '$lib/types';

import { makeAsset, makeEditContext } from '../../../test-utils/workspace';
import { makeChipFacts } from '../../../test-utils/chipFacts';
import { workspaceCapabilities } from '../capabilities';
import { imagesView } from '../imagesView';
import { settingsChips } from '../settingsChips';
import { openKind } from '../../../test-utils/draftKind';
import { editFormData } from './editForm';

const editContext = makeEditContext;

describe('the Image tab\'s kind', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('is New until Edit is chosen, and Edit is a workflow of its own', () => {
    expect(draft.state.workflow).toBe('txt2img');
    openKind('image', 'edit', editContext());
    expect(draft.state.workflow).toBe('edit');
    openKind('image', 'new', editContext());
    expect(draft.state.workflow).toBe('txt2img');
  });

  it('is not chosen off the Image tab', () => {
    const context = editContext();
    draft.setTab('video');
    openKind('image', 'edit', context);
    expect(draft.state.workflow).toBe('txt2vid');
  });

  it('shows Kind as the first chip on a phone, and names the open one', () => {
    const context = editContext();
    openKind('image', 'edit', context);
    const facts = (caps: ReturnType<typeof workspaceCapabilities>) => makeChipFacts({ caps, state: draft.state, width: 832, height: 1216, modelLabel: 'Z-Image' });
    const chips = settingsChips(facts(workspaceCapabilities(context, draft.state)));
    expect(chips[0]).toMatchObject({ key: 'kind', value: 'Edit', icon: 'edit' });
    expect(chips.map((chip) => chip.key)).not.toContain('preset');
    expect(chips.map((chip) => chip.key)).not.toContain('size');

    openKind('image', 'new', context);
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
    openKind('image', 'edit', context);
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
    openKind('image', 'edit', context);
    expect(workspaceCapabilities(context, draft.state).promptFileMode).toBe(false);
    openKind('image', 'new', context);
    expect(workspaceCapabilities(context, draft.state).promptFileMode).toBe(true);
  });

  it('is New again, unchanged, when the kind goes back', () => {
    const context = editContext();
    draft.hydrateFromContext(context, 'zit');
    const before = workspaceCapabilities(context, draft.state);
    openKind('image', 'edit', context);
    openKind('image', 'new', context);
    expect(workspaceCapabilities(context, draft.state)).toEqual(before);
  });

  it('does not touch a video workflow', () => {
    const context = editContext();
    openKind('image', 'edit', context);
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

describe('Edit\'s model picker', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('lists Klein whether or not it is downloaded, and Edit settles on it, so the job carries the model and downloads it', () => {
    const context = editContext();
    context.image_models = context.image_models.map((model) => ({ ...model, downloaded: model.id === 'flux-lite' || model.id === 'zit' }));
    draft.hydrateFromContext(context, 'flux-lite');

    openKind('image', 'edit', context);

    expect(modelsOf(context, 'edit').map((model) => model.id)).toEqual(['klein4b', 'klein']);
    expect(draft.state.model).toBe('klein4b');
    expect(editFormData(draft.state, { sourceFile: null, maskFile: null, prompt: 'a red door', lora: '', quantize: true }).get('model')).toBe('klein4b');
  });
});

describe('the viewer and menu action', () => {
  const image = makeAsset({ file_path: '/out/a.png' });
  const video = makeAsset({ file_path: '/out/a.mp4', media_type: 'video' });

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
