import { beforeEach, describe, expect, it } from 'vitest';

import { makeEditContext, makeExtendModelsContext } from '../../test-utils/workspace';
import { draft } from './draft.svelte';
import { sceneSource, sourceInfo } from './extend.fixtures';
import { heldUploads } from './heldUploads.svelte';
import { enhancedStaleReason } from './promptEnhance';
import { memoryUploadStore } from './uploadStore';
import { modelsOf } from './workflowModels';
import type { DraftState, EnhanceSettings, SettingsPreset, WorkspaceContext } from '$lib/types';
import { openKind } from '../../test-utils/draftKind';

const SETTINGS: EnhanceSettings = { style: 'cinematic', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' };
const NEW_TEXT = { scene: 'A castle at dusk.', prompt: 'A knight rides in.', enhancedScene: 'A grey castle.', enhancedPrompt: 'The knight rides in slowly.' };

let ctx: WorkspaceContext;

/** The fields New video holds, read while New video is open. */
const NEW_KEYS: (keyof DraftState)[] = ['scene', 'prompt', 'enhancedScene', 'enhancedPrompt', 'model', 'loraString', 'steps', 'ratio', 'size', 'frameCount', 'audio', 'lowMemory', 'runs', 'seed', 'appliedPreset', 'promptSource', 'promptFilePath', 'enhanceSettingsVideo'];

/** What New video holds in the open draft. */
function newKept(): Record<string, unknown> {
  return Object.fromEntries(NEW_KEYS.map((key) => [key, draft.state[key]]));
}

/** New video on LTX, set up with text, a LoRA, settings, a preset and its own prompt file and Enhancer options. */
function setUpNewVideo(): void {
  draft.hydrateFromContext(ctx, null);
  draft.setTab('video');
  draft.onTabChange(ctx);
  draft.hydrateFromContext(ctx, 'ltx-8');
  draft.patch({
    ...NEW_TEXT, loraString: 'soft:0.5', steps: 5, audio: false, lowMemory: false, frameCount: 97, runs: 3, seed: 11, appliedPreset: 'look',
    promptSource: 'file', promptFilePath: '/prompts/new.yaml', promptFileOptionIds: ['a:0'], enhanceSettingsVideo: { ...SETTINGS, style: 'dreamy' },
  });
}

beforeEach(() => {
  localStorage.clear();
  draft.reset();
  heldUploads.useStore(memoryUploadStore());
  ctx = makeExtendModelsContext();
  ctx.loras = [{ name: 'soft', path: '/loras/soft.safetensors' }, { name: 'detail', path: '/loras/detail.safetensors' }];
});

describe('Extend, apart from New video', () => {
  it('keeps its own text, and leaves New\'s Scene, Prompt and Enhanced text as they were', () => {
    setUpNewVideo();
    openKind('video', 'extend', ctx);
    expect(draft.state).toMatchObject({ workflow: 'extend', scene: '', prompt: '', enhancedScene: '', enhancedPrompt: '' });
    draft.patch({ scene: 'A hall.', prompt: 'She runs.' });

    openKind('video', 'new', ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2vid', ...NEW_TEXT });
    openKind('video', 'extend', ctx);
    expect(draft.state).toMatchObject({ scene: 'A hall.', prompt: 'She runs.' });
  });

  it('opens on the default model that can extend, with that model\'s defaults, and offers only models that can', () => {
    setUpNewVideo();
    draft.hydrateFromContext(ctx, 'wan');

    openKind('video', 'extend', ctx);

    expect(modelsOf(ctx, 'extend').map((model) => model.id)).toEqual(['ltx-8', 'h3']);
    expect(draft.state).toMatchObject({ workflow: 'extend', model: 'ltx-8', steps: 8, runs: 1, seed: null, loraString: '', appliedPreset: null });
    openKind('video', 'new', ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2vid', model: 'wan', steps: 20 });
  });

  it('keeps New\'s model, LoRAs, settings and preset when a clip is chosen, and brings the clip\'s text, model and LoRAs to Extend only', () => {
    setUpNewVideo();
    const before = newKept();
    openKind('video', 'extend', ctx);

    draft.setExtendSourcePath('/clips/walk.mp4');
    draft.applyExtendSource(sceneSource({ model: 'h3', lora: 'detail:0.7' }), ctx, { replaceText: true });

    expect(draft.state).toMatchObject({ workflow: 'extend', scene: 'A woman on a rainy street.', model: 'h3', loraString: 'detail:0.7', steps: 12 });
    openKind('video', 'new', ctx);
    expect(newKept()).toEqual(before);
    expect(draft.state).toMatchObject({ model: 'ltx-8', loraString: 'soft:0.5', appliedPreset: 'look' });
  });

  it('leaves New\'s settings alone for the Gallery\'s Extend link, and gives Extend the clip\'s model', () => {
    setUpNewVideo();
    const before = newKept();
    const params = { workflow: 'extend', source_video: '/clips/walk.mp4', model: 'h3', lora: 'detail:0.7' };

    // The steps of the Workspace's prefill: the workflow first, then the model (without keeping settings), then the params.
    draft.loadFromUrl({ workflow: 'extend' }, ctx);
    draft.hydrateFromContext(ctx, 'h3', { keepSettings: false });
    draft.loadFromUrl(params, ctx, { replaceSlots: true });

    expect(draft.state).toMatchObject({ workflow: 'extend', model: 'h3', extendSourcePath: '/clips/walk.mp4' });
    openKind('video', 'new', ctx);
    expect(newKept()).toEqual(before);
  });

  it('is put back by a Reset without touching New, and New\'s Reset leaves Extend alone', () => {
    setUpNewVideo();
    const before = newKept();
    openKind('video', 'extend', ctx);
    draft.patch({ steps: 3, runs: 4, seed: 9, appliedPreset: 'mine', enhanceSettingsVideo: { ...SETTINGS, style: 'noir' } });

    draft.resetSelections(ctx);
    expect(draft.state).toMatchObject({ workflow: 'extend', steps: 8, runs: 1, seed: null, appliedPreset: null, enhanceSettingsVideo: null });
    openKind('video', 'new', ctx);
    expect(newKept()).toEqual(before);

    draft.resetSelections(ctx);
    openKind('video', 'extend', ctx);
    expect(draft.state).toMatchObject({ steps: 8, runs: 1, seed: null });
    draft.patch({ steps: 6, runs: 2 });
    openKind('video', 'new', ctx);
    draft.resetSelections(ctx);
    openKind('video', 'extend', ctx);
    expect(draft.state).toMatchObject({ steps: 6, runs: 2 });
  });

  it('is filled by a Reuse or a Load settings of an Extend take, and New is not', () => {
    setUpNewVideo();
    const before = newKept();

    draft.loadFromUrl({ workflow: 'extend' }, ctx);
    draft.hydrateFromContext(ctx, 'h3');
    draft.loadFromUrl({ workflow: 'extend', source_video: '/clips/walk.mp4', model: 'h3', lora: 'detail:0.7', steps: '6', seed: '4', scene: 's', prompt: 'p' }, ctx, { replaceSlots: true, exact: true });

    expect(draft.state).toMatchObject({ workflow: 'extend', model: 'h3', loraString: 'detail:0.7', steps: 6, seed: 4, scene: 's', prompt: 'p' });
    openKind('video', 'new', ctx);
    expect(newKept()).toEqual(before);
  });

  it('takes a preset into Extend only', () => {
    setUpNewVideo();
    const before = newKept();
    openKind('video', 'extend', ctx);

    draft.applyPreset(ctx, { slug: 'fast', name: 'Fast', mode: 'video', model: 'h3', settings: { steps: 4, frames: 25, ratio: '16:9', size: 'm' } } as SettingsPreset);

    expect(draft.state).toMatchObject({ workflow: 'extend', model: 'h3', steps: 4, appliedPreset: 'fast' });
    openKind('video', 'new', ctx);
    expect(newKept()).toEqual(before);
  });

  it('shows no stale notice on New for an Enhanced pair written in Extend', () => {
    setUpNewVideo();
    openKind('video', 'extend', ctx);
    draft.setExtendSourcePath('/clips/walk.mp4');
    draft.patch({ extendFrames: 48, prompt: 'She waves.', enhancedPrompt: 'She waves slowly.', enhancedFrom: { prompt: 'She waves.', scene: '', mode: 'video', sourceVideoPath: '/clips/walk.mp4', extendFrames: 48 } });
    expect(enhancedStaleReason(draft.state)).toBeNull();

    openKind('video', 'new', ctx);

    expect(draft.state).toMatchObject(NEW_TEXT);
    expect(enhancedStaleReason(draft.state)).toBeNull();
  });

  it('keeps Batch and Seed of each tab and kind apart', () => {
    draft.hydrateFromContext(ctx, null);
    draft.patch({ runs: 2, seed: 5 });
    draft.setTab('video');
    draft.onTabChange(ctx);
    expect(draft.state).toMatchObject({ runs: 1, seed: null });
    draft.patch({ runs: 4, seed: 9 });
    openKind('video', 'extend', ctx);
    expect(draft.state).toMatchObject({ runs: 1, seed: null });
    draft.patch({ runs: 7, seed: 77 });

    openKind('video', 'new', ctx);
    expect(draft.state).toMatchObject({ runs: 4, seed: 9 });
    draft.setTab('image');
    draft.onTabChange(ctx);
    expect(draft.state).toMatchObject({ runs: 2, seed: 5 });
    draft.setTab('video');
    openKind('video', 'extend', ctx);
    expect(draft.state).toMatchObject({ runs: 7, seed: 77 });
  });

  it('keeps its prompt source, prompts file and Enhancer options apart from New\'s', () => {
    setUpNewVideo();
    openKind('video', 'extend', ctx);
    expect(draft.state).toMatchObject({ promptSource: 'inline', promptFilePath: null, promptFileOptionIds: [], enhanceSettingsVideo: null });
    draft.patch({ promptSource: 'file', promptFilePath: '/prompts/extend.yaml', enhanceSettingsVideo: { ...SETTINGS, style: 'noir' } });

    openKind('video', 'new', ctx);
    expect(draft.state).toMatchObject({ promptSource: 'file', promptFilePath: '/prompts/new.yaml', promptFileOptionIds: ['a:0'], enhanceSettingsVideo: { style: 'dreamy' } });
    openKind('video', 'extend', ctx);
    draft.setTab('image');
    expect(draft.state).toMatchObject({ promptSource: 'inline', promptFilePath: null });
    draft.setTab('video');
    expect(draft.state).toMatchObject({ workflow: 'extend', promptFilePath: '/prompts/extend.yaml', enhanceSettingsVideo: { style: 'noir' } });
  });

  it('gives a machine that cannot extend New\'s own text back', () => {
    setUpNewVideo();
    openKind('video', 'extend', ctx);
    draft.patch({ scene: 'A hall.', prompt: 'She runs.' });
    const without = { ...ctx, workflow_contract: { ...ctx.workflow_contract, values: ctx.workflow_contract.values.filter((workflow) => workflow !== 'extend') } };

    draft.hydrateFromContext(without, null, { keepSettings: true });

    expect(draft.state).toMatchObject({ workflow: 'txt2vid', videoKind: 'new', ...NEW_TEXT, model: 'ltx-8', loraString: 'soft:0.5' });
  });
});

describe('Extend and Edit, each with a Batch and Seed of its own', () => {
  it('keep them when the Image tab left on Edit and the Video tab left on Extend are opened from one another', () => {
    const editable = makeEditContext();
    const both: WorkspaceContext = { ...ctx, image_models: editable.image_models, image_model_defaults: editable.image_model_defaults };
    draft.hydrateFromContext(both, 'klein');
    draft.patch({ runs: 2, seed: 5 });
    openKind('image', 'edit', both);
    draft.patch({ runs: 3, seed: 33 });
    draft.setTab('video');
    draft.onTabChange(both);
    openKind('video', 'extend', both);
    draft.patch({ runs: 7, seed: 77 });

    draft.setTab('image');
    expect(draft.state).toMatchObject({ workflow: 'edit', runs: 3, seed: 33 });
    draft.setTab('video');
    expect(draft.state).toMatchObject({ workflow: 'extend', runs: 7, seed: 77 });
    draft.setTab('image');
    openKind('image', 'new', both);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', runs: 2, seed: 5 });
  });
});
