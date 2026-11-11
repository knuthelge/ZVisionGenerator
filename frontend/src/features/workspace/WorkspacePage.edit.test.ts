// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { trackLengths } from '$lib/state/audioTrackLength.svelte';
import { editParams } from '$lib/state/assetActions';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { requestJobSettings } from '$lib/state/pendingJobSettings.svelte';
import { requestPromptRun } from '$lib/state/pendingPromptRun';
import { workspaceMemory } from '$lib/state/pageMemory';
import type { GalleryAsset, JobContext, TabText, WorkspaceContext } from '$lib/types';

import { makeEditContext } from '../../test-utils/workspace';
import { createHost, mockClipboard, mountWorkspacePage, resetWorkspaceTest, settle, submitForm } from '../../test-utils/workspaceHarness';
import { layoutProbe, workspaceApiMocks } from '../../test-utils/workspaceMocks';
import { imageEdit } from './edit/editStore.svelte';
import { sourceOfPath } from './edit/editSource';
import { openKind } from '../../test-utils/draftKind';

vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  const { layoutProbe } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    layout: {
      get mode() { return layoutProbe.compact ? 'compact' : 'wide'; },
      get compact() { return layoutProbe.compact; },
      get coarse() { return layoutProbe.compact; },
    },
  };
});

vi.mock('$lib/api/config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/config')>();
  const { configApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, updateConfig: configApiMocks.updateConfig };
});

vi.mock('$lib/api/extend', async () => {
  const { extendApiMocks } = await import('../../test-utils/workspaceMocks');
  return { inspectExtendSource: extendApiMocks.inspectExtendSource };
});

vi.mock('$lib/api/videoMemory', async () => {
  const { videoMemoryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { checkVideoMemoryFit: videoMemoryApiMocks.checkVideoMemoryFit };
});

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  const { workspaceApiMocks: mocks } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    getWorkspaceContext: mocks.getWorkspaceContext,
    getWorkspaceCoreContext: mocks.getWorkspaceCoreContext,
    submitGenerate: mocks.submitGenerate,
    planGenerate: mocks.planGenerate,
    getJobSnapshot: mocks.getJobSnapshot,
    parseUrlPrefill: mocks.parseUrlPrefill,
    checkReferencePaths: mocks.checkReferencePaths,
    listJobs: mocks.listJobs,
    clearQueue: mocks.clearQueue,
    cancelJob: mocks.cancelJob,
    skipFirstRun: mocks.skipFirstRun,
  };
});

vi.mock('$lib/api/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/gallery')>();
  const { galleryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAsset: galleryApiMocks.deleteAsset };
});

vi.mock('$lib/api/promptEnhance', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/promptEnhance')>();
  const { enhanceApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, enhancePrompt: enhanceApiMocks.enhancePrompt };
});

const trackApi = vi.hoisted(() => ({ inspectAudioTrack: vi.fn(), readFileLength: vi.fn() }));
vi.mock('$lib/api/audioTrack', () => ({ inspectAudioTrack: trackApi.inspectAudioTrack }));
vi.mock('$lib/state/audioLength', () => ({ readFileLength: trackApi.readFileLength }));

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

vi.mock('$lib/api/presets', async () => (await import('../../test-utils/workspaceMocks')).presetApiMocks);

vi.mock('$lib/api/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/models')>();
  const { modelApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getDownloads: modelApiMocks.getDownloads };
});

vi.mock('$lib/api/promptFiles', async () => {
  const { promptFileApiMocks } = await import('../../test-utils/workspaceMocks');
  return {
    inspectPromptFile: promptFileApiMocks.inspectPromptFile,
    writePromptFile: promptFileApiMocks.writePromptFile,
    appendPromptEntry: promptFileApiMocks.appendPromptEntry,
    createPromptFile: promptFileApiMocks.createPromptFile,
  };
});

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
  trackLengths.reset();
  trackApi.inspectAudioTrack.mockReset().mockRejectedValue(new Error('unreadable'));
  trackApi.readFileLength.mockReset().mockResolvedValue(null);
  layoutProbe.compact = false;
  workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1' } as JobContext);
  window.location.hash = '#/workspace';
  imageEdit.setSource(null);
  URL.createObjectURL = vi.fn(() => 'blob:edit');
  URL.revokeObjectURL = vi.fn();
});

afterEach(async () => {
  vi.useRealTimers();
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
  layoutProbe.compact = false;
});

const png = (name: string): File => new File(['png'], name, { type: 'image/png' });
const kindButton = (kind: 'new' | 'edit') => target.querySelector(`[data-testid="kind-control"] [data-value="${kind}"]`) as HTMLButtonElement;
const submitButton = () => target.querySelector('#ws-submit') as HTMLButtonElement;
const reason = () => target.querySelector('#ws-generate-reason')?.textContent?.trim() ?? null;
const summary = () => target.querySelector('[data-testid="edit-summary"]')?.textContent?.trim().replace(/\s+/g, ' ') ?? null;
const fieldsOf = (form: FormData): string[] => [...new Set(form.keys())].sort();

/** The text a tab keeps while it is not open. */
const savedText = (prompt: string): TabText => ({
  prompt, scene: '', negativePrompt: '', jsonPromptEnabled: false, jsonPrompt: '', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedHistory: [], enhancedScratched: null,
});

/**
 * Mount the Workspace on a machine that can edit, on Klein, with Edit open (or New, with Edit's text saved) and an image to edit.
 * New's prompt is "a fox in snow" and Edit's "a red door", whichever is open.
 */
async function mountEdit(options: { model?: string; source?: boolean; kind?: 'new' | 'edit' } = {}): Promise<WorkspaceContext> {
  const context = makeEditContext();
  draft.reset();
  const model = options.model ?? 'klein';
  if ((options.kind ?? 'edit') === 'edit') draft.patch({ model, workflow: 'edit', prompt: 'a red door', steps: 4, savedText: { image: savedText('a fox in snow') } });
  else draft.patch({ model, prompt: 'a fox in snow', steps: 4, savedText: { edit: savedText('a red door') } });
  if (options.source !== false) draft.setEditSource({ path: '/out/a.png' });
  app = await mountWorkspacePage(context, target);
  return context;
}

const asset = (overrides: Partial<GalleryAsset> = {}): GalleryAsset => ({
  id: 'out/a.png', url: '/media/out/a.png', thumbnail_url: '/thumbs/a.png', filename: 'a.png', file_path: '/out/a.png', media_type: 'image', width: 800, height: 600, prompt: 'a fox',
  has_reusable_config: true, reuse_state: { resolved_model: 'klein', requested_model: 'klein' }, lora: '', ...overrides,
} as unknown as GalleryAsset);

describe('what Generate sends in Edit', () => {
  it('sends the fields of a whole-picture edit and nothing of New, on Klein with nothing painted', async () => {
    await mountEdit();
    expect(draft.state.workflow).toBe('edit');

    const form = await submitForm(target);

    expect(Object.fromEntries(form)).toEqual({ mode: 'image', workflow: 'edit', edit_source_path: '/out/a.png', prompt: 'a red door', model: 'klein', lora: '', steps: '4', runs: '1' });
  });

  it('sends Edit\'s own prompt, never the Enhanced text made in New', async () => {
    await mountEdit();
    draft.patch({ savedText: { image: { ...savedText('a fox in snow'), enhancedPrompt: 'an enhanced fox', enhancedFrom: { prompt: 'a fox in snow', scene: '', mode: 'image' } } } });
    await settle();

    const form = await submitForm(target);

    expect(form.get('prompt')).toBe('a red door');
    expect(form.has('prompt_input')).toBe(false);
  });

  it('keeps each kind\'s prompt when the Kind switches, in both directions', async () => {
    await mountEdit({ kind: 'new' });
    draft.patch({ enhancedPrompt: 'an enhanced fox' });
    const box = () => target.querySelector('#ws-prompt') as HTMLTextAreaElement;
    expect(box().value).toBe('a fox in snow');

    flushSync(() => kindButton('edit').click());
    await settle();
    expect(box().value).toBe('a red door');
    draft.patch({ prompt: 'a blue door' });

    flushSync(() => kindButton('new').click());
    await settle();
    expect(box().value).toBe('a fox in snow');
    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a fox in snow', enhancedPrompt: 'an enhanced fox' });

    flushSync(() => kindButton('edit').click());
    await settle();
    expect(box().value).toBe('a blue door');
    expect(draft.state).toMatchObject({ workflow: 'edit', enhancedPrompt: '' });
  });

  it('clears Edit\'s prompt with the Clear button and leaves New\'s text as it is', async () => {
    await mountEdit();
    (target.querySelector('[data-testid="clear-box"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state.prompt).toBe('');
    expect(draft.state.savedText.image?.prompt).toBe('a fox in snow');
  });

  it('blocks Generate while Edit\'s prompt is empty, even with text in New', async () => {
    await mountEdit();
    draft.patch({ prompt: '' });
    await settle();

    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).required).toBe(true);
    expect(draft.state.savedText.image?.prompt).toBe('a fox in snow');
  });

  it('sends the painted mask with its flag', async () => {
    await mountEdit();
    const mask = png('mask.png');
    draft.commitEditPaint({ mask, share: 0.2, extension: null, amount: 256 });
    await settle();

    const form = await submitForm(target);

    expect(fieldsOf(form)).toEqual(['edit_mask_file', 'edit_painted', 'edit_source_path', 'lora', 'mode', 'model', 'prompt', 'runs', 'steps', 'workflow']);
    expect((form.get('edit_mask_file') as File).name).toBe('mask.png');
    expect(form.get('edit_painted')).toBe('true');
  });

  it('sends an extension without a mask', async () => {
    await mountEdit();
    draft.commitEditPaint({ mask: null, share: 0, extension: { side: 'top', amount: 192 }, amount: 192 });
    await settle();

    const form = await submitForm(target);

    expect([form.get('edit_extend_side'), form.get('edit_extend_amount'), form.has('edit_mask_file'), form.has('edit_painted')]).toEqual(['top', '192', false, false]);
  });

  it('sends paint and an extension both', async () => {
    await mountEdit();
    draft.commitEditPaint({ mask: png('mask.png'), share: 0.2, extension: { side: 'left', amount: 128 }, amount: 128 });
    await settle();

    const form = await submitForm(target);

    expect(fieldsOf(form)).toEqual(expect.arrayContaining(['edit_mask_file', 'edit_painted', 'edit_extend_side', 'edit_extend_amount']));
  });

  it('sends an uploaded picture as a file', async () => {
    await mountEdit({ source: false });
    const upload = png('photo.png');
    draft.setEditSource({ file: upload });
    await settle();

    const form = await submitForm(target);

    expect(form.get('edit_source_file')).toBe(upload);
    expect(form.has('edit_source_path')).toBe(false);
  });

  it('sends Video\'s own fields while the Edit kind is left on and the Video tab is open', async () => {
    await mountEdit();
    flushSync(() => draft.setTab('video'));
    await settle();

    const form = await submitForm(target);

    expect(form.get('mode')).toBe('video');
    expect(form.get('workflow')).not.toBe('edit');
    expect(fieldsOf(form).some((key) => key.startsWith('edit_'))).toBe(false);
  });
});

describe('why Generate is off in Edit', () => {
  it('asks for an image first', async () => {
    await mountEdit({ source: false });
    expect(submitButton().disabled).toBe(true);
    expect(reason()).toBe('Choose an image to edit');
  });

  it('asks Z-Image for paint, and Klein for nothing', async () => {
    await mountEdit({ model: 'zit' });
    expect(submitButton().disabled).toBe(true);
    expect(reason()).toBe('Paint the area to change');
    await unmount(app!);
    app = null;
    workspaceMemory.forget();

    await mountEdit({ model: 'klein' });
    expect(submitButton().disabled).toBe(false);
  });

  it('refuses Klein with nothing painted on a cut-out once that is known, and lets a plain image through', async () => {
    await mountEdit({ source: false });
    imageEdit.setSource({ ...sourceOfPath('/out/a.png'), cutout: true });
    draft.setEditSource({ path: '/out/a.png' });
    await settle();
    expect(reason()).toBe('Paint the area to change');

    draft.commitEditPaint({ mask: png('mask.png'), share: 0.1, extension: null, amount: 256 });
    await settle();
    expect(submitButton().disabled).toBe(false);
  });

  it('leaves Klein on while the transparency of the picture is unknown: the server refuses', async () => {
    await mountEdit();
    imageEdit.setSource({ ...sourceOfPath('/out/a.png'), cutout: null });
    await settle();
    expect(submitButton().disabled).toBe(false);
  });

  it('says to choose the picture again when an upload did not come back from the reload', async () => {
    const context = makeEditContext();
    draft.reset();
    draft.patch({ model: 'klein', workflow: 'edit', prompt: 'x', editSourceUpload: 'photo.png' });
    app = await mountWorkspacePage(context, target);
    expect(reason()).toBe('Choose this image again.');
  });
});

describe('the Edit kind', () => {
  it('offers New and Edit, and sends nothing of Edit from New', async () => {
    await mountEdit({ kind: 'new' });
    expect(kindButton('new')).not.toBeNull();
    expect(kindButton('edit')).not.toBeNull();
    expect((await submitForm(target)).get('workflow')).not.toBe('edit');

    kindButton('edit').click();
    await settle();
    expect(draft.state.workflow).toBe('edit');
    expect(target.querySelector('[data-testid="edit-source-picker"], [data-testid="edit-open-painter"]')).not.toBeNull();
  });

  it('moves a model that cannot edit to one that can', async () => {
    await mountEdit({ model: 'flux-lite', kind: 'new' });
    kindButton('edit').click();
    await settle();
    expect(draft.state.model).toBe('zit');
  });

  it('shows the painted share and the extension the draft holds, and the Reset puts them back', async () => {
    await mountEdit();
    draft.commitEditPaint({ mask: png('mask.png'), share: 0.18, extension: { side: 'top', amount: 256 }, amount: 256 });
    await settle();
    expect(summary()).toContain('Painted area: 18% of the image');
    expect(summary()).toContain('Extends top by 256 px');

    (target.querySelector('#ws-reset') as HTMLButtonElement).click();
    await settle();

    expect(draft.state).toMatchObject({ editSourcePath: null, editMaskUpload: null, editExtendSide: null });
    expect(heldUploads.editMask).toBeNull();
    expect(target.querySelector('[data-testid="edit-source-picker"]')).not.toBeNull();
  });
});

/** The state of a draft with an edit in progress: Edit's image, paint and extension, held mask and the painter's history, apart from New's text. */
function paintedEditDraft(): void {
  draft.reset();
  draft.patch({ model: 'klein', prompt: 'a fox in snow', enhancedPrompt: 'an enhanced fox', enhancedFrom: { prompt: 'a fox in snow', scene: '', mode: 'image' }, steps: 4, savedText: { edit: savedText('a red door') } });
  draft.setEditSource({ path: '/out/old.png' });
  draft.commitEditPaint({ mask: png('mask.png'), share: 0.3, extension: { side: 'left', amount: 128 }, amount: 128 });
  imageEdit.setSource(sourceOfPath('/out/old.png'));
  imageEdit.addStroke({ erase: false, size: 100, points: [100, 100] });
}

/** The Edit slot as the draft and the held files keep it. */
function editSlot() {
  const { editSourcePath, editSourceUpload, editMaskUpload, editPaintedShare, editExtendSide, editExtendAmount, editRepaint } = draft.state;
  return { editSourcePath, editSourceUpload, editMaskUpload, editPaintedShare, editExtendSide, editExtendAmount, editRepaint, mask: heldUploads.editMask?.name ?? null, canUndo: imageEdit.canUndo, epoch: draft.editEpoch };
}

/** Mount the Workspace with a prefill the address carries, as a Reuse, a Retry or an Edit action from the Gallery does. */
async function mountWith(params: Record<string, string>): Promise<void> {
  workspaceApiMocks.parseUrlPrefill.mockReturnValue(params);
  app = await mountWorkspacePage(makeEditContext(), target);
}

describe('Edit from a viewer', () => {
  it('opens Edit with that image and no paint, clearing a mask, an extension and the edit prompt from an earlier image, and leaving New\'s text alone', async () => {
    paintedEditDraft();

    await mountWith(editParams(asset()));

    expect(draft.state).toMatchObject({
      workflow: 'edit', editSourcePath: '/out/a.png', editMaskUpload: null, editPaintedShare: 0, editExtendSide: null, editRepaint: false, prompt: '', enhancedPrompt: '', model: 'klein',
    });
    expect(draft.state.savedText.image).toMatchObject({ prompt: 'a fox in snow', enhancedPrompt: 'an enhanced fox' });
    expect(heldUploads.editMask).toBeNull();
    expect(imageEdit.source?.path).toBe('/out/a.png');
  });

  it('keeps the model it has when the image\'s own cannot edit, and drops that model\'s LoRAs', async () => {
    draft.reset();
    draft.patch({ model: 'klein', workflow: 'edit', savedSettings: { image: { model: 'flux-lite', steps: 9 } } });

    await mountWith(editParams(asset({ lora: 'detail:0.8', reuse_state: { resolved_model: 'flux-lite', requested_model: 'flux-lite' } as GalleryAsset['reuse_state'] })));

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein', loraString: '', editSourcePath: '/out/a.png' });
    expect(draft.state.savedSettings.image).toMatchObject({ model: 'flux-lite', steps: 9 });
  });

  it.each([
    ['Video', 'video'],
    ['Audio', 'audio'],
  ] as const)('from the %s tab leaves that tab\'s model and LoRAs, and opens Edit on a model that can edit', async (_name, tab) => {
    const context = makeEditContext();
    context.loras = [{ name: 'detail', path: '/loras/detail.safetensors' }];
    context.audio_models = [{ id: 'qwen3_tts', label: 'Qwen3 TTS', kind: 'speech', limits: { max_voice_seconds: 20 } } as unknown as NonNullable<WorkspaceContext['audio_models']>[number]];
    context.workflow_contract.values = [...context.workflow_contract.values, 'speech'];
    draft.reset();
    // New sits on a model that cannot edit; the other tab has a model and LoRAs of its own.
    draft.patch({ model: 'flux-lite', prompt: 'a fox' });
    draft.setTab(tab);
    draft.onTabChange(context);
    if (tab === 'video') draft.patch({ loraString: 'detail:0.6' });
    const { model, loraString, audioModels } = draft.state;
    expect(draft.state.workflow).toBe(tab === 'audio' ? 'speech' : 'txt2vid');
    expect(model).not.toBe('flux-lite');
    workspaceApiMocks.parseUrlPrefill.mockReturnValue(editParams(asset({ reuse_state: { resolved_model: 'flux-lite', requested_model: 'flux-lite' } as GalleryAsset['reuse_state'] })));
    app = await mountWorkspacePage(context, target);

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'zit', editSourcePath: '/out/a.png' });
    flushSync(() => draft.setTab(tab));
    draft.onTabChange(context);
    expect(draft.state).toMatchObject({ model, loraString, audioModels });
  });
});

describe('the prompt file, which belongs to New', () => {
  it('shows the Triggers menu in Edit while New is on a prompt file, and no prompt-file controls', async () => {
    const context = makeEditContext();
    context.loras = [{ name: 'detail', path: '/loras/detail.safetensors', trigger: 'dtl1' } as WorkspaceContext['loras'][number]];
    draft.reset();
    draft.patch({ model: 'klein', workflow: 'edit', promptSource: 'file', promptFilePath: '/prompts/a.yaml', promptFileOptionIds: ['a:0'], loraString: 'detail:0.8', prompt: 'a red door' });
    app = await mountWorkspacePage(context, target);

    expect(target.querySelector('#ws-triggers')).not.toBeNull();
    expect(target.querySelector('#ws-prompt')).not.toBeNull();
    expect(target.querySelector('[data-testid="prompt-file-controls"], #ws-prompt-file')).toBeNull();
    expect(draft.state).toMatchObject({ promptSource: 'file', promptFilePath: '/prompts/a.yaml' });
  });

  it('is not what Generate waits for in Edit', async () => {
    await mountEdit();
    draft.patch({ promptSource: 'file', promptFilePath: null, promptFileOptionIds: [] });
    await settle();

    const form = await submitForm(target);

    expect(form.get('prompt')).toBe('a red door');
  });
});

describe('Edit from the Gallery', () => {
  it('keeps Edit\'s settings when the image was made with the model Edit is on, and changes them with the model', async () => {
    draft.reset();
    draft.patch({ model: 'klein', workflow: 'edit', steps: 9, seed: 21 });
    workspaceApiMocks.parseUrlPrefill.mockReturnValue(editParams(asset()));
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'klein', steps: 9, seed: 21, editSourcePath: '/out/a.png' });
    await unmount(app);
    app = null;

    draft.reset();
    draft.patch({ model: 'klein', workflow: 'edit', steps: 9, seed: 21 });
    workspaceApiMocks.parseUrlPrefill.mockReturnValue(editParams(asset({ reuse_state: { resolved_model: 'zit', requested_model: 'zit' } as GalleryAsset['reuse_state'] })));
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(draft.state).toMatchObject({ workflow: 'edit', model: 'zit', steps: 28, editSourcePath: '/out/a.png' });
  });
});

describe('Reuse and Retry of an edit', () => {
  const reuse = (params: Record<string, string>) => {
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'edit', reuse: 'true', model: 'klein', prompt: 'a red door', steps: '4', seed: '9', lora: '', quantize: '', ...params });
  };

  it('lands in Edit with the picture, the text and the settings, and asks for the paint again', async () => {
    reuse({ edit_source_path: '/out/a.png', edit_repaint: 'true' });
    draft.reset();
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(draft.state).toMatchObject({ workflow: 'edit', editSourcePath: '/out/a.png', editRepaint: true, prompt: 'a red door', steps: 4, seed: 9, model: 'klein' });
    expect(kindButton('edit').getAttribute('aria-pressed')).toBe('true');
    expect(reason()).toBe('Paint the area again');
    expect(submitButton().disabled).toBe(true);
  });

  it('turns Generate on once paint is added', async () => {
    reuse({ edit_source_path: '/out/a.png', edit_repaint: 'true' });
    draft.reset();
    app = await mountWorkspacePage(makeEditContext(), target);

    draft.commitEditPaint({ mask: png('mask.png'), share: 0.1, extension: null, amount: 256 });
    await settle();

    expect(draft.state.editRepaint).toBe(false);
    expect(submitButton().disabled).toBe(false);
  });

  it('restores a saved extension, which stands on its own', async () => {
    reuse({ edit_source_path: '/out/a.png', edit_extend_side: 'bottom', edit_extend_amount: '128' });
    draft.reset();
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(draft.state).toMatchObject({ editExtendSide: 'bottom', editExtendAmount: 128, editRepaint: false });
    expect(summary()).toContain('Extends bottom by 128 px');
    expect(submitButton().disabled).toBe(false);
  });

  it('keeps the text and settings when the picture is gone, and empties the slot of the picture that was there', async () => {
    paintedEditDraft();
    reuse({ edit_source_path: '', edit_repaint: 'true' });
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(draft.state).toMatchObject({ workflow: 'edit', editSourcePath: null, editSourceUpload: null, editMaskUpload: null, editRepaint: true, prompt: 'a red door', steps: 4 });
    expect(heldUploads.editMask).toBeNull();
    expect(reason()).toBe('Choose an image to edit');
  });

  it('fills Edit\'s prompt and leaves New\'s prompt and Enhanced text untouched, from New and from the Video tab', async () => {
    reuse({ edit_source_path: '/out/a.png', edit_repaint: 'true' });
    paintedEditDraft();
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(draft.state).toMatchObject({ workflow: 'edit', prompt: 'a red door', enhancedPrompt: '' });
    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('a red door');
    expect(reason()).toBe('Paint the area again');
    flushSync(() => kindButton('new').click());
    await settle();
    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a fox in snow', enhancedPrompt: 'an enhanced fox' });
  });

  it('does the same for a Retry from the Job log and for Load settings, which carry the prompt in the job\'s settings', async () => {
    const settings = { mode: 'image', workflow: 'edit', model: 'klein', prompt: 'a red door', edit_source_path: '/out/a.png', edit_painted: 'true', steps: '4', seed: '9', runs: '1', lora: '' };
    paintedEditDraft();
    app = await mountWorkspacePage(makeEditContext(), target);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a fox in snow' });

    requestJobSettings(settings);
    await settle();

    expect(draft.state).toMatchObject({ workflow: 'edit', prompt: 'a red door', enhancedPrompt: '', editSourcePath: '/out/a.png', editRepaint: true, steps: 4, seed: 9 });
    expect(draft.state.savedText.image).toMatchObject({ prompt: 'a fox in snow', enhancedPrompt: 'an enhanced fox' });
    expect(reason()).toBe('Paint the area again');
  });

  it('is the Edit slot\'s alone: a Reuse or Retry of an edit replaces its image and paint, and a Reuse of anything else never does', async () => {
    paintedEditDraft();
    const before = editSlot();
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'txt2img', reuse: 'true', model: 'zit', prompt: 'a cat', steps: '8', seed: '3', lora: '', quantize: '', ratio: '2:3', size: 'm' });
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a cat', model: 'zit', steps: 8 });
    expect(editSlot()).toEqual(before);
  });

  it.each([
    ['txt2img', { image_path: '/in/start.png' }],
    ['txt2vid', { scene: 'a castle' }],
    ['speech', { speech_text: 'hello' }],
  ])('leaves the Edit slot alone for a %s prefill, and the Image tab\'s kind with it', async (workflow, extra) => {
    paintedEditDraft();
    openKind('image', 'edit', makeEditContext());
    draft.setTab('video');
    const before = editSlot();
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow, reuse: 'true', ...extra });
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(editSlot()).toEqual(before);
    if (workflow !== 'txt2img') {
      expect(draft.state.imageKind).toBe('edit');
      flushSync(() => draft.setTab('image'));
      expect(draft.state.workflow).toBe('edit');
    }
  });
});

describe('Generate this one, from the Prompts page', () => {
  it('moves an Image tab left on Edit to New and queues the chosen prompt as a text-to-image run, keeping Edit\'s image and paint', async () => {
    requestPromptRun({ path: '/server/prompts.yaml', optionId: 'portrait:1' });
    paintedEditDraft();
    openKind('image', 'edit', makeEditContext());

    app = await mountWorkspacePage(makeEditContext(), target);
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const form = workspaceApiMocks.submitGenerate.mock.calls[0][0] as FormData;
    expect([form.get('mode'), form.get('workflow'), form.get('prompt_source'), form.get('prompts_file')]).toEqual(['image', 'txt2img', 'file', '/server/prompts.yaml']);
    expect(form.getAll('prompt_option_id')).toEqual(['portrait:1']);
    expect(fieldsOf(form).some((key) => key.startsWith('edit_'))).toBe(false);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', editSourcePath: '/out/old.png', editMaskUpload: 'mask.png' });
    expect(heldUploads.editMask?.name).toBe('mask.png');
  });
});

describe('a reload', () => {
  it('shows the picture and the paint summary and keeps Generate on, with the held mask back from the store', async () => {
    await mountEdit();
    draft.commitEditPaint({ mask: png('mask.png'), share: 0.18, extension: null, amount: 256 });
    draft.saveDraft();
    await heldUploads.persisted();
    const { store } = { store: heldUploads.store };
    await unmount(app!);
    app = null;
    workspaceMemory.forget();
    imageEdit.setSource(null);
    heldUploads.useStore(store);

    draft.loadDraft();
    app = await mountWorkspacePage(makeEditContext(), target);

    expect(draft.state).toMatchObject({ workflow: 'edit', editSourcePath: '/out/a.png', editMaskUpload: 'mask.png' });
    expect(kindButton('edit').getAttribute('aria-pressed')).toBe('true');
    expect(heldUploads.editMask?.name).toBe('mask.png');
    expect(imageEdit.source?.path).toBe('/out/a.png');
    expect(summary()).toContain('Painted area: 18% of the image');
    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).placeholder).toBe('What should be here?');
    expect(submitButton().disabled).toBe(false);
    const form = await submitForm(target);
    expect((form.get('edit_mask_file') as File).name).toBe('mask.png');
  });

  it('asks to paint again when the mask did not come back', async () => {
    const context = makeEditContext();
    draft.reset();
    draft.patch({ model: 'klein', workflow: 'edit', prompt: 'x', steps: 4 });
    draft.setEditSource({ path: '/out/a.png' });
    draft.patch({ editMaskUpload: 'mask.png', editPaintedShare: 0.2 });
    app = await mountWorkspacePage(context, target);

    expect(heldUploads.editMask).toBeNull();
    expect(reason()).toBe('Paint the area again');
  });
});

describe('Copy CLI for an edit', () => {
  it('copies the command of the edit, with the caveat for painted paint', async () => {
    await mountEdit();
    draft.commitEditPaint({ mask: png('mask.png'), share: 0.2, extension: { side: 'top', amount: 256 }, amount: 256 });
    await settle();
    const writeText = mockClipboard();

    (target.querySelector('#ws-copy-cli') as HTMLButtonElement).click();
    await settle();

    expect(writeText).toHaveBeenCalledWith('ziv image --model klein --edit /out/a.png --prompt \'a red door\' --steps 4 --extend top=256 --mask mask.png');
  });
});
