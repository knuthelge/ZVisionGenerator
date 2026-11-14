// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { draft } from '$lib/state/draft.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import type { JobContext, JobDownloadPlan, RowDownloadPlan, WorkspaceContext } from '$lib/types';

import { makeContext, withEnhancer } from '../../test-utils/workspace';
import { answerConfirm, createHost, mountWorkspacePage, openEnhanceRun, resetWorkspaceTest, settle, submitForm } from '../../test-utils/workspaceHarness';
import { EMPTY_JOB_PLAN, enhanceApiMocks, modelApiMocks, promptFileApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

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
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAssets: galleryApiMocks.deleteAssets };
});

vi.mock('$lib/api/promptEnhance', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/promptEnhance')>();
  const { enhanceApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, enhancePrompt: enhanceApiMocks.enhancePrompt };
});

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

vi.mock('$lib/api/presets', async () => (await import('../../test-utils/workspaceMocks')).presetApiMocks);

vi.mock('$lib/api/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/models')>();
  const { modelApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getDownloads: modelApiMocks.getDownloads, getDownloadPlan: modelApiMocks.getDownloadPlan };
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
  workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1' } as JobContext);
  window.location.hash = '#/workspace';
});

afterEach(async () => {
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
});

async function mountWorkspace(context: WorkspaceContext = makeContext()): Promise<void> {
  app = await mountWorkspacePage(context, target);
}

const KLEIN = { label: 'FLUX.2 Klein 9B', model: 'klein9b', bytes: 35.1e9 };
const ENHANCER = { label: 'Prompt enhancer', model: 'org/llm', bytes: 2.4e9 };
const SPACE = { path: '/data', needed_bytes: 37.5e9, free_bytes: 812e9, enough: true };
const plan = (...parts: JobDownloadPlan['parts']): JobDownloadPlan => ({ parts, total_bytes: parts.reduce((sum, part) => sum + (part.bytes ?? 0), 0), space: SPACE, unchecked: false, licences: [] });

const dialogs = () => document.querySelectorAll('[role="alertdialog"]');
const generateButton = () => target.querySelector('#ws-submit') as HTMLButtonElement;

describe('Generate asks before a job downloads', () => {
  it('shows no dialog and submits when nothing is missing', async () => {
    await mountWorkspace();
    draft.update('prompt', 'a fox');

    await submitForm(target);

    expect(workspaceApiMocks.planGenerate).toHaveBeenCalledTimes(1);
    expect(dialogs()).toHaveLength(0);
    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
  });

  it('shows one dialog for every missing model and submits once it is confirmed', async () => {
    workspaceApiMocks.planGenerate.mockResolvedValue(plan(ENHANCER, KLEIN));
    await mountWorkspace();
    draft.update('prompt', 'a fox');

    await submitForm(target);

    expect(dialogs()).toHaveLength(1);
    expect(dialogs()[0].textContent).toContain('Download 2 models?');
    expect(dialogs()[0].textContent).toContain('FLUX.2 Klein 9B');
    expect(dialogs()[0].textContent).toContain('2.4 GB');
    expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();

    await answerConfirm(true);

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    expect(dialogs()).toHaveLength(0);
  });

  it('plans and submits the same form', async () => {
    workspaceApiMocks.planGenerate.mockResolvedValue(plan(KLEIN));
    await mountWorkspace();
    draft.update('prompt', 'a fox');

    await submitForm(target);
    await answerConfirm(true);

    const planned = workspaceApiMocks.planGenerate.mock.calls[0][0];
    const submitted = workspaceApiMocks.submitGenerate.mock.calls[0][0];
    expect(Object.fromEntries(planned.entries())).toEqual(Object.fromEntries(submitted.entries()));
  });

  it('queues nothing when the dialog is cancelled, and frees Generate again', async () => {
    workspaceApiMocks.planGenerate.mockResolvedValue(plan(KLEIN));
    await mountWorkspace();
    draft.update('prompt', 'a fox');

    await submitForm(target);
    await answerConfirm(false);

    expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
    expect(dialogs()).toHaveLength(0);
    expect(generateButton().disabled).toBe(false);
    expect(toasts).toHaveLength(0);
  });

  it('holds Generate busy while the plan is read', async () => {
    let resolvePlan: (value: JobDownloadPlan) => void = () => undefined;
    workspaceApiMocks.planGenerate.mockReturnValue(new Promise((resolve) => { resolvePlan = resolve; }));
    await mountWorkspace();
    draft.update('prompt', 'a fox');

    void submitForm(target);
    await settle();
    expect(generateButton().disabled).toBe(true);
    expect(generateButton().textContent).toContain('Submitting');

    // A second press while it waits plans nothing more.
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    expect(workspaceApiMocks.planGenerate).toHaveBeenCalledTimes(1);

    resolvePlan(EMPTY_JOB_PLAN);
    await settle();
    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    expect(generateButton().disabled).toBe(false);
  });

  it('shows the server\'s reason and queues nothing when the plan is refused', async () => {
    workspaceApiMocks.planGenerate.mockRejectedValue(new ApiError('424', 424, 'Hugging Face refused org/klein'));
    await mountWorkspace();
    draft.update('prompt', 'a fox');

    await submitForm(target);

    expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
    expect(toasts.some((toast) => toast.type === 'error')).toBe(true);
    expect(generateButton().disabled).toBe(false);
  });

  it('asks for a prompt file\'s job once, with the file\'s form', async () => {
    promptFileApiMocks.inspectPromptFile.mockResolvedValue({
      path: '/server/rain.yaml',
      options: [0, 1].map((index) => ({ id: `rain:${index}`, set_name: 'rain', source_index: index, label: `rain #${index + 1}`, prompt_preview: `Prompt ${index}`, negative_preview: null })),
    });
    workspaceApiMocks.planGenerate.mockResolvedValue(plan(KLEIN));
    await mountWorkspace();
    draft.patch({ promptSource: 'file', promptFilePath: '/server/rain.yaml', promptFileOptionIds: ['rain:0', 'rain:1'] });
    await settle();

    await submitForm(target);

    expect(dialogs()).toHaveLength(1);
    expect(workspaceApiMocks.planGenerate.mock.calls[0][0].get('prompt_source')).toBe('file');
    await answerConfirm(true);
    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
  });

  it('asks the same way on the video tab', async () => {
    workspaceApiMocks.planGenerate.mockResolvedValue(plan({ ...KLEIN, label: 'LTX 2.3 8B', model: 'ltx-8' }));
    draft.update('workflow', 'txt2vid');
    await mountWorkspace();
    draft.update('prompt', 'a lake');

    await submitForm(target);

    expect(workspaceApiMocks.planGenerate.mock.calls[0][0].get('mode')).toBe('video');
    expect(dialogs()[0].textContent).toContain('Download LTX 2.3 8B?');
  });
});

describe('Enhance asks before it downloads its model', () => {
  const rowPlan = (overrides: Partial<RowDownloadPlan> = {}): RowDownloadPlan => ({
    label: 'Fast enhancer model',
    parts: [{ label: 'Fast enhancer model', model: 'fast', bytes: 2.4e9 }],
    total_bytes: 2.4e9,
    space: SPACE,
    unchecked: false,
    licences: [],
    ...overrides,
  });

  function enhancerContext(change: (contract: NonNullable<WorkspaceContext['prompt_enhancer']>) => void): WorkspaceContext {
    const context = withEnhancer(makeContext());
    change(context.prompt_enhancer!);
    return context;
  }

  beforeEach(() => {
    enhanceApiMocks.enhancePrompt.mockResolvedValue();
    draft.update('prompt', 'a fox');
  });

  it('asks about the missing model, then enhances when the user agrees', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(rowPlan());
    await mountWorkspace(enhancerContext((contract) => { contract.downloaded = false; }));

    (await openEnhanceRun()).click();
    await settle();

    expect(modelApiMocks.getDownloadPlan).toHaveBeenCalledWith('enhancer', 'fast');
    expect(dialogs()).toHaveLength(1);
    expect(dialogs()[0].textContent).toContain('Download Fast enhancer model?');
    expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();

    await answerConfirm(true);
    expect(enhanceApiMocks.enhancePrompt).toHaveBeenCalledTimes(1);
  });

  it('does not enhance when the dialog is cancelled', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(rowPlan());
    await mountWorkspace(enhancerContext((contract) => { contract.downloaded = false; }));

    (await openEnhanceRun()).click();
    await settle();
    await answerConfirm(false);

    expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();
    expect((document.querySelector('#ws-enhance-run') as HTMLButtonElement).disabled).toBe(false);
  });

  it('asks about the Smart model when Smart is chosen', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(rowPlan({ label: 'Smart enhancer model' }));
    draft.update('enhanceModel', 'smart');
    await mountWorkspace(enhancerContext((contract) => { contract.smart = { downloaded: false, download_size_label: '19 GB', unavailable: null }; }));

    (await openEnhanceRun()).click();
    await settle();

    expect(modelApiMocks.getDownloadPlan).toHaveBeenCalledWith('enhancer', 'smart');
    await answerConfirm(true);
    expect(enhanceApiMocks.enhancePrompt).toHaveBeenCalledTimes(1);
  });

  it('asks nothing when the model is on disk', async () => {
    await mountWorkspace(enhancerContext(() => undefined));

    (await openEnhanceRun()).click();
    await settle();

    expect(modelApiMocks.getDownloadPlan).not.toHaveBeenCalled();
    expect(dialogs()).toHaveLength(0);
    expect(enhanceApiMocks.enhancePrompt).toHaveBeenCalledTimes(1);
  });

  it('asks nothing when the model is already downloading', async () => {
    modelApiMocks.getDownloadPlan.mockResolvedValue(rowPlan({ parts: [], total_bytes: 0, space: null }));
    await mountWorkspace(enhancerContext((contract) => { contract.downloaded = false; }));

    (await openEnhanceRun()).click();
    await settle();

    expect(dialogs()).toHaveLength(0);
    expect(enhanceApiMocks.enhancePrompt).toHaveBeenCalledTimes(1);
  });

  it("still asks, with Download anyway, when the size can't be checked", async () => {
    modelApiMocks.getDownloadPlan.mockRejectedValue(new ApiError('503', 503, 'Could not reach Hugging Face'));
    await mountWorkspace(enhancerContext((contract) => { contract.downloaded = false; }));

    (await openEnhanceRun()).click();
    await settle();

    expect(dialogs()[0].textContent).toContain("Couldn't check the size");
    expect((document.querySelector('[role="alertdialog"] [data-action="confirm"]') as HTMLElement).textContent).toContain('Download anyway');
  });
});
