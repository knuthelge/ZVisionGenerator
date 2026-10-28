// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { downloadStore } from '$lib/state/downloads.svelte';
import { pendingTriggers } from '$lib/state/loraTriggers.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import type { DownloadItem, LoraInfo, CuratedEntry, WorkspaceContext } from '$lib/types';

import { makeContext, makeIdeogramDefaults, makeImageDefaults, withEnhancer } from '../../test-utils/workspace';
import { createHost, fakeDebounceTimers, field, mountWorkspacePage, pressKey, resetWorkspaceTest, settle, type } from '../../test-utils/workspaceHarness';
import { discoverApiMocks, layoutProbe, modelApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  const { layoutProbe: probe } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    layout: {
      get mode() { return probe.compact ? 'compact' : 'wide'; },
      get compact() { return probe.compact; },
      get coarse() { return probe.compact || probe.coarse; },
    },
  };
});
vi.mock('$lib/api/config', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/config')>()), updateConfig: (await import('../../test-utils/workspaceMocks')).configApiMocks.updateConfig }));
vi.mock('$lib/api/extend', async () => ({ inspectExtendSource: (await import('../../test-utils/workspaceMocks')).extendApiMocks.inspectExtendSource }));
vi.mock('$lib/api/videoMemory', async () => ({ checkVideoMemoryFit: (await import('../../test-utils/workspaceMocks')).videoMemoryApiMocks.checkVideoMemoryFit }));
vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  const { workspaceApiMocks: mocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getWorkspaceContext: mocks.getWorkspaceContext, getWorkspaceCoreContext: mocks.getWorkspaceCoreContext, submitGenerate: mocks.submitGenerate, getJobSnapshot: mocks.getJobSnapshot, parseUrlPrefill: mocks.parseUrlPrefill, listJobs: mocks.listJobs, clearQueue: mocks.clearQueue, cancelJob: mocks.cancelJob, skipFirstRun: mocks.skipFirstRun };
});
vi.mock('$lib/api/gallery', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/gallery')>()), getGallery: (await import('../../test-utils/workspaceMocks')).galleryApiMocks.getGallery }));
vi.mock('$lib/api/recentPrompts', async () => (await import('../../test-utils/workspaceMocks')).recentPromptsApiMocks);
vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);
vi.mock('$lib/api/presets', async () => (await import('../../test-utils/workspaceMocks')).presetApiMocks);
vi.mock('$lib/api/discover', async () => (await import('../../test-utils/workspaceMocks')).discoverApiMocks);
vi.mock('$lib/api/models', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/models')>()), getDownloads: (await import('../../test-utils/workspaceMocks')).modelApiMocks.getDownloads, deleteLora: (await import('../../test-utils/workspaceMocks')).modelApiMocks.deleteLora }));
vi.mock('$lib/api/promptFiles', async () => {
  const { promptFileApiMocks } = await import('../../test-utils/workspaceMocks');
  return { inspectPromptFile: promptFileApiMocks.inspectPromptFile, writePromptFile: promptFileApiMocks.writePromptFile, appendPromptEntry: promptFileApiMocks.appendPromptEntry, createPromptFile: promptFileApiMocks.createPromptFile };
});

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
});

afterEach(async () => {
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

const lora = (name: string, extra: Partial<LoraInfo> = {}): LoraInfo => ({ name, path: `/l/${name}.safetensors`, family: 'zimage', size_variant: null, ...extra });
const entry = (overrides: Partial<CuratedEntry> = {}): CuratedEntry => ({
  id: 'o-style', name: 'style', family: 'zimage', size_variant: null, version: null, display_name: 'Style', author: 'o', description: 'A soft look', section: 'style',
  what_it_does: null, workflow: null, repo: 'o/style', revision: 'a'.repeat(40), file: 'style.safetensors', size_bytes: 255_000_000, size_label: '255 MB', trigger: 'sty1e', weight: 0.8,
  licence: 'apache-2.0', licence_label: 'Apache 2.0', non_commercial: false, downloads: 10, platforms: [], note: null, previews: [], preview_videos: [],
  model_card_url: 'https://huggingface.co/o/style', state: 'available', download: null, ...overrides,
});
const item = (state: DownloadItem['state'], extra: Partial<DownloadItem> = {}): DownloadItem => ({ id: 'd1', kind: 'lora', name: 'style', label: 'LoRA style', state, message: null, bytes_done: 40, bytes_total: 100, curated_id: 'o-style', ...extra });

function zimageContext(loras: LoraInfo[] = []): WorkspaceContext {
  const context = makeContext({ loras });
  context.image_model_defaults = { ...context.image_model_defaults, zit: makeImageDefaults({ family: 'zimage' }), ideo: makeIdeogramDefaults({ family: 'ideogram4' }) };
  context.image_models = [...context.image_models, { id: 'ideo', label: 'ideo', type: 'image' }];
  return context;
}

function curatedWith(entries: CuratedEntry[]): void {
  discoverApiMocks.getCurated.mockResolvedValue({ family: 'zimage', size: null, version: null, hf_enabled: false, entries });
}

async function mountWorkspace(context: WorkspaceContext): Promise<void> {
  draft.update('workflow', 'txt2img');
  draft.update('model', 'zit');
  app = await mountWorkspacePage(context, target);
}

const addLoraButton = () => Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Add LoRA')) as HTMLButtonElement;
const menu = () => document.querySelector('[role="menu"][aria-label="Available LoRAs"]') as HTMLElement | null;
const drawer = () => document.querySelector('[role="dialog"][aria-label="Discover LoRAs"]') as HTMLElement | null;
const submit = () => target.querySelector('#ws-submit') as HTMLButtonElement;
const triggersButton = () => target.querySelector('#ws-triggers') as HTMLButtonElement | null;
const triggersMenu = () => document.querySelector('[role="menu"][aria-label="Triggers"]') as HTMLElement | null;
const triggerItems = () => Array.from(triggersMenu()?.querySelectorAll('[role="menuitemradio"]') ?? []) as HTMLElement[];
const triggerWords = () => triggerItems().map((el) => `${el.getAttribute('aria-checked') === 'true' ? '+' : '-'}${el.textContent?.trim()}`);
/** The count on the Triggers button, or '' when it reads just Triggers. */
const triggersCount = () => triggersButton()?.querySelector('.triggers-count')?.textContent ?? '';
const openTriggers = async (): Promise<void> => { flushSync(() => triggersButton()!.click()); await settle(); };
const useSwitch = () => drawer()!.querySelector('input[role="switch"]') as HTMLInputElement;
const clickIn = (root: ParentNode, selector: string): void => { flushSync(() => (root.querySelector(selector) as HTMLElement).click()); };

/** Open a tile of the drawer and turn "Use in LoRAs" on. */
async function useFromDrawer(name: string): Promise<void> {
  const card = Array.from(drawer()!.querySelectorAll('.disc-name')).find((el) => el.textContent?.trim() === name)!.closest('button')!;
  flushSync(() => card.click());
  await settle();
  flushSync(() => useSwitch().click());
  await settle();
  await settle();
}

describe('Discover LoRAs: entry points', () => {
  it('ends the Add LoRA menu with Discover LoRAs…, with no count, and opens the drawer', async () => {
    curatedWith([entry(), entry({ id: 'b', name: 'b', display_name: 'B' })]);
    await mountWorkspace(zimageContext([lora('mine')]));

    flushSync(() => addLoraButton().click());
    await settle();
    const items = Array.from(menu()!.querySelectorAll('[data-action]')).map((el) => el.getAttribute('data-action'));
    expect(items.at(-1)).toBe('discover-loras');
    expect(menu()!.querySelector('[data-action="discover-loras"]')?.textContent?.trim()).toBe('Discover LoRAs…');

    flushSync(() => (menu()!.querySelector('[data-action="discover-loras"]') as HTMLElement).click());
    await settle();

    expect(drawer()).not.toBeNull();
    expect(menu()).toBeNull();
  });

  it('opens the drawer straight away from Add LoRA when there are no LoRA files', async () => {
    curatedWith([entry()]);
    await mountWorkspace(zimageContext([]));

    expect(addLoraButton().disabled).toBe(false);
    flushSync(() => addLoraButton().click());
    await settle();

    expect(menu()).toBeNull();
    expect(drawer()).not.toBeNull();
  });

  it('opens the drawer when the address asks for it, and applies no prefill for that', async () => {
    curatedWith([entry()]);
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ discover: 'loras' });
    await mountWorkspace(zimageContext([]));
    await settle();

    expect(drawer()).not.toBeNull();
    expect(window.location.hash).toBe('#/workspace');
  });

  it('is locked to the Workspace model: the family and size of the model in the form', async () => {
    curatedWith([entry()]);
    const context = zimageContext([]);
    await mountWorkspace(context);
    flushSync(() => addLoraButton().click());
    await settle();

    expect(discoverApiMocks.getCurated).toHaveBeenLastCalledWith({ family: 'zimage', size: null, version: null });
    expect(drawer()!.textContent).toContain('for zit');
  });
});

describe('Discover LoRAs: Use in LoRAs', () => {
  it('keeps the drawer open, puts the chip on at the recommended weight and puts nothing into the prompt', async () => {
    curatedWith([entry()]);
    discoverApiMocks.startDiscoverDownload.mockResolvedValue({ name: 'style', status: 'queued', renamed: false, item: item('running'), trigger: 'sty1e', trigger_source: 'curated' });
    await mountWorkspace(zimageContext([]));
    flushSync(() => addLoraButton().click());
    await settle();

    await useFromDrawer('Style');

    expect(discoverApiMocks.startDiscoverDownload).toHaveBeenCalledWith({ curated_id: 'o-style' });
    expect(drawer()).not.toBeNull();
    expect(draft.state.loraString).toBe('style:0.8');
    expect(field('ws-prompt').value).toBe('');
    expect(pendingTriggers.get('style')).toBe('sty1e');
  });

  it('applies an installed entry at once, with no download, and turning the switch off takes the chip away', async () => {
    curatedWith([entry({ state: 'installed' })]);
    await mountWorkspace(zimageContext([lora('style', { trigger: 'sty1e', trigger_source: 'curated' })]));
    flushSync(() => addLoraButton().click());
    await settle();
    flushSync(() => (menu()!.querySelector('[data-action="discover-loras"]') as HTMLElement).click());
    await settle();

    await useFromDrawer('Style');

    expect(discoverApiMocks.startDiscoverDownload).not.toHaveBeenCalled();
    expect(draft.state.loraString).toBe('style:0.8');
    expect(field('ws-prompt').value).toBe('');
    flushSync(() => useSwitch().click());
    await settle();
    expect(draft.state.loraString).toBe('');
  });

  it('takes a deleted LoRA off the chips and out of what every model remembers', async () => {
    curatedWith([entry({ state: 'installed' })]);
    modelApiMocks.deleteLora.mockResolvedValue({ tone: 'success', message: 'Deleted', detail: '' });
    await mountWorkspace(zimageContext([lora('style')]));
    draft.patch({ loraString: 'style:1', lorasByFamily: { flux1: 'style:0.5,other:1' } });
    flushSync(() => addLoraButton().click());
    await settle();
    flushSync(() => (menu()!.querySelector('[data-action="discover-loras"]') as HTMLElement).click());
    await settle();
    const card = Array.from(drawer()!.querySelectorAll('.disc-name')).find((el) => el.textContent?.trim() === 'Style')!.closest('button')!;
    flushSync(() => card.click());
    await settle();

    flushSync(() => (Array.from(drawer()!.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Delete file') as HTMLButtonElement).click());
    await settle();
    // The confirmation is a dialog of its own.
    flushSync(() => (Array.from(document.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Delete') as HTMLButtonElement).click());
    await settle();

    expect(modelApiMocks.deleteLora).toHaveBeenCalledWith('style');
    expect(draft.state.loraString).toBe('');
    expect(draft.state.lorasByFamily).toEqual({ flux1: 'other:1' });
  });
});

describe('Generate waits for a LoRA download', () => {
  async function withDownload(state: DownloadItem['state'], extra: Partial<DownloadItem> = {}): Promise<void> {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item(state, extra)] });
    await mountWorkspace(zimageContext([]));
    draft.update('loraString', 'style:0.8');
    await settle();
  }

  it('disables Generate with the name and the percentage, the full text on the tooltip and the label', async () => {
    await withDownload('running');

    expect(submit().disabled).toBe(true);
    expect(submit().textContent).toContain('Waiting for style');
    expect(submit().textContent).toContain('40 %');
    expect(submit().getAttribute('aria-label')).toBe('Waiting for style · 40 %');
    expect(submit().title).toBe('Waiting for style · 40 %');
  });

  it('shows the chip as downloading, and × on it removes the chip and never cancels the download', async () => {
    await withDownload('running');
    const chip = target.querySelector('.lora-chip') as HTMLElement;
    expect(chip.dataset.download).toBe('downloading');
    const cancel = vi.spyOn(downloadStore, 'cancel').mockResolvedValue();

    clickIn(chip, '.lora-remove');
    await settle();

    expect(cancel).not.toHaveBeenCalled();
    expect(draft.state.loraString).toBe('');
    expect(downloadStore.active('lora', 'style')).not.toBeNull();
  });

  it('never submits while waiting, the keyboard shortcut included', async () => {
    await withDownload('running');
    workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'j' } as never);
    type('ws-prompt', 'a fox');

    pressKey('Enter', { metaKey: true, ctrlKey: true });
    await settle();
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
  });

  it('is Generate again once the download finishes and the file is listed', async () => {
    await withDownload('running');
    expect(submit().disabled).toBe(true);

    // The finished download makes the page read the LoRA list again, which is what makes the file count as installed.
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item('done')] });
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(zimageContext([lora('style')]));
    await downloadStore.refresh();
    await settle();

    expect(submit().disabled).toBe(false);
    expect((target.querySelector('.lora-chip') as HTMLElement).dataset.download).toBeUndefined();
    expect(draft.state.loraString).toBe('style:0.8');
  });

  it('stops blocking Generate on a finished download whose file never shows up, after a bounded wait', async () => {
    fakeDebounceTimers();
    await withDownload('done');
    expect(submit().disabled).toBe(true);
    workspaceApiMocks.getWorkspaceCoreContext.mockReset();
    workspaceApiMocks.getWorkspaceCoreContext.mockRejectedValue(new Error('offline'));

    await vi.advanceTimersByTimeAsync(10_000);
    await settle();
    expect(submit().disabled).toBe(true);
    await vi.advanceTimersByTimeAsync(120_000);
    await settle();

    expect(submit().disabled).toBe(false);
    expect((target.querySelector('.lora-chip') as HTMLElement).dataset.download).toBeUndefined();
    expect(draft.state.loraString).toBe('style:0.8');
    vi.useRealTimers();
  });

  it('turns the chip red on a failure, with the reason, Retry download and Remove, and the reason above a disabled Generate', async () => {
    await withDownload('failed', { message: 'The connection dropped.' });
    const chip = target.querySelector('.lora-chip') as HTMLElement;
    expect(chip.dataset.download).toBe('failed');
    expect(submit().disabled).toBe(true);
    expect(target.querySelector('[data-testid="lora-failure"]')?.textContent).toContain("style didn't download. Retry it or remove it from LoRAs.");
    expect(Array.from(target.querySelectorAll('[data-testid="lora-failure"] button')).map((b) => b.textContent)).toEqual(['Retry']);

    clickIn(chip, '.lora-failed');
    await settle();
    const open = document.querySelector('[role="menu"][aria-label="Download failed"]') as HTMLElement;
    expect(open.textContent).toContain('The connection dropped.');
    expect(Array.from(open.querySelectorAll('[role="menuitem"]')).map((el) => el.textContent?.trim())).toEqual(['Retry download', 'Remove']);

    discoverApiMocks.startDiscoverDownload.mockResolvedValue({ name: 'style', status: 'queued', renamed: false, item: item('running'), trigger: null, trigger_source: null });
    flushSync(() => (open.querySelector('[data-action="retry-lora"]') as HTMLElement).click());
    await settle();
    expect(discoverApiMocks.startDiscoverDownload).toHaveBeenCalledWith({ curated_id: 'o-style' });
  });

  it('removes a failed chip with Remove', async () => {
    await withDownload('failed', { message: 'The connection dropped.' });
    clickIn(target.querySelector('.lora-chip') as HTMLElement, '.lora-failed');
    await settle();

    flushSync(() => (document.querySelector('[data-action="remove-lora"]') as HTMLElement).click());
    await settle();

    expect(draft.state.loraString).toBe('');
    expect(submit().disabled).toBe(false);
  });
});

describe('A failed download', () => {
  const failedA = () => item('failed', { id: 'd1', name: 'style', message: 'The connection dropped.' });
  const runningB = () => item('running', { id: 'dl-2', name: 'other', label: 'LoRA other', curated_id: null, bytes_done: 0, bytes_total: 100 });

  async function withFailureAndDownload(): Promise<void> {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [failedA(), runningB()] });
    await mountWorkspace(zimageContext([]));
    draft.update('loraString', 'style:0.8,other:1');
    await settle();
  }

  it('wins over Waiting on the sidebar Generate: one message, and Generate reads Generate, disabled', async () => {
    await withFailureAndDownload();

    expect(submit().disabled).toBe(true);
    expect(submit().textContent).not.toContain('Waiting');
    expect(submit().textContent).toContain('Generate');
    expect(target.querySelector('[data-testid="lora-failure"]')?.textContent).toContain("style didn't download.");
  });

  it('wins over Waiting on the phone Generate and its compact label', async () => {
    layoutProbe.compact = true;
    await withFailureAndDownload();

    const generate = target.querySelector('#ws-submit-compact') as HTMLButtonElement;
    expect(generate.disabled).toBe(true);
    expect(generate.textContent).not.toContain('Waiting');
    expect(generate.textContent).toContain('Generate');
    expect(generate.getAttribute('aria-label')).toBeNull();
    expect(target.querySelector('[data-testid="lora-failure"]')?.textContent).toContain("style didn't download.");
    // The settings strip says the same thing, not a download on its way.
    expect(target.querySelector('.settings-strip')?.textContent).not.toContain('%');
  });

  it('raises no toast for a LoRA whose chip shows the error slot, and one for a LoRA that has no chip', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item('running'), item('running', { id: 'dl-2', name: 'other', label: 'LoRA other', curated_id: null })] });
    await mountWorkspace(zimageContext([]));
    draft.update('loraString', 'style:0.8');
    await settle();

    modelApiMocks.getDownloads.mockResolvedValue({ items: [failedA(), item('failed', { id: 'dl-2', name: 'other', label: 'LoRA other', curated_id: null, message: 'Disk full.' })] });
    await downloadStore.refresh();
    await settle();

    const messages = toasts.map((toast) => toast.message);
    expect(messages).toContain('Disk full.');
    expect(messages).not.toContain('The connection dropped.');
  });

  it('is announced by a toast again once the Workspace is left', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item('running')] });
    await mountWorkspace(zimageContext([]));
    draft.update('loraString', 'style:0.8');
    await settle();
    await unmount(app!);
    app = null;

    modelApiMocks.getDownloads.mockResolvedValue({ items: [failedA()] });
    await downloadStore.refresh();

    expect(toasts.map((toast) => toast.message)).toContain('The connection dropped.');
  });

  it('keeps the phone toasts clear of the strip and the error slot by publishing the height of the controls', async () => {
    layoutProbe.compact = true;
    await mountWorkspace(zimageContext([]));

    expect(target.querySelector('.compact-controls')).not.toBeNull();
    expect(document.documentElement.style.getPropertyValue('--toast-dock')).toMatch(/^\d+px$/);
  });
});

describe('The Triggers menu in the prompt box', () => {
  const triggerContext = () => zimageContext([
    lora('tech', { trigger: 't3chnic4lly', trigger_source: 'curated', title: 'Technically Color' }),
    lora('clay', { trigger: 'Clay animation, Clay', trigger_source: 'card' }),
  ]);

  it('is not there when no LoRA in use has a trigger', async () => {
    await mountWorkspace(triggerContext());
    expect(triggersButton()).toBeNull();
    draft.update('loraString', 'unknown:1');
    await settle();

    expect(triggersButton()).toBeNull();
  });

  it('sits in the tool row by the history and Save buttons and counts the words the prompt lacks, reading just Triggers at 0', async () => {
    await mountWorkspace(triggerContext());
    type('ws-prompt', 'a lighthouse');
    draft.update('loraString', 'tech:1,clay:1');
    await settle();

    expect(triggersButton()!.textContent?.trim()).toBe('Triggers3');
    expect(triggersCount()).toBe('3');
    const row = triggersButton()!.parentElement!;
    expect(row.contains(target.querySelector('#ws-recent-prompts'))).toBe(true);
    type('ws-prompt', 't3chnic4lly, Clay animation, clay');
    await settle();
    expect(triggersButton()!.textContent?.trim()).toBe('Triggers');
    expect(triggersCount()).toBe('');
  });

  it('groups the words by LoRA, checks the ones in the prompt, and a tap on an unchecked word puts it first', async () => {
    await mountWorkspace(triggerContext());
    type('ws-prompt', 'T3CHNIC4LLY, a lighthouse');
    draft.update('loraString', 'tech:1,clay:1');
    await settle();

    await openTriggers();
    expect(Array.from(triggersMenu()!.querySelectorAll('.action-menu-heading')).map((el) => el.textContent?.trim())).toEqual(['Technically Color', 'clay']);
    expect(triggerWords()).toEqual(['+t3chnic4lly', '-Clay animation', '-Clay']);

    flushSync(() => triggerItems()[1].click());
    await settle();
    expect(field('ws-prompt').value).toBe('Clay animation, T3CHNIC4LLY, a lighthouse');
    // The word Clay is in "Clay animation" now, so nothing is missing.
    expect(triggersCount()).toBe('');
  });

  it('does nothing for a word that is checked', async () => {
    await mountWorkspace(triggerContext());
    type('ws-prompt', 't3chnic4lly, a fox');
    draft.update('loraString', 'tech:1');
    await settle();
    await openTriggers();

    flushSync(() => triggerItems()[0].click());
    await settle();

    expect(field('ws-prompt').value).toBe('t3chnic4lly, a fox');
  });

  it('offers the words of a LoRA that is still downloading', async () => {
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item('running')] });
    await mountWorkspace(triggerContext());
    pendingTriggers.set('style', 'sty1e, soft');
    draft.update('loraString', 'style:1');
    await settle();
    await openTriggers();

    expect(triggerWords()).toEqual(['-sty1e', '-soft']);
  });

  it('follows the text Generate sends: a word goes into the Enhanced text and its tab opens', async () => {
    await mountWorkspace(withEnhancer(triggerContext()));
    type('ws-prompt', 'mine');
    draft.patch({ enhancedPrompt: 'a lighthouse at dusk', enhancedFrom: { prompt: 'mine', scene: '', mode: 'image' }, loraString: 'tech:1' });
    await settle();
    await openTriggers();

    flushSync(() => triggerItems()[0].click());
    await settle();

    expect(draft.state.enhancedPrompt).toBe('t3chnic4lly, a lighthouse at dusk');
    expect(draft.state.prompt).toBe('mine');
    expect(target.querySelector('#ws-tab-enhanced')?.getAttribute('aria-selected')).toBe('true');
  });

  it('is hidden for a prompt from a file', async () => {
    await mountWorkspace(triggerContext());
    draft.patch({ promptSource: 'file', loraString: 'tech:1' });
    await settle();

    expect(triggersButton()).toBeNull();
  });
});

describe('Discover LoRAs on a phone', () => {
  it('says only "Waiting · 40 %" on Generate, with the full text kept for the label', async () => {
    layoutProbe.compact = true;
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item('running')] });
    await mountWorkspace(zimageContext([]));
    draft.update('loraString', 'style:0.8');
    await settle();

    const generate = target.querySelector('#ws-submit-compact') as HTMLButtonElement;
    expect(generate.disabled).toBe(true);
    expect(generate.textContent).toContain('Waiting');
    expect(generate.textContent).toContain('40 %');
    expect(generate.getAttribute('aria-label')).toBe('Waiting for style · 40 %');
  });

  it('shows the reason with Retry in the composer error slot when a download failed', async () => {
    layoutProbe.compact = true;
    modelApiMocks.getDownloads.mockResolvedValue({ items: [item('failed', { message: 'The connection dropped.' })] });
    discoverApiMocks.startDiscoverDownload.mockResolvedValue({ name: 'style', status: 'queued', renamed: false, item: item('running'), trigger: null, trigger_source: null });
    await mountWorkspace(zimageContext([]));
    draft.update('loraString', 'style:0.8');
    await settle();

    const slot = target.querySelector('[data-testid="lora-failure"]') as HTMLElement;
    expect(slot.textContent).toContain("style didn't download.");
    flushSync(() => (Array.from(slot.querySelectorAll('button')).find((b) => b.textContent === 'Retry') as HTMLButtonElement).click());
    await settle();
    expect(discoverApiMocks.startDiscoverDownload).toHaveBeenCalledWith({ curated_id: 'o-style' });
  });

  it('closes All settings before Discover LoRAs opens from the toolbar menu', async () => {
    layoutProbe.compact = true;
    curatedWith([entry()]);
    await mountWorkspace(zimageContext([lora('mine')]));
    flushSync(() => (target.querySelector('button[aria-label="All settings"]') as HTMLButtonElement).click());
    await settle();
    expect(document.querySelector('[role="dialog"][aria-label="All settings"]')).not.toBeNull();

    flushSync(() => addLoraButton().click());
    await settle();
    flushSync(() => (document.querySelector('[data-action="discover-loras"]') as HTMLElement).click());
    await settle();
    await settle();

    expect(drawer()).not.toBeNull();
    expect(draft.state.sidebarCollapsed).toBe(false);
  });
});
