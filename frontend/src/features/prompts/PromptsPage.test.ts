// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { draft } from '$lib/state/draft.svelte';
import { takePromptRun } from '$lib/state/pendingPromptRun';
import { router } from '$lib/state/router.svelte';
import type { PromptDocument, PromptDocumentLoad, PromptDocumentPreview, PromptDocumentSaved } from '$lib/types';
import { loadUnsaved, saveUnsaved } from './storage';

const api = vi.hoisted(() => ({
  loadPromptDocument: vi.fn(),
  savePromptDocument: vi.fn(),
  previewPromptDocument: vi.fn(),
  openPathPicker: vi.fn(),
  writePromptFile: vi.fn(),
  createPromptFile: vi.fn(),
}));

vi.mock('$lib/api/promptFiles', () => api);
vi.mock('$lib/state/toasts.svelte', () => ({ addToast: vi.fn() }));

import PromptsPage from './PromptsPage.svelte';

const PATH = '/prompts/portraits.yaml';

function sample(): PromptDocument {
  return {
    snippets: [{ id: 'n0', name: 'light', value: { kind: 'text', text: 'soft light' } }],
    sets: [
      { id: 's0', name: 'portrait', entries: [
        { id: 's0.e0', prompt: { kind: 'text', text: 'a woman, $light, {red|blue} dress' }, negative: null, active: true, enhance: null },
        { id: 's0.e1', prompt: { kind: 'text', text: 'a man' }, negative: null, active: false, enhance: null },
      ] },
    ],
  };
}

function loaded(document: PromptDocument | null = sample(), extra: Partial<PromptDocumentLoad> = {}): PromptDocumentLoad {
  return {
    path: PATH,
    revision: 'r1',
    raw_text: 'portrait: …',
    ...(document ? { document } : {}),
    enhance_matrix: { axes: [], defaults: { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [] } },
    ...extra,
  };
}

function saved(document: PromptDocument, extra: Partial<PromptDocumentSaved> = {}): PromptDocumentSaved {
  return { path: PATH, revision: 'r2', raw_text: 'saved', document, ids: {}, option_id_map: {}, warnings: [], ...extra };
}

const PREVIEW: PromptDocumentPreview = {
  entries: { 's0.e0': { prompt: 'a woman, soft light, {red|blue} dress', negative: null } },
  problems: [],
  snippet_uses: { n0: 1 },
  rolled: null,
};

async function settle(): Promise<void> {
  for (let i = 0; i < 3; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
  flushSync();
}

function button(label: string): HTMLButtonElement {
  const found = Array.from(document.querySelectorAll('button')).find((item) => item.textContent?.trim().startsWith(label));
  if (!found) throw new Error(`No button "${label}"`);
  return found;
}

describe('PromptsPage', () => {
  let target: HTMLElement;
  let app: ReturnType<typeof mount> | null = null;

  async function mountPage(): Promise<void> {
    app = mount(PromptsPage, { target });
    await settle();
  }

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    for (const mock of Object.values(api)) mock.mockReset();
    api.loadPromptDocument.mockResolvedValue(loaded());
    api.previewPromptDocument.mockResolvedValue(PREVIEW);
    router.replace('prompts', { path: PATH });
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) await unmount(app);
    app = null;
    target.remove();
    document.querySelectorAll('[role="alertdialog"]').forEach((node) => node.closest('div')?.remove());
  });

  it('opens the file from the URL and shows its snippets, sets and preview', async () => {
    await mountPage();

    expect(api.loadPromptDocument).toHaveBeenCalledWith(PATH);
    expect(target.querySelector<HTMLInputElement>('input[aria-label="Set name"]')?.value).toBe('portrait');
    expect(target.querySelectorAll('[data-entry-id]')).toHaveLength(2);
    expect(target.querySelector('.tok-snippet')?.textContent).toBe('$light');
    expect(target.querySelectorAll('.tok-choice .tok-option')).toHaveLength(2);
    expect(target.querySelector('[aria-label="Preview"]')?.textContent).toContain('portrait:0');
    expect(target.textContent).toContain('1×');
  });

  it('saves edits and remaps the Workspace selection for the file', async () => {
    draft.patch({ promptFilePath: PATH, promptFileOptionIds: ['portrait:0'] });
    api.savePromptDocument.mockImplementation(async ({ document }: { document: PromptDocument }) => saved(document, { option_id_map: { 'portrait:0': 'portrait:1' } }));
    await mountPage();

    button('＋ New set').click();
    await settle();
    button('Save').click();
    await settle();

    expect(api.savePromptDocument).toHaveBeenCalledTimes(1);
    const request = api.savePromptDocument.mock.calls[0][0];
    expect(request).toMatchObject({ path: PATH, revision: 'r1' });
    expect(request.document.sets.map((set: { name: string }) => set.name)).toEqual(['portrait', 'new_set']);
    expect(request.force).toBeUndefined();
    expect(draft.state.promptFileOptionIds).toEqual(['portrait:1']);
  });

  it('overwrites a file that changed on disk only after confirmation', async () => {
    api.savePromptDocument
      .mockRejectedValueOnce(new ApiError('conflict', 409, 'The prompt file changed on disk since it was opened.'))
      .mockImplementationOnce(async ({ document }: { document: PromptDocument }) => saved(document));
    await mountPage();

    button('＋ New set').click();
    await settle();
    button('Save').click();
    await settle();
    (document.querySelector('[role="alertdialog"] [data-action="confirm"]') as HTMLButtonElement).click();
    await settle();

    expect(api.savePromptDocument).toHaveBeenCalledTimes(2);
    expect(api.savePromptDocument.mock.calls[1][0]).toMatchObject({ force: true, base_text: 'portrait: …' });
  });

  it('opens a file that cannot be built in the repair view', async () => {
    api.loadPromptDocument.mockResolvedValueOnce(loaded(null, { problem: 'The file is not valid YAML: line 2', raw_text: 'portrait: [\n' }));
    api.writePromptFile.mockResolvedValue({ path: PATH, options: [] });
    await mountPage();

    expect(target.textContent).toContain('line 2');
    const box = target.querySelector<HTMLTextAreaElement>('textarea[aria-label="Prompt file text"]')!;
    expect(box.value).toBe('portrait: [\n');
    box.value = 'portrait: []\n';
    box.dispatchEvent(new Event('input', { bubbles: true }));
    button('Save and open').click();
    await settle();

    expect(api.writePromptFile).toHaveBeenCalledWith(PATH, 'portrait: []\n');
    expect(api.loadPromptDocument).toHaveBeenCalledTimes(2);
  });

  it('never saves the builder document from the repair view', async () => {
    api.loadPromptDocument.mockResolvedValueOnce(loaded()).mockResolvedValueOnce(loaded(null, { path: '/prompts/broken.yaml', problem: 'bad', raw_text: 'x: [\n', revision: 'r9' }));
    await mountPage();
    button('＋ New set').click();
    await settle();
    router.replace('prompts', { path: '/prompts/broken.yaml' });
    await settle();
    expect(target.textContent).toContain("can't be opened");

    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 's', metaKey: true, bubbles: true }));
    await settle();

    expect(api.savePromptDocument).not.toHaveBeenCalled();
    // The previous file's edits are kept under its own name, never under the broken file's.
    expect(loadUnsaved(PATH)).not.toBeNull();
    expect(loadUnsaved('/prompts/broken.yaml')).toBeNull();
    expect(target.querySelector('.file-dirty')).toBeNull();
  });

  it('keeps unsaved edits when switching files right away', async () => {
    const other = '/prompts/other.yaml';
    api.loadPromptDocument.mockResolvedValueOnce(loaded()).mockResolvedValueOnce(loaded(sample(), { path: other, revision: 'r5' }));
    await mountPage();
    button('＋ New set').click();
    await settle();

    router.replace('prompts', { path: other });
    await settle();

    expect(loadUnsaved(PATH)?.document.sets.map((set) => set.name)).toEqual(['portrait', 'new_set']);
  });

  it('queues one prompt in the Workspace from Generate this one', async () => {
    await mountPage();

    const generate = target.querySelector('[data-entry-id="s0.e0"]')!.querySelector<HTMLButtonElement>('button[aria-label="Generate this one"]')!;
    generate.click();
    await settle();

    expect(takePromptRun()).toEqual({ path: PATH, optionId: 'portrait:0' });
    expect(router.page).toBe('workspace');
    expect(target.querySelector('[data-entry-id="s0.e1"] button[aria-label="Generate this one"]')?.hasAttribute('disabled')).toBe(true);
  });

  it('offers to restore unsaved changes from an earlier visit', async () => {
    const edited = sample();
    edited.sets[0].name = 'people';
    saveUnsaved(PATH, { revision: 'r1', document: edited });
    await mountPage();

    button('Restore').click();
    await settle();

    expect(target.querySelector<HTMLInputElement>('input[aria-label="Set name"]')?.value).toBe('people');
    expect(target.textContent).toContain('Unsaved changes');
  });

  it('drops unsaved changes made on an older revision', async () => {
    saveUnsaved(PATH, { revision: 'old', document: sample() });
    await mountPage();

    expect(Array.from(target.querySelectorAll('button')).some((item) => item.textContent?.trim() === 'Restore')).toBe(false);
    expect(localStorage.getItem('ziv.promptBuilder.unsaved')).toBe('{}');
  });

  it('undoes a change with ⌘Z outside text fields', async () => {
    await mountPage();

    button('＋ New set').click();
    await settle();
    expect(target.querySelectorAll('input[aria-label="Set name"]')).toHaveLength(2);
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', metaKey: true, bubbles: true }));
    await settle();

    expect(target.querySelectorAll('input[aria-label="Set name"]')).toHaveLength(1);
  });
});
