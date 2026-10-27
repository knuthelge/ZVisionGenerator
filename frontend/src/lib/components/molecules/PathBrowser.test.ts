import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { BrowseEntry, BrowseListing } from '$lib/types';

const browseApiMocks = vi.hoisted(() => ({
  listDirectory: vi.fn(),
  browsePlaces: vi.fn(),
  createFolder: vi.fn(),
}));

vi.mock('$lib/api/browse', () => browseApiMocks);

import { readRecentPaths } from '$lib/state/pathBrowser';
import { mountComponent, settle } from '../../../test-utils/mount';
import { placesFor } from '../../../test-utils/pathBrowser';
import PathBrowser from './PathBrowser.svelte';

function entry(name: string, folder: string, kind: BrowseEntry['kind'] = 'file', selectable = true): BrowseEntry {
  return { name, path: `${folder}/${name}`.replace('//', '/'), kind, size: kind === 'file' ? 2048 : null, modified: 0, selectable };
}

function listing(path: string, entries: BrowseEntry[], extra: Partial<BrowseListing> = {}): BrowseListing {
  const parent = path === '/' ? null : path.slice(0, path.lastIndexOf('/')) || '/';
  return { path, parent, separator: '/', focus: null, entries, hidden_count: 0, ...extra };
}

const FOLDERS: Record<string, BrowseListing> = {
  '/models': listing('/models', [entry('loras', '/models', 'directory'), entry('base.safetensors', '/models')]),
  '/models/loras': listing('/models/loras', [entry('ink.safetensors', '/models/loras'), entry('notes.txt', '/models/loras', 'file', false)], { hidden_count: 2 }),
  '/': listing('/', [entry('models', '/', 'directory')]),
};

describe('PathBrowser', () => {
  beforeEach(() => {
    localStorage.clear();
    for (const mock of Object.values(browseApiMocks)) mock.mockReset();
    browseApiMocks.browsePlaces.mockImplementation(async (purpose: 'lora_file' | 'output_directory') => ({
      ...placesFor(purpose),
      places: [{ id: 'home', label: 'Home', path: '/', default: false }, { id: 'loras', label: 'LoRAs', path: '/models/loras', default: true }],
    }));
    browseApiMocks.listDirectory.mockImplementation(async (_purpose: string, path: string) => {
      const found = FOLDERS[path || '/models/loras'];
      if (!found) throw new Error(`No folder at ${path}.`);
      return found;
    });
  });

  async function open(props: Record<string, unknown> = {}): Promise<{ onchoose: ReturnType<typeof vi.fn> }> {
    const onchoose = vi.fn();
    mountComponent(PathBrowser, { open: true, purpose: 'lora_file', initialPath: null, onchoose, ...props });
    await settle();
    return { onchoose };
  }

  const rows = () => Array.from(document.body.querySelectorAll<HTMLElement>('[role="option"]'));
  const rowNames = () => rows().map((row) => row.querySelector('.truncate')?.textContent);
  const buttonByLabel = (label: string) => Array.from(document.body.querySelectorAll<HTMLButtonElement>('button')).find((item) => item.getAttribute('aria-label') === label || item.textContent?.trim() === label)!;
  const list = () => document.body.querySelector<HTMLElement>('[role="listbox"]')!;
  const key = (el: HTMLElement, name: string, init: KeyboardEventInit = {}) => flushSync(() => el.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true, ...init })));

  it('opens at the purpose place and lists what the server sent', async () => {
    await open();

    expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('lora_file', '', { hidden: false, all: false });
    expect(rowNames()).toEqual(['ink.safetensors', 'notes.txt']);
    expect(document.body.textContent).toContain('2 other files hidden');
  });

  it('chooses a file by double-click, closes and remembers it', async () => {
    const { onchoose } = await open();

    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));

    expect(onchoose).toHaveBeenCalledWith('/models/loras/ink.safetensors');
    expect(document.body.querySelector('[data-testid="path-browser"]')).toBeNull();
    expect(readRecentPaths('lora_file')).toEqual(['/models/loras/ink.safetensors']);
  });

  it('keeps Choose off for a file the purpose cannot take', async () => {
    await open();

    flushSync(() => rows()[1].click());

    expect(buttonByLabel('Choose file').disabled).toBe(true);
  });

  it('opens folders, goes up and comes back', async () => {
    await open({ initialPath: '/models' });

    flushSync(() => rows()[0].click());
    await settle();
    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '/models/loras', expect.anything());

    buttonByLabel('Up one folder').click();
    await settle();
    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '/models', expect.anything());

    buttonByLabel('Back').click();
    await settle();
    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '/models/loras', expect.anything());
  });

  it('moves with the arrow keys and takes the highlighted row with Enter', async () => {
    const { onchoose } = await open({ initialPath: '/models' });

    key(list(), 'ArrowDown');
    key(list(), 'Enter');

    expect(onchoose).toHaveBeenCalledWith('/models/base.safetensors');
  });

  it('goes to a typed path', async () => {
    await open();

    buttonByLabel('Type a path').click();
    flushSync();
    const path = document.body.querySelector<HTMLInputElement>('input[aria-label="Path"]')!;
    path.value = '/models';
    path.dispatchEvent(new Event('input', { bubbles: true }));
    key(path, 'Enter');
    await settle();

    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '/models', expect.anything());
    expect(rowNames()).toEqual(['loras', 'base.safetensors']);
  });

  it('shows a folder it cannot open in place and keeps the last one', async () => {
    await open();

    buttonByLabel('Home').click();
    browseApiMocks.listDirectory.mockRejectedValueOnce(new Error('Permission denied: /root.'));
    document.body.querySelector<HTMLButtonElement>('[aria-label="Folder path"] button')!.click();
    await settle();

    expect(document.body.querySelector('[role="alert"]')?.textContent).toBe('Permission denied: /root.');
  });

  it('asks for every file after Show all', async () => {
    await open();

    buttonByLabel('Show all').click();
    await settle();

    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '/models/loras', { hidden: false, all: true });
  });

  it('does not submit the form it sits in when Enter is pressed in the filter', async () => {
    const form = document.createElement('form');
    const submit = vi.fn((event: Event) => event.preventDefault());
    form.addEventListener('submit', submit);
    form.appendChild(document.createElement('button'));
    const onchoose = vi.fn();
    mountComponent(PathBrowser, { open: true, purpose: 'lora_file', initialPath: null, onchoose }, form);
    await settle();

    const filter = document.body.querySelector<HTMLInputElement>('input[aria-label="Filter this folder"]')!;
    filter.value = 'ink';
    filter.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const enter = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
    flushSync(() => filter.dispatchEvent(enter));

    expect(enter.defaultPrevented).toBe(true);
    expect(submit).not.toHaveBeenCalled();
    expect(onchoose).toHaveBeenCalledWith('/models/loras/ink.safetensors');
  });

  it('keeps the dialog open when a typed path cannot be opened', async () => {
    const { onchoose } = await open({ initialPath: '/models/loras' });

    buttonByLabel('Type a path').click();
    flushSync();
    const path = document.body.querySelector<HTMLInputElement>('input[aria-label="Path"]')!;
    path.value = '/models/missing.safetensors';
    path.dispatchEvent(new Event('input', { bubbles: true }));
    key(path, 'Enter');
    await settle();

    expect(onchoose).not.toHaveBeenCalled();
    expect(document.body.querySelector('[role="alert"]')?.textContent).toBe('No folder at /models/missing.safetensors.');
  });

  it('opens the purpose place, saying why, when the field path is gone', async () => {
    await open({ initialPath: '/gone/clip.safetensors' });

    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '', expect.anything());
    expect(rowNames()).toEqual(['ink.safetensors', 'notes.txt']);
    expect(document.body.querySelector('[role="alert"]')?.textContent).toBe('No folder at /gone/clip.safetensors.');
  });

  it('opens a folder once on double-click', async () => {
    await open({ initialPath: '/models' });

    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 })));
    await settle();
    // The second click of the double-click lands on the new listing's first row.
    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 2 })));
    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
    await settle();

    expect(browseApiMocks.listDirectory.mock.calls.map((call) => call[1])).toEqual(['/models', '/models/loras']);
  });

  describe('for a folder', () => {
    it('uses the folder it is in', async () => {
      const { onchoose } = await open({ purpose: 'output_directory', initialPath: '/models' });

      expect(buttonByLabel('Use “models”').disabled).toBe(false);
      buttonByLabel('Use “models”').click();

      expect(onchoose).toHaveBeenCalledWith('/models');
    });

    it('creates a folder and opens it', async () => {
      browseApiMocks.createFolder.mockResolvedValue({ path: '/models/loras' });
      await open({ purpose: 'output_directory', initialPath: '/models' });

      buttonByLabel('New folder').click();
      flushSync();
      const name = document.body.querySelector<HTMLInputElement>('input[aria-label="New folder name"]')!;
      name.value = 'renders';
      name.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      key(name, 'Enter');
      await settle();

      expect(browseApiMocks.createFolder).toHaveBeenCalledWith('output_directory', '/models', 'renders');
      expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('output_directory', '/models/loras', expect.anything());
    });
  });
});
