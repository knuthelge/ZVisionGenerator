import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { BrowseEntry, BrowseListing } from '$lib/types';

const browseApiMocks = vi.hoisted(() => ({
  listDirectory: vi.fn(),
  browsePlaces: vi.fn(),
  createFolder: vi.fn(),
  browsePreviewUrl: vi.fn(),
  browseMediaUrl: vi.fn(),
  browseInfo: vi.fn(),
}));
const layoutMock = vi.hoisted(() => ({ compact: false, mode: 'wide', coarse: false }));

const projectApiMocks = vi.hoisted(() => ({ listProjects: vi.fn(), createProject: vi.fn() }));

vi.mock('$lib/api/browse', () => browseApiMocks);
vi.mock('$lib/api/projects', () => projectApiMocks);
vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));

import { projectPick } from '$lib/state/projectPick.svelte';
import { projectsStore } from '$lib/state/projects.svelte';
import { formatBytes, formatModified, formatModifiedShort, readBrowseView, readRecentPaths, SORT_CHOICES } from '$lib/state/pathBrowser';
import { mountComponent, settle } from '../../../test-utils/mount';
import { placesFor, stubBrowseInfo, stubBrowseMediaUrl, stubBrowsePreviewUrl } from '../../../test-utils/pathBrowser';
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
  '/out/Night Bus': listing('/out/Night Bus', [entry('bus.png', '/out/Night Bus')]),
};

describe('PathBrowser', () => {
  beforeEach(() => {
    localStorage.clear();
    for (const mock of Object.values(browseApiMocks)) mock.mockReset();
    Object.assign(layoutMock, { compact: false, mode: 'wide', coarse: false });
    browseApiMocks.browsePreviewUrl.mockImplementation(stubBrowsePreviewUrl);
    browseApiMocks.browseMediaUrl.mockImplementation(stubBrowseMediaUrl);
    browseApiMocks.browseInfo.mockImplementation(stubBrowseInfo);
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

  describe('in a project', () => {
    beforeEach(async () => {
      projectApiMocks.listProjects.mockResolvedValue([{ name: 'Night Bus', path: '/out/Night Bus' }]);
      await projectsStore.load();
      projectPick.set('Night Bus');
    });

    afterEach(() => projectPick.set(null));

    it('opens an empty media field in the picked project\'s folder', async () => {
      await open({ purpose: 'reference_image' });

      expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('reference_image', '/out/Night Bus', expect.anything());
    });

    it('opens a field that has a value at that value', async () => {
      await open({ purpose: 'source_video', initialPath: '/models' });

      expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('source_video', '/models', expect.anything());
    });

    it('leaves a purpose that holds no project media at its own place', async () => {
      await open({ purpose: 'prompt_file' });

      expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('prompt_file', '', expect.anything());
    });

    it('opens at the purpose\'s place with All files', async () => {
      projectPick.set(null);
      await open({ purpose: 'reference_image' });

      expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('reference_image', '', expect.anything());
    });
  });

  it('chooses a file by double-click, closes and remembers it', async () => {
    const { onchoose } = await open();

    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 })));
    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 2 })));
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

  it('opens a folder once on double-click and chooses nothing in the folder it opens', async () => {
    const { onchoose } = await open({ initialPath: '/models' });

    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 })));
    await settle();
    // The second click of the double-click lands on the new listing's first row.
    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 2 })));
    flushSync(() => rows()[0].dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
    await settle();

    expect(browseApiMocks.listDirectory.mock.calls.map((call) => call[1])).toEqual(['/models', '/models/loras']);
    expect(onchoose).not.toHaveBeenCalled();
  });

  it('opens the dialog once the purpose is known, so it opens at its final size', async () => {
    let answer: (value: unknown) => void = () => {};
    browseApiMocks.browsePlaces.mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }));
    mountComponent(PathBrowser, { open: true, purpose: 'lora_file', initialPath: null, onchoose: vi.fn() });
    await settle();

    expect(document.body.querySelector('[data-testid="path-browser"]')).toBeNull();

    answer({ ...placesFor('lora_file'), places: [{ id: 'loras', label: 'LoRAs', path: '/models/loras', default: true }] });
    await settle();

    expect(document.body.querySelector('[data-testid="path-browser"]')).not.toBeNull();
  });

  it('opens the dialog with the reason when the purpose cannot be loaded', async () => {
    browseApiMocks.browsePlaces.mockRejectedValueOnce(new Error('Server down.'));
    browseApiMocks.listDirectory.mockRejectedValue(new Error('Server down.'));
    mountComponent(PathBrowser, { open: true, purpose: 'lora_file', initialPath: null, onchoose: vi.fn() });
    await settle();

    expect(document.body.querySelector('[role="alert"]')?.textContent).toBe('Server down.');
  });

  it('opens the dialog from the folder listing when the places cannot be loaded, and still closes', async () => {
    browseApiMocks.browsePlaces.mockRejectedValueOnce(new Error('Places unavailable.'));
    const onclose = vi.fn();
    mountComponent(PathBrowser, { open: true, purpose: 'lora_file', initialPath: null, onchoose: vi.fn(), onclose });
    await settle();

    expect(document.body.querySelector('[data-testid="path-browser"]')).not.toBeNull();
    expect(rowNames()).toEqual(['ink.safetensors', 'notes.txt']);
    flushSync(() => buttonByLabel('Close dialog').click());
    expect(onclose).toHaveBeenCalled();
  });

  describe('the scroll position', () => {
    const pane = () => list().parentElement!;

    it('returns to the top in a folder it opens, and is left alone when the same folder reloads', async () => {
      await open({ initialPath: '/models' });
      pane().scrollTop = 120;

      flushSync(() => rows()[0].click());
      await settle();
      expect(pane().scrollTop).toBe(0);

      pane().scrollTop = 120;
      buttonByLabel('Show all').click();
      await settle();
      expect(pane().scrollTop).toBe(120);
    });
  });

  describe('the filter field', () => {
    it('takes the first match with Enter on a computer', async () => {
      const { onchoose } = await open();
      const filter = document.body.querySelector<HTMLInputElement>('input[aria-label="Filter this folder"]')!;
      filter.focus();

      key(filter, 'Enter');

      expect(onchoose).toHaveBeenCalledWith('/models/loras/ink.safetensors');
    });
  });

  describe('sorting', () => {
    const DAY = 86400;
    const NOW = Date.now() / 1000;
    const sized = (name: string, size: number, modified: number, kind: BrowseEntry['kind'] = 'file'): BrowseEntry => ({
      name,
      path: `/pics/${name}`,
      kind,
      size: kind === 'file' ? size : null,
      modified,
      // As the server says: a folder is selectable only when the purpose chooses folders.
      selectable: kind === 'file',
    });
    // Server order is by name; take-10 is the oldest and the biggest.
    const PICS = listing('/pics', [
      sized('archive', 0, NOW - 90 * DAY, 'directory'),
      sized('take-1.png', 300, NOW - 10 * DAY),
      sized('take-2.png', 100, NOW - 3 * DAY),
      sized('take-10.png', 900, NOW - 40 * DAY),
      sized('zoo.png', 500, NOW - 60),
    ]);

    beforeEach(() => {
      browseApiMocks.listDirectory.mockImplementation(async () => PICS);
      browseApiMocks.browsePlaces.mockImplementation(async (name: 'reference_image') => placesFor(name));
    });

    const heading = (label: string) => Array.from(document.body.querySelectorAll<HTMLButtonElement>('[role="toolbar"][aria-label="Sort by"] button')).find((item) => item.textContent?.trim() === label)!;
    const click = async (el: HTMLElement) => {
      flushSync(() => el.click());
      await settle();
    };
    const selectedName = () => document.body.querySelector<HTMLElement>('[role="option"][aria-selected="true"]')?.dataset.name;

    it('lists newest first by default, with folders on top', async () => {
      await open({ purpose: 'reference_image' });

      expect(rowNames()).toEqual(['archive', 'zoo.png', 'take-2.png', 'take-1.png', 'take-10.png']);
      expect(heading('Modified').getAttribute('aria-pressed')).toBe('true');
      expect(heading('Modified').getAttribute('aria-label')).toBe('Modified, sorted descending');
      expect(heading('Name').getAttribute('aria-pressed')).toBe('false');
      expect(heading('Name').hasAttribute('aria-label')).toBe(false);
    });

    it('sorts by a heading and reverses on a second click', async () => {
      await open({ purpose: 'reference_image' });

      await click(heading('Name'));
      expect(rowNames()).toEqual(['archive', 'take-1.png', 'take-2.png', 'take-10.png', 'zoo.png']);
      expect(heading('Name').getAttribute('aria-label')).toBe('Name, sorted ascending');

      await click(heading('Name'));
      expect(rowNames()).toEqual(['archive', 'zoo.png', 'take-10.png', 'take-2.png', 'take-1.png']);
      expect(heading('Name').getAttribute('aria-label')).toBe('Name, sorted descending');

      await click(heading('Size'));
      expect(rowNames()).toEqual(['archive', 'take-10.png', 'zoo.png', 'take-1.png', 'take-2.png']);

      await click(heading('Size'));
      expect(rowNames()).toEqual(['archive', 'take-2.png', 'take-1.png', 'zoo.png', 'take-10.png']);

      await click(heading('Modified'));
      expect(rowNames()).toEqual(['archive', 'zoo.png', 'take-2.png', 'take-1.png', 'take-10.png']);

      await click(heading('Modified'));
      expect(rowNames()).toEqual(['archive', 'take-10.png', 'take-1.png', 'take-2.png', 'zoo.png']);
    });

    it('keeps the same file selected after a re-sort, and Enter chooses it', async () => {
      const { onchoose } = await open({ purpose: 'reference_image' });
      key(list(), 'ArrowDown');
      expect(selectedName()).toBe('take-2.png');

      await click(heading('Size'));
      expect(rowNames()).toEqual(['archive', 'take-10.png', 'zoo.png', 'take-1.png', 'take-2.png']);
      expect(selectedName()).toBe('take-2.png');
      expect(list().getAttribute('aria-activedescendant')).toBe('path-browser-row-4');

      key(list(), 'Enter');

      expect(onchoose).toHaveBeenCalledWith('/pics/take-2.png');
    });

    it('selects the newest file first and moves through the sorted order', async () => {
      const { onchoose } = await open({ purpose: 'reference_image' });

      expect(selectedName()).toBe('zoo.png');
      key(list(), 'ArrowDown');
      key(list(), 'Enter');

      expect(onchoose).toHaveBeenCalledWith('/pics/take-2.png');
    });

    it('jumps to a typed letter along the sorted order', async () => {
      await open({ purpose: 'reference_image' });
      key(list(), 'End');
      expect(selectedName()).toBe('take-10.png');

      key(list(), 't');
      expect(selectedName()).toBe('take-2.png');
      key(list(), 't');
      expect(selectedName()).toBe('take-1.png');

      await click(heading('Name'));
      key(list(), 'Home');
      key(list(), 't');
      expect(selectedName()).toBe('take-1.png');
      key(list(), 't');
      expect(selectedName()).toBe('take-2.png');
      key(list(), 't');
      expect(selectedName()).toBe('take-10.png');
    });

    it('opens on the file the field names, wherever the order puts it', async () => {
      browseApiMocks.listDirectory.mockImplementation(async () => ({ ...PICS, focus: 'take-10.png' }));
      const { onchoose } = await open({ purpose: 'reference_image', initialPath: '/pics/take-10.png' });

      expect(selectedName()).toBe('take-10.png');
      key(list(), 'Enter');

      expect(onchoose).toHaveBeenCalledWith('/pics/take-10.png');
    });

    it('writes the order to the one remembered key and reads it for every picker', async () => {
      const first = mountComponent(PathBrowser, { open: true, purpose: 'reference_image', initialPath: null, onchoose: vi.fn() });
      await settle();
      await click(heading('Size'));
      expect(JSON.parse(localStorage.getItem('ziv-browse-view-v1')!)).toMatchObject({ sort: 'size', dir: 'desc' });
      await first.unmount();

      browseApiMocks.browsePlaces.mockImplementation(async () => placesFor('lora_file'));
      await open({ purpose: 'lora_file' });

      expect(heading('Size').getAttribute('aria-pressed')).toBe('true');
      expect(rowNames()).toEqual(['archive', 'take-10.png', 'zoo.png', 'take-1.png', 'take-2.png']);
    });

    it('keeps the list and preview parts of the remembered view when sorting', async () => {
      localStorage.setItem('ziv-browse-view-v1', JSON.stringify({ sort: 'name', dir: 'asc', view: 'list', preview: false }));
      await open({ purpose: 'reference_image' });

      await click(heading('Name'));

      expect(readBrowseView()).toEqual({ sort: 'name', dir: 'desc', view: 'list', preview: false });
    });

    it('shows the time in the date and the full name on hover and in the footer', async () => {
      await open({ purpose: 'reference_image' });

      const modified = rows().map((row) => row.querySelector('[data-testid="entry-modified"]')?.textContent);
      const dateOf = (name: string) => formatModified(PICS.entries.find((entry) => entry.name === name)!.modified!);
      expect(modified[1]).toBe(dateOf('zoo.png'));
      expect(modified[2]).toBe(dateOf('take-2.png'));
      expect(rows()[1].querySelector('[data-testid="entry-name"]')?.getAttribute('title')).toBe('zoo.png');
      expect(document.body.querySelector('[data-testid="chosen"]')?.textContent).toBe('zoo.png');
    });

    it('sorts every picker, with its heading buttons and no thumbnails', async () => {
      browseApiMocks.listDirectory.mockImplementation(async () => PICS);
      await open({ purpose: 'lora_file' });

      await click(heading('Name'));

      expect(rowNames()).toEqual(['archive', 'take-1.png', 'take-2.png', 'take-10.png', 'zoo.png']);
    });
  });

  describe('for the image, audio and video pickers', () => {
    const NOW = Date.now() / 1000;
    const media = (name: string, ageSeconds: number, kind: BrowseEntry['kind'] = 'file'): BrowseEntry => ({
      name,
      path: `/pics/${name}`,
      kind,
      size: kind === 'file' ? 2048 : null,
      modified: NOW - ageSeconds,
      selectable: kind === 'file',
    });
    // Newest first: fox, owl, cat, with a folder on top.
    const PICS = listing('/pics', [media('archive', 9000, 'directory'), media('cat.png', 300), media('fox.png', 10), media('owl.png', 100)]);
    const previewPane = () => document.body.querySelector<HTMLElement>('[data-testid="file-preview"]');
    const click = async (el: HTMLElement) => {
      flushSync(() => el.click());
      await settle();
    };
    const previewButton = () => document.body.querySelector<HTMLButtonElement>('[aria-keyshortcuts="Alt+P"]')!;
    const selectedName = () => document.body.querySelector<HTMLElement>('[role="option"][aria-selected="true"]')?.dataset.name;

    beforeEach(() => {
      browseApiMocks.listDirectory.mockImplementation(async () => PICS);
      browseApiMocks.browsePlaces.mockImplementation(async (name: 'reference_image') => placesFor(name));
    });

    describe('thumbnails and the switch', () => {
      it('shows a thumbnail for each file and keeps the icon on a folder', async () => {
        await open({ purpose: 'reference_image' });

        const thumbs = Array.from(rows()[0].parentElement!.querySelectorAll('img')).map((img) => img.getAttribute('src'));
        expect(thumbs).toEqual(['/preview/reference_image/pics/fox.png', '/preview/reference_image/pics/owl.png', '/preview/reference_image/pics/cat.png']);
        expect(rows()[0].querySelector('img')).toBeNull();
      });

      it('keeps the icon of a file the picker does not take', async () => {
        browseApiMocks.listDirectory.mockImplementation(async () => ({ ...PICS, entries: [...PICS.entries, { ...media('notes.txt', 5), selectable: false }] }));
        await open({ purpose: 'reference_image' });

        const notes = rows().find((row) => row.dataset.name === 'notes.txt')!;
        expect(notes.querySelector('img')).toBeNull();
      });

      it('shows no thumbnails, switch or preview in a LoRA picker and asks the server for none', async () => {
        await open({ purpose: 'lora_file' });

        expect(document.body.querySelectorAll('[role="option"] img')).toHaveLength(0);
        expect(document.body.querySelector('[role="group"][aria-label="View"]')).toBeNull();
        expect(document.body.querySelector('[aria-keyshortcuts="Alt+P"]')).toBeNull();
        expect(previewPane()).toBeNull();
        flushSync(() => key(list(), 'ArrowDown'));
        await settle(300);
        expect(browseApiMocks.browsePreviewUrl).not.toHaveBeenCalled();
        expect(browseApiMocks.browseMediaUrl).not.toHaveBeenCalled();
        expect(browseApiMocks.browseInfo).not.toHaveBeenCalled();
      });

      it('shows no thumbnails or switch in a folder picker', async () => {
        await open({ purpose: 'output_directory' });

        expect(document.body.querySelector('[role="group"][aria-label="View"]')).toBeNull();
        expect(browseApiMocks.browsePreviewUrl).not.toHaveBeenCalled();
      });

      it('shows the List | Grid switch with List pressed first', async () => {
        await open({ purpose: 'reference_image' });

        const group = document.body.querySelector('[role="group"][aria-label="View"]')!;
        expect(group.querySelector('[data-value="list"]')?.getAttribute('aria-pressed')).toBe('true');
        expect(group.querySelector('[data-value="grid"]')?.getAttribute('aria-pressed')).toBe('false');
      });
    });

    describe('the grid', () => {
      const gridButton = () => document.body.querySelector<HTMLElement>('[role="group"][aria-label="View"] [data-value="grid"]')!;
      const tiles = () => rows();
      const tileNames = () => tiles().map((tile) => tile.dataset.name);
      const sortButton = () => document.body.querySelector<HTMLButtonElement>('[aria-label^="Sort:"]')!;
      const choiceLabel = (id: string) => SORT_CHOICES.find((choice) => choice.id === id)!.label;

      async function openGrid(): Promise<void> {
        await open({ purpose: 'reference_image' });
        await click(gridButton());
      }

      it('shows tiles with the name and a short date, remembers the view and drops the headings', async () => {
        await openGrid();

        expect(list().getAttribute('data-view')).toBe('grid');
        expect(tileNames()).toEqual(['archive', 'fox.png', 'owl.png', 'cat.png']);
        expect(tiles()[1].querySelector('img')?.getAttribute('src')).toBe('/preview/reference_image/pics/fox.png');
        const meta = tiles()[1].querySelector('[data-testid="entry-meta"]')?.textContent;
        expect(meta).toBe(formatModifiedShort(PICS.entries.find((entry) => entry.name === 'fox.png')!.modified!));
        expect(document.body.querySelector('[role="toolbar"][aria-label="Sort by"]')).toBeNull();
        expect(readBrowseView().view).toBe('grid');
      });

      it('opens on the remembered view', async () => {
        localStorage.setItem('ziv-browse-view-v1', JSON.stringify({ sort: 'modified', dir: 'desc', view: 'grid', preview: true }));
        await open({ purpose: 'reference_image' });

        expect(tiles()).toHaveLength(4);
      });

      it('moves by column and by row with the arrow keys', async () => {
        const original = window.getComputedStyle;
        vi.spyOn(window, 'getComputedStyle').mockImplementation((el: Element, pseudo?: string | null) => {
          const style = original(el, pseudo);
          return el === list() ? ({ gridTemplateColumns: '132px 132px' } as CSSStyleDeclaration) : style;
        });
        await openGrid();
        // Tiles in two columns: archive fox / owl cat.
        expect(selectedName()).toBe('fox.png');

        key(list(), 'ArrowDown');
        expect(selectedName()).toBe('cat.png');
        key(list(), 'ArrowLeft');
        expect(selectedName()).toBe('owl.png');
        key(list(), 'ArrowUp');
        expect(selectedName()).toBe('archive');
        key(list(), 'ArrowRight');
        expect(selectedName()).toBe('fox.png');
        key(list(), 'ArrowUp');
        expect(selectedName()).toBe('fox.png');
        key(list(), 'End');
        expect(selectedName()).toBe('cat.png');
        key(list(), 'Home');
        expect(selectedName()).toBe('archive');
        vi.restoreAllMocks();
      });

      it('opens a folder tile once on double-click and chooses nothing in the folder it opens', async () => {
        const { onchoose } = await open({ purpose: 'reference_image' });
        await click(gridButton());
        const opened = browseApiMocks.listDirectory.mock.calls.length;

        flushSync(() => tiles()[0].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 })));
        await settle();
        // The second click lands on the tile now under the pointer: the first file of the new listing.
        flushSync(() => tiles()[1].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 2 })));
        flushSync(() => tiles()[1].dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
        await settle();

        expect(browseApiMocks.listDirectory.mock.calls.length).toBe(opened + 1);
        expect(onchoose).not.toHaveBeenCalled();
      });

      it('chooses a tile that was double-clicked', async () => {
        const { onchoose } = await open({ purpose: 'reference_image' });
        await click(gridButton());

        flushSync(() => tiles()[2].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 })));
        flushSync(() => tiles()[2].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 2 })));
        flushSync(() => tiles()[2].dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));

        expect(onchoose).toHaveBeenCalledWith('/pics/owl.png');
      });

      it('keeps the focus in the sort menu from the keyboard and picks an order without closing the picker', async () => {
        await openGrid();

        sortButton().focus();
        await click(sortButton());
        const menu = document.body.querySelector<HTMLElement>('[role="menu"]')!;
        expect(menu.contains(document.activeElement)).toBe(true);

        key(document.activeElement as HTMLElement, 'ArrowDown');
        await settle();
        expect(menu.contains(document.activeElement)).toBe(true);

        await click(document.activeElement as HTMLElement);

        expect(document.body.querySelector('[data-testid="path-browser"]')).not.toBeNull();
        expect(document.body.querySelector('[role="menu"]')).toBeNull();
        expect(readBrowseView().sort).toBe('name');
      });

      it('chooses the selected tile with Enter and by double-click', async () => {
        const { onchoose } = await open({ purpose: 'reference_image' });
        await click(gridButton());

        flushSync(() => tiles()[2].dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 })));
        key(list(), 'Enter');

        expect(onchoose).toHaveBeenCalledWith('/pics/owl.png');
      });

      it('sorts from one menu button holding the six orders, and keeps the view and the preview', async () => {
        await openGrid();
        expect(sortButton().getAttribute('aria-label')).toBe(`Sort: ${choiceLabel('modified-desc')}`);

        await click(sortButton());
        const menu = document.body.querySelector('[role="menu"]')!;
        expect(Array.from(menu.querySelectorAll('[role="menuitemradio"]')).map((item) => item.getAttribute('data-action'))).toEqual(['name-asc', 'name-desc', 'modified-desc', 'modified-asc', 'size-desc', 'size-asc']);
        expect(menu.querySelector('[aria-checked="true"]')?.getAttribute('data-action')).toBe('modified-desc');

        await click(menu.querySelector<HTMLElement>('[data-action="name-asc"]')!);

        expect(tileNames()).toEqual(['archive', 'cat.png', 'fox.png', 'owl.png']);
        expect(sortButton().getAttribute('aria-label')).toBe(`Sort: ${choiceLabel('name-asc')}`);
        expect(readBrowseView()).toEqual({ sort: 'name', dir: 'asc', view: 'grid', preview: true });
      });

      it('keeps the hidden-files note, with its Show all button, beside the tiles and not among them', async () => {
        browseApiMocks.listDirectory.mockImplementation(async () => ({ ...PICS, hidden_count: 3 }));
        await openGrid();

        const note = document.body.querySelector<HTMLElement>('[data-testid="hidden-note"]')!;
        expect(note.closest('[role="option"]')).toBeNull();
        expect(tileNames()).toEqual(['archive', 'fox.png', 'owl.png', 'cat.png']);

        await click(note.querySelector('button')!);

        expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('reference_image', '/pics', { hidden: false, all: true });
      });

      it('says nothing matches a filter in the grid, with no tiles', async () => {
        await openGrid();
        const filter = document.body.querySelector<HTMLInputElement>('input[aria-label="Filter this folder"]')!;

        filter.value = 'zzz';
        filter.dispatchEvent(new Event('input', { bubbles: true }));
        flushSync();

        expect(tiles()).toHaveLength(0);
        expect(document.body.querySelector('[data-testid="empty-note"]')).not.toBeNull();
      });

      it('shows placeholder tiles, not entries, while the folder loads', async () => {
        browseApiMocks.listDirectory.mockImplementation(async () => ({ ...PICS, hidden_count: 3 }));
        await openGrid();
        browseApiMocks.listDirectory.mockImplementation(() => new Promise(() => {}));

        await click(document.body.querySelector<HTMLElement>('[data-testid="hidden-note"] button')!);
        await settle(200);

        expect(document.body.querySelectorAll('[data-testid="skeleton"]').length).toBeGreaterThan(1);
        expect(tiles()).toHaveLength(0);
        expect(list().getAttribute('data-view')).toBe('grid');
      });

      it('goes back to the list with its headings', async () => {
        await openGrid();

        await click(document.body.querySelector<HTMLElement>('[role="group"][aria-label="View"] [data-value="list"]')!);

        expect(document.body.querySelector('[role="toolbar"][aria-label="Sort by"]')).not.toBeNull();
        expect(rows()).toHaveLength(4);
        expect(readBrowseView().view).toBe('list');
      });
    });

    describe('the preview', () => {
      beforeEach(() => vi.useFakeTimers());
      afterEach(() => vi.useRealTimers());

      it('shows the selected file in a pane on first use and follows the arrow keys', async () => {
        await open({ purpose: 'reference_image' });
        await settle(200);

        expect(previewPane()?.querySelector('[data-testid="preview-name"]')?.textContent).toBe('fox.png');
        expect(previewPane()?.querySelector('img')?.getAttribute('src')).toBe('/preview/reference_image/pics/fox.png');
        expect(previewButton().getAttribute('aria-pressed')).toBe('true');

        key(list(), 'ArrowDown');
        await settle(200);

        expect(previewPane()?.querySelector('[data-testid="preview-name"]')?.textContent).toBe('owl.png');
      });

      it('shows a folder as its icon', async () => {
        await open({ purpose: 'reference_image' });
        key(list(), 'Home');
        await settle(200);

        expect(previewPane()?.querySelector('[data-testid="preview-name"]')?.textContent).toBe('archive');
        expect(previewPane()?.querySelector('[data-testid="preview-icon"]')).not.toBeNull();
        expect(browseApiMocks.browseInfo).not.toHaveBeenCalledWith('reference_image', '/pics/archive', expect.anything());
      });

      it('toggles with Alt+P, matched by code, and remembers it', async () => {
        await open({ purpose: 'reference_image' });
        expect(previewPane()).not.toBeNull();

        // On macOS Option+P types a character, so the key is not "p".
        key(list(), 'π', { code: 'KeyP', altKey: true });
        expect(previewPane()).toBeNull();
        expect(previewButton().getAttribute('aria-pressed')).toBe('false');
        expect(readBrowseView().preview).toBe(false);

        key(list(), 'π', { code: 'KeyP', altKey: true });
        expect(previewPane()).not.toBeNull();
        expect(readBrowseView().preview).toBe(true);
      });

      it('ignores P without Alt, and Alt+P with Ctrl', async () => {
        await open({ purpose: 'reference_image' });
        key(list(), 'p', { code: 'KeyP' });
        key(list(), 'p', { code: 'KeyP', altKey: true, ctrlKey: true });
        expect(previewPane()).not.toBeNull();
      });

      it('does nothing on Alt+P in a picker without previews', async () => {
        await open({ purpose: 'lora_file' });

        key(list(), 'π', { code: 'KeyP', altKey: true });

        expect(previewPane()).toBeNull();
        expect(readBrowseView().preview).toBe(true);
        expect(localStorage.getItem('ziv-browse-view-v1')).toBeNull();
      });

      it('toggles with the Preview button, and opens hidden when it was left hidden', async () => {
        localStorage.setItem('ziv-browse-view-v1', JSON.stringify({ sort: 'modified', dir: 'desc', view: 'list', preview: false }));
        await open({ purpose: 'reference_image' });
        expect(previewPane()).toBeNull();

        await click(previewButton());

        expect(previewPane()).not.toBeNull();
        expect(readBrowseView().preview).toBe(true);
      });

      it('puts the preview in a strip under the list on a portrait tablet', async () => {
        Object.assign(layoutMock, { mode: 'tall' });
        await open({ purpose: 'reference_image' });
        await settle(200);

        expect(document.body.querySelector('[data-testid="file-preview"][data-arrangement="strip"]')).not.toBeNull();
        expect(document.body.querySelector('[data-testid="file-preview"][data-arrangement="pane"]')).toBeNull();
      });

      it('puts it beside the list on a wide screen', async () => {
        await open({ purpose: 'reference_image' });

        expect(document.body.querySelector('[data-testid="file-preview"][data-arrangement="pane"]')).not.toBeNull();
        expect(document.body.querySelector('[data-testid="file-preview"][data-arrangement="strip"]')).toBeNull();
      });
    });
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
