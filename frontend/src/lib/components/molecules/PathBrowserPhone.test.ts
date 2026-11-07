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

// The real layout store over a controllable matchMedia, so a phone held sideways is a screen where only the height query matches.
const screen = vi.hoisted(() => {
  type Listener = (event: { matches: boolean }) => void;
  const state = { width: false, height: false };
  const lists = new Map<string, { matches: () => boolean; listeners: Set<Listener> }>();
  const matchesNow = (query: string) =>
    query.split(',').some((part) => (part.includes('max-width: 639px') && state.width) || (part.includes('max-height: 499px') && state.height));
  vi.stubGlobal('matchMedia', (query: string) => {
    let list = lists.get(query);
    if (!list) {
      list = { matches: () => matchesNow(query), listeners: new Set() };
      lists.set(query, list);
    }
    const entry = list;
    return {
      get matches() { return entry.matches(); },
      media: query,
      addEventListener: (_type: string, listener: Listener) => { entry.listeners.add(listener); },
      removeEventListener: (_type: string, listener: Listener) => { entry.listeners.delete(listener); },
    };
  });
  return {
    /** Put the screen at *next*; every query that now answers differently tells its listeners. */
    set(next: { width: boolean; height: boolean }) {
      const before = new Map(Array.from(lists, ([query, list]) => [query, list.matches()]));
      Object.assign(state, next);
      lists.forEach((list, query) => {
        const now = list.matches();
        if (now !== before.get(query)) list.listeners.forEach((listener) => listener({ matches: now }));
      });
    },
  };
});

vi.mock('$lib/api/browse', () => browseApiMocks);

import { formatBytes, formatModified, itemCountLabel, readBrowseView } from '$lib/state/pathBrowser';
import { mountComponent, settle } from '../../../test-utils/mount';
import { placesFor, stubBrowseInfo, stubBrowseMediaUrl, stubBrowsePreviewUrl } from '../../../test-utils/pathBrowser';
import PathBrowser from './PathBrowser.svelte';

const NOW = Date.now() / 1000;
const LONG_NAME = 'fox-portrait_2026-11-05_21-38-51.png';

function media(name: string, ageSeconds: number, kind: BrowseEntry['kind'] = 'file'): BrowseEntry {
  return { name, path: `/pics/${name}`, kind, size: kind === 'file' ? 2048 : null, modified: NOW - ageSeconds, selectable: kind === 'file' };
}

function listing(path: string, entries: BrowseEntry[], extra: Partial<BrowseListing> = {}): BrowseListing {
  return { path, parent: '/', separator: '/', focus: null, entries, hidden_count: 0, ...extra };
}

// Newest first: the long-named fox, owl, cat, with a folder on top.
const PICS = listing('/pics', [media('archive', 9000, 'directory'), media('cat.png', 300), media(LONG_NAME, 10), media('owl.png', 100)]);
const MANY = listing('/pics', Array.from({ length: 9 }, (_, index) => media(`pic-${index}.png`, index * 10 + 10)));
const FOLDERS = listing('/pics', [media('archive', 9000, 'directory'), media('older', 100, 'directory')]);
const LORAS = listing('/pics', [media('ink.safetensors', 10)]);

const PORTRAIT = { width: true, height: false };
const LANDSCAPE = { width: false, height: true };

describe('PathBrowser on a phone', () => {
  beforeEach(() => {
    localStorage.clear();
    for (const mock of Object.values(browseApiMocks)) mock.mockReset();
    browseApiMocks.browsePreviewUrl.mockImplementation(stubBrowsePreviewUrl);
    browseApiMocks.browseMediaUrl.mockImplementation(stubBrowseMediaUrl);
    browseApiMocks.browseInfo.mockImplementation(stubBrowseInfo);
    browseApiMocks.browsePlaces.mockImplementation(async (name: 'reference_image') => ({
      ...placesFor(name),
      places: [{ id: 'home', label: 'Home', path: '/', default: false }, { id: 'output', label: 'Outputs', path: '/pics', default: true }],
    }));
    browseApiMocks.listDirectory.mockImplementation(async () => PICS);
  });

  afterEach(() => {
    screen.set({ width: false, height: false });
    vi.restoreAllMocks();
  });

  async function open(viewport: { width: boolean; height: boolean }, props: Record<string, unknown> = {}): Promise<{ onchoose: ReturnType<typeof vi.fn> }> {
    screen.set(viewport);
    const onchoose = vi.fn();
    mountComponent(PathBrowser, { open: true, purpose: 'reference_image', initialPath: null, onchoose, ...props });
    await settle();
    return { onchoose };
  }

  const rows = () => Array.from(document.body.querySelectorAll<HTMLElement>('[role="option"]'));
  const rowFor = (name: string) => rows().find((row) => row.dataset.name === name)!;
  const rowNames = () => rows().map((row) => row.dataset.name);
  const metaOf = (name: string) => rowFor(name).querySelector('[data-testid="entry-meta"]')?.textContent;
  const list = () => document.body.querySelector<HTMLElement>('[role="listbox"]')!;
  const toolbar = () => document.body.querySelector<HTMLElement>('[data-testid="browser-toolbar"]')!;
  const byLabel = (label: string, root: ParentNode = document.body) => root.querySelector<HTMLElement>(`[aria-label="${label}"]`);
  const sheet = (name: string) => document.body.querySelector<HTMLElement>(`[role="dialog"][aria-label="${name}"], [role="menu"][aria-label="${name}"]`);
  const sheetLayer = (name: string) => sheet(name)?.closest<HTMLElement>('[data-layer]')?.dataset.layer;
  const click = async (el: HTMLElement | null) => {
    flushSync(() => el!.click());
    await settle();
  };
  const tap = (el: HTMLElement, detail: number) => flushSync(() => el.dispatchEvent(new MouseEvent('click', { bubbles: true, detail })));
  const key = (el: HTMLElement, name: string, init: KeyboardEventInit = {}) => flushSync(() => el.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true, ...init })));
  const sheetItems = (name: string) => Array.from(sheet(name)?.querySelectorAll<HTMLElement>('[role="menuitemradio"]') ?? []).map((item) => item.dataset.action);
  const SORT_ACTIONS = ['name-asc', 'name-desc', 'modified-desc', 'modified-asc', 'size-desc', 'size-asc'];
  const isFilter = (item: Element) => item.querySelector('input[aria-label="Filter this folder"]') !== null;
  const toolbarLabels = () => Array.from(toolbar().querySelectorAll('button, [role="group"], label')).map((item) => item.getAttribute('aria-label') ?? (isFilter(item) ? 'Filter' : ''));

  describe.each([
    ['upright', PORTRAIT],
    ['sideways', LANDSCAPE],
  ])('held %s', (_label, viewport) => {
    it('takes the phone layout and drops the desktop arrangement', async () => {
      await open(viewport);

      expect(document.body.querySelector('[data-testid="path-browser"]')?.getAttribute('data-layout')).toBe(viewport === PORTRAIT ? 'portrait' : 'landscape');
      expect(document.body.querySelector('nav[aria-label="Places"]')).toBeNull();
      expect(document.body.querySelector('[role="toolbar"][aria-label="Sort by"]')).toBeNull();
      expect(document.body.querySelector('[data-testid="file-preview"]')).toBeNull();
      expect(document.body.querySelector('[aria-keyshortcuts="Alt+P"]')).toBeNull();
    });

    it('opens a Places sheet that holds the places, recent files and the Hidden files switch last', async () => {
      localStorage.setItem('ziv-recent-paths-v1', JSON.stringify({ reference_image: ['/pics/owl.png'] }));
      await open(viewport);

      await click(byLabel('Places', toolbar()));

      const places = sheet('Places')!;
      expect(sheetLayer('Places')).toBe('modal-top');
      const labels = Array.from(places.querySelectorAll('button, [role="switch"]')).map((item) => item.getAttribute('role') === 'switch' ? 'switch' : item.textContent?.trim() || item.getAttribute('aria-label'));
      expect(labels.filter((label) => label !== 'Close')).toEqual(['Home', 'Outputs', 'owl.png', 'switch']);
      expect(Array.from(places.querySelectorAll('[aria-current="location"]')).map((item) => item.textContent?.trim())).toEqual(['Outputs']);
    });

    it('turns Hidden files on from the Places sheet and reloads the folder', async () => {
      await open(viewport);
      await click(byLabel('Places', toolbar()));

      const toggle = sheet('Places')!.querySelector<HTMLInputElement>('[role="switch"]')!;
      toggle.checked = true;
      flushSync(() => toggle.dispatchEvent(new Event('change', { bubbles: true })));
      await settle();

      expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('reference_image', '/pics', { hidden: true, all: false });
      expect(localStorage.getItem('ziv-browse-hidden-v1')).toBe('true');
    });

    it('goes to a place from the sheet and closes it', async () => {
      await open(viewport);
      await click(byLabel('Places', toolbar()));

      const home = Array.from(sheet('Places')!.querySelectorAll('button')).find((item) => item.textContent?.trim() === 'Home')!;
      await click(home);
      await settle(300);

      expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('reference_image', '/', expect.anything());
      expect(sheet('Places')).toBeNull();
    });

    it('shows New folder as a toolbar icon in a folder picker and creates a folder', async () => {
      browseApiMocks.browsePlaces.mockImplementation(async (name: 'output_directory') => placesFor(name));
      browseApiMocks.listDirectory.mockImplementation(async () => FOLDERS);
      browseApiMocks.createFolder.mockResolvedValue({ path: '/pics/renders' });
      await open(viewport, { purpose: 'output_directory' });

      const newFolder = byLabel('New folder', toolbar()) as HTMLButtonElement;
      const buttons = Array.from(toolbar().querySelectorAll('button, label'));
      expect(buttons.indexOf(newFolder)).toBeGreaterThan(buttons.findIndex(isFilter));
      expect(toolbar().querySelector('[role="group"][aria-label="View"]')).toBeNull();

      await click(newFolder);
      const name = document.body.querySelector<HTMLInputElement>('input[aria-label="New folder name"]')!;
      name.value = 'renders';
      name.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      key(name, 'Enter');
      await settle();

      expect(browseApiMocks.createFolder).toHaveBeenCalledWith('output_directory', '/pics', 'renders');
    });

    it('names the file on the Choose button, with the full name for screen readers', async () => {
      const { onchoose } = await open(viewport);
      const choose = byLabel(`Choose ${LONG_NAME}`) as HTMLButtonElement;

      expect(choose).not.toBeNull();
      expect(choose.disabled).toBe(false);
      // The visible text is shorter than the name; upright it still starts with the name, sideways it carries none of it.
      expect(choose.textContent!.length).toBeLessThan(LONG_NAME.length);
      if (viewport === PORTRAIT) {
        expect(choose.textContent).toContain(LONG_NAME.slice(0, 8));
      } else {
        expect(choose.textContent).not.toContain(LONG_NAME.slice(0, 8));
        expect(toolbar().contains(choose)).toBe(true);
      }

      await click(choose);
      expect(onchoose).toHaveBeenCalledWith(`/pics/${LONG_NAME}`);
    });

    it('keeps Use “folder” on the button of a folder picker', async () => {
      browseApiMocks.browsePlaces.mockImplementation(async (name: 'output_directory') => placesFor(name));
      browseApiMocks.listDirectory.mockImplementation(async () => FOLDERS);
      const { onchoose } = await open(viewport, { purpose: 'output_directory' });

      const use = byLabel('Use “pics”') as HTMLButtonElement;
      expect(use.textContent?.includes('pics')).toBe(viewport === PORTRAIT);
      await click(use);

      expect(onchoose).toHaveBeenCalledWith('/pics');
    });

    it('shows rows with a 40 px thumbnail and a second line of size and date', async () => {
      await open(viewport);

      const fox = rowFor(LONG_NAME);
      expect(fox.querySelector('img')?.getAttribute('src')).toBe(`/preview/reference_image/pics/${LONG_NAME}`);
      expect(metaOf(LONG_NAME)).toContain(formatBytes(2048));
      expect(metaOf(LONG_NAME)).toContain(formatModified(PICS.entries.find((entry) => entry.name === LONG_NAME)!.modified!));
      expect(fox.querySelector('[data-testid="entry-size"]')).toBeNull();
      expect(metaOf('archive')).toBe(formatModified(PICS.entries.find((entry) => entry.name === 'archive')!.modified!));
    });

    it('closes the keyboard on Enter in the filter instead of choosing the first match', async () => {
      const { onchoose } = await open(viewport);
      const filter = byLabel('Filter this folder', toolbar()) as HTMLInputElement;
      filter.focus();
      filter.value = 'owl';
      filter.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      expect(document.activeElement).toBe(filter);

      key(filter, 'Enter');

      expect(document.activeElement).not.toBe(filter);
      expect(onchoose).not.toHaveBeenCalled();
      expect(rowNames()).toEqual(['owl.png']);
    });

    describe('the preview sheet', () => {
      const previewSheet = () => sheet('Preview');
      const thumbButton = (name: string) => rowFor(name).querySelector<HTMLElement>('[data-testid="row-preview"]');

      it('opens from the thumbnail of a row and shows the file, selected, over the browser', async () => {
        await open(viewport);

        await click(thumbButton('owl.png'));
        await settle(200);

        expect(previewSheet()).not.toBeNull();
        expect(sheetLayer('Preview')).toBe('modal-top');
        expect(previewSheet()!.querySelector('[data-testid="preview-name"]')?.textContent).toBe('owl.png');
        expect(previewSheet()!.querySelector('img')?.getAttribute('src')).toBe('/preview/reference_image/pics/owl.png');
        expect(rowFor('owl.png').getAttribute('aria-selected')).toBe('true');
        expect(browseApiMocks.browseInfo).toHaveBeenCalledWith('reference_image', '/pics/owl.png', expect.anything());
      });

      it('keeps the thumbnail button out of keyboards and screen readers', async () => {
        await open(viewport);

        const button = thumbButton('owl.png')!;
        expect(button.getAttribute('tabindex')).toBe('-1');
        expect(button.getAttribute('aria-hidden')).toBe('true');
        expect(thumbButton('archive')).toBeNull();
      });

      it('opens at once on a second tap of the selected row, not the first', async () => {
        await open(viewport);

        tap(rowFor('owl.png'), 1);
        await settle();
        expect(previewSheet()).toBeNull();

        tap(rowFor('owl.png'), 1);
        await settle();
        expect(previewSheet()).not.toBeNull();
        expect(previewSheet()!.querySelector('[data-testid="preview-name"]')?.textContent).toBe('owl.png');
      });

      it('only selects the file that was selected on opening with the first tap', async () => {
        await open(viewport);
        expect(rowFor(LONG_NAME).getAttribute('aria-selected')).toBe('true');

        tap(rowFor(LONG_NAME), 1);
        await settle();
        expect(previewSheet()).toBeNull();

        tap(rowFor(LONG_NAME), 1);
        await settle();
        expect(previewSheet()).not.toBeNull();
      });

      it('opens a folder once on a quick double tap', async () => {
        browseApiMocks.listDirectory.mockImplementation(async (_purpose: string, path: string) => (path === '/pics/archive' ? FOLDERS : PICS));
        await open(viewport);

        tap(rowFor('archive'), 1);
        await settle();
        // The second tap lands on the first row of the folder just opened.
        tap(rows()[0], 2);
        await settle();

        expect(browseApiMocks.listDirectory.mock.calls.map((call) => call[1])).toEqual(['', '/pics/archive']);
      });

      it('does not choose on a double tap', async () => {
        const { onchoose } = await open(viewport);
        tap(rowFor('owl.png'), 1);
        await settle();

        flushSync(() => rowFor('owl.png').dispatchEvent(new MouseEvent('dblclick', { bubbles: true, detail: 2 })));
        await settle();

        expect(onchoose).not.toHaveBeenCalled();
      });

      it('opens on Space for the selected file', async () => {
        await open(viewport);

        key(list(), ' ');
        await settle();
        expect(previewSheet()).not.toBeNull();
        expect(previewSheet()!.querySelector('[data-testid="preview-name"]')?.textContent).toBe(LONG_NAME);
      });

      it('does nothing on Space over a folder', async () => {
        await open(viewport);
        key(list(), 'Home');
        expect(rowFor('archive').getAttribute('aria-selected')).toBe('true');

        const space = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
        flushSync(() => list().dispatchEvent(space));
        await settle();

        expect(space.defaultPrevented).toBe(true);
        expect(previewSheet()).toBeNull();
      });

      it('chooses the file from the sheet with Choose this file', async () => {
        const { onchoose } = await open(viewport);
        await click(thumbButton('cat.png'));

        const choose = Array.from(previewSheet()!.querySelectorAll('button')).find((item) => item.textContent?.trim() === 'Choose this file')!;
        await click(choose);
        await settle(300);

        expect(onchoose).toHaveBeenCalledWith('/pics/cat.png');
        expect(previewSheet()).toBeNull();
        expect(document.body.querySelector('[data-testid="path-browser"]')).toBeNull();
      });

      it('closes with its own Close and leaves the browser open on the same file', async () => {
        await open(viewport);
        await click(thumbButton('cat.png'));

        await click(byLabel('Close', previewSheet()!));
        await settle(300);

        expect(previewSheet()).toBeNull();
        expect(document.body.querySelector('[data-testid="path-browser"]')).not.toBeNull();
        expect(rowFor('cat.png').getAttribute('aria-selected')).toBe('true');
      });

      it('does not exist in a picker without previews', async () => {
        browseApiMocks.browsePlaces.mockImplementation(async (name: 'lora_file') => placesFor(name));
        browseApiMocks.listDirectory.mockImplementation(async () => LORAS);
        await open(viewport, { purpose: 'lora_file' });

        expect(document.body.querySelector('[data-testid="row-preview"]')).toBeNull();
        tap(rowFor('ink.safetensors'), 1);
        tap(rowFor('ink.safetensors'), 1);
        key(list(), ' ');
        await settle(350);

        expect(previewSheet()).toBeNull();
        expect(browseApiMocks.browseInfo).not.toHaveBeenCalled();
        expect(browseApiMocks.browsePreviewUrl).not.toHaveBeenCalled();
        expect(metaOf('ink.safetensors')).toContain(formatBytes(2048));
      });
    });

    describe('the grid', () => {
      beforeEach(() => {
        localStorage.setItem('ziv-browse-view-v1', JSON.stringify({ sort: 'modified', dir: 'desc', view: 'grid', preview: true }));
        browseApiMocks.listDirectory.mockImplementation(async () => MANY);
      });
      const columns = viewport === PORTRAIT ? 3 : 4;

      it(`has ${columns} columns`, async () => {
        await open(viewport);
        expect(list().getAttribute('data-view')).toBe('grid');
        const selected = () => rows().findIndex((row) => row.getAttribute('aria-selected') === 'true');
        expect(selected()).toBe(0);

        key(list(), 'ArrowDown');

        expect(selected()).toBe(columns);
      });

      it('selects on a tap, opens the preview on a second tap and on Space', async () => {
        await open(viewport);

        tap(rows()[4], 1);
        await settle(350);
        expect(rows()[4].getAttribute('aria-selected')).toBe('true');
        expect(sheet('Preview')).toBeNull();

        tap(rows()[4], 1);
        await settle(350);
        expect(sheet('Preview')!.querySelector('[data-testid="preview-name"]')?.textContent).toBe('pic-4.png');

        await click(byLabel('Close', sheet('Preview')!));
        await settle(300);
        key(list(), ' ');
        await settle();
        expect(sheet('Preview')).not.toBeNull();
      });

      it('has no thumbnail button, since the tile is the thumbnail', async () => {
        await open(viewport);
        expect(document.body.querySelector('[data-testid="row-preview"]')).toBeNull();
      });
    });
  });

  describe('held upright', () => {
    it('has a toolbar of Places, Back, Up, the path, List | Grid and Filter, and keeps the path pencil', async () => {
      await open(PORTRAIT);

      const labels = toolbarLabels();
      const order = ['Places', 'Back', 'Up one folder', 'View', 'Filter'].map((label) => labels.indexOf(label));
      expect(order.every((position) => position >= 0)).toBe(true);
      expect([...order].sort((a, b) => a - b)).toEqual(order);
      expect(byLabel('Type a path', toolbar())).not.toBeNull();
      expect(byLabel('Folder path', toolbar())).not.toBeNull();
    });

    it('says how many items and in what order, above the list', async () => {
      await open(PORTRAIT);

      const row = byLabel('Sort: Newest first')!;
      expect(row.querySelector('[data-testid="sort-count"]')?.textContent).toBe(itemCountLabel(3, 'image'));
    });

    it('opens the sort sheet with the six orders and applies and remembers one', async () => {
      await open(PORTRAIT);

      await click(byLabel('Sort: Newest first'));
      expect(sheetItems('Sort by')).toEqual(SORT_ACTIONS);
      await click(sheet('Sort by')!.querySelector<HTMLElement>('[data-action="modified-asc"]'));

      expect(rowNames()).toEqual(['archive', 'cat.png', 'owl.png', LONG_NAME]);
      expect(readBrowseView()).toMatchObject({ sort: 'modified', dir: 'asc' });
      expect(byLabel('Sort: Oldest first')).not.toBeNull();
    });

    it('counts folders in a folder picker and files elsewhere', async () => {
      browseApiMocks.browsePlaces.mockImplementation(async (name: 'output_directory') => placesFor(name));
      browseApiMocks.listDirectory.mockImplementation(async () => FOLDERS);
      await open(PORTRAIT, { purpose: 'output_directory' });

      expect(byLabel('Sort: Newest first')!.querySelector('[data-testid="sort-count"]')?.textContent).toBe(itemCountLabel(2, 'folder'));
    });

    it('keeps only Choose in the footer', async () => {
      await open(PORTRAIT);

      const footer = document.body.querySelector('[data-testid="footer"]')!;
      expect(Array.from(footer.querySelectorAll('button')).map((item) => item.getAttribute('aria-label'))).toEqual([`Choose ${LONG_NAME}`]);
      expect(footer.querySelector('[role="switch"]')).toBeNull();
      expect(footer.querySelector('[data-testid="extensions"]')).toBeNull();
    });

    it('types a path from the pencil', async () => {
      await open(PORTRAIT);

      await click(byLabel('Type a path'));

      expect(document.body.querySelector('input[aria-label="Path"]')).not.toBeNull();
    });
  });

  describe('held sideways', () => {
    it('folds everything into one toolbar with a sort button and a short Choose, and no footer', async () => {
      await open(LANDSCAPE);

      expect(document.body.querySelector('[data-testid="footer"]')).toBeNull();
      expect(document.body.querySelector('[data-testid="sort-row"]')).toBeNull();
      const labels = toolbarLabels();
      const order = ['Places', 'Back', 'Up one folder', 'Sort: Newest first', 'View', 'Filter', `Choose ${LONG_NAME}`].map((label) => labels.indexOf(label));
      expect(order.every((position) => position >= 0)).toBe(true);
      expect([...order].sort((a, b) => a - b)).toEqual(order);
    });

    it('has no path pencil', async () => {
      await open(LANDSCAPE);

      expect(byLabel('Type a path')).toBeNull();
      expect(byLabel('Folder path', toolbar())).not.toBeNull();
    });

    it('opens the same sort sheet from the toolbar', async () => {
      await open(LANDSCAPE);

      await click(byLabel('Sort: Newest first', toolbar()));

      expect(sheetItems('Sort by')).toEqual(SORT_ACTIONS);
    });

    it('puts New folder before Choose in a folder picker', async () => {
      browseApiMocks.browsePlaces.mockImplementation(async (name: 'output_directory') => placesFor(name));
      browseApiMocks.listDirectory.mockImplementation(async () => FOLDERS);
      await open(LANDSCAPE, { purpose: 'output_directory' });

      const buttons = Array.from(toolbar().querySelectorAll('button'));
      expect(buttons.findIndex((item) => item.getAttribute('aria-label') === 'New folder')).toBeLessThan(buttons.findIndex((item) => item.getAttribute('aria-label') === 'Use “pics”'));
    });
  });

  it('follows the screen turning: sideways to upright moves Choose to the footer and brings back the sort row', async () => {
    await open(LANDSCAPE);
    expect(document.body.querySelector('[data-testid="footer"]')).toBeNull();

    screen.set(PORTRAIT);
    await settle();

    expect(document.body.querySelector('[data-testid="footer"] [aria-label^="Choose "]')).not.toBeNull();
    expect(document.body.querySelector('[data-testid="sort-row"]')).not.toBeNull();
  });

  it('ignores Alt+P, since the phone has no pane', async () => {
    await open(PORTRAIT);

    key(list(), 'π', { code: 'KeyP', altKey: true });
    await settle();

    expect(readBrowseView().preview).toBe(true);
    expect(localStorage.getItem('ziv-browse-view-v1')).toBeNull();
    expect(document.body.querySelector('[data-testid="file-preview"]')).toBeNull();
  });
});
