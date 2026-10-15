// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const browseApiMocks = vi.hoisted(() => ({
  listDirectory: vi.fn(),
  browsePlaces: vi.fn(),
  createFolder: vi.fn(),
}));

vi.mock('$lib/api/browse', () => browseApiMocks);

import { rememberRecentPath } from '$lib/state/pathBrowser';
import { placesFor } from '../../../test-utils/pathBrowser';
import PathInput from './PathInput.svelte';

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 200));
  flushSync();
  await new Promise((resolve) => setTimeout(resolve, 0));
  flushSync();
}

const MODELS = {
  path: '/models', parent: '/', separator: '/', focus: null, hidden_count: 0,
  entries: [
    { name: 'loras', path: '/models/loras', kind: 'directory', size: null, modified: null, selectable: false },
    { name: 'local', path: '/models/local', kind: 'directory', size: null, modified: null, selectable: false },
    { name: 'base.safetensors', path: '/models/base.safetensors', kind: 'file', size: 1, modified: null, selectable: true },
  ],
};

describe('PathInput', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;
  let oninput: ReturnType<typeof vi.fn>;
  let onpick: ReturnType<typeof vi.fn>;
  let onkeydown: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    localStorage.clear();
    for (const mock of Object.values(browseApiMocks)) mock.mockReset();
    browseApiMocks.listDirectory.mockResolvedValue(MODELS);
    browseApiMocks.browsePlaces.mockResolvedValue({ ...placesFor('lora_file'), places: [{ id: 'home', label: 'Home', path: '/home/user', default: false }] });
    target = document.createElement('div');
    document.body.appendChild(target);
    oninput = vi.fn();
    onpick = vi.fn();
    onkeydown = vi.fn();
    app = flushSync(() => mount(PathInput, { target, props: { id: 'p', value: '', purpose: 'lora_file', oninput, onpick, onkeydown } }));
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
  });

  const input = () => target.querySelector<HTMLInputElement>('#p')!;
  const options = () => Array.from(document.querySelectorAll<HTMLElement>('#p-completions [role="option"]'));
  const labels = () => options().map((option) => option.querySelector('.label')?.textContent);
  const key = (name: string) => {
    const event = new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true });
    flushSync(() => input().dispatchEvent(event));
    return event;
  };

  async function type(text: string): Promise<void> {
    input().focus();
    input().value = text;
    input().dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
  }

  it('lists the typed folder and suggests names starting with the rest', async () => {
    await type('/models/lo');

    expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('lora_file', '/models/', { hidden: false });
    expect(labels()).toEqual(['loras/', 'local/']);
    expect(input().getAttribute('aria-expanded')).toBe('true');
  });

  it('completes the first folder with Tab and keeps completing inside it', async () => {
    await type('/models/lo');

    key('Tab');
    await settle();

    expect(input().value).toBe('/models/loras/');
    expect(oninput).toHaveBeenLastCalledWith('/models/loras/');
    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '/models/loras/', { hidden: false });
  });

  it('takes a highlighted file with Enter and hands other keys on', async () => {
    await type('/models/ba');

    key('ArrowDown');
    const enter = key('Enter');

    expect(enter.defaultPrevented).toBe(true);
    expect(onpick).toHaveBeenCalledWith('/models/base.safetensors');
    expect(onkeydown).not.toHaveBeenCalled();

    key('Enter');
    expect(onkeydown).toHaveBeenCalledTimes(1);
  });

  it('lists dot names once the typed name starts with a dot', async () => {
    await type('/home/user/.z');

    expect(browseApiMocks.listDirectory).toHaveBeenLastCalledWith('lora_file', '/home/user/', { hidden: true });
  });

  it('completes from a fresh list when Tab comes before the list caught up with typing', async () => {
    await type('/models/lo');
    browseApiMocks.listDirectory.mockResolvedValue({ ...MODELS, entries: MODELS.entries.filter((entry) => entry.name === 'local') });
    input().value = '/models/loc';
    input().dispatchEvent(new Event('input', { bubbles: true }));

    key('Tab');
    await settle();

    expect(input().value).toBe('/models/local/');
  });

  it('suggests nothing for a path that is not absolute', async () => {
    await type('models/lo');

    expect(browseApiMocks.listDirectory).not.toHaveBeenCalled();
    expect(options()).toHaveLength(0);
  });

  it('stays closed on focus and offers recent paths and places on ArrowDown in an empty box', async () => {
    rememberRecentPath('lora_file', '/models/loras/ink.safetensors');

    input().focus();
    await settle();
    expect(options()).toHaveLength(0);

    key('ArrowDown');
    await settle();

    expect(labels()).toEqual(['ink.safetensors', 'Home']);
    flushSync(() => options()[0].click());
    expect(onpick).toHaveBeenCalledWith('/models/loras/ink.safetensors');
  });
});
