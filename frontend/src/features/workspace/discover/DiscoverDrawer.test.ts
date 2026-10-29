import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { downloadStore } from '$lib/state/downloads.svelte';
import type { HfCheck, HfPage, HfRepo, CuratedEntry, CuratedList, LoraInfo, WorkspaceContext } from '$lib/types';
import { makeContext } from '../../../test-utils/workspace';

const layoutMock = vi.hoisted(() => ({ compact: false, mode: 'wide', coarse: false }));
const api = vi.hoisted(() => ({
  getCurated: vi.fn(),
  getHfList: vi.fn(),
  getHfRepo: vi.fn(),
  getHfCheck: vi.fn(),
  setDiscoverSettings: vi.fn(),
  setLoraTrigger: vi.fn(),
  startDiscoverDownload: vi.fn(),
}));
const models = vi.hoisted(() => ({ deleteLora: vi.fn(), getDownloads: vi.fn(async (): Promise<{ items: unknown[] }> => ({ items: [] })) }));
const confirm = vi.hoisted(() => ({ requestConfirm: vi.fn() }));
const toasts = vi.hoisted(() => ({ addToast: vi.fn(() => 'toast') }));
const jobs = vi.hoisted(() => ({ active: false }));

vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: layoutMock }));
vi.mock('$lib/api/discover', () => api);
vi.mock('$lib/api/models', () => models);
vi.mock('$lib/components/molecules/confirm.svelte', () => confirm);
vi.mock('$lib/state/toasts.svelte', () => toasts);
// `jobs.active` is backed by a rune so a test can end the job while the drawer is open.
vi.mock('$lib/state/job.svelte', async () => {
  const { reactiveProps } = await import('../../../test-utils/reactiveProps.svelte');
  const live = reactiveProps({ active: jobs.active });
  Object.defineProperty(jobs, 'active', { get: () => live.active, set: (value: boolean) => { live.active = value; }, configurable: true });
  return { jobStore: { get jobsActive() { return live.active; } } };
});

import { mountComponent, settle } from '../../../test-utils/mount';
import { pressKey } from '../../../test-utils/workspaceHarness';
import DiscoverDrawer from './DiscoverDrawer.svelte';
import { INPUT_DEBOUNCE_MS } from '$lib/state/debounce';

const entry = (overrides: Partial<CuratedEntry> = {}): CuratedEntry => ({
  id: 'o-style', name: 'style', family: 'zimage', size_variant: null, version: null, display_name: 'Style', author: 'o', description: 'A soft look', section: 'style',
  what_it_does: null, workflow: null, repo: 'o/style', revision: 'a'.repeat(40), file: 'style.safetensors', size_bytes: 255_000_000, size_label: '255 MB', trigger: 'sty1e, soft', weight: 0.8,
  licence: 'apache-2.0', licence_label: 'Apache 2.0', non_commercial: false, downloads: 1529, platforms: [], note: null, previews: ['https://huggingface.co/o/style/1.png', 'https://huggingface.co/o/style/2.png'], preview_videos: [],
  model_card_url: 'https://huggingface.co/o/style', state: 'available', download: null, ...overrides,
});
const curatedList = (entries: CuratedEntry[], overrides: Partial<CuratedList> = {}): CuratedList => ({ family: 'zimage', size: null, version: null, hf_enabled: false, entries, ...overrides });
const target = { family: 'zimage', size: null, version: null, name: 'Z-Image Turbo', familyName: 'Z-Image' };
const repoDetail = (overrides: Partial<HfRepo> = {}): HfRepo => ({
  repo: 'x/clay', name: 'Clay', author: 'x', revision: 'b'.repeat(40), downloads: 10, likes: 2, licence: 'mit', licence_label: 'MIT', base_model: 'Tongyi-MAI/Z-Image-Turbo', instance_prompt: 'clayish', previews: [], preview_videos: [],
  model_card_url: 'https://huggingface.co/x/clay', preselected: 'clay.safetensors', other_files: [],
  files: [{ file: 'clay.safetensors', size_bytes: 1000, size_label: '1 KB', sha256: null, step: null, marked: false, curated_id: null, save_as: 'clay', renamed: false, state: 'available' }], ...overrides,
});

interface Spies { onuse: ReturnType<typeof vi.fn>; onstop: ReturnType<typeof vi.fn>; ondeleted: ReturnType<typeof vi.fn>; onloraschange: ReturnType<typeof vi.fn> }

async function mountDrawer(entries: CuratedEntry[], options: { context?: WorkspaceContext; list?: Partial<CuratedList>; chips?: { name: string; weight: number }[]; target?: { family: string; size: string | null; version: string | null; name: string; familyName: string } } = {}): Promise<Spies> {
  api.getCurated.mockResolvedValue(curatedList(entries, options.list));
  const spies = { onuse: vi.fn(), onstop: vi.fn(), ondeleted: vi.fn(), onloraschange: vi.fn() };
  mountComponent(DiscoverDrawer, { open: true, context: options.context ?? makeContext(), target: options.target ?? target, chips: options.chips ?? [], ...spies });
  await settle();
  await settle();
  return spies;
}

const dialog = () => document.body.querySelector('[role="dialog"]') as HTMLElement | null;
const text = () => dialog()?.textContent ?? '';
const clickByText = (label: string, root: ParentNode = document.body): void => {
  const found = Array.from(root.querySelectorAll('button')).find((b) => b.textContent?.trim().startsWith(label) && !b.disabled);
  if (!found) throw new Error(`No enabled button "${label}"`);
  flushSync(() => found.click());
};
const buttons = (root: ParentNode = document.body) => Array.from(root.querySelectorAll('button')).map((b) => b.textContent?.trim() ?? '');
const input = (label: string) => document.body.querySelector(`input[aria-label="${label}"]`) as HTMLInputElement;
const typeInto = (el: HTMLInputElement, value: string): void => { el.value = value; flushSync(() => el.dispatchEvent(new Event('input', { bubbles: true }))); };
/** The "Use in LoRAs" switch of the detail. */
const useSwitch = () => dialog()!.querySelector('input[role="switch"]') as HTMLInputElement;
const flipSwitch = async (): Promise<void> => { flushSync(() => useSwitch().click()); await settle(); await settle(); };
const lora = (name: string, overrides: Partial<LoraInfo> = {}): LoraInfo => ({ name, path: `/l/${name}.safetensors`, family: 'zimage', ...overrides });
/** Open a tile or a row by the name it shows. */
const openDetail = async (name: string): Promise<void> => {
  const found = Array.from(document.body.querySelectorAll('.disc-name')).find((el) => el.textContent?.trim() === name)?.closest('button');
  if (!found) throw new Error(`No card "${name}"`);
  flushSync(() => found.click());
  await settle();
  await settle();
};

beforeEach(() => {
  downloadStore.reset();
  for (const mock of [...Object.values(api), models.deleteLora, confirm.requestConfirm, toasts.addToast]) mock.mockReset();
  api.getHfList.mockResolvedValue({ items: [], next_cursor: null, base_models: [] } satisfies HfPage);
  models.getDownloads.mockResolvedValue({ items: [] });
  Object.assign(layoutMock, { compact: false, mode: 'wide', coarse: false });
  jobs.active = false;
  Object.defineProperty(navigator, 'onLine', { value: true, configurable: true });
});

describe('Discover LoRAs drawer: tabs', () => {
  it('opens on Curated, with Installed, Curated (counted) and Hugging Face, and "for <model>" at the top right', async () => {
    await mountDrawer([entry(), entry({ id: 'b', name: 'b', display_name: 'B' })], { context: makeContext({ loras: [lora('style'), lora('other', { family: 'krea2' })] }) });

    expect(text()).toContain('Discover LoRAs');
    expect(text()).toContain('Installed 1');
    expect(text()).toContain('Curated 2');
    expect(text()).toContain('Hugging Face');
    expect(text()).toContain('for Z-Image Turbo');
    expect(dialog()!.querySelector('#disc-tab-curated')?.getAttribute('aria-selected')).toBe('true');
    expect(api.getCurated).toHaveBeenCalledWith({ family: 'zimage', size: null, version: null });
  });

  it('opens on Hugging Face, with its question, for a model with nothing curated', async () => {
    await mountDrawer([]);

    expect(buttons(dialog()!)).toContain('Show Hugging Face results');
    expect(dialog()!.querySelector('#disc-tab-hf')?.getAttribute('aria-selected')).toBe('true');
  });
});

describe('Discover LoRAs drawer: Curated', () => {
  it('shows styles and tools mixed in one grid of tiles, with a Tool tag, what a tool does, author and downloads and no size', async () => {
    await mountDrawer([entry(), entry({ id: 'o-tool', name: 'tool', display_name: 'Shatter', section: 'tool', what_it_does: 'Shatters the shot', workflow: 'img2vid', size_label: '390 MB' })]);

    const tiles = Array.from(dialog()!.querySelectorAll('.disc-card')).map((tile) => tile.textContent ?? '');
    expect(tiles).toHaveLength(2);
    expect(tiles[0]).toContain('A soft look');
    expect(tiles[0]).toContain('1.5k');
    expect(tiles[0]).not.toContain('255 MB');
    expect(tiles[0]).not.toContain('Tool');
    expect(tiles[1]).toContain('Shatter');
    expect(tiles[1]).toContain('Tool');
    expect(tiles[1]).toContain('Shatters the shot');
    expect(tiles[1]).not.toContain('390 MB');
    expect(dialog()!.querySelectorAll('.disc-row')).toHaveLength(0);
  });

  it('badges what is installed, in use and non-commercial', async () => {
    const context = makeContext({ loras: [lora('here'), lora('used')] });
    await mountDrawer(
      [entry({ id: 'a', name: 'here', display_name: 'Here' }), entry({ id: 'b', name: 'used', display_name: 'Used' }), entry({ id: 'c', name: 'nc', display_name: 'Nc', non_commercial: true })],
      { context, chips: [{ name: 'used', weight: 1 }] },
    );

    const tiles = Array.from(dialog()!.querySelectorAll('.disc-card')).map((tile) => tile.textContent ?? '');
    expect(tiles[0]).toContain('Installed');
    expect(tiles[1]).toContain('In use');
    expect(tiles[1]).not.toContain('Installed');
    expect(tiles[2]).toContain('Non-commercial');
  });

  it('always has search on top and an All, Styles and Tools filter', async () => {
    await mountDrawer([entry(), entry({ id: 't', name: 't', display_name: 'Shatter', section: 'tool', what_it_does: 'Shatters' })]);

    expect(input('Filter curated LoRAs').placeholder).toBe('Filter 2 curated LoRAs');
    const names = () => Array.from(dialog()!.querySelectorAll('.disc-card .disc-name')).map((el) => el.textContent);
    clickByText('Tools');
    await settle();
    expect(names()).toEqual(['Shatter']);
    clickByText('Styles');
    await settle();
    expect(names()).toEqual(['Style']);
    clickByText('All');
    await settle();
    expect(names()).toEqual(['Style', 'Shatter']);
  });

  it('carries the words to Hugging Face when nothing matches, and ends the list with More on Hugging Face', async () => {
    await mountDrawer([entry()], { list: { hf_enabled: true } });
    expect(buttons(dialog()!)).toContain('More on Hugging Face');

    typeInto(input('Filter curated LoRAs'), 'watercolor');
    await settle();
    expect(text()).toContain('Nothing curated matches “watercolor”.');
    expect(buttons(dialog()!)).not.toContain('More on Hugging Face');
    clickByText('Search Hugging Face for');
    await settle(INPUT_DEBOUNCE_MS + 10);

    expect(input('Search Hugging Face').value).toBe('watercolor');
    expect(api.getHfList).toHaveBeenLastCalledWith({ family: 'zimage', size: null, version: null }, expect.objectContaining({ q: 'watercolor' }));
  });

  it('More on Hugging Face opens that tab with no words', async () => {
    await mountDrawer([entry()], { list: { hf_enabled: true } });

    clickByText('More on Hugging Face');
    await settle(INPUT_DEBOUNCE_MS + 10);

    expect(dialog()!.querySelector('#disc-tab-hf')?.getAttribute('aria-selected')).toBe('true');
    expect(input('Search Hugging Face').value).toBe('');
  });
});

describe('Discover LoRAs drawer: Installed', () => {
  const gotoInstalled = async (): Promise<void> => { clickByText('Installed'); await settle(); };
  const hub = { name: 'Pixel Art Style', author: 'tarn59', downloads: 14000, licence: null, licence_label: null, model_card_url: 'https://huggingface.co/tarn59/pix', thumbnail_url: '/api/discover/installed-thumbnails/pix?v=1' };

  it('lists this model\'s LoRAs as tiles, a curated one like Curated, a download with what it kept, a file of your own by its name', async () => {
    const context = makeContext({ loras: [lora('style', { curated_id: 'o-style' }), lora('pix', { hub }), lora('my_style'), lora('krea_one', { family: 'krea2' })] });
    await mountDrawer([entry()], { context, chips: [{ name: 'style', weight: 1 }] });
    await gotoInstalled();

    const tile = (name: string) => Array.from(dialog()!.querySelectorAll('.disc-card')).find((card) => card.querySelector('.disc-name')?.textContent === name)?.textContent ?? '';
    expect(dialog()!.querySelectorAll('.disc-card')).toHaveLength(3);
    expect(tile('Style')).toContain('In use');
    expect(tile('Style')).toContain('A soft look');
    expect(tile('my_style')).not.toContain('Installed');
    expect(tile('Pixel Art Style')).toContain('tarn59');
    expect(tile('Pixel Art Style')).toContain('14k');
    expect(tile('my_style')).not.toBe('');
    expect(text()).not.toContain('krea_one');
    expect(input('Filter installed LoRAs').placeholder).toBe('Filter 3 installed LoRAs');
  });

  it('puts the files of unknown model under a collapsed "Other models (N)" row at the bottom', async () => {
    const context = makeContext({ loras: [lora('style'), lora('mystery', { family: null }), lora('mystery_two', { family: null })] });
    await mountDrawer([entry()], { context });
    await gotoInstalled();

    expect(text()).toContain('Other models (2)');
    expect(text()).toContain('Installed 1');
    expect(text()).not.toContain('mystery');
    clickByText('Other models');
    await settle();
    expect(text()).toContain('mystery');
    expect(text()).toContain('mystery_two');
  });

  it('says there is nothing for the model, with a button to Curated', async () => {
    await mountDrawer([entry()], { context: makeContext({ loras: [lora('krea_one', { family: 'krea2' })] }) });
    await gotoInstalled();

    expect(text()).toContain('No LoRAs for Z-Image Turbo yet.');
    clickByText('Curated');
    await settle();
    expect(dialog()!.querySelector('#disc-tab-curated')?.getAttribute('aria-selected')).toBe('true');
  });

  it('opens a curated file as its curated detail, a Hugging Face download from what it kept (offline), and your own file with the rows that apply', async () => {
    const context = makeContext({ loras: [lora('style', { curated_id: 'o-style', size_label: '255 MB' }), lora('pix', { hub, size_label: '85 MB', trigger: 'pix, art', source: { repo: 'tarn59/pix', revision: 'a', file: 'p.safetensors', sha256: null, curated_id: null } }), lora('my_style', { size_label: '163 MB' })] });
    await mountDrawer([entry()], { context });
    await gotoInstalled();

    await openDetail('Style');
    expect(text()).toContain('A soft look');
    expect(text()).toContain('File name');
    clickByText('Installed');
    await settle();

    await openDetail('Pixel Art Style');
    expect(api.getHfRepo).not.toHaveBeenCalled();
    expect(text()).toContain('tarn59/pix');
    expect(text()).toContain('pix, art');
    expect(text()).toContain('85 MB');
    expect(text()).toContain('14,000');
    expect(text()).toContain('Made for');
    expect(dialog()!.querySelector('a[href="https://huggingface.co/tarn59/pix"]')?.textContent).toBe('Model card');
    clickByText('Installed');
    await settle();

    await openDetail('my_style');
    expect(text()).toContain('163 MB');
    expect(text()).not.toContain('Weight');
    expect(text()).not.toContain('Downloads');
    expect(text()).not.toContain('Licence');
    expect(text()).not.toContain('Model card');
    expect(buttons(dialog()!)).toContain('Edit');
  });
});

describe('Discover LoRAs drawer: the detail', () => {
  it('shows made for, trigger words, saves as, size, the non-commercial note, the Model card link and the copyable command', async () => {
    await mountDrawer([entry({ non_commercial: true })]);
    await openDetail('Style');

    expect(dialog()!.querySelector('.disc-bar h3')?.textContent).toBe('Style');
    expect(dialog()!.querySelector('.disc-lead')?.textContent).toBe('A soft look');
    expect(text()).toContain('Made for');
    expect(dialog()!.querySelector('.disc-kv .ok')?.textContent).toContain('Z-Image Turbo');
    expect(dialog()!.querySelector('.disc-code')?.textContent).toBe('sty1e, soft');
    expect(text()).toContain('Saves as');
    expect(text()).toContain('255 MB');
    expect(text()).toContain('Non-commercial');
    expect(dialog()!.querySelector('a[href="https://huggingface.co/o/style"]')?.textContent).toBe('Model card');
    expect(dialog()!.querySelector('.disc-cli code')?.textContent).toBe('ziv-model lora --hf o/style --file style.safetensors --name style');
  });

  it('says None given for a LoRA without a trigger, and shows no Edit until the file is here', async () => {
    await mountDrawer([entry({ trigger: null })]);
    await openDetail('Style');

    expect(text()).toContain('None given');
    expect(buttons(dialog()!)).not.toContain('Edit');
  });

  it('steps through the images with the arrows', async () => {
    await mountDrawer([entry()]);
    await openDetail('Style');
    const hero = () => dialog()!.querySelector('.disc-hero img') as HTMLImageElement;

    expect(hero().alt).toContain('Example 1 of 2');
    flushSync(() => (dialog()!.querySelector('button[aria-label="Next image"]') as HTMLButtonElement).click());
    expect(hero().alt).toContain('Example 2 of 2');
    expect((dialog()!.querySelector('button[aria-label="Next image"]') as HTMLButtonElement).disabled).toBe(true);
  });

  it('closes the detail on Escape first, and the drawer on the next one', async () => {
    await mountDrawer([entry()]);
    await openDetail('Style');
    expect(text()).toContain('Saves as');

    pressKey('Escape');
    await settle();
    expect(dialog()).not.toBeNull();
    expect(text()).not.toContain('Saves as');
    expect(text()).toContain('Curated 1');

    pressKey('Escape');
    await settle();
    expect(dialog()).toBeNull();
  });

  it('keeps the list as it was when going back', async () => {
    const many = Array.from({ length: 25 }, (_, i) => entry({ id: `s${i}`, name: `s${i}`, display_name: `Style ${i}` }));
    await mountDrawer(many);
    typeInto(input('Filter curated LoRAs'), 'Style 7');
    await settle();
    await openDetail('Style 7');

    clickByText('Curated');
    await settle();

    expect(input('Filter curated LoRAs').value).toBe('Style 7');
  });
});

describe('Discover LoRAs drawer: the footer', () => {
  const lane = (overrides: Record<string, unknown> = {}) => ({ id: 'dl-1', kind: 'lora', name: 'style', label: 'LoRA style', state: 'queued', message: null, curated_id: 'o-style', ...overrides });

  it('offers Download and an off switch for a file that is not here, and the drawer stays open', async () => {
    const item = lane();
    api.startDiscoverDownload.mockResolvedValue({ name: 'style', status: 'queued', renamed: false, item, trigger: 'sty1e, soft', trigger_source: 'curated' });
    models.getDownloads.mockResolvedValue({ items: [item] } as never);
    const spies = await mountDrawer([entry()]);
    await openDetail('Style');
    expect(buttons(dialog()!)).toContain('Download');
    expect(useSwitch().checked).toBe(false);

    clickByText('Download');
    await settle();

    expect(api.startDiscoverDownload).toHaveBeenCalledWith({ curated_id: 'o-style' });
    expect(downloadStore.active('lora', 'style')).not.toBeNull();
    expect(spies.onuse).not.toHaveBeenCalled();
    expect(dialog()).not.toBeNull();
    expect(text()).toContain('Downloading');
  });

  it('turning the switch on downloads the file and puts it on your chips at the recommended weight, with its trigger, and the drawer stays open', async () => {
    const item = lane();
    api.startDiscoverDownload.mockResolvedValue({ name: 'style', status: 'queued', renamed: false, item, trigger: 'sty1e, soft', trigger_source: 'curated' });
    models.getDownloads.mockResolvedValue({ items: [item] } as never);
    const spies = await mountDrawer([entry()]);
    await openDetail('Style');

    await flipSwitch();

    expect(api.startDiscoverDownload).toHaveBeenCalledWith({ curated_id: 'o-style' });
    expect(spies.onuse).toHaveBeenCalledWith({ name: 'style', weight: 0.8, trigger: 'sty1e, soft', downloading: true });
    expect(dialog()).not.toBeNull();
  });

  it('reads Downloading with the percent while the file arrives, and the switch can still be turned on', async () => {
    const item = lane({ state: 'running', bytes_done: 40, bytes_total: 100 });
    models.getDownloads.mockResolvedValue({ items: [item] } as never);
    const spies = await mountDrawer([entry()]);
    downloadStore.startSync();
    await settle();
    await openDetail('Style');

    expect(buttons(dialog()!)).toContain('Downloading 40 %');
    await flipSwitch();

    expect(api.startDiscoverDownload).not.toHaveBeenCalled();
    expect(spies.onuse).toHaveBeenCalledWith({ name: 'style', weight: 0.8, trigger: 'sty1e, soft', downloading: true });
  });

  it('turns the switch on for a file that is here without a download, with the trigger it has', async () => {
    const context = makeContext({ loras: [lora('style', { trigger: 'typed one', trigger_source: 'user' })] });
    const spies = await mountDrawer([entry({ state: 'installed' })], { context });
    await openDetail('Style');
    expect(buttons(dialog()!)).toContain('Delete file');

    await flipSwitch();

    expect(api.startDiscoverDownload).not.toHaveBeenCalled();
    expect(spies.onuse).toHaveBeenCalledWith({ name: 'style', weight: 0.8, trigger: 'typed one', downloading: false });
  });

  it('shows the switch on for a LoRA in use, and turning it off takes the chip away', async () => {
    const context = makeContext({ loras: [lora('style')] });
    const spies = await mountDrawer([entry({ state: 'installed' })], { context, chips: [{ name: 'style', weight: 1 }] });
    await openDetail('Style');
    expect(useSwitch().checked).toBe(true);

    await flipSwitch();

    expect(spies.onstop).toHaveBeenCalledWith('style');
    expect(dialog()).not.toBeNull();
  });

  it('turning the switch off never stops a download', async () => {
    const item = lane({ state: 'running', bytes_done: 40, bytes_total: 100 });
    models.getDownloads.mockResolvedValue({ items: [item] } as never);
    const spies = await mountDrawer([entry()], { chips: [{ name: 'style', weight: 1 }] });
    downloadStore.startSync();
    await settle();
    await openDetail('Style');
    const cancel = vi.spyOn(downloadStore, 'cancel');

    await flipSwitch();

    expect(spies.onstop).toHaveBeenCalledWith('style');
    expect(cancel).not.toHaveBeenCalled();
    expect(downloadStore.active('lora', 'style')).not.toBeNull();
  });

  it('reads Retry download after a failure and starts the download again', async () => {
    const failed = lane({ state: 'failed', message: 'The connection dropped.' });
    models.getDownloads.mockResolvedValue({ items: [failed] } as never);
    api.startDiscoverDownload.mockResolvedValue({ name: 'style', status: 'queued', renamed: false, item: lane({ id: 'dl-2' }), trigger: null, trigger_source: null });
    await mountDrawer([entry()]);
    downloadStore.startSync();
    await settle();
    await openDetail('Style');

    clickByText('Retry download');
    await settle();

    expect(api.startDiscoverDownload).toHaveBeenCalledWith({ curated_id: 'o-style' });
  });

  it('says why when the server refuses, hands nothing back and leaves the switch off', async () => {
    api.startDiscoverDownload.mockRejectedValue(new ApiError('Conflict', 409, 'A different file called style is already in your LoRAs.'));
    const spies = await mountDrawer([entry()]);
    await openDetail('Style');

    await flipSwitch();

    expect(text()).toContain('A different file called style is already in your LoRAs.');
    expect(spies.onuse).not.toHaveBeenCalled();
    expect(useSwitch().checked).toBe(false);
  });

  it('is offline: Download and the switch are off for a file that is not here, with the reason', async () => {
    Object.defineProperty(navigator, 'onLine', { value: false, configurable: true });
    await mountDrawer([entry()], { context: makeContext({ loras: [] }) });
    window.dispatchEvent(new Event('offline'));
    await openDetail('Style');

    expect(text()).toContain("You're offline.");
    expect((Array.from(dialog()!.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Download'))!.disabled).toBe(true);
    expect(useSwitch().disabled).toBe(true);
  });

  it('is offline with the file here: the switch still works', async () => {
    Object.defineProperty(navigator, 'onLine', { value: false, configurable: true });
    const context = makeContext({ loras: [lora('style')] });
    const spies = await mountDrawer([entry({ state: 'installed' })], { context });
    window.dispatchEvent(new Event('offline'));
    await openDetail('Style');

    expect(useSwitch().disabled).toBe(false);
    await flipSwitch();
    expect(spies.onuse).toHaveBeenCalledWith(expect.objectContaining({ name: 'style', downloading: false }));
  });
});

describe('Discover LoRAs drawer: a name a different file holds', () => {
  const own = () => makeContext({ loras: [lora('style', { trigger: 'my own words', trigger_source: 'user' })] });

  it('does not count the file as installed or in use: no badge on the tile, no chip', async () => {
    await mountDrawer([entry({ state: 'name_taken' })], { context: own(), chips: [{ name: 'style', weight: 1 }] });

    const tile = document.body.querySelector('.disc-card')!;
    expect(tile.textContent).not.toContain('Installed');
    expect(tile.textContent).not.toContain('In use');
  });

  it('offers no Delete file, no download and no switch, says why, and shows the entry\'s own trigger', async () => {
    const spies = await mountDrawer([entry({ state: 'name_taken' })], { context: own(), chips: [{ name: 'style', weight: 1 }] });
    await openDetail('Style');

    expect(buttons(dialog()!)).not.toContain('Delete file');
    expect(Array.from(dialog()!.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Download')?.disabled).toBe(true);
    expect(useSwitch().disabled).toBe(true);
    expect(useSwitch().checked).toBe(false);
    expect(text()).toContain('You already have a different file called style.');
    expect(text()).toContain('sty1e, soft');
    expect(text()).not.toContain('my own words');
    expect(api.startDiscoverDownload).not.toHaveBeenCalled();
    expect(spies.onuse).not.toHaveBeenCalled();
  });
});

describe('Discover LoRAs drawer: Delete file', () => {
  const installedContext = () => makeContext({ loras: [lora('style')] });
  const deleteFile = async (): Promise<void> => { clickByText('Delete file'); await settle(); };

  it('confirms by name, deletes and hands the name back', async () => {
    confirm.requestConfirm.mockResolvedValue(true);
    models.deleteLora.mockResolvedValue({ tone: 'success', message: 'Deleted', detail: '' });
    const spies = await mountDrawer([entry({ state: 'installed' })], { context: installedContext() });
    await openDetail('Style');

    await deleteFile();

    expect(confirm.requestConfirm).toHaveBeenCalledWith({ question: 'Delete style?', info: undefined, confirmLabel: 'Delete', danger: true });
    expect(models.deleteLora).toHaveBeenCalledWith('style');
    expect(spies.ondeleted).toHaveBeenCalledWith('style');
  });

  it('forgets the finished download of the file, so the entry can be downloaded again instead of reading Downloading 100 %', async () => {
    confirm.requestConfirm.mockResolvedValue(true);
    models.deleteLora.mockResolvedValue({ tone: 'success', message: 'Deleted', detail: '' });
    const done = { id: 'dl-1', kind: 'lora', name: 'style', label: 'LoRA style', state: 'done', message: null, curated_id: 'o-style' };
    models.getDownloads.mockResolvedValue({ items: [done] } as never);
    const spies = await mountDrawer([entry({ state: 'installed' })], { context: installedContext() });
    await downloadStore.refresh();
    await openDetail('Style');
    // The next read of the lane has not come back yet: the page forgets the item without waiting for it.
    models.getDownloads.mockReturnValue(new Promise(() => undefined));

    await deleteFile();

    expect(spies.ondeleted).toHaveBeenCalledWith('style');
    expect(downloadStore.loraNames()).toEqual([]);
  });

  it('adds that it is removed from your LoRAs only when it is in use', async () => {
    confirm.requestConfirm.mockResolvedValue(false);
    await mountDrawer([entry({ state: 'installed' })], { context: installedContext(), chips: [{ name: 'style', weight: 1 }] });
    await openDetail('Style');

    await deleteFile();

    expect(confirm.requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: 'Delete style?', info: "It's also removed from your LoRAs." }));
  });

  it('does nothing when the confirmation is declined', async () => {
    confirm.requestConfirm.mockResolvedValue(false);
    const spies = await mountDrawer([entry({ state: 'installed' })], { context: installedContext() });
    await openDetail('Style');

    await deleteFile();

    expect(models.deleteLora).not.toHaveBeenCalled();
    expect(spies.ondeleted).not.toHaveBeenCalled();
  });

  it('turns Delete off with the reason while a job runs, and back on when the job ends', async () => {
    jobs.active = true;
    await mountDrawer([entry({ state: 'installed' })], { context: installedContext() });
    await openDetail('Style');
    const button = () => Array.from(dialog()!.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Delete file')!;
    expect(button().disabled).toBe(true);
    expect(text()).toContain('You can delete this when the current job is done.');

    jobs.active = false;
    await settle();

    expect(button().disabled).toBe(false);
    expect(text()).not.toContain('You can delete this when the current job is done.');
  });

  it('keeps the file and shows the server reason when the delete is refused', async () => {
    confirm.requestConfirm.mockResolvedValue(true);
    models.deleteLora.mockRejectedValue(new ApiError('Conflict', 409, 'Delete is off while a job runs.'));
    const spies = await mountDrawer([entry({ state: 'installed' })], { context: installedContext() });
    await openDetail('Style');

    await deleteFile();

    expect(spies.ondeleted).not.toHaveBeenCalled();
    expect(text()).toContain('Delete is off while a job runs.');
  });
});

describe('Discover LoRAs drawer: a typed trigger', () => {
  it('sets the trigger of an installed LoRA from Edit and reads the rows again', async () => {
    api.setLoraTrigger.mockResolvedValue({ name: 'style', trigger: 'mine', trigger_source: 'user', user_trigger: 'mine' });
    const context = makeContext({ loras: [lora('style')] });
    const spies = await mountDrawer([entry({ state: 'installed' })], { context });
    await openDetail('Style');

    clickByText('Edit');
    await settle();
    typeInto(input('Trigger text'), '  mine, other ');
    clickByText('Save');
    await settle();

    expect(api.setLoraTrigger).toHaveBeenCalledWith('style', 'mine, other');
    expect(spies.onloraschange).toHaveBeenCalled();
  });
});

describe('Discover LoRAs drawer: Hugging Face', () => {
  it('asks before showing Hugging Face, saves the answer on the server and then lists rows with author, downloads, likes, files and licence but no description or filter', async () => {
    api.setDiscoverSettings.mockResolvedValue({ hf_enabled: true });
    api.getHfList.mockResolvedValue({ items: [{ repo: 'x/clay', name: 'Clay', author: 'x', downloads: 1200, likes: 5, files: 3, licence: null, thumbnail: null, curated_id: 'o-style' }], next_cursor: 'c1', base_models: [] });
    await mountDrawer([entry()]);

    clickByText('Hugging Face');
    await settle();
    expect(buttons(dialog()!)).toEqual(expect.arrayContaining(['Not now', 'Show Hugging Face results']));
    expect(api.getHfList).not.toHaveBeenCalled();

    clickByText('Show Hugging Face results');
    await settle();
    await settle();

    expect(api.setDiscoverSettings).toHaveBeenCalledOnce();
    expect(api.setDiscoverSettings).toHaveBeenCalledWith(true);
    expect(buttons(dialog()!)).not.toContain('Show Hugging Face results');
    const row = dialog()!.querySelector('.disc-row')!.textContent ?? '';
    expect(row).toContain('Clay');
    expect(row).toContain('Curated');
    expect(row).toContain('x');
    expect(row).toContain('1.2k');
    expect(row).toContain('♥ 5');
    expect(row).toContain('3 files');
    expect(row).toContain('No licence');
    expect(row).not.toContain('x/clay');
    expect(buttons(dialog()!)).toContain('Show more');
    expect(buttons(dialog()!)).not.toContain('Styles');
  });

  it('compares Curated with Hugging Face in a table with column and row headers', async () => {
    await mountDrawer([entry()]);
    clickByText('Hugging Face');
    await settle();

    const table = dialog()!.querySelector('table')!;
    const columns = Array.from(table.querySelectorAll('th[scope="col"]')).map((th) => th.textContent?.trim());
    const rows = Array.from(table.querySelectorAll('th[scope="row"]')).map((th) => th.textContent?.trim());
    expect(columns).toEqual(expect.arrayContaining(['Curated', 'Hugging Face']));
    expect(rows).toEqual(['Checked by', 'Licence', 'Previews']);
  });

  it('goes back to Curated on Not now and saves nothing', async () => {
    await mountDrawer([entry()]);
    clickByText('Hugging Face');
    await settle();
    clickByText('Not now');
    await settle();
    expect(dialog()!.querySelector('#disc-tab-curated')?.getAttribute('aria-selected')).toBe('true');
    expect(api.setDiscoverSettings).not.toHaveBeenCalled();
  });

  it('shows why the answer could not be saved', async () => {
    api.setDiscoverSettings.mockRejectedValue(new ApiError('Bad', 500, 'Could not write the settings.'));
    await mountDrawer([entry()]);
    clickByText('Hugging Face');
    await settle();

    clickByText('Show Hugging Face results');
    await settle();

    expect(text()).toContain('Could not write the settings.');
    expect(api.getHfList).not.toHaveBeenCalled();
  });

  it('turns the results off with Turn off, saves false and brings the question back', async () => {
    api.setDiscoverSettings.mockResolvedValue({ hf_enabled: false });
    await mountDrawer([entry()], { list: { hf_enabled: true } });
    clickByText('Hugging Face');
    await settle();
    await settle();
    expect(buttons(dialog()!)).not.toContain('Show Hugging Face results');

    clickByText('Turn off');
    await settle();
    await settle();

    expect(api.setDiscoverSettings).toHaveBeenCalledOnce();
    expect(api.setDiscoverSettings).toHaveBeenCalledWith(false);
    expect(buttons(dialog()!)).toContain('Show Hugging Face results');
    expect(input('Search Hugging Face')).toBeNull();
  });

  it('shows why the results could not be turned off and keeps them', async () => {
    api.setDiscoverSettings.mockRejectedValue(new ApiError('Bad', 500, 'Could not write the settings.'));
    await mountDrawer([entry()], { list: { hf_enabled: true } });
    clickByText('Hugging Face');
    await settle();
    await settle();

    clickByText('Turn off');
    await settle();

    expect(text()).toContain('Could not write the settings.');
    expect(input('Search Hugging Face')).not.toBeNull();
  });

  it('searches after a pause, sorts, and appends the next page on Show more', async () => {
    api.getHfList.mockResolvedValueOnce({ items: [{ repo: 'x/one', name: 'One', author: 'x', downloads: 1, likes: 1, files: 1, licence: 'mit', thumbnail: null, curated_id: null }], next_cursor: 'c1', base_models: [] })
      .mockResolvedValue({ items: [{ repo: 'x/two', name: 'Two', author: 'x', downloads: 1, likes: 1, files: 1, licence: 'mit', thumbnail: null, curated_id: null }], next_cursor: null, base_models: [] });
    await mountDrawer([entry()], { list: { hf_enabled: true } });
    clickByText('Hugging Face');
    await settle();
    await settle();

    clickByText('Show more');
    await settle();
    expect(api.getHfList).toHaveBeenLastCalledWith({ family: 'zimage', size: null, version: null }, expect.objectContaining({ cursor: 'c1' }));
    expect(text()).toContain('One');
    expect(text()).toContain('Two');
    expect(buttons(dialog()!)).not.toContain('Show more');

    api.getHfList.mockClear();
    typeInto(input('Search Hugging Face'), 'clay');
    await settle(INPUT_DEBOUNCE_MS - 50);
    expect(api.getHfList).not.toHaveBeenCalled();
    await settle(100);
    expect(api.getHfList).toHaveBeenCalledWith({ family: 'zimage', size: null, version: null }, expect.objectContaining({ q: 'clay', cursor: null }));
  });

  it('asks for the adapters of the LTX version of the Workspace model', async () => {
    await mountDrawer([], { list: { hf_enabled: true }, target: { family: 'ltx', size: null, version: '2.5', name: 'LTX-2.5', familyName: 'LTX' } });
    clickByText('Hugging Face');
    await settle();
    await settle();

    expect(api.getHfList).toHaveBeenCalledWith({ family: 'ltx', size: null, version: '2.5' }, expect.objectContaining({ cursor: null }));
  });

  it('lists a repository once even when a page repeats it or the next page returns it again', async () => {
    const row = (repo: string) => ({ repo, name: repo, author: 'x', downloads: 1, likes: 1, files: 1, licence: 'mit', thumbnail: null, curated_id: null });
    api.getHfList.mockResolvedValueOnce({ items: [row('x/one'), row('x/one'), row('x/two')], next_cursor: 'c1', base_models: [] })
      .mockResolvedValue({ items: [row('x/two'), row('x/three'), row('x/three')], next_cursor: null, base_models: [] });
    await mountDrawer([entry()], { list: { hf_enabled: true } });
    clickByText('Hugging Face');
    await settle();
    await settle();
    expect(dialog()!.querySelectorAll('.disc-row')).toHaveLength(2);

    clickByText('Show more');
    await settle();

    expect(Array.from(dialog()!.querySelectorAll('.disc-row .disc-name')).map((el) => el.textContent?.trim())).toEqual(['x/one', 'x/two', 'x/three']);
  });

  it('sorts from a menu on a phone, where the sort control is an icon button', async () => {
    Object.assign(layoutMock, { compact: true, mode: 'compact', coarse: true });
    await mountDrawer([entry()], { list: { hf_enabled: true } });
    clickByText('Hugging Face');
    await settle();
    await settle();

    const sort = dialog()!.querySelector('button[aria-label="Sort by"]') as HTMLButtonElement;
    expect(sort.textContent?.trim()).toBe('');
    flushSync(() => sort.click());
    await settle();
    api.getHfList.mockClear();
    const newest = Array.from(document.body.querySelectorAll('[role="menuitemradio"]')).find((el) => el.textContent?.includes('Newest')) as HTMLElement;
    flushSync(() => newest.click());
    await settle(INPUT_DEBOUNCE_MS + 10);

    expect(api.getHfList).toHaveBeenCalledWith({ family: 'zimage', size: null, version: null }, expect.objectContaining({ sort: 'newest' }));
  });

  async function openRepo(check: HfCheck | Error, repo: HfRepo = repoDetail()): Promise<Spies> {
    api.getHfList.mockResolvedValue({ items: [{ repo: repo.repo, name: repo.name, author: 'x', downloads: 1, likes: 1, files: 1, licence: 'mit', thumbnail: null, curated_id: null }], next_cursor: null, base_models: [] });
    api.getHfRepo.mockResolvedValue(repo);
    if (check instanceof Error) api.getHfCheck.mockRejectedValue(check);
    else api.getHfCheck.mockResolvedValue(check);
    const spies = await mountDrawer([entry()], { list: { hf_enabled: true } });
    clickByText('Hugging Face');
    await settle();
    await settle();
    await openDetail('Clay');
    return spies;
  }
  const match: HfCheck = { status: 'match', family: 'zimage', size_variant: null, expected: { family: 'zimage', size_variant: null } };

  it('shows the repository name, the family check, the trigger from the card, weight 1 and the name it saves as', async () => {
    await openRepo(match);

    expect(api.getHfCheck).toHaveBeenCalledWith('x/clay', 'clay.safetensors', { family: 'zimage', size: null });
    expect(dialog()!.querySelector('.disc-lead')?.textContent).toBe('x/clay');
    expect(dialog()!.querySelector('.disc-kv .ok')?.textContent).toContain('Z-Image Turbo');
    expect(dialog()!.querySelector('.disc-code')?.textContent).toBe('clayish');
    expect(text()).toContain('Weight');
    expect(text()).toContain('Saves as');
    expect(buttons(dialog()!)).toContain('Download');
  });

  it('warns on another model with an amber icon and that model\'s name, and offers the normal footer', async () => {
    api.startDiscoverDownload.mockResolvedValue({ name: 'clay', status: 'queued', renamed: false, item: null, trigger: 'clayish', trigger_source: 'card' });
    const spies = await openRepo({ status: 'mismatch', family: 'krea2', size_variant: null, expected: { family: 'zimage', size_variant: null } });

    expect(dialog()!.querySelector('.disc-kv .w')?.textContent).toContain('Krea 2');
    expect(buttons(dialog()!)).toContain('Download');
    expect(useSwitch().disabled).toBe(false);

    await flipSwitch();
    expect(api.startDiscoverDownload).toHaveBeenCalledWith({ repo: 'x/clay', file: 'clay.safetensors', name: 'clay' });
    expect(spies.onuse).toHaveBeenCalledWith({ name: 'clay', weight: 1, trigger: 'clayish', downloading: true });
  });

  it('leaves the Made for row out when it cannot tell or the check fails, and keeps the footer on', async () => {
    await openRepo(new ApiError('Bad', 502, 'Hugging Face is down.'));

    expect(text()).not.toContain('Made for');
    expect((Array.from(dialog()!.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Download'))!.disabled).toBe(false);
  });

  it('folds training snapshots under Other files, lists real variants, and checks the file chosen', async () => {
    const files = [
      { file: 'a.safetensors', size_bytes: 1, size_label: '1 KB', sha256: null, step: null, marked: true, curated_id: null, save_as: 'clay', renamed: false, state: 'available' as const },
      { file: 'b.safetensors', size_bytes: 2, size_label: '2 KB', sha256: null, step: null, marked: false, curated_id: null, save_as: 'clay', renamed: false, state: 'available' as const },
    ];
    const snapshot = { file: 'clay_000100.safetensors', size_bytes: 3, size_label: '3 KB', sha256: null, step: 100, marked: false, curated_id: null, save_as: 'clay', renamed: false, state: 'available' as const };
    await openRepo(match, repoDetail({ preselected: 'a.safetensors', files, other_files: [snapshot] }));

    expect(text()).toContain('Other files (1)');
    expect(dialog()!.querySelectorAll('input[name="disc-file"]')).toHaveLength(3);
    api.getHfCheck.mockClear();
    flushSync(() => (dialog()!.querySelector('input[value="b.safetensors"]') as HTMLInputElement).click());
    await settle();
    expect(api.getHfCheck).toHaveBeenCalledWith('x/clay', 'b.safetensors', { family: 'zimage', size: null });
  });

  describe('a repository with two variant files', () => {
    const variant = (file: string, saveAs: string, state: 'available' | 'downloading' | 'installed', overrides: Record<string, unknown> = {}) => ({
      file, size_bytes: 1, size_label: '1 KB', sha256: null, step: null, marked: false, curated_id: null, save_as: saveAs, renamed: saveAs !== 'clay', state, ...overrides,
    });
    const twoFiles = (a: ReturnType<typeof variant>, b: ReturnType<typeof variant>) => repoDetail({ preselected: 'a.safetensors', files: [a, b] });
    const pickB = async (): Promise<void> => {
      flushSync(() => (dialog()!.querySelector('input[value="b.safetensors"]') as HTMLInputElement).click());
      await settle();
    };
    const savesAs = () => Array.from(dialog()!.querySelectorAll('.disc-kv')).find((row) => row.textContent?.startsWith('Saves as'))?.querySelector('.v')?.textContent;
    const laneItem = (state: string, overrides: Record<string, unknown> = {}) => ({ id: 'dl-1', kind: 'lora', name: 'clay', label: 'LoRA clay', state, message: null, curated_id: null, ...overrides });

    it('reads the repository again when a download starts, so the other file takes its own name and shows nothing of the download', async () => {
      const item = laneItem('running', { bytes_done: 40, bytes_total: 100 });
      api.startDiscoverDownload.mockResolvedValue({ name: 'clay', status: 'running', renamed: false, item, trigger: 'clayish', trigger_source: 'card' });
      models.getDownloads.mockResolvedValue({ items: [item] } as never);
      await openRepo(match, twoFiles(variant('a.safetensors', 'clay', 'available'), variant('b.safetensors', 'clay', 'available')));
      expect(savesAs()).toBe('clay');
      api.getHfRepo.mockClear();
      api.getHfRepo.mockResolvedValue(twoFiles(variant('a.safetensors', 'clay', 'downloading'), variant('b.safetensors', 'clay-2', 'available')));

      clickByText('Download');
      await settle();
      await settle();

      expect(api.getHfRepo).toHaveBeenCalledWith('x/clay');
      expect(buttons(dialog()!)).toContain('Downloading 40 %');
      await pickB();
      expect(savesAs()).toBe('clay-2');
      expect(buttons(dialog()!)).toContain('Download');
      expect(buttons(dialog()!).some((label) => label.startsWith('Downloading'))).toBe(false);
      expect(useSwitch().checked).toBe(false);

      api.startDiscoverDownload.mockClear();
      clickByText('Download');
      await settle();
      expect(api.startDiscoverDownload).toHaveBeenCalledWith({ repo: 'x/clay', file: 'b.safetensors', name: 'clay-2' });
    });

    it('reads the repository again when the download finishes, so the other file is not shown as installed or deletable', async () => {
      const running = laneItem('running', { bytes_done: 40, bytes_total: 100 });
      models.getDownloads.mockResolvedValue({ items: [running] } as never);
      await openRepo(match, twoFiles(variant('a.safetensors', 'clay', 'downloading'), variant('b.safetensors', 'clay-2', 'available')));
      downloadStore.startSync();
      await settle();
      await pickB();
      api.getHfRepo.mockClear();
      api.getHfRepo.mockResolvedValue(twoFiles(variant('a.safetensors', 'clay', 'installed'), variant('b.safetensors', 'clay-2', 'available')));

      models.getDownloads.mockResolvedValue({ items: [laneItem('done')] } as never);
      await downloadStore.refresh();
      await settle();
      await settle();

      expect(api.getHfRepo).toHaveBeenCalledWith('x/clay');
      expect(savesAs()).toBe('clay-2');
      expect(buttons(dialog()!)).toContain('Download');
      expect(buttons(dialog()!)).not.toContain('Delete file');
    });
  });

  it('has no File row for a repository with one file', async () => {
    await openRepo(match);

    expect(dialog()!.querySelectorAll('input[name="disc-file"]')).toHaveLength(0);
    expect(text()).not.toContain('File to download');
  });

  it('opens the curated detail for a repository that is one of ours', async () => {
    api.getHfList.mockResolvedValue({ items: [{ repo: 'o/style', name: 'Style', author: 'o', downloads: 1, likes: 1, files: 1, licence: 'mit', thumbnail: null, curated_id: 'o-style' }], next_cursor: null, base_models: [] });
    await mountDrawer([entry()], { list: { hf_enabled: true } });
    clickByText('Hugging Face');
    await settle();
    await settle();

    await openDetail('Style');

    expect(api.getHfRepo).not.toHaveBeenCalled();
    expect(text()).toContain('sty1e, soft');
    expect(text()).toContain('Apache 2.0');
  });
});

describe('Discover LoRAs drawer: a model with no known family', () => {
  it('says so and asks for nothing', async () => {
    api.getCurated.mockResolvedValue(curatedList([]));
    mountComponent(DiscoverDrawer, { open: true, context: makeContext(), target: null, chips: [], onuse: vi.fn(), onstop: vi.fn(), ondeleted: vi.fn(), onloraschange: vi.fn() });
    await settle();

    expect(text()).toContain('Discover needs a model whose LoRA family is known.');
    expect(api.getCurated).not.toHaveBeenCalled();
  });
});
