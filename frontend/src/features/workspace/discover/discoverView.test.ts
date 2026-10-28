import { describe, expect, it } from 'vitest';
import type { DownloadItem, HfCheck, LoraInfo, CuratedEntry, WorkspaceContext } from '$lib/types';
import { makeContext } from '../../../test-utils/workspace';
import {
  cliLine,
  compactCount,
  curatedDetail,
  curatedTile,
  discoverTarget,
  downloadingLabel,
  filterCurated,
  filterInstalled,
  footerView,
  hfDetail,
  hfRowFacts,
  installedDetail,
  installedForModel,
  installedMadeFor,
  installedTile,
  loraStatus,
  madeFor,
  type FooterFacts,
  type LoraStatus,
  type UseFacts,
} from './discoverView';

const entry = (overrides: Partial<CuratedEntry> = {}): CuratedEntry => ({
  id: 'o-style', name: 'style', family: 'zimage', size_variant: null, version: null, display_name: 'Style', author: 'o', description: 'A soft look', section: 'style',
  what_it_does: null, workflow: null, repo: 'o/style', revision: 'a'.repeat(40), file: 'style.safetensors', size_bytes: 1000, size_label: '1 KB', trigger: 'sty1e, soft', weight: 0.8,
  licence: 'apache-2.0', licence_label: 'Apache 2.0', non_commercial: false, downloads: 1529, platforms: [], note: null, previews: ['https://huggingface.co/o/style/1.png'], preview_videos: [],
  model_card_url: 'https://huggingface.co/o/style', state: 'available', download: null, ...overrides,
});
const item = (overrides: Partial<DownloadItem> = {}): DownloadItem => ({ id: 'd', kind: 'lora', name: 'style', label: 'LoRA style', state: 'running', message: null, bytes_done: 1, bytes_total: 2, ...overrides });
const file = (name: string, overrides: Partial<LoraInfo> = {}): LoraInfo => ({ name, path: `/l/${name}.safetensors`, family: 'zimage', ...overrides });
const target = { family: 'zimage', size: null, version: null, name: 'Z-Image Turbo', familyName: 'Z-Image' };

describe('discoverTarget', () => {
  const context = (): WorkspaceContext => makeContext({
    image_model_defaults: { zit: { family: 'zimage' }, klein: { family: 'flux2_klein', size_variant: '4b' } } as never,
    video_model_defaults: { 'ltx-4': { family: 'ltx' }, 'ltx25-4': { family: 'ltx' }, h3: { family: 'h3' } } as never,
  });

  it('locks to the family, with the size for Klein', () => {
    expect(discoverTarget(context(), 'klein', 'Klein 4B')).toMatchObject({ family: 'flux2_klein', size: '4b', version: null });
  });

  it('adds the LTX version the server read from the model, and 2.3 when it did not say', () => {
    const withVersions = (versions: Record<string, '2.3' | '2.5' | undefined>): WorkspaceContext => {
      const base = context();
      return { ...base, video_models: Object.entries(versions).map(([id, version]) => ({ id, label: id, type: 'video' as const, version })) };
    };
    const ctx = withVersions({ 'ltx-4': '2.3', 'ltx25-4': '2.5', 'my-merge': '2.5', 'ltx25-lookalike': '2.3', 'no-field': undefined });
    ctx.video_model_defaults = { ...ctx.video_model_defaults, 'my-merge': { family: 'ltx' }, 'ltx25-lookalike': { family: 'ltx' }, 'no-field': { family: 'ltx' } } as never;

    expect(discoverTarget(ctx, 'ltx-4', 'LTX-2.3 4-bit')).toMatchObject({ family: 'ltx', version: '2.3', name: 'LTX-2.3' });
    expect(discoverTarget(ctx, 'ltx25-4', 'LTX-2.5 4-bit')).toMatchObject({ version: '2.5', name: 'LTX-2.5' });
    // A converted model with a name of its own is 2.5 by what the server read; a name that only looks like 2.5 is not.
    expect(discoverTarget(ctx, 'my-merge', 'my-merge')).toMatchObject({ version: '2.5', name: 'LTX-2.5' });
    expect(discoverTarget(ctx, 'ltx25-lookalike', 'x')).toMatchObject({ version: '2.3' });
    expect(discoverTarget(ctx, 'no-field', 'x')).toMatchObject({ version: '2.3' });
  });

  it('names the model by its label outside LTX', () => {
    expect(discoverTarget(context(), 'zit', 'Z-Image Turbo')?.name).toBe('Z-Image Turbo');
  });

  it('is null when the model is unknown', () => {
    expect(discoverTarget(context(), 'nothing', 'x')).toBeNull();
  });
});

describe('loraStatus', () => {
  const facts = (overrides: Partial<UseFacts> = {}): UseFacts => ({ chips: [], installed: [], items: [], ...overrides });
  const chips = [{ name: 'style', weight: 1 }];

  it('is available, installed or in use by where the file and the chip are', () => {
    expect(loraStatus('style', 'available', facts())).toEqual({ file: 'available', percent: null, inUse: false });
    expect(loraStatus('style', 'available', facts({ installed: [file('style')] }))).toMatchObject({ file: 'installed', inUse: false });
    expect(loraStatus('style', 'installed', facts())).toMatchObject({ file: 'installed' });
    expect(loraStatus('style', 'available', facts({ chips, installed: [file('style')] }))).toMatchObject({ file: 'installed', inUse: true });
  });

  it('follows the lane for a file on its way, with how far it is, and for one that failed', () => {
    expect(loraStatus('style', 'available', facts({ items: [item({ bytes_done: 40, bytes_total: 100 })] }))).toEqual({ file: 'downloading', percent: 40, inUse: false });
    expect(loraStatus('style', 'available', facts({ items: [item({ state: 'failed', message: 'no' })] }))).toMatchObject({ file: 'failed' });
    expect(loraStatus('style', 'downloading', facts())).toMatchObject({ file: 'downloading', percent: null });
  });

  it('keeps a chip in use whatever its file is doing', () => {
    expect(loraStatus('style', 'available', facts({ chips, items: [item()] }))).toMatchObject({ file: 'downloading', inUse: true });
    expect(loraStatus('style', 'available', facts({ chips, items: [item({ state: 'failed', message: 'no' })] }))).toMatchObject({ file: 'failed', inUse: true });
  });

  it("is taken when the server says a different file holds the entry's name: not installed, not in use, whatever files and chips say", () => {
    expect(loraStatus('style', 'name_taken', facts())).toEqual({ file: 'taken', percent: null, inUse: false });
    expect(loraStatus('style', 'name_taken', facts({ chips, installed: [file('style')] }))).toEqual({ file: 'taken', percent: null, inUse: false });
  });
});

describe('filterCurated', () => {
  const list = [
    entry(),
    entry({ id: 'b', name: 'clay', display_name: 'Clay Animation', description: 'Stop-motion figures', trigger: 'crtanim', author: 'artist', what_it_does: null }),
    entry({ id: 't', name: 'shatter', display_name: 'Shatter', section: 'tool', what_it_does: 'Shatters the shot' }),
  ];

  it('needs every word, in any case and order, across name, description, what it does, author and trigger', () => {
    expect(filterCurated(list, 'MOTION clay', 'all').map((e) => e.id)).toEqual(['b']);
    expect(filterCurated(list, 'crtanim', 'all').map((e) => e.id)).toEqual(['b']);
    expect(filterCurated(list, 'artist', 'all').map((e) => e.id)).toEqual(['b']);
    expect(filterCurated(list, 'shatters', 'all').map((e) => e.id)).toEqual(['t']);
    expect(filterCurated(list, 'watercolor', 'all')).toEqual([]);
  });

  it('keeps everything for a blank query, in curated order, and narrows to styles or tools', () => {
    expect(filterCurated(list, '  ', 'all').map((e) => e.id)).toEqual(['o-style', 'b', 't']);
    expect(filterCurated(list, '', 'style').map((e) => e.id)).toEqual(['o-style', 'b']);
    expect(filterCurated(list, '', 'tool').map((e) => e.id)).toEqual(['t']);
    expect(filterCurated(list, 'shot', 'style')).toEqual([]);
  });
});

describe('tiles', () => {
  const status: LoraStatus = { file: 'available', percent: null, inUse: false };

  it('shows a style with its description, author and downloads, and a tool with what it does and its tag', () => {
    expect(curatedTile(entry(), status)).toMatchObject({ title: 'Style', tool: false, line: 'A soft look', author: 'o', downloads: 1529, image: 'https://huggingface.co/o/style/1.png' });
    expect(curatedTile(entry({ section: 'tool', what_it_does: 'Shatters the shot' }), status)).toMatchObject({ tool: true, line: 'Shatters the shot', image: 'https://huggingface.co/o/style/1.png' });
  });

  it('describes an installed file by its curated entry, by what a Hugging Face download kept, or by its name alone', () => {
    const curated = file('style', { curated_id: 'o-style' });
    const hub = file('pix', { hub: { name: 'Pixel Art Style', author: 'tarn59', downloads: 14000, licence: null, licence_label: null, model_card_url: 'u', thumbnail_url: '/api/discover/installed-thumbnails/pix?v=1' } });

    expect(installedTile(curated, [entry()], status)).toMatchObject({ title: 'Style', line: 'A soft look' });
    expect(installedTile(hub, [entry()], status)).toMatchObject({ title: 'Pixel Art Style', author: 'tarn59', downloads: 14000, image: '/api/discover/installed-thumbnails/pix?v=1', line: null });
    expect(installedTile(file('my_style'), [entry()], status)).toMatchObject({ title: 'my_style', author: null, downloads: null, image: null });
  });
});

describe('installed LoRAs of the model', () => {
  const loras = [file('b_style'), file('a_style'), file('unknown', { family: null }), file('krea', { family: 'krea2' }), file('klein9', { family: 'flux2_klein', size_variant: '9b' })];

  it('lists the ones that fit by name, the ones of unknown model apart, and leaves out those of another model', () => {
    const { main, other } = installedForModel(loras, target);

    expect(main.map((lora) => lora.name)).toEqual(['a_style', 'b_style']);
    expect(other.map((lora) => lora.name)).toEqual(['unknown']);
  });

  it('narrows a Klein size', () => {
    const klein4 = { family: 'flux2_klein', size: '4b', version: null, name: 'Klein 4B', familyName: 'Klein' };

    expect(installedForModel(loras, klein4).main).toEqual([]);
    expect(installedForModel(loras, { ...klein4, size: '9b' }).main.map((lora) => lora.name)).toEqual(['klein9']);
  });

  it('shows files of your own and Hugging Face downloads under All only, and searches name, title and author', () => {
    const entries = [entry({ id: 'o-style', name: 'style' }), entry({ id: 'o-tool', name: 'tool', section: 'tool', display_name: 'Tooly' })];
    const list = [file('style', { curated_id: 'o-style' }), file('tool', { curated_id: 'o-tool' }), file('my_style'), file('pix', { hub: { name: 'Pixel Art', author: 'tarn59', downloads: 1, licence: null, licence_label: null, model_card_url: 'u', thumbnail_url: null } })];

    expect(filterInstalled(list, entries, '', 'all').map((lora) => lora.name)).toEqual(['style', 'tool', 'my_style', 'pix']);
    expect(filterInstalled(list, entries, '', 'style').map((lora) => lora.name)).toEqual(['style']);
    expect(filterInstalled(list, entries, '', 'tool').map((lora) => lora.name)).toEqual(['tool']);
    expect(filterInstalled(list, entries, 'tarn59', 'all').map((lora) => lora.name)).toEqual(['pix']);
    expect(filterInstalled(list, entries, 'my_', 'all').map((lora) => lora.name)).toEqual(['my_style']);
  });
});

describe('small formatters', () => {
  it('writes counts compactly', () => {
    expect([950, 1200, 12400].map(compactCount)).toEqual(['950', '1.2k', '12k']);
  });

  it('writes the CLI line with the name it saves as, quoting only what needs it', () => {
    expect(cliLine('o/r', 'a b.safetensors', 'n')).toBe("ziv-model lora --hf o/r --file 'a b.safetensors' --name n");
  });

  it('writes the download progress of the footer button as unknown while the size is not known', () => {
    expect(downloadingLabel(null)).toBe('Downloading…');
  });

  it('leaves out of a listing row what is plain: one file, a licence', () => {
    expect(hfRowFacts({ files: 1, licence: 'mit' })).toEqual({ files: false, noLicence: false });
    expect(hfRowFacts({ files: 5, licence: null })).toEqual({ files: true, noLicence: true });
  });
});

describe('details', () => {
  it('leads a tool with what it does and a style with its description, and says which workflow a tool needs', () => {
    expect(curatedDetail(entry({ section: 'tool', what_it_does: 'Shatters the shot', workflow: 'img2vid' }), 'LTX-2.3')).toMatchObject({ lead: 'Shatters the shot', tool: true, worksWith: 'Image to Video' });
    expect(curatedDetail(entry({ section: 'tool', what_it_does: 'Enhances detail', workflow: null }), 'Z-Image')).toMatchObject({ worksWith: 'Every workflow of Z-Image' });
    expect(curatedDetail(entry(), 'Z-Image')).toMatchObject({ lead: 'A soft look', tool: false, worksWith: null, weight: 0.8, downloads: 1529, licence: 'Apache 2.0', trigger: 'sty1e, soft' });
  });

  it('shows a Hugging Face file with the repository name under its title, weight 1 and its facts', () => {
    const repo = { repo: 'o/r', name: 'R', downloads: 77, previews: [], preview_videos: [], licence_label: null, instance_prompt: 'a, b', model_card_url: 'u' } as never;
    const hf = { file: 'f.safetensors', size_label: '1 MB', save_as: 'r-2', renamed: true };

    expect(hfDetail(repo, hf as never)).toMatchObject({ title: 'R', lead: 'o/r', leadIsRepo: true, weight: 1, downloads: 77, licence: null, trigger: 'a, b', saveAs: 'r-2', modelCardUrl: 'u' });
    expect(hfDetail(repo, hf as never).cli).toBe('ziv-model lora --hf o/r --file f.safetensors --name r-2');
  });

  it('says what the empty picture box means: the author added no images, or only videos; a file of your own says nothing', () => {
    expect(curatedDetail(entry({ previews: [], preview_videos: [] }), 'x').emptyNote).toBe('The author added no preview images');
    expect(curatedDetail(entry({ previews: [], preview_videos: ['https://huggingface.co/o/v.mp4'] }), 'x').emptyNote).toBe('The author added only preview videos');
  });

  it('shows an installed Hugging Face download from what it kept, and a file of your own with the rows that apply to it', () => {
    const hub = { name: 'Pixel Art', author: 'tarn59', downloads: 14000, licence: 'mit', licence_label: 'MIT', model_card_url: 'u', thumbnail_url: '/i.webp' };

    expect(installedDetail(file('pix', { hub, source: { repo: 'tarn59/pix', revision: 'a', file: 'p.safetensors', sha256: null, curated_id: null }, size_label: '85 MB', trigger: 'pix' }))).toMatchObject({
      title: 'Pixel Art', lead: 'tarn59/pix', weight: 1, downloads: 14000, licence: 'MIT', previews: ['/i.webp'], size: '85 MB', saveAs: 'pix', modelCardUrl: 'u',
    });
    const own = installedDetail(file('my_style', { size_label: '163 MB' }));
    expect(own).toMatchObject({ title: 'my_style', lead: null, weight: null, downloads: null, licence: undefined, modelCardUrl: null, cli: null, previews: [], emptyNote: null });
  });
});

describe('madeFor', () => {
  const check = (overrides: Partial<HfCheck>): HfCheck => ({ status: 'match', family: 'zimage', size_variant: null, expected: { family: 'zimage', size_variant: null }, ...overrides });

  it('says the model on a match and the other model on a mismatch, and nothing while the answer is unknown', () => {
    expect(madeFor(null, target)).toBeNull();
    expect(madeFor(check({}), target)).toEqual({ kind: 'match', text: 'Z-Image Turbo' });
    expect(madeFor(check({ status: 'mismatch', family: 'krea2' }), target)).toEqual({ kind: 'mismatch', text: 'Krea 2' });
    expect(madeFor(check({ status: 'mismatch', family: null }), target)).toEqual({ kind: 'mismatch', text: 'Another model' });
    expect(madeFor(check({ status: 'unknown', family: null }), target)).toBeNull();
  });

  it('reads an installed file by the family the app detected in it', () => {
    expect(installedMadeFor(file('a'), target)).toEqual({ kind: 'match', text: 'Z-Image Turbo' });
    expect(installedMadeFor(file('a', { family: 'krea2' }), target)).toEqual({ kind: 'mismatch', text: 'Krea 2' });
    expect(installedMadeFor(file('a', { family: null }), target)).toBeNull();
  });
});

describe('footerView', () => {
  const facts = (overrides: Partial<FooterFacts> = {}, status: Partial<LoraStatus> = {}): FooterFacts => ({ status: { file: 'available', percent: null, inUse: false, ...status }, name: 'style', offline: false, deleteBlocked: false, busy: false, ...overrides });

  it('offers Download and an off switch for a file that is not here', () => {
    expect(footerView(facts())).toEqual({ action: { kind: 'download', label: 'Download', disabled: false }, use: { on: false, disabled: false }, note: null });
  });

  it('offers Delete file and the switch for a file that is here, with the switch on when a chip uses it', () => {
    expect(footerView(facts({}, { file: 'installed' }))).toMatchObject({ action: { kind: 'delete', label: 'Delete file', disabled: false }, use: { on: false } });
    expect(footerView(facts({}, { file: 'installed', inUse: true })).use.on).toBe(true);
  });

  it('turns Delete off with the reason while a job runs', () => {
    expect(footerView(facts({ deleteBlocked: true }, { file: 'installed' }))).toMatchObject({ action: { kind: 'delete', disabled: true }, note: 'You can delete this when the current job is done.' });
  });

  it('reads Downloading with the percent while the file arrives, and leaves the switch alone', () => {
    expect(footerView(facts({}, { file: 'downloading', percent: 40, inUse: true }))).toEqual({ action: { kind: 'downloading', label: 'Downloading 40 %', disabled: true }, use: { on: true, disabled: false }, note: null });
  });

  it('offers Retry download after a failure', () => {
    expect(footerView(facts({}, { file: 'failed' }))).toMatchObject({ action: { kind: 'retry', label: 'Retry download', disabled: false } });
  });

  it("is offline: a file that is not here can't download or be switched on, one that is here still can", () => {
    expect(footerView(facts({ offline: true }))).toMatchObject({ action: { kind: 'download', disabled: true }, use: { disabled: true }, note: "You're offline." });
    expect(footerView(facts({ offline: true }, { file: 'failed' }))).toMatchObject({ action: { disabled: true }, use: { disabled: true }, note: "You're offline." });
    expect(footerView(facts({ offline: true }, { file: 'installed' }))).toMatchObject({ action: { kind: 'delete', disabled: false }, use: { disabled: false }, note: null });
  });

  it('offers no download, no delete and no switch for a name a different file holds, and says so', () => {
    expect(footerView(facts({}, { file: 'taken' }))).toEqual({
      action: { kind: 'download', label: 'Download', disabled: true },
      use: { on: false, disabled: true },
      note: 'You already have a different file called style.',
    });
    expect(footerView(facts({ offline: true }, { file: 'taken' })).note).toBe('You already have a different file called style.');
  });

  it('turns everything off while a request is on its way', () => {
    expect(footerView(facts({ busy: true }, { file: 'installed' }))).toMatchObject({ action: { disabled: true }, use: { disabled: true } });
  });
});
