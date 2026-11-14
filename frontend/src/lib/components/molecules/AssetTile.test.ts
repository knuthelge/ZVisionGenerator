import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { audioPeaks } from '$lib/state/audioPeaks.svelte';

import type { GalleryAsset } from '$lib/types';

import AssetTile from './AssetTile.svelte';
import { makeAudioAsset, makeImportedAudio, makeSpeechAsset } from '../../../test-utils/audio';
import { mountComponent, settle } from '../../../test-utils/mount';

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/a.png',
    url: '/media/out/a.png',
    thumbnail_url: '/media/out/a-thumb.png',
    filename: 'a.png',
    created_at: '2026-10-03T12:00:00Z',
    project: null,
    workflow: 'txt2img',
    prompt: 'a fox',
    model: 'zit',
    media_type: 'image',
    file_path: '/outputs/out/a.png',
    has_reusable_config: true,
    reuse_workspace_url: '#/workspace?workflow=txt2img',
    ...overrides,
  };
}

function menuActions(): string[] {
  return Array.from(document.querySelectorAll('[role="menu"] [role="menuitem"]')).map((item) => item.getAttribute('data-action') ?? '');
}

const mocks = vi.hoisted(() => ({ getAudioPeaks: vi.fn(), listProjects: vi.fn(), createProject: vi.fn() }));
vi.mock('$lib/api/gallery', () => ({ getAudioPeaks: mocks.getAudioPeaks }));
vi.mock('$lib/api/projects', () => ({ listProjects: mocks.listProjects, createProject: mocks.createProject }));

describe('AssetTile', () => {
  let page: ReturnType<typeof mountComponent>;
  let target: HTMLElement;

  function mountTile(props: Record<string, unknown>): void {
    page = mountComponent(AssetTile, props);
    target = page.target;
  }

  function openMenu(asset: GalleryAsset): void {
    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
  }

  describe('meta line with showProject', () => {
    const foot = () => target.querySelector('.asset-tile-foot p:nth-child(2)')?.textContent ?? '';
    const date = new Date(makeAsset().created_at).toLocaleDateString();

    it('leads with the project in place of the workflow word', () => {
      mountTile({ asset: makeAsset({ project: 'Night Bus' }), showProject: true });
      expect(foot()).toBe(`Night Bus · ${date}`);
    });

    it('keeps the workflow word for a file in no project, and without the prop', () => {
      mountTile({ asset: makeAsset(), showProject: true });
      expect(foot()).toBe(`txt2img · ${date}`);
      page.unmount();
      mountTile({ asset: makeAsset({ project: 'Night Bus' }) });
      expect(foot()).toBe(`txt2img · ${date}`);
    });

    it('leads an audio tile\'s foot with the project', () => {
      mountTile({ asset: makeSpeechAsset({ project: 'Night Bus' }), showProject: true });
      expect(foot()).toContain('Night Bus · ');
    });

    it('gives a strip tile no foot', () => {
      mountTile({ asset: makeAsset({ project: 'Night Bus' }), density: 'compact', showProject: true });
      expect(target.querySelector('.asset-tile-foot')).toBeNull();
    });
  });

  it('puts the checkerboard on the image of a transparent asset only', () => {
    mountTile({ asset: makeAsset({ transparent: true }) });
    const img = target.querySelector('img') as HTMLImageElement;
    expect(img.classList.contains('ui-checkerboard')).toBe(true);
    expect(img.dataset.transparent).toBe('true');
    expect(target.querySelector('article')?.getAttribute('data-transparent')).toBe('true');
  });

  it.each([{ transparent: false }, {}])('leaves an opaque asset (%j) without the checkerboard', (overrides) => {
    mountTile({ asset: makeAsset(overrides) });
    const img = target.querySelector('img') as HTMLImageElement;
    expect(img.classList.contains('ui-checkerboard')).toBe(false);
    expect(img.dataset.transparent).toBe('false');
  });

  it('keeps the checkerboard in every density', () => {
    for (const density of ['card', 'compact', 'stage']) {
      mountTile({ asset: makeAsset({ transparent: true }), density });
      expect(target.querySelector('img')?.classList.contains('ui-checkerboard')).toBe(true);
    }
  });

  it('never puts the checkerboard on a video', () => {
    mountTile({ asset: makeAsset({ transparent: true, media_type: 'video' }) });
    expect(target.querySelector('.ui-checkerboard')).toBeNull();
  });

  it('previews from the media and the expand action with the clicked trigger', () => {
    const asset = makeAsset();
    const onpreview = vi.fn();
    mountTile({ asset, onpreview });

    const media = target.querySelector(`button[aria-label="View ${asset.filename}"]`) as HTMLButtonElement;
    media.click();
    expect(onpreview).toHaveBeenLastCalledWith(asset, media);

    (target.querySelector(`button[aria-label="View ${asset.filename} fullscreen"]`) as HTMLButtonElement).click();
    expect(onpreview).toHaveBeenCalledTimes(2);
  });

  it('reuses settings, and disables reuse without a reusable config', async () => {
    const onreuse = vi.fn();
    const asset = makeAsset();
    mountTile({ asset, onreuse });
    (target.querySelector(`button[aria-label="Reuse settings from ${asset.filename}"]`) as HTMLButtonElement).click();
    expect(onreuse).toHaveBeenCalledWith(asset);

    await page.unmount();
    mountTile({ asset: makeAsset({ has_reusable_config: false }), onreuse });
    expect((target.querySelector('button[aria-label="Reusable settings unavailable"]') as HTMLButtonElement).disabled).toBe(true);
  });

  it('leads the menu with Reuse settings when reuse is offered, disabled without a reusable config', async () => {
    const onreuse = vi.fn();
    const asset = makeAsset();
    mountTile({ asset, onreuse, ondelete: vi.fn() });
    openMenu(asset);
    expect(menuActions()).toEqual(['reuse', 'download', 'delete']);
    (document.querySelector('[data-action="reuse"]') as HTMLButtonElement).click();
    expect(onreuse).toHaveBeenCalledWith(asset);
    flushSync();

    const plain = makeAsset({ id: 'out/b.png', filename: 'b.png', has_reusable_config: false });
    await page.unmount();
    mountTile({ asset: plain, onreuse });
    openMenu(plain);
    expect(document.querySelector('[data-action="reuse"]')?.getAttribute('aria-disabled')).toBe('true');
  });

  it('offers reference, download and delete for images in the overflow menu', () => {
    const asset = makeAsset();
    const onreference = vi.fn();
    const ondelete = vi.fn();
    mountTile({ asset, onreference, ondelete });

    openMenu(asset);
    expect(menuActions()).toEqual(['reference-image', 'reference-reference', 'reference-video', 'reference-last-frame', 'download', 'delete']);
    expect(document.querySelector('[data-action="download"]')?.getAttribute('href')).toBe(asset.url);

    (document.querySelector('[data-action="reference-video"]') as HTMLButtonElement).click();
    flushSync();
    expect(onreference).toHaveBeenCalledWith(asset, 'video');
    expect(document.querySelector('[role="menu"]')).toBeNull();

    openMenu(asset);
    (document.querySelector('[data-action="delete"]') as HTMLButtonElement).click();
    expect(ondelete).toHaveBeenCalledWith(asset);
  });

  it('shows the use-as hints and disables a target with its reason as the tooltip', () => {
    const asset = makeAsset();
    const onreference = vi.fn();
    mountTile({
      asset,
      onreference,
      useAs: { reference: { disabled: false, hint: 'next slot' }, image: { disabled: true, hint: "Ideogram 4 can't use a start image" } },
    });

    openMenu(asset);
    const image = document.querySelector('[data-action="reference-image"]') as HTMLElement;
    expect(document.querySelector('[data-action="reference-reference"]')?.textContent).toContain('next slot');
    expect(image.getAttribute('aria-disabled')).toBe('true');
    expect(image.getAttribute('title')).toBe("Ideogram 4 can't use a start image");
    image.click();
    flushSync();
    expect(onreference).not.toHaveBeenCalled();
  });

  it('runs an upscale from the menu, disabling a factor over a limit', () => {
    const asset = makeAsset({
      upscale: {
        factors: [
          { factor: 2, width: 1664, height: 2432, allowed: true, reason: null },
          { factor: 4, width: 3328, height: 4864, allowed: false, reason: 'Too large.' },
        ],
        seedvr2: null,
      },
    });
    const onupscale = vi.fn();
    mountTile({ asset, onreference: vi.fn(), onupscale, ondelete: vi.fn() });

    openMenu(asset);
    expect(document.querySelector('[data-action="upscale-creative-4"]')?.getAttribute('aria-disabled')).toBe('true');

    (document.querySelector('[data-action="upscale-creative-2"]') as HTMLButtonElement).click();
    flushSync();
    expect(onupscale).toHaveBeenCalledWith(asset, 2, 'creative');
  });

  it('hides the reference actions for videos', () => {
    const asset = makeAsset({ media_type: 'video', filename: 'a.mp4', url: '/media/out/a.mp4' });
    mountTile({ asset, onreference: vi.fn(), ondelete: vi.fn() });
    openMenu(asset);
    expect(menuActions()).toEqual(['download', 'delete']);
  });

  describe('extend', () => {
    const video = (overrides: Partial<GalleryAsset> = {}): GalleryAsset =>
      makeAsset({ media_type: 'video', filename: 'a.mp4', url: '/media/out/a.mp4', extend: { available: true, reason: null }, ...overrides });

    it('lists Extend first in a video menu, ahead of Download and Delete', () => {
      const asset = video();
      const onextend = vi.fn();
      mountTile({ asset, onextend, ondelete: vi.fn(), onreference: vi.fn() });
      openMenu(asset);
      expect(menuActions()).toEqual(['extend', 'download', 'delete']);

      (document.querySelector('[data-action="extend"]') as HTMLButtonElement).click();
      flushSync();
      expect(onextend).toHaveBeenCalledWith(asset);
      expect(document.querySelector('[role="menu"]')).toBeNull();
    });

    it('disables Extend with the reason when the video cannot be extended', () => {
      const asset = video({ extend: { available: false, reason: "Upscaled videos can't be extended yet." } });
      const onextend = vi.fn();
      mountTile({ asset, onextend });
      openMenu(asset);

      const item = document.querySelector('[data-action="extend"]') as HTMLElement;
      expect(item.getAttribute('aria-disabled')).toBe('true');
      expect(item.textContent).toContain("Upscaled videos can't be extended yet.");
      item.click();
      expect(onextend).not.toHaveBeenCalled();
    });

    it('has no Extend item for an image, an unavailable video, or without a handler', async () => {
      const cases: Array<[GalleryAsset, boolean]> = [
        [makeAsset({ extend: { available: true, reason: null } }), true],
        [video({ extend: null }), true],
        [video(), false],
      ];
      for (const [asset, withHandler] of cases) {
        mountTile({ asset, onextend: withHandler ? vi.fn() : undefined });
        openMenu(asset);
        expect(menuActions()).not.toContain('extend');
        await page.unmount();
        document.body.innerHTML = '';
      }
    });

  });

  it('closes the menu with Escape and returns focus to its button', async () => {
    const asset = makeAsset();
    mountTile({ asset, ondelete: vi.fn() });
    openMenu(asset);
    await Promise.resolve();

    document.querySelector('[role="menu"]')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    await Promise.resolve();

    expect(document.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement?.getAttribute('aria-label')).toBe(`More actions for ${asset.filename}`);
  });

  it('shows a selection checkbox only when selection is supported', async () => {
    const asset = makeAsset();
    const onselect = vi.fn();
    mountTile({ asset });
    expect(target.querySelector('input[type="checkbox"]')).toBeNull();

    await page.unmount();
    mountTile({ asset, onselect });
    const checkbox = target.querySelector(`input[aria-label="Select ${asset.filename}"]`) as HTMLInputElement;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event('change', { bubbles: true }));
    expect(onselect).toHaveBeenCalledWith(asset, true);
  });

  it('shows the full-size file on the stage and the thumbnail elsewhere', async () => {
    const asset = makeAsset();
    mountTile({ asset, density: 'compact' });
    expect(target.querySelector('img')?.getAttribute('src')).toBe(asset.thumbnail_url);

    await page.unmount();
    mountTile({ asset, density: 'stage' });
    expect(target.querySelector('img')?.getAttribute('src')).toBe(asset.url);
  });

  describe('a video', () => {
    const clip = () => makeAsset({ id: 'out/clip.mp4', url: '/media/out/clip.mp4', thumbnail_url: '/thumbs/out/clip.mp4', filename: 'clip.mp4', media_type: 'video' });

    it.each(['card', 'compact'])('is its still with a play badge in a %s tile, never a player', (density) => {
      mountTile({ asset: clip(), density });
      expect(target.querySelector('video')).toBeNull();
      expect(target.querySelector('img')?.getAttribute('src')).toBe('/thumbs/out/clip.mp4');
      expect(target.querySelector('[data-testid="play-badge"]')).not.toBeNull();
    });

    it('plays the file itself on the stage', () => {
      mountTile({ asset: clip(), density: 'stage' });
      expect(target.querySelector('video')?.getAttribute('src')).toBe('/media/out/clip.mp4');
      expect(target.querySelector('video')?.getAttribute('poster')).toBe('/thumbs/out/clip.mp4');
      expect(target.querySelector('img')).toBeNull();
      expect(target.querySelector('[data-testid="play-badge"]')).toBeNull();
    });

    it('plays in the app\'s player on the stage: no native controls, muted at the start, and the whole player opted out of the swipe', () => {
      mountTile({ asset: clip(), density: 'stage' });
      const video = target.querySelector('video')!;

      expect(video.controls).toBe(false);
      expect(video.muted).toBe(true);
      expect(target.querySelector('[data-media-player]')?.getAttribute('data-size')).toBe('stage');
      expect(target.querySelector('[data-media-player]')?.hasAttribute('data-own-drag')).toBe(true);
    });

    it('tells the page when the video has its first frame, so the stage can reveal it', () => {
      const onmediaload = vi.fn();
      mountTile({ asset: clip(), density: 'stage', onmediaload });

      target.querySelector('video')!.dispatchEvent(new Event('loadeddata'));

      expect(onmediaload).toHaveBeenCalledOnce();
    });

    it('shows an icon when its still cannot be loaded', () => {
      mountTile({ asset: clip() });
      flushSync(() => target.querySelector('img')!.dispatchEvent(new Event('error')));
      expect(target.querySelector('img')).toBeNull();
      expect(target.querySelector('[data-testid="video-fallback"]')).not.toBeNull();
    });
  });

  it('draws an image from its thumbnail and falls back to the file when that cannot be loaded', () => {
    const asset = makeAsset();
    mountTile({ asset });
    expect(target.querySelector('img')?.getAttribute('src')).toBe(asset.thumbnail_url);

    flushSync(() => target.querySelector('img')!.dispatchEvent(new Event('error')));

    expect(target.querySelector('img')?.getAttribute('src')).toBe(asset.url);
  });

  it('opens the menu on a right-click, but leaves a video its own menu', async () => {
    const asset = makeAsset();
    mountTile({ asset, ondelete: vi.fn() });
    const rightClick = (el: Element) => { const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 40, clientY: 50 }); el.dispatchEvent(event); flushSync(); return event; };
    expect(rightClick(target.querySelector('img')!).defaultPrevented).toBe(true);
    expect(document.querySelector('[role="menu"]')).not.toBeNull();
    await page.unmount();
    document.body.innerHTML = '';

    const video = makeAsset({ id: 'out/clip.mp4', filename: 'clip.mp4', media_type: 'video' });
    mountTile({ asset: video, density: 'stage', ondelete: vi.fn() });
    expect(rightClick(target.querySelector('video')!).defaultPrevented).toBe(false);
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });
});

describe('AssetTile with audio', () => {
  let page: ReturnType<typeof mountComponent>;
  let target: HTMLElement;

  function mountTile(props: Record<string, unknown>): void {
    page = mountComponent(AssetTile, props);
    target = page.target;
  }

  beforeEach(() => {
    audioPeaks.reset();
    mocks.getAudioPeaks.mockReset();
  });

  it('asks the server for the waveform of a take the listing sent without one, and draws it', async () => {
    mocks.getAudioPeaks.mockResolvedValue(Array.from({ length: 100 }, () => 255));
    mountTile({ asset: makeAudioAsset({ peaks: null }) });
    const barHeights = () => Array.from(target.querySelectorAll('.audio-cover .bar')).map((bar) => bar.getAttribute('height'));
    const flat = barHeights();

    await new Promise((resolve) => setTimeout(resolve, 0));
    flushSync();

    expect(mocks.getAudioPeaks).toHaveBeenCalledExactlyOnceWith('out/song.flac');
    expect(barHeights()).not.toEqual(flat);
  });

  it('does not ask when the listing carried the waveform', () => {
    mountTile({ asset: makeAudioAsset() });

    expect(mocks.getAudioPeaks).not.toHaveBeenCalled();
  });

  const words = () => target.querySelector('[data-testid="audio-words"]')?.textContent ?? null;
  const cover = () => target.querySelector<HTMLElement>('[data-testid="audio-cover"]')!;

  it('shows a song\'s style and its lyric lines, never the [section] lines', () => {
    mountTile({ asset: makeAudioAsset() });

    expect(words()).toBe('Dreamy synth-pop, warm analog pads\nStreetlights hum a quiet tune\nHold the night, don\'t let it go');
    expect(words()).not.toContain('[verse]');
    expect(words()).not.toContain('[chorus]');
  });

  it('shows the kind icon, the length and a waveform, and the kind and date in the foot', () => {
    mountTile({ asset: makeSpeechAsset() });

    expect(words()).toBe('Welcome back to the workshop.');
    expect(target.querySelector('[data-testid="audio-length"]')?.textContent).toBe('0:21');
    expect(cover().querySelectorAll('svg rect.bar').length).toBe(100);
    expect(target.querySelector('.asset-tile-foot')?.textContent).toContain('speech · ');
    expect(target.querySelector('img')).toBeNull();
  });

  it('shows only the icon and waveform of a file without settings, whose foot says so', () => {
    mountTile({ asset: makeImportedAudio() });

    expect(cover().dataset.bare).toBe('true');
    expect(words()).toBeNull();
    expect(target.querySelector('.asset-tile-foot')?.textContent).toContain('audio · no settings');
    expect(target.querySelector('[data-testid="audio-length"]')).not.toBeNull();
  });

  it('shows a strip tile as the bare layout with its length, whatever the words', () => {
    mountTile({ asset: makeAudioAsset(), density: 'compact' });

    expect(cover().dataset.bare).toBe('true');
    expect(words()).toBeNull();
    expect(target.querySelector('[data-testid="audio-length"]')?.textContent).toBe('2:00');
    expect(target.querySelector('.asset-tile-foot')).toBeNull();
  });

  it('never plays in a grid or strip, and opens the viewer on a click', () => {
    const onpreview = vi.fn();
    mountTile({ asset: makeAudioAsset(), onpreview });

    expect(target.querySelector('audio')).toBeNull();
    target.querySelector<HTMLButtonElement>('button.asset-tile-media')!.click();
    expect(onpreview).toHaveBeenCalledOnce();
  });

  it('plays in the stage panel, with Reuse and the menu in its corner and no fullscreen button', () => {
    mountTile({ asset: makeAudioAsset(), density: 'stage', onpreview: vi.fn(), onreuse: vi.fn() });

    expect(target.querySelector('audio')?.getAttribute('src')).toBe('/media/out/song.flac');
    expect(target.querySelector('[data-media-player][data-kind="audio"][data-size="stage"]')).not.toBeNull();
    expect(target.querySelector('button[aria-label^="Reuse settings from"]')).not.toBeNull();
    expect(target.querySelector('button[aria-label^="More actions"]')).not.toBeNull();
    expect(target.querySelector('.asset-tile-expand')).toBeNull();
  });

  it('disables Reuse for a file without settings', () => {
    mountTile({ asset: makeImportedAudio(), onreuse: vi.fn() });

    expect(target.querySelector<HTMLButtonElement>('button[aria-label="Reusable settings unavailable"]')?.disabled).toBe(true);
  });
});

describe('AssetTile Move', () => {
  let page: ReturnType<typeof mountComponent>;
  let target: HTMLElement;

  function mountTile(props: Record<string, unknown>): void {
    page = mountComponent(AssetTile, props);
    target = page.target;
  }

  function openMenu(asset: GalleryAsset): void {
    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
  }

  const NIGHT_BUS = { name: 'Night Bus', path: '/out/Night Bus' };
  const ZOO = { name: 'Zoo', path: '/out/Zoo' };
  const moveRows = () => Array.from(document.querySelectorAll<HTMLElement>('[role="menu"] [role="menuitemradio"]'));

  async function chooseMove(asset: GalleryAsset): Promise<void> {
    openMenu(asset);
    document.querySelector<HTMLElement>('[data-action="move"]')!.click();
    await settle();
  }

  beforeEach(() => mocks.listProjects.mockReset().mockResolvedValue([NIGHT_BUS, ZOO]));

  it('lists Move to project in the menu only with an onmove handler', () => {
    const asset = makeAsset();
    mountTile({ asset });
    openMenu(asset);
    expect(menuActions()).not.toContain('move');
    page.unmount();
    mountTile({ asset, onmove: vi.fn() });
    openMenu(asset);
    expect(menuActions().indexOf('move')).toBe(menuActions().indexOf('download') - 1);
  });

  it('opens the project list after the menu closes, with the current location checked and disabled', async () => {
    const asset = makeAsset({ project: 'Night Bus' });
    mountTile({ asset, onmove: vi.fn() });
    await chooseMove(asset);

    expect(document.querySelector('[role="menu"][aria-label="Move to project"]')).not.toBeNull();
    expect(document.querySelectorAll('[role="menu"]')).toHaveLength(1);
    expect(document.querySelector('.action-menu-heading')?.textContent).toBe('Move to');
    const rows = moveRows();
    expect(rows.map((row) => row.textContent?.trim())).toEqual(['No project', 'Night Bus', 'Zoo']);
    const current = rows.find((row) => row.getAttribute('aria-checked') === 'true')!;
    expect(current.textContent?.trim()).toBe('Night Bus');
    expect(current.getAttribute('aria-disabled')).toBe('true');
    expect(target.querySelector('.asset-tile')?.getAttribute('data-menu-open')).toBe('true');
  });

  it('checks No project for a file at the top level', async () => {
    const asset = makeAsset();
    mountTile({ asset, onmove: vi.fn() });
    await chooseMove(asset);

    const current = moveRows().find((row) => row.getAttribute('aria-checked') === 'true')!;
    expect(current.textContent?.trim()).toBe('No project');
  });

  it('moves the asset to the chosen project, or out of every project', async () => {
    const onmove = vi.fn();
    const asset = makeAsset({ project: 'Night Bus' });
    mountTile({ asset, onmove });
    await chooseMove(asset);

    moveRows().find((row) => row.textContent?.trim() === 'Zoo')!.click();
    await settle();
    expect(onmove).toHaveBeenCalledWith(asset, 'Zoo');
    expect(document.querySelector('[role="menu"]')).toBeNull();

    await chooseMove(asset);
    moveRows().find((row) => row.textContent?.trim() === 'No project')!.click();
    await settle();
    expect(onmove).toHaveBeenLastCalledWith(asset, null);
  });

  it('creates a project and moves the file into it from New project', async () => {
    const onmove = vi.fn();
    mocks.createProject.mockResolvedValue({ name: 'Bridges', path: '/out/Bridges' });
    const asset = makeAsset();
    mountTile({ asset, onmove });
    await chooseMove(asset);

    document.querySelector<HTMLElement>('[data-action="project-new"]')!.click();
    await settle();
    const field = document.querySelector<HTMLInputElement>('#project-name')!;
    field.value = 'Bridges';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(document.querySelector('[data-action="create-project"]')?.textContent).toContain('Create and move');
    document.querySelector<HTMLButtonElement>('[data-action="create-project"]')!.click();
    await settle();

    expect(mocks.createProject).toHaveBeenCalledWith('Bridges');
    expect(onmove).toHaveBeenCalledWith(asset, 'Bridges');
  });
});

describe('AssetTile Select mode', () => {
  let page: ReturnType<typeof mountComponent>;
  let target: HTMLElement;
  const asset = makeAsset();

  function mountTile(props: Record<string, unknown>): void {
    page = mountComponent(AssetTile, { asset, ...props });
    target = page.target;
  }

  const mediaButton = (name: string) => target.querySelector(`button[aria-label="${name} ${asset.filename}"]`) as HTMLButtonElement | null;
  const checkbox = () => target.querySelector(`input[type="checkbox"][aria-label="Select ${asset.filename}"]`) as HTMLInputElement;

  async function openMenu(): Promise<string[]> {
    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    await settle();
    return menuActions();
  }

  describe('outside the mode', () => {
    it('names the media View, without a pressed state, and a click opens the viewer', () => {
      const onpreview = vi.fn();
      const onmediaclick = vi.fn();
      const onselect = vi.fn();
      mountTile({ onpreview, onmediaclick, onselect });
      const media = mediaButton('View')!;
      expect(media.hasAttribute('aria-pressed')).toBe(false);
      media.click();
      expect(onpreview).toHaveBeenCalledWith(asset, media);
      expect(onmediaclick).not.toHaveBeenCalled();
      expect(onselect).not.toHaveBeenCalled();
    });

    it('keeps the checkbox named Select, and ticking it starts the mode with this tile', () => {
      const onselect = vi.fn();
      const onstartselect = vi.fn();
      mountTile({ onselect, onstartselect });
      checkbox().click();
      expect(onstartselect).toHaveBeenCalledWith(asset);
      expect(onselect).not.toHaveBeenCalled();
    });

    it('ticks through onselect when nothing can start the mode', () => {
      const onselect = vi.fn();
      mountTile({ onselect });
      checkbox().click();
      expect(onselect).toHaveBeenCalledWith(asset, true);
    });

    it('leads the menu with Select, which starts the mode and returns focus to the media', async () => {
      const onstartselect = vi.fn();
      mountTile({ onstartselect, onpreview: vi.fn() });
      expect((await openMenu())[0]).toBe('select');
      document.querySelector<HTMLElement>('[data-action="select"]')!.click();
      await settle();
      expect(onstartselect).toHaveBeenCalledWith(asset);
      expect(document.activeElement).toBe(mediaButton('View'));
    });

    it('has no lead on a tile that cannot start the mode', async () => {
      mountTile({ onpreview: vi.fn(), onreuse: vi.fn() });
      expect((await openMenu())[0]).toBe('reuse');
    });
  });

  describe('in the mode', () => {
    it('names the media Select with the selected state as aria-pressed', async () => {
      mountTile({ selectMode: true, selected: false, onselect: vi.fn() });
      expect(mediaButton('Select')?.getAttribute('aria-pressed')).toBe('false');
      expect(mediaButton('View')).toBeNull();
      await page.unmount();
      mountTile({ selectMode: true, selected: true, onselect: vi.fn() });
      expect(mediaButton('Select')?.getAttribute('aria-pressed')).toBe('true');
    });

    it('hands a click on the media to onmediaclick with its event, and does not open the viewer', () => {
      const onpreview = vi.fn();
      const onmediaclick = vi.fn();
      mountTile({ selectMode: true, onpreview, onmediaclick, onselect: vi.fn() });
      mediaButton('Select')!.dispatchEvent(new MouseEvent('click', { shiftKey: true, bubbles: true }));
      expect(onmediaclick).toHaveBeenCalledOnce();
      expect(onmediaclick.mock.calls[0][0]).toBe(asset);
      expect((onmediaclick.mock.calls[0][1] as MouseEvent).shiftKey).toBe(true);
      expect(onpreview).not.toHaveBeenCalled();
    });

    it('opens the viewer on Enter, without a toggle', () => {
      const onpreview = vi.fn();
      const onmediaclick = vi.fn();
      mountTile({ selectMode: true, onpreview, onmediaclick, onselect: vi.fn() });
      const media = mediaButton('Select')!;
      const event = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
      media.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(true);
      expect(onpreview).toHaveBeenCalledWith(asset, media);
      expect(onmediaclick).not.toHaveBeenCalled();
    });

    it('prevents a Shift mousedown only in the mode, so no page text gets selected', async () => {
      mountTile({ onpreview: vi.fn() });
      const outside = new MouseEvent('mousedown', { shiftKey: true, bubbles: true, cancelable: true });
      mediaButton('View')!.dispatchEvent(outside);
      expect(outside.defaultPrevented).toBe(false);
      await page.unmount();
      mountTile({ selectMode: true, onselect: vi.fn() });
      const shift = new MouseEvent('mousedown', { shiftKey: true, bubbles: true, cancelable: true });
      mediaButton('Select')!.dispatchEvent(shift);
      expect(shift.defaultPrevented).toBe(true);
      const plain = new MouseEvent('mousedown', { bubbles: true, cancelable: true });
      mediaButton('Select')!.dispatchEvent(plain);
      expect(plain.defaultPrevented).toBe(false);
    });

    it('toggles through onselect when the checkbox is ticked, never starting the mode again', () => {
      const onselect = vi.fn();
      const onstartselect = vi.fn();
      mountTile({ selectMode: true, onselect, onstartselect });
      checkbox().click();
      expect(onselect).toHaveBeenCalledWith(asset, true);
      expect(onstartselect).not.toHaveBeenCalled();
    });

    it('leads the menu with View, which opens the viewer and leaves the selection alone', async () => {
      const onpreview = vi.fn();
      const onstartselect = vi.fn();
      const onselect = vi.fn();
      mountTile({ selectMode: true, selected: true, onpreview, onstartselect, onselect });
      expect((await openMenu())[0]).toBe('view');
      document.querySelector<HTMLElement>('[data-action="view"]')!.click();
      await settle();
      expect(onpreview).toHaveBeenCalledOnce();
      expect(onstartselect).not.toHaveBeenCalled();
      expect(onselect).not.toHaveBeenCalled();
    });

    it('keeps the expand button opening the viewer', () => {
      const onpreview = vi.fn();
      mountTile({ selectMode: true, onpreview, onselect: vi.fn() });
      mediaButtonFullscreen().click();
      expect(onpreview).toHaveBeenCalledOnce();
    });
  });

  function mediaButtonFullscreen(): HTMLButtonElement {
    return target.querySelector(`button[aria-label="View ${asset.filename} fullscreen"]`) as HTMLButtonElement;
  }
});

describe('AssetTile star', () => {
  let page: ReturnType<typeof mountComponent>;
  let target: HTMLElement;
  const asset = makeAsset();
  const starButton = () => target.querySelector(`button[aria-label="Star ${asset.filename}"]`) as HTMLButtonElement | null;

  function mountTile(props: Record<string, unknown>): void {
    page = mountComponent(AssetTile, { asset, ...props });
    target = page.target;
  }

  describe('on a grid tile', () => {
    it('is a toggle named Star <filename>, pressed while starred, and absent without onstar', async () => {
      mountTile({ onstar: vi.fn() });
      expect(starButton()?.getAttribute('aria-pressed')).toBe('false');
      await page.unmount();
      mountTile({ asset: makeAsset({ starred: true }), onstar: vi.fn() });
      expect(starButton()?.getAttribute('aria-pressed')).toBe('true');
      await page.unmount();
      mountTile({});
      expect(starButton()).toBeNull();
    });

    it('stars an unstarred file and unstars a starred one', async () => {
      const onstar = vi.fn();
      mountTile({ onstar });
      starButton()!.click();
      expect(onstar).toHaveBeenCalledWith(asset, true);
      await page.unmount();
      const starred = makeAsset({ starred: true });
      mountTile({ asset: starred, onstar });
      starButton()!.click();
      expect(onstar).toHaveBeenLastCalledWith(starred, false);
    });

    it('never touches the selection or opens the viewer, in Select mode too', () => {
      const onstar = vi.fn();
      const onselect = vi.fn();
      const onmediaclick = vi.fn();
      const onpreview = vi.fn();
      const onstartselect = vi.fn();
      mountTile({ onstar, onselect, onmediaclick, onpreview, onstartselect, selectMode: true, selected: true });
      starButton()!.click();
      expect(onstar).toHaveBeenCalledOnce();
      expect(onselect).not.toHaveBeenCalled();
      expect(onmediaclick).not.toHaveBeenCalled();
      expect(onstartselect).not.toHaveBeenCalled();
      expect(onpreview).not.toHaveBeenCalled();
    });

    it('keeps Space and X from reaching the grid, which selects the focused tile with them', () => {
      mountTile({ onstar: vi.fn(), onpreview: vi.fn() });
      const reached = vi.fn();
      target.addEventListener('keydown', reached);
      for (const key of [' ', 'x', 'X']) starButton()!.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
      expect(reached).not.toHaveBeenCalled();
      starButton()!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
      expect(reached).toHaveBeenCalledOnce();
    });
  });

  describe('on a strip tile', () => {
    it('shows a mark, with no button, for a starred file and nothing for an unstarred one', async () => {
      mountTile({ asset: makeAsset({ starred: true }), density: 'compact', onstar: vi.fn() });
      expect(target.querySelector('[data-testid="starred-mark"]')).not.toBeNull();
      expect(target.querySelector('[data-testid="starred-mark"]')?.closest('button')).toBeNull();
      expect(starButton()).toBeNull();
      await page.unmount();
      mountTile({ density: 'compact', onstar: vi.fn() });
      expect(target.querySelector('[data-testid="starred-mark"]')).toBeNull();
    });
  });

  describe('on the stage', () => {
    it('is a button, and no mark', () => {
      const onstar = vi.fn();
      mountTile({ density: 'stage', onstar, onpreview: vi.fn() });
      starButton()!.click();
      expect(onstar).toHaveBeenCalledWith(asset, true);
      expect(target.querySelector('[data-testid="starred-mark"]')).toBeNull();
    });
  });
});
