import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import AssetViewer from './AssetViewer.svelte';
import { escapeLayerOpen } from '$lib/actions/escapeStack';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { requestConfirm } from './confirm.svelte';

// The touch layer (neighbour panes) follows the pointer; a test switches it on without a real media query.
const touch = vi.hoisted(() => ({ coarse: false, compact: false }));
vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  return {
    ...actual,
    layout: new Proxy(actual.layout, {
      get: (target, key) => (key === 'coarse' && touch.coarse) || (key === 'compact' && touch.compact) ? true : Reflect.get(target, key),
    }),
  };
});
const projectApi = vi.hoisted(() => ({ listProjects: vi.fn(), createProject: vi.fn() }));
vi.mock('$lib/api/projects', () => projectApi);

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/asset-a.png',
    url: '/media/out/asset-a.png',
    thumbnail_url: '/media/out/asset-a.png',
    filename: 'asset-a.png',
    created_at: '2026-04-30T12:00:00Z',
    project: null,
    workflow: 'txt2img',
    prompt: 'Test asset',
    model: 'zit',
    media_type: 'image',
    file_path: '/outputs/out/asset-a.png',
    seed: 1234,
    has_reusable_config: true,
    reuse_workspace_url: '#/workspace?workflow=txt2img',
    ...overrides,
  };
}

const assetB = makeAsset({ id: 'out/asset-b.png', url: '/media/out/asset-b.png', filename: 'asset-b.png', prompt: 'Second asset' });

describe('AssetViewer', () => {
  let page: ReturnType<typeof mountComponent>;

  beforeEach(() => localStorage.clear());

  function mountViewer(props: Record<string, unknown>): void {
    const assets = (props.assets as GalleryAsset[] | undefined) ?? [makeAsset(), assetB];
    page = mountComponent(AssetViewer, { assets, currentId: assets[0]?.id ?? null, open: true, onclose: vi.fn(), onnavigate: vi.fn(), ...props });
  }

  function button(label: string): HTMLButtonElement | null {
    return document.querySelector(`button[aria-label="${label}"]`);
  }

  it('disables previous on the first asset and next on the last', async () => {
    mountViewer({});
    expect(button('Previous asset')?.disabled).toBe(true);
    expect(button('Next asset')?.disabled).toBe(false);

    await page.unmount();
    mountViewer({ currentId: assetB.id });
    expect(button('Previous asset')?.disabled).toBe(false);
    expect(button('Next asset')?.disabled).toBe(true);
  });

  it('navigates with the arrow keys and closes with Escape', () => {
    const onnavigate = vi.fn();
    const onclose = vi.fn();
    mountViewer({ onnavigate, onclose });

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(onnavigate).toHaveBeenCalledWith(1);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(onclose).toHaveBeenCalled();
  });

  it('toggles the details panel with I and remembers the choice', async () => {
    mountViewer({});
    expect(document.querySelector('#asset-viewer-details')).toBeNull();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'i', bubbles: true }));
    flushSync();
    const details = document.querySelector('#asset-viewer-details');
    expect(details?.textContent).toContain('Test asset');
    expect(details?.textContent).toContain('1234');

    await page.unmount();
    mountViewer({});
    expect(document.querySelector('#asset-viewer-details')).not.toBeNull();
  });

  describe('scene in the details', () => {
    function openDetails(asset: GalleryAsset): HTMLElement {
      mountViewer({ assets: [asset] });
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'i', bubbles: true }));
      flushSync();
      return document.querySelector('#asset-viewer-details') as HTMLElement;
    }

    const video = { media_type: 'video', workflow: 'txt2vid', filename: 'clip.mp4' } as const;

    it('shows the scene beside a prompt that differs from it', () => {
      const details = openDetails(makeAsset({ ...video, scene: 'A castle at dusk.', prompt: 'A knight rides in.' }));
      expect(details.querySelector('[data-testid="viewer-scene"]')?.textContent).toContain('A castle at dusk.');
      expect(details.textContent).toContain('A knight rides in.');
    });

    it('shows a scene-only video once', () => {
      // The gallery reports the scene as the prompt of a video made without one.
      const details = openDetails(makeAsset({ ...video, scene: 'A castle at dusk.', prompt: 'A castle at dusk.' }));
      expect(details.querySelector('[data-testid="viewer-scene"]')).toBeNull();
      expect(details.textContent?.match(/A castle at dusk\./g)).toHaveLength(1);
    });

    it('shows how an enhanced prompt was made and the own prompt it came from', () => {
      const enhance = { summary: 'Fast · Cinematic, Longer', tier: 'fast', model: 'org/fast', format: 'text', settings: { style: 'cinematic' }, prompt_input: 'a knight', scene_input: null } as const;
      const details = openDetails(makeAsset({ prompt: 'A knight rides in at dusk.', enhance }));
      const section = details.querySelector('[data-testid="viewer-enhance"]');
      expect(section?.textContent).toContain('Fast · Cinematic, Longer');
      expect(section?.textContent).toContain('a knight');
    });

    it('shows no enhance section for an asset that was not enhanced', () => {
      const details = openDetails(makeAsset({ prompt: 'A fox.' }));
      expect(details.querySelector('[data-testid="viewer-enhance"]')).toBeNull();
    });

    it('shows no scene section for an asset without one', () => {
      const details = openDetails(makeAsset({ prompt: 'A fox.' }));
      expect(details.querySelector('[data-testid="viewer-scene"]')).toBeNull();
    });
  });

  it('runs reuse, reference and delete on the shown asset', () => {
    const onreuse = vi.fn();
    const onreference = vi.fn();
    const ondelete = vi.fn();
    mountViewer({ currentId: assetB.id, onreuse, onreference, ondelete });

    (document.querySelector('[data-action="reuse"]') as HTMLButtonElement).click();
    expect(onreuse).toHaveBeenCalledWith(assetB);

    (document.querySelector('[data-action="reference"]') as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    expect(onreference).toHaveBeenCalledWith(assetB, 'image');

    (document.querySelector('[data-action="delete"]') as HTMLButtonElement).click();
    expect(ondelete).toHaveBeenCalledWith(assetB);
  });

  it('lists the four Use as targets with their hints, and disables one with its reason', () => {
    const onreference = vi.fn();
    mountViewer({
      onreference,
      useAs: {
        reference: { disabled: true, hint: 'no room' },
        last: { disabled: true, hint: 'LTX 2.3 has no last frame' },
      },
    });

    (document.querySelector('[data-action="reference"]') as HTMLButtonElement).click();
    flushSync();
    const items = [...document.querySelectorAll<HTMLElement>('[role="menu"] [data-action^="reference-"]')];
    expect(items.map((item) => item.getAttribute('data-action'))).toEqual(['reference-image', 'reference-reference', 'reference-video', 'reference-last-frame']);
    const reference = items[1];
    expect(reference.getAttribute('aria-disabled')).toBe('true');
    expect(reference.getAttribute('title')).toBe('no room');
    expect(reference.textContent).toContain('no room');
    expect(items[3].getAttribute('aria-disabled')).toBe('true');
    expect(items[0].hasAttribute('aria-disabled')).toBe(false);

    reference.click();
    flushSync();
    expect(onreference).not.toHaveBeenCalled();
  });

  it('shows the next image number on an open Reference target and picks it', () => {
    const onreference = vi.fn();
    mountViewer({ onreference, useAs: { reference: { disabled: false, hint: 'next slot' } } });

    (document.querySelector('[data-action="reference"]') as HTMLButtonElement).click();
    flushSync();
    const reference = document.querySelector<HTMLElement>('[role="menu"] [data-action="reference-reference"]')!;
    expect(reference.textContent).toContain('next slot');
    reference.click();
    expect(onreference).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/asset-a.png' }), 'reference');
  });

  it('stays on the nearest asset when the open one leaves the list and closes when it empties', async () => {
    const onclose = vi.fn();
    const onnavigate = vi.fn();
    const props = reactiveProps({ assets: [makeAsset(), assetB], currentId: assetB.id, open: true, onclose, onnavigate });
    page = mountComponent(AssetViewer, props);
    expect(document.querySelector('#asset-viewer-title')?.textContent).toBe('asset-b.png');

    props.assets = [makeAsset()];
    flushSync();
    expect(document.querySelector('#asset-viewer-title')?.textContent).toBe('asset-a.png');
    expect(onnavigate).toHaveBeenLastCalledWith(0);

    props.assets = [];
    flushSync();
    expect(onclose).toHaveBeenCalled();
    expect(document.querySelector('[data-testid="asset-viewer"]')).toBeNull();
  });

  it('keeps the open asset when others land before or after it', () => {
    const props = reactiveProps({ assets: [makeAsset(), assetB], currentId: assetB.id, open: true, onclose: vi.fn(), onnavigate: vi.fn() });
    page = mountComponent(AssetViewer, props);
    const title = () => document.querySelector('#asset-viewer-title')?.textContent;
    const counter = () => document.querySelector('.viewer-title small')?.textContent;
    expect(counter()).toContain('2 of 2');

    props.assets = [makeAsset({ id: 'out/new.png', filename: 'new.png' }), makeAsset(), assetB];
    flushSync();
    expect(title()).toBe('asset-b.png');
    expect(counter()).toContain('3 of 3');

    props.assets = [...props.assets as GalleryAsset[], makeAsset({ id: 'out/older.png', filename: 'older.png' })];
    flushSync();
    expect(title()).toBe('asset-b.png');
    expect(counter()).toContain('3 of 4');
  });

  it('shows a video with its still as the poster until it plays', () => {
    const clip = makeAsset({ id: 'out/clip.mp4', url: '/media/out/clip.mp4', thumbnail_url: '/thumbs/out/clip.mp4', filename: 'clip.mp4', media_type: 'video' });
    mountViewer({ assets: [clip], currentId: clip.id });

    const video = document.querySelector('video');
    expect(video?.getAttribute('src')).toBe('/media/out/clip.mp4');
    expect(video?.getAttribute('poster')).toBe('/thumbs/out/clip.mp4');
  });

  it('draws a video as its still in the strip and beside the shown asset, and an image beside it in full', () => {
    touch.coarse = true;
    try {
      const clip = makeAsset({ id: 'out/clip.mp4', url: '/media/out/clip.mp4', thumbnail_url: '/thumbs/out/clip.mp4', filename: 'clip.mp4', media_type: 'video' });
      const photo = makeAsset({ id: 'out/photo.png', url: '/media/out/photo.png', thumbnail_url: '/thumbs/out/photo.png', filename: 'photo.png' });
      mountViewer({ assets: [clip, photo, makeAsset({ id: 'out/c.png', url: '/media/out/c.png', filename: 'c.png' })], currentId: photo.id });

      expect(document.querySelectorAll('.viewer-film video')).toHaveLength(0);
      expect(document.querySelector('[data-film-index="0"] img')?.getAttribute('src')).toBe('/thumbs/out/clip.mp4');
      expect(document.querySelector('[data-film-index="0"] [data-testid="film-play-badge"]')).not.toBeNull();
      expect(document.querySelectorAll('[data-testid="film-play-badge"]')).toHaveLength(1);
      expect(document.querySelector('.viewer-pane-prev .viewer-neighbour')?.getAttribute('src')).toBe('/thumbs/out/clip.mp4');
      expect(document.querySelector('.viewer-pane-next .viewer-neighbour')?.getAttribute('src')).toBe('/media/out/c.png');
    } finally {
      touch.coarse = false;
    }
  });

  it('lists every asset in the thumbnail strip and asks for more near the end', async () => {
    const many = Array.from({ length: 30 }, (_, i) => makeAsset({ id: `out/${i}.png`, url: `/media/out/${i}.png`, filename: `${i}.png` }));
    const onnearend = vi.fn();
    mountViewer({ assets: many, currentId: many[10].id, onnearend });
    expect(document.querySelectorAll('[data-film-index]')).toHaveLength(30);
    expect(onnearend).not.toHaveBeenCalled();

    await page.unmount();
    mountViewer({ assets: many, currentId: many[28].id, onnearend });
    expect(onnearend).toHaveBeenCalled();
  });

  function press(key: string, init: KeyboardEventInit = {}): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
    document.dispatchEvent(event);
    flushSync();
    return event;
  }

  it.each(['Delete', 'Backspace'])('deletes the shown asset with %s', (key) => {
    const ondelete = vi.fn();
    mountViewer({ currentId: assetB.id, ondelete });
    expect(press(key).defaultPrevented).toBe(true);
    expect(ondelete).toHaveBeenCalledWith(assetB, { confirm: true });
  });

  it('ignores a held Delete key', () => {
    const ondelete = vi.fn();
    mountViewer({ ondelete });
    expect(press('Delete', { repeat: true }).defaultPrevented).toBe(true);
    expect(ondelete).not.toHaveBeenCalled();
  });

  it('keeps a confirmation dialog on top: Escape closes only the dialog and arrows do not page', async () => {
    const onnavigate = vi.fn();
    const onclose = vi.fn();
    let answer: Promise<boolean> | null = null;
    mountViewer({ onnavigate, onclose, ondelete: (asset: GalleryAsset) => { answer = requestConfirm({ question: `Delete "${asset.filename}"?`, confirmLabel: 'Delete' }); } });
    press('Delete');
    await new Promise((resolve) => setTimeout(resolve, 0));
    flushSync();
    expect(document.querySelector('[role="alertdialog"]')).not.toBeNull();

    press('ArrowRight');
    expect(onnavigate).not.toHaveBeenCalled();
    press('Escape');
    await expect(answer!).resolves.toBe(false);
    expect(onclose).not.toHaveBeenCalled();
    expect(document.querySelector('[data-testid="asset-viewer"]')).not.toBeNull();
  });

  it('deletes without asking on Shift+Delete', () => {
    const ondelete = vi.fn();
    mountViewer({ ondelete });
    press('Delete', { shiftKey: true });
    expect(ondelete).toHaveBeenCalledWith(makeAsset(), { confirm: false });
  });

  it('jumps to the first and last asset with Home and End', () => {
    const many = Array.from({ length: 5 }, (_, i) => makeAsset({ id: `out/${i}.png`, filename: `${i}.png` }));
    const onnavigate = vi.fn();
    mountViewer({ assets: many, currentId: many[2].id, onnavigate });
    press('End');
    expect(onnavigate).toHaveBeenLastCalledWith(4);
    press('Home');
    expect(onnavigate).toHaveBeenLastCalledWith(0);
  });

  it('opens the reference menu with E on an image, leaving other keys to it while open', () => {
    const onreference = vi.fn();
    const ondelete = vi.fn();
    const onextend = vi.fn();
    mountViewer({ onreference, ondelete, onextend });
    press('e');
    expect(document.querySelector('[role="menu"]')).not.toBeNull();
    expect(onextend).not.toHaveBeenCalled();
    press('Delete');
    expect(ondelete).not.toHaveBeenCalled();
  });

  describe('extend', () => {
    const clip = (overrides: Partial<GalleryAsset> = {}): GalleryAsset =>
      makeAsset({ id: 'out/clip.mp4', url: '/media/out/clip.mp4', filename: 'clip.mp4', media_type: 'video', extend: { available: true, reason: null }, ...overrides });

    function extendButton(): HTMLButtonElement | null {
      return document.querySelector('[data-action="extend"]');
    }

    it('shows an Extend button for a video and runs it on the shown asset', () => {
      const onextend = vi.fn();
      const asset = clip();
      mountViewer({ assets: [asset], onextend });
      expect(extendButton()?.disabled).toBe(false);
      extendButton()!.click();
      expect(onextend).toHaveBeenCalledWith(asset);
    });

    it('has no Extend button for an image, without a handler, or when extending is unavailable here', async () => {
      mountViewer({ assets: [makeAsset({ extend: { available: true, reason: null } })], onextend: vi.fn() });
      expect(extendButton()).toBeNull();

      await page.unmount();
      mountViewer({ assets: [clip()] });
      expect(extendButton()).toBeNull();

      await page.unmount();
      mountViewer({ assets: [clip({ extend: null })], onextend: vi.fn() });
      expect(extendButton()).toBeNull();
    });

    it('disables the button with the reason as its tooltip when the video cannot be extended', () => {
      const onextend = vi.fn();
      mountViewer({ assets: [clip({ extend: { available: false, reason: 'The file is missing.' } })], onextend });
      expect(extendButton()?.disabled).toBe(true);
      expect(extendButton()?.getAttribute('title')).toBe('The file is missing.');
      extendButton()!.click();
      expect(onextend).not.toHaveBeenCalled();
    });

    it('extends with E on a video instead of opening the reference menu', () => {
      const onextend = vi.fn();
      const asset = clip();
      mountViewer({ assets: [asset], onextend, onreference: vi.fn() });
      expect(press('e').defaultPrevented).toBe(true);
      expect(onextend).toHaveBeenCalledWith(asset);
      expect(document.querySelector('[role="menu"]')).toBeNull();
    });

    it('says why with E when the video cannot be extended', () => {
      const onextend = vi.fn();
      mountViewer({ assets: [clip({ extend: { available: false, reason: "Upscaled videos can't be extended yet." } })], onextend });
      press('e');
      expect(onextend).not.toHaveBeenCalled();
      expect(toasts.map((toast) => [toast.type, toast.message])).toContainEqual(['warning', "Upscaled videos can't be extended yet."]);
    });

    it('leaves E alone for a video that cannot be extended on this machine', () => {
      const onextend = vi.fn();
      mountViewer({ assets: [clip({ extend: null })], onextend });
      expect(press('e').defaultPrevented).toBe(false);
      expect(onextend).not.toHaveBeenCalled();
    });

  });

  it('copies the prompt with C', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    mountViewer({});
    press('c');
    expect(writeText).toHaveBeenCalledWith('Test asset');
  });

  it('ignores the delete key while that asset is already being deleted', () => {
    const ondelete = vi.fn();
    mountViewer({ ondelete, deletingIds: new Set([makeAsset().id]) });
    expect(press('Delete').defaultPrevented).toBe(false);
    expect(ondelete).not.toHaveBeenCalled();
  });

  it('reuses settings with R only when the asset has reusable settings', async () => {
    const onreuse = vi.fn();
    mountViewer({ onreuse });
    press('r');
    expect(onreuse).toHaveBeenCalledWith(makeAsset());

    await page.unmount();
    onreuse.mockClear();
    mountViewer({ onreuse, assets: [makeAsset({ has_reusable_config: false })] });
    expect(press('R').defaultPrevented).toBe(false);
    expect(onreuse).not.toHaveBeenCalled();
  });

  it('downloads the shown asset with D', () => {
    mountViewer({});
    const link = document.querySelector<HTMLAnchorElement>('[data-action="download"]')!;
    const click = vi.spyOn(link, 'click').mockImplementation(() => {});
    press('d');
    expect(click).toHaveBeenCalled();
  });

  it('toggles the shortcut list with ? and closes it with Escape before the viewer', () => {
    const onclose = vi.fn();
    mountViewer({ onclose });
    press('?', { shiftKey: true });
    expect(document.querySelector('#asset-viewer-shortcuts')).not.toBeNull();

    press('Escape');
    expect(document.querySelector('#asset-viewer-shortcuts')).toBeNull();
    expect(onclose).not.toHaveBeenCalled();

    press('Escape');
    expect(onclose).toHaveBeenCalled();
  });

  it('takes the shortcut list down with it when the viewer closes, leaving no layer behind', () => {
    const props = reactiveProps({ assets: [makeAsset(), assetB], currentId: 'out/asset-a.png', open: true, onclose: vi.fn(), onnavigate: vi.fn() });
    page = mountComponent(AssetViewer, props);
    press('?', { shiftKey: true });
    expect(document.querySelector('#asset-viewer-shortcuts')).not.toBeNull();
    expect(escapeLayerOpen()).toBe(true);

    props.open = false;
    flushSync();
    expect(escapeLayerOpen()).toBe(false);

    props.open = true;
    flushSync();
    expect(document.querySelector('#asset-viewer-shortcuts')).toBeNull();
  });

  it('leaves the delete key alone while typing in a field', () => {
    const ondelete = vi.fn();
    mountViewer({ ondelete });
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Backspace', bubbles: true }));
    expect(ondelete).not.toHaveBeenCalled();
  });

  it('leaves modified shortcuts such as Ctrl+I to the browser', () => {
    mountViewer({});
    const event = new KeyboardEvent('keydown', { key: 'i', ctrlKey: true, bubbles: true, cancelable: true });
    document.dispatchEvent(event);
    flushSync();
    expect(event.defaultPrevented).toBe(false);
    expect(document.querySelector('#asset-viewer-details')).toBeNull();
  });
  const upscale = {
    factors: [
      { factor: 2 as const, width: 1664, height: 2432, allowed: true, reason: null },
      { factor: 4 as const, width: 3328, height: 4864, allowed: false, reason: 'Over the 16-megapixel upscale limit.' },
    ],
    seedvr2: null,
  };
  const seedvr2Upscale = {
    ...upscale,
    seedvr2: {
      options: [
        { factor: 2 as const, width: 1664, height: 2432, allowed: true as const, reason: null, warning: null },
        { factor: 4 as const, width: 3328, height: 4864, allowed: true as const, reason: null, warning: 'May need more memory than this Mac has.' },
      ],
      model: 'SeedVR2 7B q8',
      downloaded: true,
      download_size_label: null,
    },
  };

  function upscaleItems(): HTMLElement[] {
    return Array.from(document.querySelectorAll<HTMLElement>('[role="menu"][aria-label="Upscale"] [role="menuitem"]'));
  }

  it('shows the upscale menu with each factor and size, disabling one over a limit', () => {
    const onupscale = vi.fn();
    mountViewer({ assets: [makeAsset({ upscale })], onupscale });

    flushSync(() => document.querySelector<HTMLButtonElement>('[data-action="upscale"]')!.click());

    const items = upscaleItems();
    expect(items.map((item) => item.textContent?.trim())).toEqual(['2× creative → 1664×2432', '4× creative → 3328×4864']);
    expect(items[1].getAttribute('aria-disabled')).toBe('true');
    expect(items[1].getAttribute('title')).toBe('Over the 16-megapixel upscale limit.');
    flushSync(() => items[0].click());
    expect(onupscale).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/asset-a.png' }), 2, 'creative');
  });

  const videoUpscale = (warning: string | null = null) => ({
    factors: [],
    seedvr2: { options: [{ factor: 2 as const, width: 1536, height: 896, allowed: true as const, reason: null, warning }], model: 'SeedVR2 7B fp8', downloaded: true, download_size_label: null },
  });

  it('hides the upscale menu without a handler, for a video with only creative options, and without options', async () => {
    const cases: Array<[Partial<GalleryAsset>, boolean]> = [
      [{ upscale }, false],
      [{ media_type: 'video', upscale }, true],
      [{ media_type: 'video', upscale: null }, true],
    ];
    for (const [overrides, withHandler] of cases) {
      mountViewer({ assets: [makeAsset(overrides)], onupscale: withHandler ? vi.fn() : undefined });
      expect(document.querySelector('[data-action="upscale"]')).toBeNull();
      await page.unmount();
    }
  });

  it('upscales a video with the SeedVR2 2x item from the menu', () => {
    const onupscale = vi.fn();
    mountViewer({ assets: [makeAsset({ media_type: 'video', upscale: videoUpscale() })], onupscale });
    flushSync(() => document.querySelector<HTMLButtonElement>('[data-action="upscale"]')!.click());
    flushSync(() => upscaleItems()[0].click());
    expect(onupscale).toHaveBeenCalledWith(expect.objectContaining({ media_type: 'video' }), 2, 'seedvr2');
  });

  const ltxVideoUpscale = (warning: string | null = null) => ({
    factors: [],
    seedvr2: null,
    ltx: { options: [{ factor: 2 as const, width: 1536, height: 896, allowed: true as const, reason: null, warning }], model: 'LTX-2.3 4-bit', downloaded: true, download_size_label: null },
  });

  it('upscales a video on macOS with the LTX 2x item from the menu', () => {
    const onupscale = vi.fn();
    mountViewer({ assets: [makeAsset({ media_type: 'video', upscale: ltxVideoUpscale() })], onupscale });
    flushSync(() => document.querySelector<HTMLButtonElement>('[data-action="upscale"]')!.click());
    expect(upscaleItems().map((item) => item.textContent?.trim())).toEqual(['2× → 1536×896']);
    flushSync(() => upscaleItems()[0].click());
    expect(onupscale).toHaveBeenCalledWith(expect.objectContaining({ media_type: 'video' }), 2, 'ltx');
  });

  const noCreative = (seedvr2: typeof seedvr2Upscale.seedvr2 | null) => ({
    factors: upscale.factors.map((option) => ({ ...option, allowed: false, reason: 'This model cannot upscale.' })),
    seedvr2,
  });

  it.each([
    { name: 'creative: 2x runs, a disallowed 4x is refused', upscale, video: false, keys: ['2', '4'], calls: [[2, 'creative']], warnings: ['Over the 16-megapixel upscale limit.'] },
    {
      name: 'SeedVR2: 2x and 4x run, the high-memory size warns first',
      upscale: seedvr2Upscale, video: false, keys: ['2', '4'], calls: [[2, 'seedvr2'], [4, 'seedvr2']], warnings: ['May need more memory than this Mac has.'],
    },
    { name: 'video: only 2x runs, with its warning', upscale: videoUpscale('May need more memory.'), video: true, keys: ['4', '2'], calls: [[2, 'seedvr2']], warnings: ['May need more memory.'] },
    { name: 'LTX video on macOS: only 2x runs, with its warning', upscale: ltxVideoUpscale('May need more memory.'), video: true, keys: ['4', '2'], calls: [[2, 'ltx']], warnings: ['May need more memory.'] },
    { name: 'SeedVR2 runs when every creative factor is disabled', upscale: noCreative(seedvr2Upscale.seedvr2), video: false, keys: ['2'], calls: [[2, 'seedvr2']], warnings: [] },
    { name: 'disabled creative factors alone explain instead of upscaling', upscale: noCreative(null), video: false, keys: ['2'], calls: [], warnings: ['This model cannot upscale.'] },
  ])('runs the X then digit chord, $name', ({ upscale: options, video, keys, calls, warnings }) => {
    const onupscale = vi.fn();
    toasts.length = 0;
    mountViewer({ assets: [makeAsset({ upscale: options, ...(video ? { media_type: 'video' as const } : {}) })], onupscale });

    for (const key of keys) {
      press('x');
      press(key);
    }

    expect(onupscale.mock.calls.map(([, factor, method]) => [factor, method])).toEqual(calls);
    expect(toasts.map((toast) => toast.message)).toEqual(warnings);
  });

  it('starts the upscale chord only from a plain X', () => {
    const onupscale = vi.fn();
    mountViewer({ assets: [makeAsset({ upscale: seedvr2Upscale })], onupscale });

    press('2');
    press('X', { shiftKey: true });
    press('2');

    expect(onupscale).not.toHaveBeenCalled();
  });

  it('lists the SeedVR2 items, a separator and the creative items in the upscale menu', () => {
    const onupscale = vi.fn();
    mountViewer({ assets: [makeAsset({ upscale: seedvr2Upscale })], onupscale });

    flushSync(() => document.querySelector<HTMLButtonElement>('[data-action="upscale"]')!.click());

    const items = upscaleItems();
    expect(items.map((item) => item.getAttribute('data-action'))).toEqual(['upscale-seedvr2-2', 'upscale-seedvr2-4', 'upscale-creative-2', 'upscale-creative-4']);
    expect(document.querySelectorAll('[role="menu"][aria-label="Upscale"] [role="separator"]')).toHaveLength(1);
    flushSync(() => items[1].click());
    expect(onupscale).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/asset-a.png' }), 4, 'seedvr2');
  });

  it('navigates to the upscale source from the details', () => {
    const source = makeAsset({ id: 'out/source.png', filename: 'source.png' });
    const onnavigate = vi.fn();
    const upscaled = makeAsset({
      id: 'out/source_2x.png',
      filename: 'source_2x.png',
      details: {
        recorded_workflow: 'upscale',
        negative_prompt: 'blurry',
        scheduler: 'beta',
        generation: { time: 12.3, upscale: { factor: 2, denoise: 0.4, steps: 3, pre_sharpen: 0.6 }, sharpen: 1.2 },
        source: { path: '/outputs/out/source.png', width: 832, height: 1216, id: 'out/source.png' },
      },
    });
    mountViewer({ assets: [upscaled, source], onnavigate });
    press('i');

    const details = document.querySelector('#asset-viewer-details')!;

    flushSync(() => details.querySelector<HTMLButtonElement>('[data-testid="viewer-source"] button')!.click());
    expect(onnavigate).toHaveBeenCalledWith(1);
  });

  it('lists every recorded LoRA on its own row with the full entry on hover', () => {
    const lora = '/Users/me/.ziv/loras/style.safetensors:0.8,C:\\loras\\detail.safetensors:1,legacy';
    mountViewer({ assets: [makeAsset({ lora })] });
    press('i');

    const rows = Array.from(document.querySelectorAll('#asset-viewer-details dl div'));
    const loraRows = rows.filter((row) => row.querySelector('dt')?.textContent?.startsWith('LoRA'));
    expect(loraRows.map((row) => [row.querySelector('dt')?.textContent, row.querySelector('dd')?.textContent])).toEqual([
      ['LoRA 1', 'style · 0.8'],
      ['LoRA 2', 'detail · 1'],
      ['LoRA 3', 'legacy'],
    ]);
    expect(loraRows[0].querySelector('dd')?.getAttribute('title')).toBe('/Users/me/.ziv/loras/style.safetensors:0.8');
  });

  describe('checkerboard', () => {
    const cutOut = (id: string) => makeAsset({ id: `out/${id}.png`, url: `/media/out/${id}.png`, thumbnail_url: `/media/out/${id}-thumb.png`, filename: `${id}.png`, transparent: true });
    const opaque = makeAsset({ id: 'out/plain.png', url: '/media/out/plain.png', thumbnail_url: '/media/out/plain-thumb.png', filename: 'plain.png' });
    const checkered = (selector: string) => document.querySelector(selector)?.classList.contains('ui-checkerboard');

    it('sits behind the main image and the stand-in of a transparent asset, on the images themselves', () => {
      mountViewer({ assets: [cutOut('a'), opaque], currentId: 'out/a.png' });
      expect(document.querySelector<HTMLElement>('.viewer-media')?.dataset.transparent).toBe('true');
      expect(checkered('.viewer-media')).toBe(true);
      expect(document.querySelector<HTMLElement>('.viewer-standin')?.dataset.transparent).toBe('true');
      expect(checkered('.viewer-standin')).toBe(true);
    });

    it('leaves an opaque asset without it', () => {
      mountViewer({ assets: [cutOut('a'), opaque], currentId: opaque.id });
      expect(document.querySelector<HTMLElement>('.viewer-media')?.dataset.transparent).toBe('false');
      expect(checkered('.viewer-media')).toBe(false);
      expect(checkered('.viewer-standin')).toBe(false);
    });

    it('marks only the transparent thumbnails in the filmstrip', () => {
      mountViewer({ assets: [cutOut('a'), opaque, cutOut('b')], currentId: opaque.id });
      const states = Array.from(document.querySelectorAll<HTMLImageElement>('.viewer-film img')).map((img) => img.classList.contains('ui-checkerboard'));
      expect(states).toEqual([true, false, true]);
    });

    it('marks the neighbours waiting beside the shown asset in the touch layer', () => {
      touch.coarse = true;
      try {
        mountViewer({ assets: [cutOut('a'), opaque, cutOut('b')], currentId: 'out/plain.png' });
        const prev = document.querySelector('.viewer-pane-prev .viewer-neighbour');
        const next = document.querySelector('.viewer-pane-next .viewer-neighbour');
        expect(prev?.classList.contains('ui-checkerboard')).toBe(true);
        expect(next?.classList.contains('ui-checkerboard')).toBe(true);
      } finally {
        touch.coarse = false;
      }
    });

    it('rides along on the opening flight of a transparent asset', () => {
      const rect = { left: 10, top: 10, width: 200, height: 200, right: 210, bottom: 210, x: 10, y: 10, toJSON: () => ({}) } as DOMRect;
      const box = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(rect);
      const animate = vi.fn(() => ({}) as Animation);
      Element.prototype.animate = animate;
      const origin = document.createElement('button');
      document.body.appendChild(origin);
      try {
        mountViewer({ assets: [cutOut('a'), opaque], currentId: 'out/a.png', origin });
        const flight = document.querySelector<HTMLElement>('[data-testid="viewer-flight"]');
        expect(flight?.dataset.transparent).toBe('true');
        expect(flight?.classList.contains('ui-checkerboard')).toBe(true);
      } finally {
        box.mockRestore();
        Reflect.deleteProperty(Element.prototype, 'animate');
        origin.remove();
      }
    });

    it('leaves the flight of an opaque asset plain', () => {
      const rect = { left: 10, top: 10, width: 200, height: 200, right: 210, bottom: 210, x: 10, y: 10, toJSON: () => ({}) } as DOMRect;
      const box = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(rect);
      Element.prototype.animate = vi.fn(() => ({}) as Animation);
      const origin = document.createElement('button');
      document.body.appendChild(origin);
      try {
        mountViewer({ assets: [opaque, cutOut('a')], currentId: opaque.id, origin });
        const flight = document.querySelector<HTMLElement>('[data-testid="viewer-flight"]');
        expect(flight).not.toBeNull();
        expect(flight?.dataset.transparent).toBe('false');
        expect(flight?.classList.contains('ui-checkerboard')).toBe(false);
      } finally {
        box.mockRestore();
        Reflect.deleteProperty(Element.prototype, 'animate');
        origin.remove();
      }
    });
  });
});

describe('AssetViewer Move', () => {
  const ZOO = { name: 'Zoo', path: '/out/Zoo' };
  const rows = () => Array.from(document.querySelectorAll<HTMLElement>('[role="menu"] [role="menuitemradio"]'));
  const moveButton = () => document.querySelector<HTMLButtonElement>('button[data-action="move"]');

  function mountViewer(props: Record<string, unknown> = {}): void {
    const assets = [makeAsset({ project: 'Night Bus' }), assetB];
    mountComponent(AssetViewer, { assets, currentId: assets[0].id, open: true, onclose: vi.fn(), onnavigate: vi.fn(), onmove: vi.fn(), ...props });
  }

  beforeEach(() => {
    localStorage.clear();
    touch.compact = false;
    projectApi.listProjects.mockReset().mockResolvedValue([{ name: 'Night Bus', path: '/out/Night Bus' }, ZOO]);
  });

  it('shows a labelled Move button between Use as and Download, and none without a handler', () => {
    mountViewer({ onreference: vi.fn() });
    const actions = Array.from(document.querySelectorAll('.viewer-bar [data-action]')).map((el) => el.getAttribute('data-action'));
    expect(actions.indexOf('move')).toBe(actions.indexOf('download') - 1);
    expect(moveButton()?.title).toBe('Move to project');
    expect(moveButton()?.textContent).toContain('Move');
    expect(moveButton()?.getAttribute('aria-haspopup')).toBe('menu');
  });

  it('has no Move button when nothing handles a move', async () => {
    mountViewer({ onmove: undefined });
    expect(moveButton()).toBeNull();
  });

  it('opens the project list at the button, with the current project checked and disabled', async () => {
    mountViewer();
    moveButton()!.click();
    await settle();

    expect(moveButton()?.getAttribute('aria-expanded')).toBe('true');
    const checked = rows().find((row) => row.getAttribute('aria-checked') === 'true')!;
    expect(checked.textContent?.trim()).toBe('Night Bus');
    expect(checked.getAttribute('aria-disabled')).toBe('true');
    expect(document.querySelector('.action-menu-heading')?.textContent).toBe('Move to');
  });

  it('moves the open asset to the chosen project', async () => {
    const onmove = vi.fn();
    mountViewer({ onmove });
    moveButton()!.click();
    await settle();
    rows().find((row) => row.textContent?.trim() === 'Zoo')!.click();
    await settle();

    expect(onmove).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/asset-a.png' }), 'Zoo');
  });

  it('opens the list at the toolbar button when Move is chosen from the right-click menu', async () => {
    mountViewer();
    const anchorRect = vi.spyOn(moveButton()!, 'getBoundingClientRect');
    document.querySelector('.viewer-stage')!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 300, clientY: 200 }));
    await settle();
    document.querySelector<HTMLElement>('[role="menu"] [data-action="move"]')!.click();
    await settle();

    expect(rows().map((row) => row.textContent?.trim())).toEqual(['No project', 'Night Bus', 'Zoo']);
    expect(document.querySelectorAll('[role="menu"]')).toHaveLength(1);
    expect(anchorRect).toHaveBeenCalled();
  });

  it('leaves the viewer keys alone while the Move list is open', async () => {
    const onnavigate = vi.fn();
    mountViewer({ onnavigate });
    moveButton()!.click();
    await settle();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));

    expect(onnavigate).not.toHaveBeenCalled();
  });

  it('keeps Move in the actions sheet on a phone, with no toolbar button and no heading', async () => {
    touch.compact = true;
    mountViewer();
    expect(moveButton()).toBeNull();
    document.querySelector<HTMLButtonElement>('button[data-action="more"]')!.click();
    await settle();
    document.querySelector<HTMLElement>('[role="menu"] [data-action="move"]')!.click();
    await settle();

    expect(rows().map((row) => row.textContent?.trim())).toEqual(['No project', 'Night Bus', 'Zoo']);
    expect(document.querySelector('.action-menu-heading')).toBeNull();
  });
});
