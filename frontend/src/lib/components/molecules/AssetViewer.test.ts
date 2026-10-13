// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import AssetViewer from './AssetViewer.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { requestConfirm } from './confirm.svelte';

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/asset-a.png',
    url: '/media/out/asset-a.png',
    thumbnail_url: '/media/out/asset-a.png',
    filename: 'asset-a.png',
    created_at: '2026-04-30T12:00:00Z',
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
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    localStorage.clear();
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
  });

  function mountViewer(props: Record<string, unknown>): void {
    app = flushSync(() => mount(AssetViewer, {
      target,
      props: { assets: [makeAsset(), assetB], currentIndex: 0, open: true, onclose: vi.fn(), onnavigate: vi.fn(), ...props },
    }));
  }

  function button(label: string): HTMLButtonElement | null {
    return document.querySelector(`button[aria-label="${label}"]`);
  }

  it('disables previous on the first asset and next on the last', async () => {
    mountViewer({});
    expect(button('Previous asset')?.disabled).toBe(true);
    expect(button('Next asset')?.disabled).toBe(false);

    await unmount(app!);
    mountViewer({ currentIndex: 1 });
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

    await unmount(app!);
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

    it('shows no scene section for an asset without one', () => {
      const details = openDetails(makeAsset({ prompt: 'A fox.' }));
      expect(details.querySelector('[data-testid="viewer-scene"]')).toBeNull();
    });
  });

  it('runs reuse, reference and delete on the shown asset', () => {
    const onreuse = vi.fn();
    const onreference = vi.fn();
    const ondelete = vi.fn();
    mountViewer({ currentIndex: 1, onreuse, onreference, ondelete });

    (document.querySelector('[data-action="reuse"]') as HTMLButtonElement).click();
    expect(onreuse).toHaveBeenCalledWith(assetB);

    (document.querySelector('[data-action="reference"]') as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    expect(onreference).toHaveBeenCalledWith(assetB, 'image');

    (document.querySelector('[data-action="delete"]') as HTMLButtonElement).click();
    expect(ondelete).toHaveBeenCalledWith(assetB);
  });

  it('stays on the nearest asset when the list shrinks and closes when it empties', async () => {
    const onclose = vi.fn();
    mountViewer({ currentIndex: 5, onclose });
    expect(document.querySelector('#asset-viewer-title')?.textContent).toBe('asset-b.png');

    await unmount(app!);
    mountViewer({ assets: [], onclose });
    flushSync();
    expect(onclose).toHaveBeenCalled();
    expect(document.querySelector('[data-testid="asset-viewer"]')).toBeNull();
  });

  it('lists every asset in the thumbnail strip and asks for more near the end', () => {
    const many = Array.from({ length: 30 }, (_, i) => makeAsset({ id: `out/${i}.png`, url: `/media/out/${i}.png`, filename: `${i}.png` }));
    const onnearend = vi.fn();
    mountViewer({ assets: many, currentIndex: 10, onnearend });
    expect(document.querySelectorAll('[data-film-index]')).toHaveLength(30);
    expect(onnearend).not.toHaveBeenCalled();

    flushSync(() => { void unmount(app!); });
    mountViewer({ assets: many, currentIndex: 28, onnearend });
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
    mountViewer({ currentIndex: 1, ondelete });
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
    mountViewer({ assets: many, currentIndex: 2, onnavigate });
    press('End');
    expect(onnavigate).toHaveBeenLastCalledWith(4);
    press('Home');
    expect(onnavigate).toHaveBeenLastCalledWith(0);
  });

  it('opens the reference menu with E and leaves other keys to it while open', () => {
    const onreference = vi.fn();
    const ondelete = vi.fn();
    mountViewer({ onreference, ondelete });
    press('e');
    expect(document.querySelector('[role="menu"]')).not.toBeNull();
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
      expect(extendButton()?.getAttribute('title')).toBe('Extend this video (E)');
      extendButton()!.click();
      expect(onextend).toHaveBeenCalledWith(asset);
    });

    it('has no Extend button for an image, without a handler, or when extending is unavailable here', () => {
      mountViewer({ assets: [makeAsset({ extend: { available: true, reason: null } })], onextend: vi.fn() });
      expect(extendButton()).toBeNull();

      flushSync(() => { void unmount(app!); });
      mountViewer({ assets: [clip()] });
      expect(extendButton()).toBeNull();

      flushSync(() => { void unmount(app!); });
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

    it('still opens the reference menu with E on an image', () => {
      const onextend = vi.fn();
      mountViewer({ assets: [makeAsset()], onextend, onreference: vi.fn() });
      press('e');
      expect(document.querySelector('[role="menu"]')).not.toBeNull();
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

    await unmount(app!);
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

  it('hides the upscale menu without a handler', () => {
    mountViewer({ assets: [makeAsset({ upscale })] });
    expect(document.querySelector('[data-action="upscale"]')).toBeNull();
  });

  it('hides the upscale menu for videos', () => {
    mountViewer({ assets: [makeAsset({ media_type: 'video', upscale })], onupscale: vi.fn() });
    expect(document.querySelector('[data-action="upscale"]')).toBeNull();
  });

  it('upscales 2× with X then 2 and refuses a disallowed 4× with X then 4', () => {
    const onupscale = vi.fn();
    const ondelete = vi.fn();
    mountViewer({ assets: [makeAsset({ upscale })], onupscale, ondelete });

    press('x');
    press('2');
    expect(onupscale).toHaveBeenCalledTimes(1);
    expect(onupscale).toHaveBeenLastCalledWith(expect.objectContaining({ id: 'out/asset-a.png' }), 2, 'creative');

    press('x');
    press('4');
    expect(onupscale).toHaveBeenCalledTimes(1);
  });

  it('runs the SeedVR2 upscale with X then 2 or 4 where it is offered, even for a size flagged as high memory', () => {
    const onupscale = vi.fn();
    mountViewer({ assets: [makeAsset({ upscale: seedvr2Upscale })], onupscale });

    press('x');
    press('2');
    press('x');
    press('4');

    expect(onupscale.mock.calls.map(([, factor, method]) => [factor, method])).toEqual([[2, 'seedvr2'], [4, 'seedvr2']]);
  });

  it('warns when the chord starts a SeedVR2 upscale flagged as high memory, and stays quiet otherwise', () => {
    const onupscale = vi.fn();
    toasts.length = 0;
    mountViewer({ assets: [makeAsset({ upscale: seedvr2Upscale })], onupscale });

    press('x');
    press('2');
    expect(onupscale).toHaveBeenCalledTimes(1);
    expect(toasts).toEqual([]);

    press('x');
    press('4');
    expect(onupscale).toHaveBeenCalledTimes(2);
    expect(toasts.map((toast) => [toast.message, toast.type])).toEqual([['May need more memory than this Mac has.', 'warning']]);
  });

  it('does not start the upscale chord from Shift+X', () => {
    const onupscale = vi.fn();
    mountViewer({ assets: [makeAsset({ upscale: seedvr2Upscale })], onupscale });

    press('X', { shiftKey: true });
    press('2');

    expect(onupscale).not.toHaveBeenCalled();
  });

  it('runs the SeedVR2 upscale from the chord when every creative factor is disabled', () => {
    const onupscale = vi.fn();
    const noCreative = {
      ...seedvr2Upscale,
      factors: upscale.factors.map((option) => ({ ...option, allowed: false, reason: 'This model cannot upscale.' })),
    };
    mountViewer({ assets: [makeAsset({ upscale: noCreative })], onupscale });

    press('x');
    press('2');

    expect(onupscale.mock.calls.map(([, factor, method]) => [factor, method])).toEqual([[2, 'seedvr2']]);
  });

  it('explains instead of upscaling from the chord when only disabled creative factors exist', () => {
    const onupscale = vi.fn();
    const noCreative = { factors: upscale.factors.map((option) => ({ ...option, allowed: false, reason: 'This model cannot upscale.' })), seedvr2: null };
    toasts.length = 0;
    mountViewer({ assets: [makeAsset({ upscale: noCreative })], onupscale });

    press('x');
    press('2');

    expect(onupscale).not.toHaveBeenCalled();
    expect(toasts.map((toast) => toast.message)).toEqual(['This model cannot upscale.']);
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

  it('leaves 2 and 4 alone without a preceding X', () => {
    const onupscale = vi.fn();
    mountViewer({ assets: [makeAsset({ upscale })], onupscale });
    press('2');
    expect(onupscale).not.toHaveBeenCalled();
  });

  it('shows recorded settings, post-processing and the upscale source in the details', () => {
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
    const facts = Object.fromEntries(Array.from(details.querySelectorAll('dl div')).map((row) => [row.querySelector('dt')?.textContent, row.querySelector('dd')?.textContent]));
    expect(facts.Workflow).toBe('Upscale');
    expect(facts.Scheduler).toBe('beta');
    expect(facts.Upscale).toBe('2× · denoise 0.4 · 3 steps · pre-sharpen 0.6');
    expect(facts.Sharpen).toBe('1.2');
    expect(facts['Generation time']).toBe('12.3 s');
    expect(details.textContent).toContain('blurry');

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
});
