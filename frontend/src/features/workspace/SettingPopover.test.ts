// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { downloadStore } from '$lib/state/downloads.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import type { GalleryAsset, ModelOption, WorkspaceContext } from '$lib/types';
import SettingPopover from './SettingPopover.svelte';

const models: ModelOption[] = [
  { id: 'zit', label: 'Z-Image Turbo', type: 'image' },
  { id: 'flux', label: 'FLUX', type: 'image' },
];

const context = {
  workflow_contract: { values: ['txt2img'], definitions: { txt2img: { mode: 'image', visible_controls: [], clear_fields: [] } } },
  image_model_defaults: {},
  video_model_defaults: {},
  defaults: null,
  loras: [],
  prompt_sources: [],
  scheduler_options: [],
} as unknown as WorkspaceContext;

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

beforeEach(() => {
  localStorage.clear();
  draft.reset();
  // jsdom has no Web Animations API, which the popover's enter transition calls.
  Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(async () => {
  if (app) {
    await unmount(app);
    app = null;
  }
  target.remove();
  galleryStore.reset();
  delete (Element.prototype as Partial<Element>).animate;
});

describe('SettingPopover model list', () => {
  beforeEach(() => draft.update('model', 'zit'));

  function mountPopover(onModelChange: (model: string) => void): void {
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: {
        key: 'model', context, busy: false, models, quantizeOptions: null, imageFile: null, referencePreviewUrl: null, lastSeed: null,
        onImageFileChange: vi.fn(), onModelChange, onUseLatestAsReference: vi.fn(), onclose: vi.fn(),
      },
    }));
  }

  it('hands the chosen model to onModelChange', () => {
    const onModelChange = vi.fn();
    mountPopover(onModelChange);

    (target.querySelector('[role="radio"][aria-checked="false"]') as HTMLButtonElement).click();

    expect(onModelChange).toHaveBeenCalledExactlyOnceWith('flux');
  });

  it('does nothing when the current model is chosen again', () => {
    const onModelChange = vi.fn();
    mountPopover(onModelChange);

    (target.querySelector('[role="radio"][aria-checked="true"]') as HTMLButtonElement).click();

    expect(onModelChange).not.toHaveBeenCalled();
  });

  describe('not-downloaded note', () => {
    const missing: ModelOption[] = [{ id: 'zit', label: 'Z-Image Turbo', type: 'image', downloaded: false }];

    function mountMissing(): void {
      app = flushSync(() => mount(SettingPopover, {
        target,
        props: {
          key: 'model', context, busy: false, models: missing, quantizeOptions: null, imageFile: null, referencePreviewUrl: null, lastSeed: null,
          onImageFileChange: vi.fn(), onModelChange: vi.fn(), onUseLatestAsReference: vi.fn(), onclose: vi.fn(),
        },
      }));
    }

    afterEach(() => vi.restoreAllMocks());

    it('is shown for a missing model with no download on its way', () => {
      mountMissing();

      expect(target.querySelector('[data-testid="model-not-downloaded"]')).not.toBeNull();
    });

    it('is hidden while that model is downloading', () => {
      vi.spyOn(downloadStore, 'active').mockReturnValue({ id: 'd1', kind: 'model', name: 'zit', label: 'Z-Image Turbo', state: 'running', message: null });
      mountMissing();

      expect(target.querySelector('[data-testid="model-not-downloaded"]')).toBeNull();
    });
  });
});

describe('SettingPopover LoRA list', () => {
  beforeEach(() => draft.update('model', 'zit'));

  const loraContext = {
    ...context,
    image_model_defaults: { zit: { family: 'zimage' } },
    loras: [
      { name: 'b-match', path: '/l/b', family: 'zimage', size_variant: null },
      { name: 'a-match', path: '/l/a', family: 'zimage', size_variant: null },
      { name: 'c-unknown', path: '/l/c', family: null, size_variant: null },
      { name: 'd-ltx', path: '/l/d', family: 'ltx', size_variant: null },
    ],
  } as unknown as WorkspaceContext;

  function mountLoras(ctx: WorkspaceContext = loraContext): void {
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: {
        key: 'loras', context: ctx, busy: false, models, quantizeOptions: null, imageFile: null, referencePreviewUrl: null, lastSeed: null,
        onImageFileChange: vi.fn(), onModelChange: vi.fn(), onUseLatestAsReference: vi.fn(), onclose: vi.fn(),
      },
    }));
  }

  const fold = () => target.querySelector('[data-action="other-models"]') as HTMLButtonElement;

  it('expands the fold in place and shows the family label beside each folded LoRA', () => {
    mountLoras();
    flushSync(() => fold().click());

    expect(fold().getAttribute('aria-expanded')).toBe('true');
    const row = Array.from(target.querySelectorAll('.pop-item')).find((el) => el.textContent?.includes('d-ltx'));
    expect(row?.textContent).toContain('LTX');
    const unknown = Array.from(target.querySelectorAll('.pop-item')).find((el) => el.textContent?.includes('c-unknown'));
    expect(unknown?.querySelector('small')).toBeNull();
  });

  it('adds the chosen LoRA to the draft', () => {
    mountLoras();
    (Array.from(target.querySelectorAll('.pop-item')).find((el) => el.textContent?.includes('b-match')) as HTMLButtonElement).click();

    expect(draft.state.loraString).toBe('b-match:1');
  });
});

describe('SettingPopover last frame', () => {
  beforeEach(() => draft.patch({ workflow: 'img2vid', model: 'h3' }));

  const videoContext = (hidden: string[]): WorkspaceContext => ({
    workflow_contract: {
      values: ['img2vid'],
      definitions: {
        img2vid: {
          mode: 'video',
          visible_controls: ['reference_image', 'reference_image_path', 'end_image', 'end_image_path', 'end_image_clear'],
          clear_fields: [],
        },
      },
    },
    image_model_defaults: {},
    video_model_defaults: { h3: { hidden_controls: hidden } },
    defaults: null,
    loras: [],
    prompt_sources: [],
    scheduler_options: [],
  }) as unknown as WorkspaceContext;

  function open(key: 'reference' | 'endReference', hidden: string[] = [], props: Record<string, unknown> = {}): void {
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: {
        key, context: videoContext(hidden), busy: false, models: [], quantizeOptions: null, imageFile: null, referencePreviewUrl: null, lastSeed: null,
        onImageFileChange: vi.fn(), onModelChange: vi.fn(), onUseLatestAsReference: vi.fn(), onclose: vi.fn(), ...props,
      },
    }));
  }

  const popover = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;

  it('picks the last frame by typing or pasting a path, and submits no field of its own', () => {
    open('endReference');

    const input = target.querySelector('#ws-pop-end-image-path') as HTMLInputElement;
    input.value = '/in/end.png';
    input.dispatchEvent(new Event('input', { bubbles: true }));

    expect(draft.state.endImagePath).toBe('/in/end.png');
    expect(input.getAttribute('name')).toBeNull();
    expect(popover().getAttribute('data-key')).toBe('endReference');
  });

  it('shows the chosen last frame by name and clears it without touching the first frame', () => {
    draft.patch({ referenceImagePath: '/in/start.png', endImagePath: '/in/end.png' });
    const onEndImageFileChange = vi.fn();
    open('endReference', [], { onEndImageFileChange });

    expect(popover().textContent).toContain('end.png');
    (popover().querySelector('button[aria-label="Clear last frame"]') as HTMLButtonElement).click();
    flushSync();

    expect(draft.state.endImagePath).toBeNull();
    expect(draft.state.referenceImagePath).toBe('/in/start.png');
    expect(onEndImageFileChange).toHaveBeenCalledWith(null);
  });

  it('offers no Clear while the last frame is empty', () => {
    open('endReference');

    expect(popover().querySelector('button[aria-label="Clear last frame"]')).toBeNull();
    expect(popover().textContent).toContain('No last frame');
  });

  it('hands an uploaded image to the last-frame callback, not the first frame', () => {
    const onImageFileChange = vi.fn();
    const onEndImageFileChange = vi.fn();
    open('endReference', [], { onImageFileChange, onEndImageFileChange });

    const input = popover().querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['x'], 'end.png', { type: 'image/png' });
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    input.dispatchEvent(new Event('change', { bubbles: true }));

    expect(onEndImageFileChange).toHaveBeenCalledWith(file);
    expect(onImageFileChange).not.toHaveBeenCalled();
  });

  it('names the first-frame popover First frame while the model has a last frame', () => {
    open('reference');

    expect(popover().getAttribute('aria-label')).toBe('First frame');
  });
});

describe('SettingPopover references', () => {
  const imageContext = (cap: number): WorkspaceContext => ({
    workflow_contract: {
      values: ['txt2img'],
      definitions: { txt2img: { mode: 'image', visible_controls: ['references', 'references_clear'], clear_fields: [] } },
    },
    image_model_defaults: { klein: { max_reference_images: cap } },
    video_model_defaults: {},
    defaults: null,
    loras: [],
    prompt_sources: [],
    scheduler_options: [],
  }) as unknown as WorkspaceContext;

  beforeEach(() => draft.patch({ workflow: 'txt2img', model: 'klein' }));

  function open(cap = 4, props: Record<string, unknown> = {}): void {
    URL.createObjectURL = vi.fn(() => 'blob:ref');
    URL.revokeObjectURL = vi.fn();
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: {
        key: 'references', context: imageContext(cap), busy: false, models: [], quantizeOptions: null, imageFile: null, referencePreviewUrl: null, lastSeed: null,
        onImageFileChange: vi.fn(), onModelChange: vi.fn(), onUseLatestAsReference: vi.fn(), onclose: vi.fn(), ...props,
      },
    }));
  }

  const popover = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;
  const removeButtons = () => Array.from(popover().querySelectorAll('button[aria-label^="Remove image"]'));

  it('offers no way to add once the list is full', () => {
    draft.update('referenceImagePaths', ['/in/a.png', '/in/b.png']);
    open(2);

    expect(popover().querySelector('.pop-list')).toBeNull();
    expect(popover().querySelector('#pop-ref-add-device')).toBeNull();
  });

  it('marks a row whose image is gone and keeps its number and its remove button', () => {
    draft.patch({ referenceImagePaths: ['/in/a.png', '/in/gone.png', ''], missingReferencePaths: ['/in/gone.png'] });
    open();

    const rows = Array.from(popover().querySelectorAll('.pop-refrow'));
    expect(rows).toHaveLength(3);
    // The flagged path and the upload that lost its file are both marked; the readable path is not.
    expect(rows.map((row) => row.querySelector('.pop-refrow-error') !== null)).toEqual([false, true, true]);
    expect(removeButtons()).toHaveLength(3);
  });

  it('replaces a missing row in place with a device image, keeping its number, even when the list is full', () => {
    draft.patch({ referenceImagePaths: ['/in/a.png', '/in/gone.png'], missingReferencePaths: ['/in/gone.png'] });
    const onReferenceFilesChange = vi.fn();
    open(2, { referenceFiles: [null, null], onReferenceFilesChange });
    expect(popover().querySelector('#pop-ref-1-replace')).toBeNull();

    (popover().querySelector('#pop-ref-2-replace') as HTMLButtonElement).click();
    flushSync();
    const file = new File(['x'], 'new.png', { type: 'image/png' });
    const input = popover().querySelector('input[type="file"]') as HTMLInputElement;
    expect(input.multiple).toBe(false);
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();

    expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '']);
    expect(draft.state.missingReferencePaths).toEqual([]);
    expect(onReferenceFilesChange).toHaveBeenCalledWith([null, file]);
  });

  it('replaces a lost upload with the latest output', () => {
    galleryStore.seed([{ id: 'latest', file_path: '/out/latest.png', media_type: 'image' } as GalleryAsset]);
    draft.patch({ referenceImagePaths: ['/in/a.png', ''] });
    open(4, { referenceFiles: [null, null] });

    (popover().querySelector('#pop-ref-2-replace') as HTMLButtonElement).click();
    flushSync();
    const latest = Array.from(popover().querySelectorAll('.pop-item')).find((item) => item.textContent?.includes('latest output')) as HTMLButtonElement;
    latest.click();
    flushSync();

    expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '/out/latest.png']);
  });

  it('removes a row and hands the page the uploads that are left', () => {
    const upload = new File(['x'], 'mine.png', { type: 'image/png' });
    draft.update('referenceImagePaths', ['/in/a.png', '']);
    const onReferenceFilesChange = vi.fn();
    open(4, { referenceFiles: [null, upload], onReferenceFilesChange });

    (removeButtons()[0] as HTMLButtonElement).click();
    flushSync();

    expect(draft.state.referenceImagePaths).toEqual(['']);
    expect(onReferenceFilesChange).toHaveBeenCalledWith([upload]);
  });

  it('adds every file chosen from the device, up to the cap', () => {
    draft.update('referenceImagePaths', ['/in/a.png']);
    const onReferenceFilesChange = vi.fn();
    open(3, { referenceFiles: [null], onReferenceFilesChange });
    const files = [new File(['1'], 'one.png', { type: 'image/png' }), new File(['2'], 'two.png', { type: 'image/png' }), new File(['3'], 'three.png', { type: 'image/png' })];

    const input = popover().querySelector('input[type="file"]') as HTMLInputElement;
    expect(input.multiple).toBe(true);
    Object.defineProperty(input, 'files', { configurable: true, value: files });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();

    expect(draft.state.referenceImagePaths).toHaveLength(3);
    expect(onReferenceFilesChange).toHaveBeenCalledWith([null, files[0], files[1]]);
  });
});
