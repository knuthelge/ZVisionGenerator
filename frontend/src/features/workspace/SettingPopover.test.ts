// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { downloadStore } from '$lib/state/downloads.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { memoryUploadStore } from '$lib/state/uploadStore';
import type { GalleryAsset, ModelOption, WorkspaceContext } from '$lib/types';
import { makeContext, makeImageDefaults } from '../../test-utils/workspace';
import { currentImagesView } from './imagesView';
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
  heldUploads.useStore(memoryUploadStore());
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
        key: 'model', context, get images() { return currentImagesView(context); }, busy: false, models, quantizeOptions: null, lastSeed: null,
        onModelChange, onclose: vi.fn(),
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
          key: 'model', context, get images() { return currentImagesView(context); }, busy: false, models: missing, quantizeOptions: null, lastSeed: null,
          onModelChange: vi.fn(), onclose: vi.fn(),
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
        key: 'loras', context: ctx, get images() { return currentImagesView(ctx); }, busy: false, models, quantizeOptions: null, lastSeed: null,
        onModelChange: vi.fn(), onclose: vi.fn(),
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

// The Images popover: the section's groups for the open tab, read from the draft and the held uploads, written through the draft's setters.
describe('SettingPopover images', () => {
  const imageContext = (): WorkspaceContext => makeContext({ image_model_defaults: { zit: makeImageDefaults({ max_reference_images: 4 }) } });

  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => 'blob:img');
    URL.revokeObjectURL = vi.fn();
    draft.update('model', 'zit');
  });

  function open(props: Record<string, unknown> = {}, context: WorkspaceContext = imageContext()): void {
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: { key: 'images', context, get images() { return currentImagesView(context); }, busy: false, models: [], quantizeOptions: null, lastSeed: null, onModelChange: vi.fn(), onclose: vi.fn(), ...props },
    }));
  }

  const popover = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement;
  const groups = () => Array.from(popover().querySelectorAll('.pop-group-label')).map((el) => el.textContent);
  const button = (name: string) => popover().querySelector(`button[aria-label="${name}"]`) as HTMLButtonElement | null;
  const buttonWithText = (text: string) => Array.from(popover().querySelectorAll('button')).find((b) => b.textContent?.trim() === text) as HTMLButtonElement | undefined;
  const sourceButton = (text: string) => Array.from(popover().querySelectorAll('button')).find((b) => b.textContent?.includes(text)) as HTMLButtonElement | undefined;
  const videoContext = () => makeContext();

  describe('in Image', () => {
    it('holds Start image, References and no Strength while there is no start image', () => {
      open();

      expect(popover().getAttribute('aria-label')).toBe('Images');
      expect(groups()).toEqual(['Start image', 'References']);
      expect(popover().textContent).toContain('No start image yet');
      expect(popover().querySelector('input[aria-label="Strength"]')).toBeNull();
      expect(popover().textContent).toContain('Add reference 1');
    });

    it('puts Strength between the Start image and the References, with its line, once a start image is set', () => {
      draft.setStartImage('image', { path: '/in/pond-sketch.png' }, imageContext());
      draft.update('referenceImageStrength', 0.8);
      open();

      expect(groups()).toEqual(['Start image', 'Strength', 'References']);
      expect(popover().textContent).toContain('pond-sketch.png');
      expect(popover().querySelector('input[aria-label="Strength"]')).not.toBeNull();
    });

    it('lists each reference as Reference N over its name, with a way to add the next', () => {
      draft.patch({ referenceImagePaths: ['/in/jacket.png', '/in/portrait.png'] });
      open();

      expect(Array.from(popover().querySelectorAll('.ref-num')).map((el) => el.textContent)).toEqual(['Reference 1', 'Reference 2']);
      expect(popover().querySelector('#pop-ref-add-device')).not.toBeNull();
    });

    it('sets the start image from the latest output and keeps the popover open', () => {
      galleryStore.seed([{ id: 'latest', file_path: '/out/latest.png', media_type: 'image' } as GalleryAsset]);
      const onclose = vi.fn();
      open({ onclose });

      sourceButton('latest output')!.click();
      flushSync();

      expect(draft.state).toMatchObject({ workflow: 'img2img', referenceImagePath: '/out/latest.png' });
      expect(groups()).toContain('Strength');
      expect(onclose).not.toHaveBeenCalled();
    });

    it('offers the clear button only while the start image holds an image', () => {
      open();
      expect(button('Clear start image')).toBeNull();

      draft.setStartImage('image', { file: new File(['x'], 'mine.png', { type: 'image/png' }) }, imageContext());
      flushSync();
      expect(button('Clear start image')).not.toBeNull();

      draft.setStartImage('image', null, imageContext());
      flushSync();
      expect(button('Clear start image')).toBeNull();
    });

    it('clears the start image and keeps the popover open', () => {
      draft.setStartImage('image', { path: '/in/start.png' }, imageContext());
      const onclose = vi.fn();
      open({ onclose });

      button('Clear start image')!.click();
      flushSync();

      expect(draft.state).toMatchObject({ workflow: 'txt2img', referenceImagePath: null });
      expect(groups()).toEqual(['Start image', 'References']);
      expect(onclose).not.toHaveBeenCalled();
    });

    it('a missing start image says to choose it again, and its sources stay in reach', () => {
      draft.setStartImage('image', { path: '/in/gone.png' }, imageContext());
      draft.patch({ missingReferencePaths: ['/in/gone.png'] });
      open();

      expect(popover().textContent).toContain('moved or been deleted');
      expect(sourceButton('Choose from this device')).toBeDefined();
      expect(sourceButton('Browse the server')).toBeDefined();
    });

    it('a missing reference offers Replace, and the replacement keeps its number', () => {
      galleryStore.seed([{ id: 'latest', file_path: '/out/latest.png', media_type: 'image' } as GalleryAsset]);
      draft.patch({ referenceImagePaths: ['/in/a.png', '/in/gone.png'], missingReferencePaths: ['/in/gone.png'] });
      open();

      expect(popover().textContent).toContain('moved or been deleted');
      expect(button('Replace reference 1')).toBeNull();
      button('Replace reference 2')!.click();
      flushSync();
      const add = popover().querySelector('.add') as HTMLElement;
      expect(add.textContent).toContain('Replace reference 2');
      (Array.from(add.querySelectorAll('button')).find((b) => b.textContent?.includes('latest output')) as HTMLButtonElement).click();
      flushSync();

      expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '/out/latest.png']);
      expect(draft.state.missingReferencePaths).toEqual([]);
    });

    it('a lost reference upload says to choose it again', async () => {
      draft.patch({ referenceImagePaths: [''] });
      await heldUploads.restore();
      open();

      expect(popover().textContent).toContain('Choose this image again.');
    });

    it('removing a reference writes the rows and the held uploads together', () => {
      const upload = new File(['x'], 'mine.png', { type: 'image/png' });
      draft.update('referenceImagePaths', ['/in/a.png', '']);
      heldUploads.setReferences([null, upload]);
      const onclose = vi.fn();
      open({ onclose });

      button('Remove reference 1')!.click();
      flushSync();

      expect(draft.state.referenceImagePaths).toEqual(['']);
      expect(heldUploads.references).toEqual([upload]);
      expect(onclose).not.toHaveBeenCalled();
    });

    it('is not offered the References group on a model without references', () => {
      open({}, makeContext());

      expect(groups()).toEqual(['Start image']);
    });

    it('closes itself when the model can use no image', () => {
      const none = makeContext({ image_model_defaults: { zit: makeImageDefaults({ supports_img2img: false, max_reference_images: 0 }) } });
      const onclose = vi.fn();
      open({ onclose }, none);

      expect(onclose).toHaveBeenCalled();
    });
  });

  describe('in Video', () => {
    beforeEach(() => draft.patch({ workflow: 'txt2vid', model: 'ltx-8' }));

    it('holds First frame and Last frame, flags a last frame kept without a first frame, and has neither Strength nor References', () => {
      draft.setLastFrame({ path: '/in/end.png' });
      open({}, videoContext());

      expect(groups()).toEqual(['First frame', 'Last frame']);
      expect(draft.state.workflow).toBe('txt2vid');
      expect(button('Clear first frame')).toBeNull();
      expect(button('Clear last frame')).not.toBeNull();
      expect(popover().querySelector('[data-unused="true"]')).not.toBeNull();
      expect(popover().querySelector('input[aria-label="Strength"]')).toBeNull();
    });

    it('offers no clear button on an empty Last frame', () => {
      open({}, videoContext());

      expect(groups()).toEqual(['First frame', 'Last frame']);
      expect(button('Clear first frame')).toBeNull();
      expect(button('Clear last frame')).toBeNull();
    });

    it('clears the first frame, leaving the Image tab\'s start image alone', () => {
      const video = videoContext();
      draft.patch({ referenceImagePath: '/in/start.png' });
      draft.setStartImage('video', { path: '/in/first.png' }, video);
      open({}, video);
      expect(draft.state.workflow).toBe('img2vid');

      button('Clear first frame')!.click();
      flushSync();

      expect(draft.state).toMatchObject({ workflow: 'txt2vid', firstFramePath: null, referenceImagePath: '/in/start.png' });
    });

    it('clears the last frame without touching the first', () => {
      const video = videoContext();
      draft.setStartImage('video', { path: '/in/first.png' }, video);
      draft.setLastFrame({ path: '/in/end.png' });
      open({}, video);

      button('Clear last frame')!.click();
      flushSync();

      expect(draft.state).toMatchObject({ workflow: 'img2vid', firstFramePath: '/in/first.png', endImagePath: null });
    });
  });

  it('is titled Images, and has no Reset: Strength resets from its own row', () => {
    open();

    expect(buttonWithText('Reset')).toBeUndefined();
  });
});
