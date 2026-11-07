// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { downloadStore } from '$lib/state/downloads.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { memoryUploadStore } from '$lib/state/uploadStore';
import type { GalleryAsset, ModelOption, WorkspaceContext } from '$lib/types';
import { FLUX_LICENCE, MATTE_LICENCE, makeContext, makeImageDefaults, makeVideoDefaults } from '../../test-utils/workspace';
import { currentAudioTrackView } from './audioTrackView';
import { currentImagesView } from './imagesView';
import SettingPopover from './SettingPopover.svelte';

const trackApi = vi.hoisted(() => ({ inspectAudioTrack: vi.fn() }));
vi.mock('$lib/api/audioTrack', () => ({ inspectAudioTrack: trackApi.inspectAudioTrack }));

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
        key: 'model', context, get images() { return currentImagesView(context); }, get audioTrack() { return currentAudioTrackView(context); }, busy: false, models, quantizeOptions: null, lastSeed: null,
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
          key: 'model', context, get images() { return currentImagesView(context); }, get audioTrack() { return currentAudioTrackView(context); }, busy: false, models: missing, quantizeOptions: null, lastSeed: null,
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

    it('reads an audio model\'s download under the audio kind', () => {
      const audio: ModelOption[] = [{ id: 'yue2', label: 'YuE2 3B', type: 'audio', downloaded: false }];
      vi.spyOn(downloadStore, 'active').mockImplementation((kind, name) => (
        kind === 'audio' && name === 'yue2' ? { id: 'd1', kind: 'audio', name: 'yue2', label: 'YuE2 3B', state: 'running', message: null } : null
      ));
      app = flushSync(() => mount(SettingPopover, {
        target,
        props: {
          key: 'model', context, get images() { return currentImagesView(context); }, get audioTrack() { return currentAudioTrackView(context); }, busy: false, models: audio, quantizeOptions: null, lastSeed: null,
          onModelChange: vi.fn(), onclose: vi.fn(),
        },
      }));

      expect(target.querySelector('[data-testid="model-not-downloaded"]')).toBeNull();
      expect(target.querySelector('[data-testid="model-download-status"]')).not.toBeNull();
    });
  });
});

describe('SettingPopover Transparent background', () => {
  const imageContext = {
    ...context,
    matte_licences: [MATTE_LICENCE],
    workflow_contract: { values: ['txt2img'], definitions: { txt2img: { mode: 'image', visible_controls: ['postprocess_transparent'], clear_fields: [] } } },
  } as unknown as WorkspaceContext;
  const onclose = vi.fn();

  function mountTransparent(busy = false): void {
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: {
        key: 'transparent', context: imageContext, get images() { return currentImagesView(imageContext); }, busy, models, quantizeOptions: null, lastSeed: null,
        onModelChange: vi.fn(), onclose,
      },
    }));
  }

  const toggle = () => target.querySelector('[role="switch"][aria-label="Transparent background"]') as HTMLInputElement | null;

  it('holds the switch, the labelled licence mark and the reason as visible text', () => {
    mountTransparent();

    expect(toggle()).not.toBeNull();
    const reason = target.querySelector('[data-testid="transparent-reason"]') as HTMLElement;
    const mark = reason.querySelector('button')!;
    expect(mark.textContent).toContain('Non-commercial');
    expect(mark.getAttribute('aria-label')).toBe(`Non-commercial. Licence: ${MATTE_LICENCE.name}`);
    expect(reason.textContent).toContain(`${MATTE_LICENCE.summary}.`);
    // The mark sits outside the switch's label, so tapping it never flips the switch.
    expect(reason.closest('label')).toBeNull();
  });

  it('turns the draft setting on and off', () => {
    mountTransparent();

    flushSync(() => toggle()!.click());
    expect(draft.state.postprocessTransparent).toBe(true);
    flushSync(() => toggle()!.click());
    expect(draft.state.postprocessTransparent).toBe(false);
  });

  it('disables the switch while a submit is in flight', () => {
    mountTransparent(true);

    expect(toggle()!.disabled).toBe(true);
  });

  it('offers Reset only while the switch is on', () => {
    draft.update('postprocessTransparent', true);
    mountTransparent();

    const reset = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Reset') ?? null;
    expect(reset).not.toBeNull();
    expect(reset!.disabled).toBe(false);
    flushSync(() => reset!.click());
    expect(draft.state.postprocessTransparent).toBe(false);
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
        key: 'loras', context: ctx, get images() { return currentImagesView(ctx); }, get audioTrack() { return currentAudioTrackView(ctx); }, busy: false, models, quantizeOptions: null, lastSeed: null,
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
      props: { key: 'images', context, get images() { return currentImagesView(context); }, get audioTrack() { return currentAudioTrackView(context); }, busy: false, models: [], quantizeOptions: null, lastSeed: null, onModelChange: vi.fn(), onclose: vi.fn(), ...props },
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

describe('SettingPopover thumbnails', () => {
  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => 'blob:img');
    URL.revokeObjectURL = vi.fn();
    draft.update('model', 'zit');
  });

  function open(key: string, ctx: WorkspaceContext): void {
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: { key, context: ctx, get images() { return currentImagesView(ctx); }, get audioTrack() { return currentAudioTrackView(ctx); }, busy: false, models: [], quantizeOptions: null, lastSeed: null, onModelChange: vi.fn(), onclose: vi.fn() },
    }));
  }

  /** The purpose and path a thumbnail's server preview asks for. */
  const asked = (img: Element | null | undefined) => {
    const url = new URL(img?.getAttribute('src') ?? '', 'http://localhost');
    return { route: url.pathname, purpose: url.searchParams.get('purpose'), path: url.searchParams.get('path'), frame: url.searchParams.get('frame') };
  };
  const imagesContext = () => makeContext({ image_model_defaults: { zit: makeImageDefaults({ max_reference_images: 4 }) } });

  it('shows the pictures of a start image and references outside the output folder', () => {
    const ctx = imagesContext();
    draft.setStartImage('image', { path: '/mnt/photos/start.png' }, ctx);
    draft.patch({ referenceImagePaths: ['/mnt/photos/jacket.png'] });
    open('images', ctx);

    const pictures = Array.from(target.querySelectorAll('img')).map((img) => asked(img));
    expect(pictures).toEqual([
      { route: '/api/browse/preview', purpose: 'reference_image', path: '/mnt/photos/start.png', frame: null },
      { route: '/api/browse/preview', purpose: 'reference_image', path: '/mnt/photos/jacket.png', frame: null },
    ]);
  });

  it('shows the last frame of a source video outside the output folder, from the server', () => {
    draft.setExtendSourcePath('/mnt/clips/walk.mp4');
    open('source', makeContext());

    const thumb = target.querySelector('[data-testid="slot-thumb"]')!;
    expect(asked(thumb.querySelector('img'))).toEqual({ route: '/api/browse/preview', purpose: 'source_video', path: '/mnt/clips/walk.mp4', frame: 'last' });
    expect(thumb.querySelector('video')).toBeNull();
  });

  it('keeps the film icon while no source video is chosen', () => {
    open('source', makeContext());

    const thumb = target.querySelector('[data-testid="slot-thumb"]')!;
    expect(thumb.querySelector('img')).toBeNull();
    expect(thumb.querySelector('svg')).not.toBeNull();
  });
});

describe('SettingPopover audio track', () => {
  function mountTrack(ctx: WorkspaceContext, onclose = vi.fn()): ReturnType<typeof vi.fn> {
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: { key: 'track', context: ctx, get images() { return currentImagesView(ctx); }, get audioTrack() { return currentAudioTrackView(ctx); }, busy: false, models: [], quantizeOptions: null, lastSeed: null, onModelChange: vi.fn(), onclose },
    }));
    return onclose;
  }

  it('closes itself in Image and on a model that follows no track', () => {
    draft.patch({ workflow: 'txt2img', model: 'zit' });
    const inImage = mountTrack(makeContext());
    flushSync();
    expect(inImage).toHaveBeenCalled();

    unmount(app!);
    app = null;
    const ctx = makeContext({ video_model_defaults: { 'ltx-8': makeVideoDefaults({ hidden_controls: ['audio_track', 'audio_track_path', 'audio_track_start'] }) } });
    draft.patch({ workflow: 'txt2vid', model: 'ltx-8' });
    const onModel = mountTrack(ctx);
    flushSync();
    expect(onModel).toHaveBeenCalled();
  });

  it('stays open on a model that follows a track, with Start at only once a track is set', () => {
    draft.patch({ workflow: 'txt2vid', model: 'ltx-8' });

    const onclose = mountTrack(makeContext());
    flushSync();

    expect(onclose).not.toHaveBeenCalled();
    expect(target.querySelector('input[aria-label="Start at, in seconds"]')).toBeNull();
    draft.setAudioTrack('video', { path: '/audio/song.mp3' });
    flushSync();
    expect(target.querySelector('input[aria-label="Start at, in seconds"]')).not.toBeNull();
  });

});
