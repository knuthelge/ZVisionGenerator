// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { memoryUploadStore } from '$lib/state/uploadStore';
import type { Workflow, WorkspaceContext } from '$lib/types';

import ImagesSection from './ImagesSection.svelte';
import { currentImagesView } from './imagesView';
import { chooseInBrowser, listingWith, placesFor } from '../../test-utils/pathBrowser';
import { makeContext, makeIdeogramDefaults, makeImageDefaults, makeVideoDefaults } from '../../test-utils/workspace';
import { addFiles, chooseSource, dragFiles, settle } from '../../test-utils/workspaceHarness';
import { browseApiMocks } from '../../test-utils/workspaceMocks';

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

const END_CONTROLS = ['end_image', 'end_image_clear', 'end_image_path'];

/** Image models without references (zit), with three (klein) and without image-to-image (ideo); video models with a last frame (h3) and without (ltx). */
function imagesContext(): WorkspaceContext {
  const context = makeContext({
    image_models: [{ id: 'zit', label: 'zit', type: 'image' }, { id: 'klein', label: 'klein', type: 'image' }, { id: 'ideo', label: 'ideo', type: 'image' }],
    image_model_defaults: { zit: makeImageDefaults(), klein: makeImageDefaults({ max_reference_images: 3 }), ideo: makeIdeogramDefaults() },
    video_models: [{ id: 'h3', label: 'h3', type: 'video' }, { id: 'ltx', label: 'ltx', type: 'video' }],
    current_video_model: 'h3',
    video_model_defaults: { h3: makeVideoDefaults(), ltx: makeVideoDefaults({ hidden_controls: END_CONTROLS }) },
  });
  for (const workflow of ['txt2img', 'img2img'] as const) context.workflow_contract.definitions[workflow].visible_controls.push('references', 'references_clear');
  return context;
}

describe('ImagesSection', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;
  let context: WorkspaceContext;

  const png = (name: string) => new File(['img'], name, { type: 'image/png' });
  const section = () => target.querySelector('[data-section="images"]');
  const summary = () => target.querySelector('.inspector-summary') as HTMLElement | null;
  const groups = () => Array.from(target.querySelectorAll('.image-group-label')).map((el) => el.textContent);
  const labels = () => Array.from(target.querySelectorAll('.inspector-label-text')).map((el) => el.textContent);
  const group = (name: string) => Array.from(target.querySelectorAll<HTMLElement>('.image-group')).find((el) => el.querySelector('.image-group-label')?.textContent === name);
  const labelsIn = (name: string) => Array.from(group(name)?.querySelectorAll('.inspector-label-text') ?? []).map((el) => el.textContent);
  const text = () => target.textContent ?? '';
  const button = (name: string) => target.querySelector(`button[aria-label="${name}"]`) as HTMLButtonElement | null;

  /** Open a tab on a model: the workflow and its model, hydrated from the context. */
  function open(workflow: Workflow, model: string): void {
    draft.update('workflow', workflow);
    draft.hydrateFromContext(context, model);
  }

  function show(): void {
    app = flushSync(() => mount(ImagesSection, { target, props: { context, get images() { return currentImagesView(context); }, busy: false } }));
  }

  beforeEach(async () => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
    galleryStore.reset();
    toasts.splice(0, toasts.length);
    browseApiMocks.browsePlaces.mockReset();
    browseApiMocks.listDirectory.mockReset();
    browseApiMocks.browsePlaces.mockResolvedValue(placesFor('reference_image'));
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test-image') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    context = imagesContext();
    await heldUploads.restore();
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
  });

  describe('the groups of a tab', () => {
    it('Image on a model without references is one Start image group with an empty Image row', () => {
      open('txt2img', 'zit');
      show();

      expect(summary()?.textContent).toBe('None');
      expect(groups()).toEqual(['Start image']);
      expect(labelsIn('Start image')).toEqual(['Image']);
      expect(text()).toContain('Browse or drop');
      expect(target.querySelector('.ui-btn-add')).toBeNull();
    });

    it('Image on a model with references adds a References group with an empty Reference 1 row', () => {
      open('txt2img', 'klein');
      show();

      expect(groups()).toEqual(['Start image', 'References']);
      expect(labelsIn('References')).toEqual(['Reference 1']);
      expect(text()).toContain('Add reference');
      // The hint waits for a reference to refer to.
      expect(text()).not.toContain('Refer to them in the prompt');
    });

    it('Video is a First frame group and a Last frame group, each one Image row', () => {
      open('txt2vid', 'h3');
      show();

      expect(groups()).toEqual(['First frame', 'Last frame']);
      expect(labelsIn('First frame')).toEqual(['Image']);
      expect(labelsIn('Last frame')).toEqual(['Image']);
    });

    it('names each group for its content, so a screen reader hears First frame and Last frame, not two Images', () => {
      open('txt2vid', 'h3');
      show();

      const named = Array.from(target.querySelectorAll<HTMLElement>('.image-group')).map((el) => document.getElementById(el.getAttribute('aria-labelledby') ?? '')?.textContent);
      expect(named).toEqual(['First frame', 'Last frame']);
    });

    it('shows the last-frame hint under the Last frame once it is set', () => {
      open('txt2vid', 'h3');
      draft.setStartImage('video', { path: '/in/first.png' }, context);
      show();
      expect(text()).not.toContain('The clip ends on this image');

      draft.setLastFrame({ path: '/in/last.png' });
      flushSync();
      expect(group('Last frame')?.textContent).toContain('The clip ends on this image; different scenes dissolve into each other.');
    });
  });

  describe('Strength', () => {
    it('is a row of the Start image group, only while a start image is set, with no caption', () => {
      open('txt2img', 'zit');
      show();
      expect(target.querySelector('#ws-image-strength')).toBeNull();

      draft.setStartImage('image', { path: '/in/start.png' }, context);
      flushSync();
      const field = target.querySelector('#ws-image-strength') as HTMLInputElement;
      expect(field).not.toBeNull();
      expect(field.hasAttribute('name')).toBe(false);
      expect(field.hasAttribute('aria-describedby')).toBe(false);
      expect(labelsIn('Start image')).toEqual(['Image', 'Strength']);
      expect(target.querySelector('.inspector-row.sub')).toBeNull();
    });

    it('changes the draft, and its reset puts back the model default', () => {
      open('txt2img', 'zit');
      draft.setStartImage('image', { path: '/in/start.png' }, context);
      show();
      const field = target.querySelector('#ws-image-strength') as HTMLInputElement;

      field.value = '0.3';
      field.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      expect(draft.state.referenceImageStrength).toBe(0.3);

      button('Reset Strength to default')!.click();
      flushSync();
      expect(draft.state.referenceImageStrength).toBe(0.5);
    });

    it('is not offered in Video', () => {
      open('txt2vid', 'h3');
      draft.setStartImage('video', { path: '/in/first.png' }, context);
      show();

      expect(target.querySelector('#ws-image-strength')).toBeNull();
    });
  });

  describe('an image the model cannot use', () => {
    it('shows no section at all on a model that takes no image', () => {
      open('txt2img', 'ideo');
      show();

      expect(section()).toBeNull();
    });

    it('hides the Last frame group on a video model without one, and shows it again on one that has it', () => {
      open('txt2vid', 'h3');
      draft.setStartImage('video', { path: '/in/first.png' }, context);
      draft.setLastFrame({ path: '/in/last.png' });
      show();
      expect(groups()).toEqual(['First frame', 'Last frame']);

      draft.hydrateFromContext(context, 'ltx');
      flushSync();
      expect(groups()).toEqual(['First frame']);
      expect(summary()?.textContent).toBe('first.png');
      expect(draft.state.endImagePath).toBe('/in/last.png');

      draft.hydrateFromContext(context, 'h3');
      flushSync();
      expect(groups()).toEqual(['First frame', 'Last frame']);
    });
  });

  describe('a last frame without a first frame', () => {
    it('is flagged unused, and the summary does not count it', () => {
      open('txt2vid', 'h3');
      draft.setLastFrame({ path: '/in/last.png' });
      show();

      expect(group('Last frame')?.querySelector('[data-unused]')?.getAttribute('data-unused')).toBe('true');
      expect(summary()?.textContent).toBe('None');
      expect(summary()?.dataset.tone).toBe('muted');
    });
  });

  describe('references', () => {
    it('are numbered Reference 1, 2, …, with one hint about naming them in the prompt', () => {
      open('txt2img', 'klein');
      draft.patch({ referenceImagePaths: ['/in/a.png', '/in/b.png'] });
      show();

      expect(labelsIn('References')).toEqual(['Reference 1', 'Reference 2', 'Reference 3']);
      expect(text().match(/Refer to them in the prompt as image 1, image 2…/g)).toHaveLength(1);
    });

    it('stop offering an empty row once the model\'s limit is reached', () => {
      open('txt2img', 'klein');
      draft.patch({ referenceImagePaths: ['/in/a.png', '/in/b.png', '/in/c.png'] });
      show();

      expect(labelsIn('References')).toEqual(['Reference 1', 'Reference 2', 'Reference 3']);
      expect(text()).not.toContain('Add reference');
    });
  });

  describe('a missing image', () => {
    it('marks a lost upload only after the stored uploads have been read back', async () => {
      open('txt2img', 'zit');
      draft.setStartImage('image', { file: png('start.png') }, context);
      heldUploads.useStore(memoryUploadStore());
      show();

      expect(heldUploads.restored).toBe(false);
      expect(summary()?.dataset.tone).toBe('active');
      expect(target.querySelector('#ws-image-error')).toBeNull();
      // The row draws from the stored marker meanwhile.
      expect(target.querySelector('[role="group"][aria-label="Image: start.png"]')).not.toBeNull();

      await heldUploads.restore();
      flushSync();
      expect(summary()?.textContent).toBe('start.png');
      expect(summary()?.dataset.tone).toBe('error');
      expect(target.querySelector('#ws-image-error')?.textContent).toBe('Choose this image again.');
    });

    it('marks a path the check flagged, the start image and a reference alike', () => {
      open('txt2img', 'klein');
      draft.setStartImage('image', { path: '/in/start.png' }, context);
      draft.patch({ referenceImagePaths: ['/in/a.png', '/in/gone.png'], missingReferencePaths: ['/in/gone.png'] });
      show();

      expect(summary()?.dataset.tone).toBe('error');
      expect(target.querySelector('#ws-image-error')).toBeNull();
      expect(target.querySelector('#ws-ref-2-error')?.textContent).toContain('moved or been deleted');

      draft.patch({ missingReferencePaths: ['/in/start.png'] });
      flushSync();
      expect(target.querySelector('#ws-ref-2-error')).toBeNull();
      expect(target.querySelector('#ws-image-error')?.textContent).toContain('moved or been deleted');
      expect(summary()?.dataset.tone).toBe('error');
    });

    it('marks the first and last frame in Video', () => {
      open('txt2vid', 'h3');
      draft.setStartImage('video', { path: '/in/first.png' }, context);
      draft.setLastFrame({ path: '/in/last.png' });
      draft.patch({ missingReferencePaths: ['/in/first.png', '/in/last.png'] });
      show();

      expect(target.querySelector('#ws-image-error')).not.toBeNull();
      expect(target.querySelector('#ws-end-image-error')).not.toBeNull();
      expect(summary()?.textContent).toBe('first.png → last.png');
      expect(summary()?.dataset.tone).toBe('error');
    });

  });

  describe('choosing an image', () => {
    it('sets the start image from a picked file through the draft, so the workflow follows', async () => {
      open('txt2img', 'zit');
      show();
      const file = png('start.png');

      await addFiles(target, 'start', file);

      expect(draft.state).toMatchObject({ workflow: 'img2img', referenceImageUpload: 'start.png', referenceImagePath: null });
      expect(heldUploads.start('image')).toBe(file);
      expect(summary()?.textContent).toBe('start.png');
    });

    it('sets the first and the last frame from picked files', async () => {
      open('txt2vid', 'h3');
      show();

      await addFiles(target, 'start', png('first.png'));
      await addFiles(target, 'last', png('last.png'));

      expect(draft.state).toMatchObject({ workflow: 'img2vid', firstFrameUpload: 'first.png', endImageUpload: 'last.png' });
      expect(heldUploads.lastFrame?.name).toBe('last.png');
    });

    it('sets the start image from the server browser', async () => {
      browseApiMocks.listDirectory.mockResolvedValue(listingWith('/pictures/cat.png'));
      open('txt2img', 'zit');
      show();

      await chooseSource(target, 'start');
      chooseInBrowser('cat.png');
      await settle();

      expect(draft.state).toMatchObject({ workflow: 'img2img', referenceImagePath: '/pictures/cat.png' });
    });

    it('forgets a held upload in the store when a server file replaces it, so it does not come back after a reload', async () => {
      browseApiMocks.listDirectory.mockResolvedValue(listingWith('/pictures/cat.png'));
      const store = memoryUploadStore();
      heldUploads.useStore(store);
      await heldUploads.restore();
      open('txt2vid', 'h3');
      show();
      await addFiles(target, 'start', png('first.png'));
      await addFiles(target, 'last', png('last.png'));
      await heldUploads.persisted();
      expect(await store.get('video')).toMatchObject({ start: { name: 'first.png' }, end: { name: 'last.png' } });

      await chooseSource(target, 'start');
      chooseInBrowser('cat.png');
      await settle();
      await chooseSource(target, 'last');
      chooseInBrowser('cat.png');
      await settle();
      await heldUploads.persisted();

      expect(draft.state).toMatchObject({ firstFramePath: '/pictures/cat.png', endImagePath: '/pictures/cat.png', firstFrameUpload: null, endImageUpload: null });
      expect(heldUploads.start('video')).toBeNull();
      expect(heldUploads.lastFrame).toBeNull();
      expect(await store.get('video')).toBeNull();
    });

    it('takes a file dropped on a slot, and turns a drop with no supported image away with a toast', async () => {
      open('txt2img', 'klein');
      show();
      const slot = target.querySelector('#ws-image-label')!.closest('.reference-drop') as HTMLElement;
      const file = png('dropped.png');

      const dropped = dragFiles(slot, 'drop', [file]);
      await settle();
      expect(dropped.defaultPrevented).toBe(true);
      expect(draft.state.referenceImageUpload).toBe('dropped.png');
      expect(heldUploads.start('image')).toBe(file);

      draft.setStartImage('image', null, context);
      flushSync();
      dragFiles(slot, 'drop', [new File(['x'], 'notes.txt', { type: 'text/plain' })]);
      await settle();
      expect(draft.state.referenceImageUpload).toBeNull();
      expect(toasts).toHaveLength(1);
    });

    it('appends several references in order from one pick, and the row after them stays open', async () => {
      open('txt2img', 'klein');
      show();

      await addFiles(target, 'reference', png('a.png'), png('b.png'));

      expect(draft.state.referenceImagePaths).toEqual(['', '']);
      expect(heldUploads.references.map((file) => file?.name)).toEqual(['a.png', 'b.png']);
      expect(labelsIn('References')).toEqual(['Reference 1', 'Reference 2', 'Reference 3']);
    });
  });

  describe('removing an image', () => {
    it('offers the clear button on a slot only while it holds an image', () => {
      open('txt2vid', 'h3');
      show();
      expect(button('Clear first frame')).toBeNull();
      expect(button('Clear last frame')).toBeNull();

      draft.setStartImage('video', { path: '/in/first.png' }, context);
      flushSync();
      expect(button('Clear first frame')).not.toBeNull();
      expect(button('Clear last frame')).toBeNull();

      draft.setLastFrame({ file: png('last.png') });
      flushSync();
      expect(button('Clear last frame')).not.toBeNull();

      draft.setStartImage('video', null, context);
      draft.setLastFrame(null);
      flushSync();
      expect(button('Clear first frame')).toBeNull();
      expect(button('Clear last frame')).toBeNull();
    });

    it('offers no clear button on an empty start image, and one once it holds an upload', async () => {
      open('txt2img', 'zit');
      show();
      expect(button('Clear start image')).toBeNull();

      await addFiles(target, 'start', png('start.png'));
      expect(button('Clear start image')).not.toBeNull();
    });

    it('empties the start image through the draft and leaves its row in place', async () => {
      open('txt2img', 'zit');
      draft.setStartImage('image', { path: '/in/start.png' }, context);
      show();

      button('Clear start image')!.click();
      await settle();

      expect(draft.state).toMatchObject({ workflow: 'txt2img', referenceImagePath: null });
      expect(groups()).toEqual(['Start image']);
      expect(target.querySelector('#ws-image-strength')).toBeNull();
    });

    it('keeps the last frame, greyed, when the first frame goes', async () => {
      open('txt2vid', 'h3');
      draft.setStartImage('video', { path: '/in/first.png' }, context);
      draft.setLastFrame({ path: '/in/last.png' });
      show();

      button('Clear first frame')!.click();
      await settle();

      expect(draft.state).toMatchObject({ workflow: 'txt2vid', firstFramePath: null, endImagePath: '/in/last.png' });
      expect(group('Last frame')?.querySelector('[data-unused]')?.getAttribute('data-unused')).toBe('true');
    });

    it('removes a reference, renumbers the rest and keeps focus in the list', async () => {
      open('txt2img', 'klein');
      draft.patch({ referenceImagePaths: ['/in/a.png', '/in/b.png'] });
      show();

      button('Remove reference 1')!.click();
      await settle();

      expect(draft.state.referenceImagePaths).toEqual(['/in/b.png']);
      expect(document.activeElement?.id).toBe('ws-ref-1-clear');
    });
  });
});
