// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { workspaceMemory } from '$lib/state/pageMemory';
import { memoryUploadStore } from '$lib/state/uploadStore';
import type { JobSnapshot, RecentPrompt, WorkflowMode, WorkspaceContext } from '$lib/types';
import { makeContext, makeIdeogramDefaults, makeImageDefaults } from '../../test-utils/workspace';

vi.mock('$lib/state/layoutMode.svelte', () => ({
  layout: { compact: true, mode: 'compact', coarse: true },
}));

const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
  parseUrlPrefill: vi.fn<() => Record<string, string>>(() => ({})),
  listJobs: vi.fn(async () => ({ active_job: null as JobSnapshot | null, queued_jobs: [] as JobSnapshot[] })),
}));

const recentPromptsMock = vi.hoisted(() => ({
  getRecentPrompts: vi.fn<(query: string, mode: WorkflowMode) => Promise<{ entries: RecentPrompt[] }>>(),
  clearRecentPrompts: vi.fn(),
}));

vi.mock('$lib/api/recentPrompts', () => recentPromptsMock);

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  return { ...actual, ...workspaceApiMocks };
});

import WorkspacePage from './WorkspacePage.svelte';

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  flushSync();
}

describe('WorkspacePage on a phone', () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    draft.reset();
    workspaceMemory.forget();
    galleryStore.reset();
    jobStore.clearJob();
    workspaceApiMocks.getWorkspaceCoreContext.mockReset();
    heldUploads.useStore(memoryUploadStore());
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
  });

  it('draws the settings strip and the composer before the context answers, as placeholders', async () => {
    draft.update('workflow', 'txt2vid');
    workspaceApiMocks.getWorkspaceCoreContext.mockImplementation(() => new Promise<WorkspaceContext>(() => undefined));

    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    const strip = target.querySelector('[data-testid="settings-strip"]') as HTMLElement | null;
    expect(strip).not.toBeNull();
    expect(strip!.getAttribute('aria-busy')).toBe('true');
    const chips = Array.from(strip!.querySelectorAll<HTMLButtonElement>('[data-chip]'));
    expect(chips.map((chip) => chip.dataset.chip)).toEqual(['preset', 'seconds', 'steps', 'guidance', 'seed', 'images', 'size', 'audio', 'loras', 'model']);
    expect(chips.every((chip) => chip.disabled)).toBe(true);
    expect((strip!.querySelector('[aria-label="All settings"]') as HTMLButtonElement).disabled).toBe(true);

    expect(target.querySelector('[data-testid="composer"]')).not.toBeNull();
    expect((target.querySelector('#ws-submit-compact') as HTMLButtonElement).disabled).toBe(true);
    expect((target.querySelector('.composer-summary') as HTMLButtonElement).disabled).toBe(true);
  });

  it('opens the recent prompts as a sheet, with the filter left unfocused so the keyboard stays down', async () => {
    recentPromptsMock.getRecentPrompts.mockResolvedValue({ entries: [{ format: 'text', prompt: 'A red fox', scene: '', negative_prompt: '', mode: 'image', used_at: Date.now() / 1000 - 60 }] });
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(makeContext());
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    (target.querySelector('#ws-recent-prompts') as HTMLButtonElement).click();
    await settle();

    const sheet = document.querySelector('[data-testid="sheet"] [role="dialog"][aria-label="Recent prompts"]');
    expect(sheet).not.toBeNull();
    expect(sheet!.querySelectorAll('[role="option"]')).toHaveLength(1);
    expect(document.activeElement).not.toBe(sheet!.querySelector('input'));

    (sheet!.querySelector('[role="option"]') as HTMLElement).click();
    await settle();
    expect(draft.state.prompt).toBe('A red fox');
  });

  describe('the Images chip', () => {
    const png = (name: string) => new File(['img'], name, { type: 'image/png' });
    const kleinContext = (): WorkspaceContext => makeContext({ image_model_defaults: { zit: makeImageDefaults({ max_reference_images: 4 }), 'flux-lite': makeImageDefaults() } });
    const chipKeys = () => Array.from(target.querySelectorAll<HTMLElement>('[data-chip]')).map((chip) => chip.dataset.chip);
    const imagesChip = () => target.querySelector('[data-chip="images"]') as HTMLButtonElement | null;
    const popover = () => target.querySelector('[data-testid="setting-popover"]') as HTMLElement | null;
    const groups = () => Array.from(popover()?.querySelectorAll('.pop-group-label') ?? []).map((el) => el.textContent);

    beforeEach(() => {
      // jsdom has no Web Animations API, which the sheet's and the popover's enter transitions call.
      Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
      Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test-image') });
      Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    });

    afterEach(() => {
      delete (Element.prototype as Partial<Element>).animate;
    });

    async function mountPhone(context: WorkspaceContext): Promise<void> {
      workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
      app = flushSync(() => mount(WorkspacePage, { target }));
      await settle();
    }

    function kleinWithImages(): WorkspaceContext {
      const context = kleinContext();
      draft.update('model', 'zit');
      draft.setStartImage('image', { path: '/in/pond.png' }, context);
      draft.update('referenceImagePaths', ['/in/lantern.png', '/in/fox.png']);
      draft.update('referenceImageStrength', 0.65);
      return context;
    }

    function videoWithFrames(): WorkspaceContext {
      draft.reset();
      draft.patch({ workflow: 'txt2vid', model: 'ltx-8' });
      const context = makeContext();
      draft.setStartImage('video', { path: '/in/dawn.png' }, context);
      draft.setLastFrame({ path: '/in/dusk.png' });
      return context;
    }

    it('offers images only through the Images chip, in either composer state', async () => {
      await mountPhone(kleinWithImages());

      const outsideImages = () => Array.from(target.querySelectorAll('input[type="file"]')).filter((el) => !el.closest('[data-section="images"]'));
      expect(outsideImages()).toEqual([]);
      (target.querySelector('.composer-summary') as HTMLButtonElement).click();
      await settle();
      expect(outsideImages()).toEqual([]);
    });

    it('is one chip for every image, between Batch and Size', async () => {
      await mountPhone(kleinWithImages());

      const keys = chipKeys();
      expect(keys.filter((key) => key === 'images')).toHaveLength(1);
      expect(keys.indexOf('images')).toBe(keys.indexOf('runs') + 1);
      expect(keys.indexOf('images')).toBe(keys.indexOf('size') - 1);
      expect(imagesChip()!.querySelector('svg')).not.toBeNull();
    });

    it('reads the section summary: the start image and references in Image, the frames in Video, None without any', async () => {
      await mountPhone(kleinWithImages());
      expect(imagesChip()!.querySelector('.setchip-v')?.textContent).toBe('pond.png · 2 references');
      expect(imagesChip()!.getAttribute('aria-label')).toBe('Images: pond.png · 2 references');

      await unmount(app!);
      app = null;
      await mountPhone(videoWithFrames());
      expect(imagesChip()!.querySelector('.setchip-v')?.textContent).toBe('dawn.png → dusk.png');

      await unmount(app!);
      app = null;
      draft.reset();
      draft.update('model', 'zit');
      await mountPhone(kleinContext());
      expect(imagesChip()!.querySelector('.setchip-v')?.textContent).toBe('None');
    });

    it('opens one popover with the groups of the Image tab, and a second tap closes it', async () => {
      await mountPhone(kleinWithImages());

      imagesChip()!.click();
      await settle();

      expect(popover()!.getAttribute('data-key')).toBe('images');
      expect(popover()!.getAttribute('aria-label')).toBe('Images');
      expect(imagesChip()!.getAttribute('aria-expanded')).toBe('true');
      expect(groups()).toEqual(['Start image', 'Strength', 'References']);
      expect(popover()!.querySelectorAll('.refrow')).toHaveLength(2);

      imagesChip()!.click();
      await settle();
      expect(popover()).toBeNull();
    });

    it('opens one popover with the groups of the Video tab', async () => {
      await mountPhone(videoWithFrames());

      imagesChip()!.click();
      await settle();

      expect(groups()).toEqual(['First frame', 'Last frame']);
      expect(popover()!.querySelector('input[aria-label="Strength"]')).toBeNull();
    });

    it('is hidden on a model that can use no image, and in Extend Video', async () => {
      const ideo = makeContext({
        image_models: [{ id: 'zit', label: 'zit', type: 'image' }, { id: 'ideo', label: 'ideo', type: 'image' }],
        image_model_defaults: { zit: makeImageDefaults({ max_reference_images: 4 }), ideo: makeIdeogramDefaults() },
      });
      draft.update('model', 'ideo');
      draft.setStartImage('image', { path: '/in/pond.png' }, null);
      draft.update('referenceImagePaths', ['/in/lantern.png']);
      await mountPhone(ideo);

      expect(imagesChip()).toBeNull();
      // The images stay in the draft for a model that can use them.
      expect(draft.state.referenceImagePath).toBe('/in/pond.png');

      await unmount(app!);
      app = null;
      const extendContext = makeContext();
      extendContext.workflow_contract.values = [...extendContext.workflow_contract.values, 'extend'];
      draft.reset();
      draft.update('workflow', 'extend');
      await mountPhone(extendContext);
      expect(imagesChip()).toBeNull();
    });

    it('a start image chosen in the popover shows on the chip at once, and the workflow follows', async () => {
      draft.update('model', 'zit');
      await mountPhone(kleinContext());

      imagesChip()!.click();
      await settle();
      const input = popover()!.querySelector('input[type="file"]') as HTMLInputElement;
      Object.defineProperty(input, 'files', { configurable: true, value: [png('start.png')] });
      input.dispatchEvent(new Event('change', { bubbles: true }));
      await settle();

      expect(draft.state.workflow).toBe('img2img');
      expect(imagesChip()!.querySelector('.setchip-v')?.textContent).toBe('start.png');
      expect(groups()).toEqual(['Start image', 'Strength', 'References']);
      expect(popover()).not.toBeNull();
    });

    it('removing a reference in the popover keeps it open', async () => {
      await mountPhone(kleinWithImages());
      imagesChip()!.click();
      await settle();

      (popover()!.querySelector('button[aria-label="Remove reference 1"]') as HTMLButtonElement).click();
      await settle();

      expect(draft.state.referenceImagePaths).toEqual(['/in/fox.png']);
      expect(popover()).not.toBeNull();
      expect(imagesChip()!.querySelector('.setchip-v')?.textContent).toBe('pond.png · 1 reference');
    });

    it('warns on the chip when an image the run uses is gone', async () => {
      const context = kleinWithImages();
      draft.update('missingReferencePaths', ['/in/fox.png']);
      await mountPhone(context);

      expect(imagesChip()!.querySelector('.setchip-warn')).not.toBeNull();
    });

    it('the All sheet shows the Images section third', async () => {
      await mountPhone(kleinWithImages());

      (target.querySelector('[aria-label="All settings"]') as HTMLButtonElement).click();
      await settle();

      const sections = Array.from(document.querySelectorAll('[data-testid="sheet"] .settings-pane [data-section]')).map((el) => (el as HTMLElement).dataset.section);
      expect(sections.slice(0, 3)).toEqual(['size', 'sampling', 'images']);
    });

    it('the All chip counts a changed Strength while the start image is used, as Reset does', async () => {
      await mountPhone(kleinWithImages());

      const count = (): number => Number(target.querySelector('[aria-label="All settings"] .setchip-n')?.textContent?.trim() ?? 0);
      const changedStrength = count();

      draft.update('referenceImageStrength', 0.5);
      await settle();
      const atDefault = count();
      expect(changedStrength).toBe(atDefault + 1);

      // Changed but with no start image it is not used, so it is not counted.
      draft.update('referenceImageStrength', 0.65);
      draft.setStartImage('image', null, kleinContext());
      await settle();
      expect(count()).toBe(atDefault);
    });
  });
});
