// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { downloadStore } from '$lib/state/downloads.svelte';
import type { ModelOption, WorkspaceContext } from '$lib/types';
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

  it('names the first-frame popover First frame only while the model has a last frame', () => {
    open('reference');
    const withSlot = popover().getAttribute('aria-label');
    void unmount(app!);
    app = null;
    target.replaceChildren();
    open('reference', ['end_image', 'end_image_path', 'end_image_clear']);

    expect(withSlot).toBe('First frame');
    expect(popover().getAttribute('aria-label')).toBe('Reference image');
  });
});
