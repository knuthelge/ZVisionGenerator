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

describe('SettingPopover model list', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    draft.update('model', 'zit');
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
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

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

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    draft.update('model', 'zit');
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

  function mountLoras(ctx: WorkspaceContext = loraContext): void {
    app = flushSync(() => mount(SettingPopover, {
      target,
      props: {
        key: 'loras', context: ctx, busy: false, models, quantizeOptions: null, imageFile: null, referencePreviewUrl: null, lastSeed: null,
        onImageFileChange: vi.fn(), onModelChange: vi.fn(), onUseLatestAsReference: vi.fn(), onclose: vi.fn(),
      },
    }));
  }

  const addRows = () => Array.from(target.querySelectorAll('.pop-list .pop-item')).map((el) => el.textContent?.trim() ?? '');
  const fold = () => target.querySelector('[data-action="other-models"]') as HTMLButtonElement;

  it('lists only the matches by name, then a collapsed Other models row', () => {
    mountLoras();

    expect(addRows().slice(0, 2)).toEqual(['a-match', 'b-match']);
    expect(fold().getAttribute('aria-expanded')).toBe('false');
    expect(target.textContent).not.toContain('d-ltx');
    expect(target.textContent).not.toContain('c-unknown');
  });

  it('expands the fold in place and shows the family label beside each folded LoRA', () => {
    mountLoras();
    flushSync(() => fold().click());

    expect(fold().getAttribute('aria-expanded')).toBe('true');
    const row = Array.from(target.querySelectorAll('.pop-item')).find((el) => el.textContent?.includes('d-ltx'));
    expect(row?.textContent).toContain('LTX');
    const unknown = Array.from(target.querySelectorAll('.pop-item')).find((el) => el.textContent?.includes('c-unknown'));
    expect(unknown?.querySelector('small')).toBeNull();
  });

  it('starts with the fold open when nothing matches', () => {
    mountLoras({ ...loraContext, loras: [{ name: 'd-ltx', path: '/l/d', family: 'ltx', size_variant: null }] } as WorkspaceContext);

    expect(fold().getAttribute('aria-expanded')).toBe('true');
    expect(target.textContent).toContain('d-ltx');
  });

  it('adds the chosen LoRA to the draft', () => {
    mountLoras();
    (Array.from(target.querySelectorAll('.pop-item')).find((el) => el.textContent?.includes('b-match')) as HTMLButtonElement).click();

    expect(draft.state.loraString).toBe('b-match:1');
  });

  it('shows no fold row when every LoRA fits the model', () => {
    mountLoras({ ...loraContext, loras: loraContext.loras.slice(0, 2) } as WorkspaceContext);

    expect(fold()).toBeNull();
  });
});

