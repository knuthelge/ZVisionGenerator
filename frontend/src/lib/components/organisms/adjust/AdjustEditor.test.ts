import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';

import { ApiError } from '$lib/api/client';
import { HOLD_MS } from '$lib/actions/hold';
import { cssFrameTransform } from '$lib/state/adjustGeometry';
import { toasts } from '$lib/state/toasts.svelte';
import type { GalleryAsset } from '$lib/types';
import { mountComponent, settle } from '../../../../test-utils/mount';
import { reactiveProps } from '../../../../test-utils/reactiveProps.svelte';
import AdjustEditor from './AdjustEditor.svelte';

const api = vi.hoisted(() => ({ saveAdjusted: vi.fn() }));
vi.mock('$lib/api/adjust', () => api);
const stores = vi.hoisted(() => ({
  addCopy: vi.fn(),
  replaceAsset: vi.fn(),
  replaceOutput: vi.fn(),
  refreshSlotsHolding: vi.fn(),
}));
vi.mock('$lib/state/gallery.svelte', () => ({ galleryStore: { addCopy: stores.addCopy, replaceAsset: stores.replaceAsset } }));
vi.mock('$lib/state/job.svelte', () => ({ jobStore: { replaceOutput: stores.replaceOutput } }));
vi.mock('$lib/state/draft.svelte', () => ({ draft: { refreshSlotsHolding: stores.refreshSlotsHolding } }));

// The layout mode follows the screen; a test picks one without a real media query.
// It is reactive, so a test can turn the screen while the editor is open. The proxy reads it when the layout is asked, never while the mock loads.
const screen = reactiveProps({ mode: null as 'wide' | 'tall' | 'compact' | null, short: false });
vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  return {
    ...actual,
    layout: new Proxy(actual.layout, {
      get: (target, key) => {
        if (screen.mode === null) return Reflect.get(target, key);
        if (key === 'mode') return screen.mode;
        if (key === 'compact') return screen.mode === 'compact';
        if (key === 'short') return screen.short;
        return Reflect.get(target, key);
      },
    }),
  };
});

const REPLACE_ASKED_KEY = 'ziv-adjust-replace-asked-v1';
/** The picture's upright size, and the room the stage gives it. */
const PICTURE = { width: 800, height: 600 };
const ROOM = { width: 640, height: 480 };

class FakeImage {
  src = '';
  naturalWidth = PICTURE.width;
  naturalHeight = PICTURE.height;
  decode(): Promise<void> {
    return Promise.resolve();
  }
}

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/a.png',
    url: '/media/out/a.png',
    thumbnail_url: '/media/out/a.png',
    filename: 'a.png',
    created_at: '2026-04-30T12:00:00Z',
    project: null,
    workflow: 'txt2img',
    prompt: 'A lake',
    model: 'zit',
    media_type: 'image',
    file_path: '/outputs/out/a.png',
    seed: 1,
    has_reusable_config: true,
    reuse_workspace_url: '#/workspace',
    ...overrides,
  };
}

const saved = makeAsset({ id: 'out/a_adjusted.png', filename: 'a_adjusted.png' });

function pointer(node: EventTarget, type: string, init: { pointerType?: string; x?: number; y?: number } = {}): void {
  const event = new MouseEvent(type, { bubbles: type !== 'pointerleave', cancelable: true, clientX: init.x ?? 10, clientY: init.y ?? 10, button: 0 });
  Object.defineProperty(event, 'pointerType', { value: init.pointerType ?? 'mouse' });
  Object.defineProperty(event, 'pointerId', { value: 1 });
  Object.defineProperty(event, 'isPrimary', { value: true });
  node.dispatchEvent(event);
  flushSync();
}

function key(type: 'keydown' | 'keyup', init: KeyboardEventInit, target: EventTarget = document.body): KeyboardEvent {
  const event = new KeyboardEvent(type, { bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(event);
  flushSync();
  return event;
}

const q = <T extends Element>(selector: string): T => document.querySelector<T>(selector)!;
const qa = <T extends Element>(selector: string): T[] => [...document.querySelectorAll<T>(selector)];
const compareButton = (): HTMLButtonElement => q('[data-testid="adjust-compare"]');
const original = (): Element | null => document.querySelector('[data-testid="adjust-original"]');
const ranges = (): HTMLInputElement[] => qa<HTMLInputElement>('input[type="range"]');
const sliderByLabel = (label: string): HTMLInputElement => ranges().find((input) => input.labels?.[0]?.textContent === label)!;
function button(label: string): HTMLButtonElement {
  const found = qa<HTMLButtonElement>('button').find((item) => item.textContent?.trim() === label);
  if (!found) throw new Error(`No button "${label}"`);
  return found;
}
/** A header of the side panel's accordion. */
const section = (label: string): HTMLButtonElement => qa<HTMLButtonElement>('h3 > button[aria-expanded]').find((item) => item.querySelector('span')?.textContent?.trim() === label)!;
const expandedSections = (): string[] => qa<HTMLButtonElement>('h3 > button[aria-expanded="true"]').map((item) => item.querySelector('span')?.textContent?.trim() ?? '');
const tab = (label: string): HTMLButtonElement => qa<HTMLButtonElement>('[role="tab"]').find((item) => item.textContent?.trim() === label)!;
const confirmButton = (action: 'confirm' | 'cancel'): HTMLButtonElement | null => document.querySelector(`[role="alertdialog"] [data-action="${action}"]`);

function setSlider(label: string, value: number): void {
  const input = sliderByLabel(label);
  input.value = String(value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

describe('AdjustEditor', () => {
  let onclose: ReturnType<typeof vi.fn>;
  let onsaved: ReturnType<typeof vi.fn>;
  const sizes = vi.spyOn(Element.prototype, 'clientWidth', 'get');
  const heights = vi.spyOn(Element.prototype, 'clientHeight', 'get');

  async function mountEditor(asset: GalleryAsset = makeAsset()): Promise<void> {
    mountComponent(AdjustEditor, { asset, onclose, onsaved });
    await settle();
  }

  beforeEach(() => {
    localStorage.clear();
    screen.mode = null;
    screen.short = false;
    toasts.splice(0, toasts.length);
    onclose = vi.fn();
    onsaved = vi.fn();
    api.saveAdjusted.mockReset().mockResolvedValue(saved);
    stores.addCopy.mockReset();
    stores.replaceAsset.mockReset();
    stores.replaceOutput.mockReset();
    stores.refreshSlotsHolding.mockReset();
    vi.stubGlobal('Image', FakeImage);
    sizes.mockReturnValue(ROOM.width);
    heights.mockReturnValue(ROOM.height);
    // jsdom has no WebGL2.
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  describe('opening', () => {
    it('is a modal dialog named for the file that owns its own drags, in the side layout', async () => {
      await mountEditor();
      const root = q<HTMLElement>('[role="dialog"]');
      expect(root.getAttribute('aria-modal')).toBe('true');
      expect(root.getAttribute('aria-label')).toBe('Adjust a.png');
      expect(root.hasAttribute('data-own-drag')).toBe(true);
      expect(root.dataset.layout).toBe('side');
    });

    it('opens on Light & colour with its header focused and seven centre-zero sliders at 0', async () => {
      await mountEditor();
      expect(expandedSections()).toEqual(['Light & colour']);
      expect(document.activeElement).toBe(section('Light & colour'));
      expect(ranges()).toHaveLength(7);
      for (const input of ranges()) {
        expect(input.min).toBe('-100');
        expect(input.max).toBe('100');
        expect(input.step).toBe('1');
        expect(input.value).toBe('0');
        expect(input.classList.contains('ui-range-centred')).toBe(true);
        expect(input.getAttribute('aria-valuetext')).toBe('0');
      }
      expect(ranges().map((input) => input.labels?.[0]?.textContent)).toEqual(['Exposure', 'Contrast', 'Highlights', 'Shadows', 'Saturation', 'Vibrance', 'Warmth']);
    });

    it('labels its controls exactly and shows the file name over the output size', async () => {
      await mountEditor();
      expect(q('.adjust-title b').textContent).toBe('a.png');
      expect(q('.adjust-title small').textContent).toBe('Adjust · 800 × 600');
      for (const label of ['Cancel', 'Replace', 'Save as copy', 'Reset all']) expect(button(label)).toBeTruthy();
      expect(compareButton().getAttribute('aria-label')).toBe('Compare with the original');
      expect(compareButton().textContent).toContain('Compare');
    });

    it('has the save buttons and Reset all off until something changes', async () => {
      await mountEditor();
      expect(button('Save as copy').disabled).toBe(true);
      expect(button('Replace').disabled).toBe(true);
      expect(button('Reset all').disabled).toBe(true);
      setSlider('Exposure', 10);
      expect(button('Save as copy').disabled).toBe(false);
      expect(button('Replace').disabled).toBe(false);
      expect(button('Reset all').disabled).toBe(false);
    });

    it('does nothing for a click in the gutter around the panel', async () => {
      await mountEditor();
      q<HTMLElement>('[role="dialog"]').click();
      expect(onclose).not.toHaveBeenCalled();
    });

    it('toasts and closes when the picture cannot be opened', async () => {
      class Broken extends FakeImage {
        decode(): Promise<void> {
          return Promise.reject(new Error('bad'));
        }
      }
      vi.stubGlobal('Image', Broken);
      await mountEditor();
      expect(onclose).toHaveBeenCalledOnce();
      expect(toasts.map((toast) => toast.message)).toEqual(['Could not open the picture']);
    });

    it('wraps Tab from the last control to the first and back', async () => {
      await mountEditor();
      const focusable = qa<HTMLElement>('[role="dialog"] button:not([disabled]), [role="dialog"] input:not([disabled]), [role="dialog"] [tabindex]:not([tabindex="-1"])');
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      last.focus();
      expect(key('keydown', { key: 'Tab' }, last).defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(first);
      expect(key('keydown', { key: 'Tab', shiftKey: true }, first).defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(last);
    });
  });

  describe('sliders', () => {
    it('shows the signed value and a signed aria-valuetext', async () => {
      await mountEditor();
      setSlider('Exposure', 24);
      setSlider('Shadows', -8);
      expect(sliderByLabel('Exposure').getAttribute('aria-valuetext')).toBe('+24');
      expect(sliderByLabel('Shadows').getAttribute('aria-valuetext')).toBe('-8');
      const values = qa('.adj-value').map((node) => node.textContent);
      expect(values).toEqual(['+24', '0', '0', '−8', '0', '0', '0']);
    });

    it('sets a slider back to 0 on a double click', async () => {
      await mountEditor();
      setSlider('Contrast', 40);
      sliderByLabel('Contrast').dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
      flushSync();
      expect(sliderByLabel('Contrast').value).toBe('0');
    });

    it('sets a slider back to 0 on two pointerups within 300 ms and 10 px, and not on two far apart', async () => {
      vi.useFakeTimers();
      await mountEditor();
      const input = sliderByLabel('Warmth');
      setSlider('Warmth', 30);
      pointer(input, 'pointerup', { x: 100 });
      vi.advanceTimersByTime(200);
      pointer(input, 'pointerup', { x: 104 });
      expect(input.value).toBe('0');

      setSlider('Warmth', 30);
      pointer(input, 'pointerup', { x: 100 });
      vi.advanceTimersByTime(200);
      pointer(input, 'pointerup', { x: 160 });
      expect(input.value).toBe('30');
      vi.advanceTimersByTime(400);
      pointer(input, 'pointerup', { x: 160 });
      expect(input.value).toBe('30');
    });

    it('sets every slider back to neutral with Reset all, which is then disabled', async () => {
      await mountEditor();
      setSlider('Exposure', 50);
      setSlider('Warmth', -20);
      button('Reset all').click();
      flushSync();
      expect(ranges().map((input) => input.value)).toEqual(['0', '0', '0', '0', '0', '0', '0']);
      expect(button('Reset all').disabled).toBe(true);
    });
  });

  describe('sections', () => {
    const SECTION_KEY = 'ziv-adjust-section-v1';
    const headers = (): HTMLButtonElement[] => qa<HTMLButtonElement>('h3 > button[aria-expanded]');
    const regionOf = (header: HTMLElement): HTMLElement | null => document.getElementById(header.getAttribute('aria-controls')!);
    const names = (): string[] => qa<HTMLElement>('.adjust-sections h3 > button > span:first-of-type').map((item) => item.textContent?.trim() ?? '');

    it('has no tabs: three disclosure headers in h3s, Crop, Light & colour and Colour mixer, each controlling a labelled region', async () => {
      await mountEditor();
      expect(document.querySelector('[role="tablist"]')).toBeNull();
      expect(document.querySelector('[role="tab"]')).toBeNull();
      expect(names()).toEqual(['Crop', 'Light & colour', 'Colour mixer']);
      expect(headers()).toHaveLength(3);
      for (const header of headers()) {
        expect(header.getAttribute('aria-controls')).toBeTruthy();
        const region = regionOf(header)!;
        expect(region.getAttribute('role')).toBe('region');
        expect(region.getAttribute('aria-labelledby')).toBe(header.id);
      }
      expect(headers().map((header) => header.getAttribute('aria-expanded'))).toEqual(['false', 'true', 'false']);
      expect(regionOf(section('Crop'))!.hidden).toBe(true);
      expect(regionOf(section('Light & colour'))!.hidden).toBe(false);
    });

    it('names each header with its summary', async () => {
      await mountEditor();
      expect(section('Crop').getAttribute('aria-label')).toBe('Crop, Full picture');
      expect(section('Light & colour').getAttribute('aria-label')).toBe('Light & colour, No changes');
      expect(section('Colour mixer').getAttribute('aria-label')).toBe('Colour mixer, No changes');
    });

    it('opens one section at a time: opening Crop closes Light & colour', async () => {
      await mountEditor();
      section('Crop').click();
      flushSync();
      expect(expandedSections()).toEqual(['Crop']);
      section('Colour mixer').click();
      flushSync();
      expect(expandedSections()).toEqual(['Colour mixer']);
      expect(ranges()).toHaveLength(8);
    });

    it('collapses the open section on its header, leaving all three closed with only the headers rendered', async () => {
      await mountEditor();
      section('Light & colour').click();
      flushSync();
      expect(expandedSections()).toEqual([]);
      expect(ranges()).toHaveLength(0);
      for (const header of headers()) {
        expect(regionOf(header)!.hidden).toBe(true);
        expect(regionOf(header)!.childElementCount).toBe(0);
      }
      expect(button('Save as copy')).toBeTruthy();
    });

    it('keeps the focus on the header it toggles', async () => {
      await mountEditor();
      for (const label of ['Colour mixer', 'Crop', 'Crop']) {
        section(label).focus();
        section(label).click();
        flushSync();
        expect(document.activeElement).toBe(section(label));
      }
    });

    it.each(['crop', 'light', 'mixer', 'none'] as const)('remembers %s across a remount, in ziv-adjust-section-v1', async (state) => {
      const first = mountComponent(AdjustEditor, { asset: makeAsset(), onclose, onsaved });
      await settle();
      // Light & colour is where it starts, so it is reached by leaving and coming back; none by closing it.
      const clicks = { crop: ['Crop'], light: ['Crop', 'Light & colour'], mixer: ['Colour mixer'], none: ['Light & colour'] }[state];
      for (const label of clicks) {
        section(label).click();
        flushSync();
      }
      expect(localStorage.getItem(SECTION_KEY)).toBe(state);
      await first.unmount();
      await mountEditor();
      const expected = { crop: ['Crop'], light: ['Light & colour'], mixer: ['Colour mixer'], none: [] }[state];
      expect(expandedSections()).toEqual(expected);
    });

    it('opens on Light & colour when the stored value is not one of the four', async () => {
      localStorage.setItem(SECTION_KEY, 'colour');
      await mountEditor();
      expect(expandedSections()).toEqual(['Light & colour']);
    });

    it('shows the crop frame only while Crop is open', async () => {
      await mountEditor();
      expect(document.querySelector('[data-testid="adjust-crop-box"]')).toBeNull();
      section('Crop').click();
      flushSync();
      expect(q('[data-testid="adjust-crop-box"]')).toBeTruthy();
      expect(q('[data-testid="adjust-announce"]').textContent).toBe('Crop frame shown');
      section('Colour mixer').click();
      flushSync();
      expect(document.querySelector('[data-testid="adjust-crop-box"]')).toBeNull();
      expect(q('[data-testid="adjust-announce"]').textContent).toBe('Crop frame hidden');
    });

    it('shows the cropped picture when Crop is closed and the whole frame when it is open, for a compare too', async () => {
      await mountEditor();
      section('Crop').click();
      flushSync();
      button('1:1').click();
      flushSync();
      compareButton().click();
      flushSync();
      // Crop open: the untouched full frame, with no crop box.
      const whole = q<HTMLImageElement>('[data-testid="adjust-fallback"]').style.transform;
      expect(document.querySelector('[data-testid="adjust-crop-box"]')).toBeNull();
      section('Crop').click();
      flushSync();
      // Crop closed: the original colours at the current crop.
      expect(original()).not.toBeNull();
      expect(q<HTMLImageElement>('[data-testid="adjust-fallback"]').style.transform).not.toBe(whole);
    });

    it('summarises each section in its header', async () => {
      await mountEditor();
      setSlider('Exposure', 10);
      section('Crop').click();
      flushSync();
      button('4:5').click();
      flushSync();
      expect(section('Crop').textContent).toContain('4:5 · 480 × 600');
      expect(section('Light & colour').textContent).toContain('1 changed');
      section('Colour mixer').click();
      flushSync();
      setSlider('Blue', 30);
      expect(section('Colour mixer').textContent).toContain('Blue +30');
      setSlider('Red', -10);
      expect(section('Colour mixer').textContent).toContain('2 colours changed');
    });

    it('tabs through the controls in the order: Compare, Reset all, crop frame, the headers with their controls, then the actions', async () => {
      await mountEditor();
      setSlider('Exposure', 10);
      section('Crop').click();
      flushSync();
      const order = (): string[] =>
        qa<HTMLElement>('[role="dialog"] button:not([disabled]), [role="dialog"] input:not([disabled]), [role="dialog"] [tabindex]:not([tabindex="-1"])')
          .filter((el) => !el.closest('[inert], [hidden]'))
          .map((el) => el.getAttribute('data-testid') === 'adjust-crop-box' ? 'frame' : el.matches('h3 > button') ? `header:${el.querySelector('span')?.textContent}` : el.matches('input') ? `slider:${el.labels?.[0]?.textContent}` : (el.getAttribute('aria-label') ?? el.textContent?.trim() ?? ''));
      const head = ['Compare with the original', 'Reset all', 'frame', 'header:Crop'];
      const tail = ['Cancel', 'Replace', 'Save as copy'];
      const crop = order();
      expect(crop.slice(0, 4)).toEqual(head);
      expect(crop.slice(-3)).toEqual(tail);
      expect(crop.slice(-5, -3)).toEqual(['header:Light & colour', 'header:Colour mixer']);
      section('Light & colour').click();
      flushSync();
      const light = order();
      expect(light.slice(0, 2)).toEqual(['Compare with the original', 'Reset all']);
      expect(light.slice(2)).toEqual([
        'header:Crop', 'header:Light & colour', 'slider:Exposure', 'slider:Contrast', 'slider:Highlights', 'slider:Shadows', 'slider:Saturation', 'slider:Vibrance', 'slider:Warmth',
        'header:Colour mixer', ...tail,
      ]);
      section('Colour mixer').click();
      flushSync();
      expect(order().slice(2)).toEqual([
        'header:Crop', 'header:Light & colour', 'header:Colour mixer', 'slider:Red', 'slider:Orange', 'slider:Yellow', 'slider:Green', 'slider:Aqua', 'slider:Blue', 'slider:Purple', 'slider:Magenta', ...tail,
      ]);
    });
  });

  describe('colour mixer', () => {
    const BANDS = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta'];

    it('has eight band sliders from -100 to +100 in steps of 1, with a swatch each, at 0', async () => {
      await mountEditor();
      section('Colour mixer').click();
      flushSync();
      expect(ranges().map((input) => input.labels?.[0]?.textContent)).toEqual(BANDS);
      for (const input of ranges()) {
        expect(input.min).toBe('-100');
        expect(input.max).toBe('100');
        expect(input.step).toBe('1');
        expect(input.value).toBe('0');
        expect(input.classList.contains('ui-range-hue')).toBe(true);
        expect(input.classList.contains('ui-range-centred')).toBe(false);
      }
      expect(ranges()[5].style.getPropertyValue('--band')).toBe('var(--color-band-blue)');
      expect(qa('.adj-row .adj-swatch')).toHaveLength(8);
      expect(q('.adjust-sections').textContent).toContain('Saturation per colour. Double-click a slider to set it back to 0.');
    });

    it('sends the bands in the request, sets one back to 0 on a double click, and Reset all zeroes them', async () => {
      await mountEditor();
      section('Colour mixer').click();
      flushSync();
      setSlider('Orange', 40);
      setSlider('Blue', -25);
      expect(button('Save as copy').disabled).toBe(false);
      sliderByLabel('Blue').dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
      flushSync();
      expect(sliderByLabel('Blue').value).toBe('0');
      button('Save as copy').click();
      await settle();
      expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ mix_orange: 40, mix_blue: 0, mix_red: 0 }));
    });

    it('counts a band alone as a change and Reset all sets it back', async () => {
      await mountEditor();
      section('Colour mixer').click();
      flushSync();
      setSlider('Aqua', 10);
      expect(button('Reset all').disabled).toBe(false);
      button('Reset all').click();
      flushSync();
      expect(sliderByLabel('Aqua').value).toBe('0');
      expect(button('Save as copy').disabled).toBe(true);
    });
  });

  describe('compare', () => {
    it('turns on with a click and stays until the next click', async () => {
      await mountEditor();
      expect(original()).toBeNull();
      pointer(compareButton(), 'pointerdown');
      pointer(compareButton(), 'pointerup');
      compareButton().click();
      expect(compareButton().getAttribute('aria-pressed')).toBe('true');
      expect(original()).not.toBeNull();
      expect(q('[data-testid="adjust-announce"]').textContent).toBe('Showing the original');
      pointer(compareButton(), 'pointerdown');
      pointer(compareButton(), 'pointerup');
      compareButton().click();
      expect(compareButton().getAttribute('aria-pressed')).toBe('false');
      expect(original()).toBeNull();
    });

    it('toggles on Enter or Space, which click the button without a pointer', async () => {
      await mountEditor();
      compareButton().click();
      flushSync();
      expect(compareButton().getAttribute('aria-pressed')).toBe('true');
    });

    it('turns off when a slider moves, and on any other edit', async () => {
      await mountEditor();
      compareButton().click();
      flushSync();
      setSlider('Exposure', 5);
      expect(original()).toBeNull();
      expect(compareButton().getAttribute('aria-pressed')).toBe('false');
      compareButton().click();
      flushSync();
      button('Reset all').click();
      flushSync();
      expect(original()).toBeNull();
    });

    it('shows the original only while the button is held past the hold time', async () => {
      vi.useFakeTimers();
      await mountEditor();
      pointer(compareButton(), 'pointerdown');
      vi.advanceTimersByTime(HOLD_MS - 10);
      flushSync();
      expect(original()).toBeNull();
      vi.advanceTimersByTime(20);
      flushSync();
      expect(original()).not.toBeNull();
      expect(compareButton().getAttribute('aria-pressed')).toBe('true');
      pointer(compareButton(), 'pointerup');
      compareButton().click();
      expect(original()).toBeNull();
      expect(compareButton().getAttribute('aria-pressed')).toBe('false');
    });

    it('ends a held button when the pointer leaves, without toggling', async () => {
      vi.useFakeTimers();
      await mountEditor();
      pointer(compareButton(), 'pointerdown');
      vi.advanceTimersByTime(HOLD_MS + 10);
      flushSync();
      expect(original()).not.toBeNull();
      pointer(compareButton(), 'pointerleave');
      expect(original()).toBeNull();
      pointer(compareButton(), 'pointerdown');
      pointer(compareButton(), 'pointerleave');
      vi.advanceTimersByTime(HOLD_MS + 10);
      flushSync();
      expect(original()).toBeNull();
    });

    it('shows the original while M is held with a range focused, and hides it on the keyup', async () => {
      await mountEditor();
      const range = sliderByLabel('Exposure');
      range.focus();
      key('keydown', { key: 'm', code: 'KeyM' }, range);
      expect(original()).not.toBeNull();
      key('keyup', { key: 'm', code: 'KeyM' }, range);
      expect(original()).toBeNull();
      expect(compareButton().getAttribute('aria-pressed')).toBe('false');
    });

    it('ignores a repeating M and changes nothing', async () => {
      await mountEditor();
      key('keydown', { key: 'm', code: 'KeyM' });
      key('keydown', { key: 'm', code: 'KeyM', repeat: true });
      expect(original()).not.toBeNull();
      key('keyup', { key: 'm', code: 'KeyM' });
      expect(original()).toBeNull();
      key('keydown', { key: 'M', code: 'KeyM', repeat: true });
      expect(original()).toBeNull();
    });

    it('ends a Shift+M hold when Shift is released before the M', async () => {
      await mountEditor();
      key('keydown', { key: 'M', code: 'KeyM', shiftKey: true });
      expect(original()).not.toBeNull();
      key('keyup', { key: 'Shift', code: 'ShiftLeft' });
      expect(original()).toBeNull();
    });

    it.each([['ctrlKey'], ['metaKey'], ['altKey']])('does nothing for M with %s', async (modifier) => {
      await mountEditor();
      key('keydown', { key: 'm', code: 'KeyM', [modifier]: true });
      expect(original()).toBeNull();
    });

    it('leaves M to a text field', async () => {
      await mountEditor();
      const field = document.createElement('input');
      field.type = 'text';
      document.body.append(field);
      key('keydown', { key: 'm', code: 'KeyM' }, field);
      expect(original()).toBeNull();
    });

    it('hides a held original when the window loses focus', async () => {
      await mountEditor();
      key('keydown', { key: 'm', code: 'KeyM' });
      expect(original()).not.toBeNull();
      window.dispatchEvent(new Event('blur'));
      flushSync();
      expect(original()).toBeNull();
    });

    it('shows the original on a long press on the picture on Light & colour, until the pointer lifts', async () => {
      vi.useFakeTimers();
      await mountEditor();
      const stage = q<HTMLElement>('.stage-fit');
      pointer(stage, 'pointerdown', { pointerType: 'touch' });
      vi.advanceTimersByTime(HOLD_MS + 10);
      flushSync();
      expect(original()).not.toBeNull();
      pointer(stage, 'pointerup', { pointerType: 'touch' });
      expect(original()).toBeNull();
    });

    it('does not take a long press on the crop box on the Crop tab for a compare', async () => {
      vi.useFakeTimers();
      await mountEditor();
      section('Crop').click();
      flushSync();
      const box = q<HTMLElement>('[data-testid="adjust-crop-box"]');
      pointer(box, 'pointerdown', { pointerType: 'touch' });
      vi.advanceTimersByTime(HOLD_MS + 10);
      flushSync();
      expect(original()).toBeNull();
      pointer(box, 'pointerup', { pointerType: 'touch' });
      // Off the crop box the same press compares: the picture is the whole frame there.
      pointer(q<HTMLElement>('.stage-fit'), 'pointerdown', { pointerType: 'touch' });
      vi.advanceTimersByTime(HOLD_MS + 10);
      flushSync();
      expect(original()).not.toBeNull();
    });

    it('keeps the current crop and turn on Light & colour, and drops the crop frame on Crop', async () => {
      await mountEditor();
      section('Crop').click();
      flushSync();
      button('1:1').click();
      flushSync();
      expect(q('[data-testid="adjust-crop-box"]')).toBeTruthy();
      compareButton().click();
      flushSync();
      expect(original()).not.toBeNull();
      expect(document.querySelector('[data-testid="adjust-crop-box"]')).toBeNull();
      section('Light & colour').click();
      flushSync();
      // The original is still the cropped square.
      const fallback = q<HTMLImageElement>('[data-testid="adjust-fallback"]');
      expect(fallback.style.transform).toBe(
        cssFrameTransform({ rotate: 0, flip: false, crop: { x: 100, y: 0, width: 600, height: 600 } }, PICTURE.width, PICTURE.height, { width: 480, height: 480 }, 'crop').transform,
      );
    });
  });

  describe('cancel', () => {
    it('closes on Escape and on Cancel when nothing changed', async () => {
      await mountEditor();
      key('keydown', { key: 'Escape' });
      await settle();
      expect(onclose).toHaveBeenCalledTimes(1);
      button('Cancel').click();
      await settle();
      expect(onclose).toHaveBeenCalledTimes(2);
    });

    it('asks before discarding changes, and Keep editing leaves the editor open', async () => {
      await mountEditor();
      setSlider('Exposure', 10);
      key('keydown', { key: 'Escape' });
      await settle();
      expect(q('[role="alertdialog"]').textContent).toContain('Discard your changes?');
      expect(confirmButton('confirm')!.textContent).toContain('Discard');
      expect(confirmButton('cancel')!.textContent).toContain('Keep editing');
      confirmButton('cancel')!.click();
      await settle();
      expect(onclose).not.toHaveBeenCalled();
      expect(document.querySelector('[role="alertdialog"]')).toBeNull();
      expect(sliderByLabel('Exposure').value).toBe('10');
    });

    it('closes only the question on a second Escape', async () => {
      await mountEditor();
      setSlider('Exposure', 10);
      button('Cancel').click();
      await settle();
      expect(document.querySelector('[role="alertdialog"]')).not.toBeNull();
      key('keydown', { key: 'Escape' });
      await settle();
      expect(document.querySelector('[role="alertdialog"]')).toBeNull();
      expect(onclose).not.toHaveBeenCalled();
    });

    it('closes after Discard', async () => {
      await mountEditor();
      setSlider('Exposure', 10);
      button('Cancel').click();
      await settle();
      confirmButton('confirm')!.click();
      await settle();
      expect(onclose).toHaveBeenCalledOnce();
      expect(api.saveAdjusted).not.toHaveBeenCalled();
    });
  });

  describe('saving', () => {
    it('posts the adjustment as a copy, reports it and closes with no toast', async () => {
      await mountEditor();
      setSlider('Exposure', 12);
      button('Save as copy').click();
      await settle();
      expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ asset_id: 'out/a.png', save: 'copy', exposure: 12, rotate: 0, flip: false, crop: null }));
      expect(stores.addCopy).toHaveBeenCalledWith(saved, 'out/a.png');
      expect(onsaved).toHaveBeenCalledWith(saved, false);
      expect(onclose).toHaveBeenCalledOnce();
      expect(toasts).toHaveLength(0);
    });

    it('is busy while the request runs: controls inert, the pressed button saying Saving…, and Escape ignored', async () => {
      let finish: (asset: GalleryAsset) => void = () => {};
      api.saveAdjusted.mockReturnValue(new Promise<GalleryAsset>((resolve) => { finish = resolve; }));
      await mountEditor();
      setSlider('Exposure', 12);
      button('Save as copy').click();
      await settle();
      const pressed = button('Saving…');
      expect(pressed.getAttribute('aria-busy')).toBe('true');
      expect(pressed.disabled).toBe(false);
      expect(pressed.textContent).toContain('Saving…');
      expect(q<HTMLElement>('.adjust-controls').inert).toBe(true);
      expect(q<HTMLElement>('.adjust-sections').inert).toBe(true);
      key('keydown', { key: 'Escape' });
      button('Cancel').click();
      await settle();
      expect(document.querySelector('[role="alertdialog"]')).toBeNull();
      expect(onclose).not.toHaveBeenCalled();
      pressed.click();
      await settle();
      expect(api.saveAdjusted).toHaveBeenCalledTimes(1);
      finish(saved);
      await settle();
      expect(onclose).toHaveBeenCalledOnce();
    });

    it('stays open with its values and a toast without an action after a 422', async () => {
      api.saveAdjusted.mockRejectedValue(new ApiError('bad', 422, 'The crop is outside the picture.'));
      await mountEditor();
      setSlider('Exposure', 12);
      button('Save as copy').click();
      await settle();
      expect(onclose).not.toHaveBeenCalled();
      expect(onsaved).not.toHaveBeenCalled();
      expect(sliderByLabel('Exposure').value).toBe('12');
      expect(toasts).toHaveLength(1);
      expect(toasts[0].message).toBe('Adjust failed: The crop is outside the picture.');
      expect(toasts[0].action).toBeUndefined();
      expect(button('Save as copy').getAttribute('aria-busy')).toBeNull();
    });

    it('stays open with its values and no toast action after a 500, and a second press saves', async () => {
      api.saveAdjusted.mockRejectedValueOnce(new ApiError('bad', 500, 'Could not save a.png: disk full'));
      await mountEditor();
      setSlider('Exposure', 12);
      button('Save as copy').click();
      await settle();
      expect(toasts).toHaveLength(1);
      expect(toasts[0].message).toBe('Adjust failed: Could not save a.png: disk full');
      expect(toasts[0].action).toBeUndefined();
      expect(onclose).not.toHaveBeenCalled();
      expect(onsaved).not.toHaveBeenCalled();
      expect(sliderByLabel('Exposure').value).toBe('12');

      api.saveAdjusted.mockResolvedValueOnce(saved);
      button('Save as copy').click();
      await settle();
      expect(api.saveAdjusted).toHaveBeenCalledTimes(2);
      expect(api.saveAdjusted.mock.calls[1][0]).toEqual(api.saveAdjusted.mock.calls[0][0]);
      expect(onsaved).toHaveBeenCalledOnce();
      expect(onclose).toHaveBeenCalledOnce();
    });

    it('asks the first time it replaces, and never again', async () => {
      await mountEditor();
      setSlider('Exposure', 12);
      button('Replace').click();
      await settle();
      expect(q('[role="alertdialog"]').textContent).toContain('Replace a.png?');
      expect(q('[role="alertdialog"]').textContent).toContain("The original can't be restored.");
      expect(confirmButton('confirm')!.textContent).toContain('Replace');
      expect(api.saveAdjusted).not.toHaveBeenCalled();
      confirmButton('confirm')!.click();
      await settle();
      expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ save: 'replace' }));
      expect(localStorage.getItem(REPLACE_ASKED_KEY)).toBe('1');
      expect(stores.replaceAsset).toHaveBeenCalledWith(saved);
      expect(onsaved).toHaveBeenCalledWith(saved, true);
    });

    it('does not replace when the question is declined', async () => {
      await mountEditor();
      setSlider('Exposure', 12);
      button('Replace').click();
      await settle();
      confirmButton('cancel')!.click();
      await settle();
      expect(api.saveAdjusted).not.toHaveBeenCalled();
      expect(localStorage.getItem(REPLACE_ASKED_KEY)).toBeNull();
    });

    it('replaces without asking once it has been answered', async () => {
      localStorage.setItem(REPLACE_ASKED_KEY, '1');
      await mountEditor();
      setSlider('Exposure', 12);
      button('Replace').click();
      await settle();
      expect(document.querySelector('[role="alertdialog"]')).toBeNull();
      expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ save: 'replace' }));
    });

    it.each([
      ['photo.jpg', 'JPEGs are saved as a PNG copy'],
      ['photo.webp', 'WebP images are saved as a PNG copy'],
    ])('shows a muted line instead of Replace for %s', async (filename, line) => {
      await mountEditor(makeAsset({ filename }));
      expect(qa('button').some((item) => item.textContent?.trim() === 'Replace')).toBe(false);
      expect(q('.adjust-type-note').textContent).toBe(line);
      setSlider('Exposure', 12);
      button('Save as copy').click();
      await settle();
      expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ save: 'copy' }));
    });

    it('does not show the muted line on a PNG', async () => {
      await mountEditor();
      expect(document.querySelector('.adjust-type-note')).toBeNull();
    });
  });

  describe('crop', () => {
    it('has eight shape chips with Swap last, and no number fields', async () => {
      await mountEditor();
      section('Crop').click();
      flushSync();
      const chips = qa<HTMLButtonElement>('.adj-shapes button.ui-chip');
      expect(chips.map((chip) => chip.textContent?.trim())).toEqual(['Free', 'Original', '1:1', '4:5', '3:2', '16:9', '9:16', 'Swap']);
      expect(chips.slice(0, 7).every((chip) => chip.hasAttribute('aria-pressed'))).toBe(true);
      expect(chips[0].getAttribute('aria-pressed')).toBe('true');
      expect(chips[7].disabled).toBe(true);
      expect(document.querySelector('input[type="number"]')).toBeNull();
      expect(document.querySelector('input[type="range"]')).toBeNull();
      for (const label of ['Rotate left', 'Rotate right', 'Flip']) expect(button(label)).toBeTruthy();
      expect(q('[data-testid="adjust-crop-size"]').textContent).toBe('Crop 800 × 600 px');
    });

    it('fits the largest box of a shape, centred, and enables Swap and relabels the swapped chip', async () => {
      await mountEditor();
      section('Crop').click();
      flushSync();
      button('4:5').click();
      flushSync();
      expect(button('4:5').getAttribute('aria-pressed')).toBe('true');
      expect(q('.adjust-title small').textContent).toBe('Adjust · 480 × 600');
      const swap = qa<HTMLButtonElement>('.adj-shapes button.swap')[0];
      expect(swap.disabled).toBe(false);
      swap.click();
      flushSync();
      expect(button('5:4').getAttribute('aria-pressed')).toBe('true');
      expect(q('.adjust-title small').textContent).toBe('Adjust · 750 × 600');
    });

    it('moves the focused crop box with the arrow keys, by 10 times that with Shift, and stops at the edge', async () => {
      await mountEditor();
      section('Crop').click();
      flushSync();
      button('1:1').click();
      flushSync();
      const box = q<HTMLElement>('[data-testid="adjust-crop-box"]');
      expect(box.getAttribute('role')).toBe('group');
      expect(box.getAttribute('aria-label')).toBe('Crop frame');
      box.focus();
      // The square starts centred at x = 100; the step is max(1, round(600 / 100)) = 6.
      expect(key('keydown', { key: 'ArrowRight' }, box).defaultPrevented).toBe(true);
      key('keydown', { key: 'ArrowRight', shiftKey: true }, box);
      key('keydown', { key: 'ArrowRight', shiftKey: true }, box);
      key('keydown', { key: 'ArrowRight', shiftKey: true }, box);
      button('Save as copy').click();
      await settle();
      // 100 + 6 + 3 * 60 would pass the right edge, so it stops at 200.
      expect(api.saveAdjusted.mock.calls[0][0].crop).toEqual({ x: 200, y: 0, width: 600, height: 600 });
    });

    it('announces the crop size 300 ms after a change', async () => {
      vi.useFakeTimers();
      await mountEditor();
      section('Crop').click();
      flushSync();
      expect(q('[data-testid="adjust-announce"]').textContent).toBe('Crop frame shown');
      button('Rotate right').click();
      flushSync();
      await settle(299);
      expect(q('[data-testid="adjust-announce"]').textContent).toBe('Crop frame shown');
      await settle(2);
      expect(q('[data-testid="adjust-announce"]').textContent).toBe('Crop 600 × 800');
    });

    it('swaps the readout on a rotate, carries a locked shape with it and sends the rotation', async () => {
      await mountEditor();
      section('Crop').click();
      flushSync();
      button('3:2').click();
      flushSync();
      button('Rotate right').click();
      flushSync();
      expect(q('.adjust-title small').textContent).toBe('Adjust · 533 × 800');
      expect(button('2:3').getAttribute('aria-pressed')).toBe('true');
      button('Flip').click();
      flushSync();
      button('Save as copy').click();
      await settle();
      const body = api.saveAdjusted.mock.calls[0][0];
      expect(body.rotate).toBe(90);
      expect(body.flip).toBe(true);
      expect(body.crop).toEqual({ x: 34, y: 0, width: 533, height: 800 });
    });

    it('counts a crop that covers the whole frame as no change', async () => {
      await mountEditor();
      section('Crop').click();
      flushSync();
      button('Original').click();
      flushSync();
      expect(button('Save as copy').disabled).toBe(true);
    });
  });

  describe('without WebGL2', () => {
    it('shows the notice and the picture turned by CSS, and still saves', async () => {
      await mountEditor();
      expect(q('.adjust-notice').textContent).toContain("Live preview isn't available in this browser; saving still applies your changes.");
      const fallback = q<HTMLImageElement>('[data-testid="adjust-fallback"]');
      const expected = cssFrameTransform({ rotate: 0, flip: false, crop: null }, PICTURE.width, PICTURE.height, { width: 640, height: 480 }, 'crop');
      expect(fallback.style.transform).toBe(expected.transform);
      section('Crop').click();
      flushSync();
      button('Rotate right').click();
      button('Rotate right').click();
      flushSync();
      const turned = q<HTMLImageElement>('[data-testid="adjust-fallback"]');
      expect(turned.style.transform).toBe(cssFrameTransform({ rotate: 180, flip: false, crop: null }, PICTURE.width, PICTURE.height, { width: 640, height: 480 }, 'frame').transform);
      button('Save as copy').click();
      await settle();
      expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ rotate: 180 }));
    });

    it('prevents the default of a lost WebGL context so the browser can restore it', async () => {
      await mountEditor();
      const event = new Event('webglcontextlost', { cancelable: true });
      q<HTMLCanvasElement>('.pic-canvas').dispatchEvent(event);
      expect(event.defaultPrevented).toBe(true);
    });
  });

  describe('layouts', () => {
    const setScreen = (mode: 'wide' | 'tall' | 'compact', short = false): void => {
      screen.mode = mode;
      screen.short = short;
    };
    const layoutOf = (): string => q<HTMLElement>('[role="dialog"]').dataset.layout!;
    const inTopBar = (label: string): boolean => qa<HTMLButtonElement>('.adjust-bar button').some((item) => item.textContent?.includes(label));
    const inDockLine = (label: string): boolean => qa<HTMLButtonElement>('.dock-line button').some((item) => item.textContent?.includes(label));
    const menuItems = (): HTMLElement[] => qa<HTMLElement>('[role="menu"] [role="menuitem"]');
    function setRange(index: number, value: number): void {
      const input = ranges()[index];
      input.value = String(value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
    }

    it.each([
      ['wide', false, 'side'],
      ['tall', false, 'bottom'],
      ['compact', false, 'phone'],
      ['compact', true, 'phone-side'],
    ] as const)('uses the %s screen (short: %s) as the %s layout', async (mode, short, expected) => {
      setScreen(mode, short);
      await mountEditor();
      expect(layoutOf()).toBe(expected);
    });

    it('keeps the top bar with Reset all on the side and bottom layouts', async () => {
      for (const mode of ['wide', 'tall'] as const) {
        setScreen(mode);
        await mountEditor();
        expect(inTopBar('Reset all')).toBe(true);
        expect(inTopBar('Compare')).toBe(true);
        expect(document.querySelector('.dock-line')).toBeNull();
        document.body.replaceChildren();
      }
    });

    it('puts Reset all in the dock header on the phone layouts, which have no side top bar', async () => {
      setScreen('compact');
      await mountEditor();
      expect(document.querySelector('.adjust-bar')).toBeNull();
      expect(inDockLine('Reset all')).toBe(true);
      expect(q('.phone-top')).toBeTruthy();
    });

    it('has no top bar at all on a phone held sideways, and the actions head the column', async () => {
      setScreen('compact', true);
      await mountEditor();
      expect(document.querySelector('.adjust-bar')).toBeNull();
      expect(document.querySelector('.phone-top')).toBeNull();
      expect(inDockLine('Reset all')).toBe(true);
      expect(qa('.phone-actions button').map((item) => item.getAttribute('aria-label') ?? item.textContent?.trim())).toEqual(['Cancel', 'Compare with the original', 'Save']);
    });

    it('keeps the picture mounted when the screen turns', async () => {
      setScreen('compact');
      await mountEditor();
      const picture = q('[data-testid="adjust-picture"]');
      screen.short = true;
      flushSync();
      await settle();
      expect(layoutOf()).toBe('phone-side');
      expect(q('[data-testid="adjust-picture"]')).toBe(picture);
    });

    describe('tablet portrait', () => {
      it('shares the first row of the dock between the tabs and the buttons, with seven sliders on Light & colour', async () => {
        setScreen('tall');
        await mountEditor();
        const row = q('.dock-row');
        expect(row.querySelector('[role="tablist"]')).toBeTruthy();
        expect(['Cancel', 'Replace', 'Save as copy'].every((label) => [...row.querySelectorAll('button')].some((item) => item.textContent?.trim() === label))).toBe(true);
        expect(ranges()).toHaveLength(7);
        expect(q('.dock-body').querySelector('.dock-sliders')).toBeTruthy();
      });

      it('lays the eight chips in one row, then the three buttons with the size', async () => {
        setScreen('tall');
        await mountEditor();
        tab('Crop').click();
        flushSync();
        expect(qa('.adj-shapes button.ui-chip')).toHaveLength(8);
        expect(q('.dock-body').textContent).toContain('Crop 800 × 600 px');
        for (const label of ['Rotate left', 'Rotate right', 'Flip']) expect(button(label)).toBeTruthy();
      });

      it('has three tabs with icons, Crop, Light & colour and Colour mixer, opening on Light & colour', async () => {
        setScreen('tall');
        await mountEditor();
        expect(qa('[role="tab"]').map((item) => item.textContent?.trim())).toEqual(['Crop', 'Light & colour', 'Colour mixer']);
        expect(qa('[role="tab"]').every((item) => item.querySelector('svg'))).toBe(true);
        expect(tab('Light & colour').getAttribute('aria-selected')).toBe('true');
        expect(document.querySelector('h3 > button[aria-expanded]')).toBeNull();
      });

      it('shows the eight bands in two columns of four on Colour mixer', async () => {
        setScreen('tall');
        await mountEditor();
        tab('Colour mixer').click();
        flushSync();
        const grid = q<HTMLElement>('.dock-body .dock-sliders');
        const labels = [...grid.querySelectorAll('input[type="range"]')].map((input) => (input as HTMLInputElement).labels?.[0]?.textContent);
        // The grid's four rows fill down first (a stylesheet rule jsdom does not load): Red to Green, then Aqua to Magenta.
        expect(labels).toEqual(['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']);
        expect(grid.children).toHaveLength(8);
      });

      it('puts the JPEG line in the row in place of Replace', async () => {
        setScreen('tall');
        await mountEditor(makeAsset({ filename: 'photo.jpg' }));
        expect(q('.dock-row .adjust-type-note').textContent).toBe('JPEGs are saved as a PNG copy');
        expect(qa('button').some((item) => item.textContent?.trim() === 'Replace')).toBe(false);
      });
    });

    describe('phone', () => {
      beforeEach(() => setScreen('compact'));

      it('shows exactly one range, and picking another tool swaps it', async () => {
        await mountEditor();
        // The panels of the other tabs are kept, hidden, so the dock never changes height.
        const shown = (): HTMLInputElement[] => ranges().filter((input) => !input.closest('[data-off]'));
        expect(shown()).toHaveLength(1);
        expect(shown()[0].getAttribute('aria-label')).toBe('Exposure');
        expect(q('.dock-line').textContent).toContain('Exposure');
        const tools = qa<HTMLButtonElement>('[role="group"][aria-label="Light & colour tools"] button');
        expect(tools).toHaveLength(7);
        expect(tools.map((item) => item.getAttribute('aria-pressed'))).toEqual(['true', 'false', 'false', 'false', 'false', 'false', 'false']);
        tools[3].click();
        flushSync();
        expect(shown()).toHaveLength(1);
        expect(shown()[0].getAttribute('aria-label')).toBe('Shadows');
        expect(q('.dock-line').textContent).toContain('Shadows');
        expect(qa<HTMLButtonElement>('[aria-label^="Shadows"]')[0].getAttribute('aria-pressed')).toBe('true');
      });

      it('lists Vibrance between Saturation and Warmth and scrolls the selected dial into view', async () => {
        const scrollIntoView = vi.fn();
        Element.prototype.scrollIntoView = scrollIntoView;
        await mountEditor();
        const names = qa<HTMLButtonElement>('[role="group"][aria-label="Light & colour tools"] button').map((item) => item.getAttribute('aria-label'));
        expect(names.slice(4)).toEqual(['Saturation 0', 'Vibrance 0', 'Warmth 0']);
        scrollIntoView.mockClear();
        qa<HTMLButtonElement>('[aria-label^="Vibrance"]')[0].click();
        flushSync();
        await settle();
        expect(scrollIntoView).toHaveBeenLastCalledWith({ inline: 'nearest', block: 'nearest' });
        expect(scrollIntoView.mock.contexts.at(-1)).toBe(q('[aria-label="Vibrance 0"]'));
        expect(ranges()[0].getAttribute('aria-label')).toBe('Vibrance');
      });

      it('names each tool with its value, draws its arc, and edits the selected tool', async () => {
        await mountEditor();
        expect(q('[aria-label="Exposure 0"]').querySelector('svg.adj-ring')).toBeNull();
        setRange(0, 24);
        expect(q('[aria-label="Exposure +24"]').querySelector('svg.adj-ring')).toBeTruthy();
        expect(q('.dock-line .adj-value').textContent).toBe('+24');
        qa<HTMLButtonElement>('[aria-label^="Contrast"]')[0].click();
        flushSync();
        setRange(0, -40);
        expect(q('[aria-label="Contrast -40"]').querySelector('svg.adj-ring')?.hasAttribute('data-neg')).toBe(true);
        expect(q('[aria-label="Exposure +24"]')).toBeTruthy();
        expect(q('.dock-line .adj-value').textContent).toBe('−40');
      });

      it('asks Save as copy first and Replace original second on a PNG', async () => {
        await mountEditor();
        setRange(0, 12);
        const save = button('Save');
        expect(save.getAttribute('aria-haspopup')).toBe('menu');
        save.click();
        await settle();
        expect(menuItems().map((item) => item.querySelector('.action-menu-name')?.textContent)).toEqual(['Save as copy', 'Replace original']);
        expect(menuItems()[0].textContent).toContain('Adds an adjusted copy; the original stays');
        expect(menuItems()[1].textContent).toContain("Overwrites a.png. It can't be restored.");
        menuItems()[0].click();
        await settle();
        expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ save: 'copy', exposure: 12 }));
        expect(onclose).toHaveBeenCalledOnce();
      });

      it('replaces through the menu after the first question', async () => {
        await mountEditor();
        setRange(0, 12);
        button('Save').click();
        await settle();
        menuItems()[1].click();
        await settle();
        expect(q('[role="alertdialog"]').textContent).toContain('Replace a.png?');
        confirmButton('confirm')!.click();
        await settle();
        expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ save: 'replace' }));
      });

      it('keeps Save off until something changed', async () => {
        await mountEditor();
        expect(button('Save').disabled).toBe(true);
        setRange(0, 5);
        expect(button('Save').disabled).toBe(false);
      });

      it('gives a JPEG a direct Save as copy with the note and no menu', async () => {
        await mountEditor(makeAsset({ filename: 'photo.jpg' }));
        setRange(0, 12);
        const save = button('Save as copy');
        expect(save.hasAttribute('aria-haspopup')).toBe(false);
        expect(q('.phone-note').textContent).toBe('JPEGs are saved as a PNG copy');
        save.click();
        await settle();
        expect(menuItems()).toHaveLength(0);
        expect(api.saveAdjusted).toHaveBeenCalledWith(expect.objectContaining({ save: 'copy' }));
      });

      it('has three tabs without icons, opening on Light & colour', async () => {
        await mountEditor();
        expect(qa('[role="tab"]').map((item) => item.textContent?.trim())).toEqual(['Crop', 'Light & colour', 'Colour mixer']);
        expect(qa('[role="tab"]').some((item) => item.querySelector('svg'))).toBe(false);
        expect(tab('Light & colour').getAttribute('aria-selected')).toBe('true');
      });

      it('shows eight swatch dials and one large slider on Colour mixer, and picking a band swaps the slider and scrolls it into view', async () => {
        const scrollIntoView = vi.fn();
        Element.prototype.scrollIntoView = scrollIntoView;
        await mountEditor();
        tab('Colour mixer').click();
        flushSync();
        await settle();
        const row = q('[role="group"][aria-label="Colour mixer"]');
        const dials = [...row.querySelectorAll<HTMLButtonElement>('button')];
        expect(dials.map((item) => item.getAttribute('aria-label'))).toEqual(['Red 0', 'Orange 0', 'Yellow 0', 'Green 0', 'Aqua 0', 'Blue 0', 'Purple 0', 'Magenta 0']);
        expect(dials.map((item) => item.getAttribute('aria-pressed'))).toEqual(['true', ...Array(7).fill('false')]);
        expect(dials[5].querySelector<HTMLElement>('.adj-swatch')!.style.getPropertyValue('--band')).toBe('var(--color-band-blue)');
        const shown = (): HTMLInputElement[] => ranges().filter((input) => !input.closest('[data-off]'));
        expect(shown()).toHaveLength(1);
        expect(shown()[0].getAttribute('aria-label')).toBe('Red saturation');
        scrollIntoView.mockClear();
        dials[5].click();
        flushSync();
        await settle();
        expect(scrollIntoView).toHaveBeenLastCalledWith({ inline: 'nearest', block: 'nearest' });
        expect(scrollIntoView.mock.contexts.at(-1)).toBe(q('[aria-label="Blue 0"]'));
        expect(shown()[0].getAttribute('aria-label')).toBe('Blue saturation');
        expect(shown()[0].classList.contains('ui-range-hue')).toBe(true);
        expect(q('.dock-line').textContent).toContain('Blue');
      });

      it('edits the selected band, draws its value arc and names it with its value', async () => {
        await mountEditor();
        tab('Colour mixer').click();
        flushSync();
        qa<HTMLButtonElement>('[aria-label^="Blue"]')[0].click();
        flushSync();
        const big = ranges().find((input) => input.getAttribute('aria-label') === 'Blue saturation')!;
        big.value = '30';
        big.dispatchEvent(new Event('input', { bubbles: true }));
        flushSync();
        expect(q('[aria-label="Blue +30"]').querySelector('svg.adj-ring')).toBeTruthy();
        expect(q('.dock-line').textContent).toContain('+30');
        expect(q('.dock-line').textContent).toContain('Reset all');
      });

      it('compares from an icon button', async () => {
        await mountEditor();
        const compareIcon = compareButton();
        expect(compareIcon.getAttribute('aria-label')).toBe('Compare with the original');
        expect(compareIcon.textContent?.trim()).toBe('');
        pointer(compareIcon, 'pointerdown', { pointerType: 'touch' });
        pointer(compareIcon, 'pointerup', { pointerType: 'touch' });
        expect(compareIcon.getAttribute('aria-pressed')).toBe('true');
        expect(original()).not.toBeNull();
      });

      it('shows eight chips with Swap last and the three buttons on Crop, and the size in the header', async () => {
        await mountEditor();
        tab('Crop').click();
        flushSync();
        const panel = q('[role="tabpanel"]:not([data-off])');
        const chips = [...panel.querySelectorAll<HTMLButtonElement>('button.ui-chip')];
        expect(chips.map((chip) => chip.textContent?.trim())).toEqual(['Free', 'Original', '1:1', '4:5', '3:2', '16:9', '9:16', 'Swap']);
        for (const label of ['Rotate left', 'Rotate right', 'Flip']) expect([...panel.querySelectorAll('button')].some((item) => item.textContent?.trim() === label)).toBe(true);
        expect(q('.dock-line').textContent).toContain('Crop');
        expect(q('[data-testid="adjust-crop-size"]').textContent).toBe('800 × 600');
      });

      it('keeps the dock in one grid cell across the tabs: the idle panel is hidden, inert and kept', async () => {
        await mountEditor();
        const cell = q('.dock-panels');
        const panels = (): HTMLElement[] => qa<HTMLElement>('.dock-panel');
        expect(panels()).toHaveLength(3);
        const before = panels();
        expect(before.every((panel) => panel.parentElement === cell && panel.getAttribute('role') === 'tabpanel')).toBe(true);
        const idleIds = (): string[] => panels().filter((panel) => panel.hasAttribute('data-off')).map((panel) => panel.id.replace(/^.*-panel-/, ''));
        expect(idleIds()).toEqual(['crop', 'mixer']);
        for (const idle of panels().filter((panel) => panel.hasAttribute('data-off'))) {
          expect(idle.inert).toBe(true);
          expect(idle.getAttribute('aria-hidden')).toBe('true');
        }
        tab('Crop').click();
        flushSync();
        expect(panels()).toEqual(before);
        expect(idleIds()).toEqual(['light', 'mixer']);
        expect(panels().find((panel) => !panel.hasAttribute('data-off'))!.inert).toBe(false);
      });

      it('makes the whole dock inert while saving, with Save saying Saving…', async () => {
        let finish: (asset: GalleryAsset) => void = () => {};
        api.saveAdjusted.mockReturnValue(new Promise<GalleryAsset>((resolve) => { finish = resolve; }));
        await mountEditor(makeAsset({ filename: 'photo.jpg' }));
        setRange(0, 12);
        button('Save as copy').click();
        await settle();
        expect(q<HTMLElement>('.phone-dock').inert).toBe(true);
        const pressed = button('Saving…');
        expect(pressed.getAttribute('aria-busy')).toBe('true');
        finish(saved);
        await settle();
      });
    });

    describe('phone held sideways', () => {
      beforeEach(() => setScreen('compact', true));

      it('keeps the tool names for screen readers and shows one range', async () => {
        await mountEditor();
        expect(ranges()).toHaveLength(1);
        const dial = q<HTMLElement>('[aria-label="Exposure 0"]');
        expect(dial.querySelector<HTMLElement>('.adj-tool-name')!.hasAttribute('data-hidden')).toBe(true);
      });

      it('shows the chips and three named round buttons on Crop', async () => {
        await mountEditor();
        tab('Crop').click();
        flushSync();
        expect(qa('.phone-col .adj-shapes button.ui-chip')).toHaveLength(8);
        expect(qa('.phone-col .adj-tools button').map((item) => item.textContent?.trim())).toEqual(['Rotate left', 'Rotate right', 'Flip']);
        button('Rotate right').click();
        flushSync();
        expect(q('[data-testid="adjust-crop-size"]').textContent).toBe('600 × 800');
      });

      it('has three tabs without icons and a swatch row with hidden names on Colour mixer', async () => {
        await mountEditor();
        expect(qa('[role="tab"]').map((item) => item.textContent?.trim())).toEqual(['Crop', 'Light & colour', 'Colour mixer']);
        tab('Colour mixer').click();
        flushSync();
        const dials = qa<HTMLButtonElement>('.phone-col [role="group"][aria-label="Colour mixer"] button');
        expect(dials).toHaveLength(8);
        expect(dials.every((item) => item.querySelector('.adj-tool-name')!.hasAttribute('data-hidden'))).toBe(true);
        expect(ranges()).toHaveLength(1);
        expect(ranges()[0].getAttribute('aria-label')).toBe('Red saturation');
      });

      it('saves from the same Save menu', async () => {
        await mountEditor();
        setRange(0, 12);
        button('Save').click();
        await settle();
        expect(menuItems()).toHaveLength(2);
      });
    });
  });
});
