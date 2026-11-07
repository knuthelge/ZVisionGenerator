// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AUDIO_MATRIX } from '../../../test-utils/audio';
import { axesOf } from '$lib/state/audioEnhance';
import type { AudioEnhanceSettings, EnhanceAxis, EnhanceSettings } from '$lib/types';

import EnhanceOptions from './EnhanceOptions.svelte';
import Host from './EnhanceOptionsHost.test.svelte';

const axes: EnhanceAxis[] = [
  { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep', description: '', group: null }, { slug: 'photo', label: 'Photo', description: '', group: null }], default: ['keep'] },
  { key: 'details', label: 'Details', multi: true, video_only: false, audio_only: false, options: [{ slug: 'lighting', label: 'Lighting', description: '', group: null }, { slug: 'composition', label: 'Composition', description: '', group: null }], default: [] },
];

describe('EnhanceOptions', () => {
  let app: Record<string, unknown> | null = null;
  const target = document.createElement('div');

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.innerHTML = '';
  });

  function mountOptions(settings: EnhanceSettings, onchange = vi.fn(), extra: Record<string, unknown> = {}): typeof onchange {
    document.body.appendChild(target);
    app = flushSync(() => mount(EnhanceOptions, { target, props: { axes, settings, mode: 'image', onchange, ...extra } }));
    return onchange;
  }

  const BASE: EnhanceSettings = { style: 'keep', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' };
  const videoAxes: EnhanceAxis[] = [
    ...axes,
    { key: 'motion', label: 'Motion', multi: true, video_only: true, audio_only: false, options: [{ slug: 'action', label: 'Action', description: '', group: null }], default: [] },
    { key: 'sound', label: 'Sound', multi: true, video_only: true, audio_only: true, options: [{ slug: 'ambient', label: 'Ambience', description: '', group: null }], default: [] },
  ];

  it.each([
    ['image', false, false, false],
    ['video', false, true, false],
    ['video', true, true, true],
  ])('mode %s with audio %s shows motion %s and sound %s', (mode, audio, motion, sound) => {
    mountOptions(BASE, vi.fn(), { axes: videoAxes, mode, audio });
    expect(target.querySelector('[data-toggle-all="motion"]') !== null).toBe(motion);
    expect(target.querySelector('[data-toggle-all="sound"]') !== null).toBe(sound);
  });

  it('reports a typed direction', () => {
    const onchange = mountOptions({ ...BASE, note: 'at' }, vi.fn(), { idPrefix: 't' });
    const input = target.querySelector('#t-note') as HTMLInputElement;
    expect(input.value).toBe('at');
    input.value = 'at night';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ note: 'at night' }));
  });

  it('offers select-all only on multi-pick axes and selects every option', () => {
    const onchange = mountOptions({ style: 'keep', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: ['lighting'], length: 'same', motion: [], sound: [], note: '' });
    expect(target.querySelector('[data-toggle-all="style"]')).toBeNull();
    const toggle = target.querySelector('[data-toggle-all="details"]') as HTMLButtonElement;
    expect(toggle.dataset.allSelected).toBe('false');
    toggle.click();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ details: ['lighting', 'composition'] }));
  });

  it('clears every option once all are selected', () => {
    const onchange = mountOptions({ style: 'keep', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: ['lighting', 'composition'], length: 'same', motion: [], sound: [], note: '' });
    const toggle = target.querySelector('[data-toggle-all="details"]') as HTMLButtonElement;
    expect(toggle.dataset.allSelected).toBe('true');
    toggle.click();
    expect(onchange).toHaveBeenCalledWith(expect.objectContaining({ details: [] }));
  });

  describe('Framing and Color', () => {
    const opt = (slug: string, label: string) => ({ slug, label, description: '', group: null });
    const single = (key: EnhanceAxis['key'], label: string, options: EnhanceAxis['options']): EnhanceAxis => ({ key, label, multi: false, video_only: false, audio_only: false, options, default: ['keep'] });
    const cameraAxes: EnhanceAxis[] = [
      ...axes,
      single('shot', 'Shot', [opt('keep', 'Keep'), opt('wide', 'Wide')]),
      single('angle', 'Angle', [opt('keep', 'Keep'), opt('low', 'Low')]),
      single('palette', 'Palette', [opt('keep', 'Keep'), opt('earth', 'Earth tones'), opt('custom', 'Custom')]),
    ];
    // Two own colours as plain strings the test never renders as literal CSS.
    const OWN = ['#' + '112233', '#' + '445566'];

    function mountHost(initial: EnhanceSettings, onchange = vi.fn()): typeof onchange {
      document.body.appendChild(target);
      app = flushSync(() => mount(Host, { target, props: { axes: cameraAxes, initial, onchange } }));
      return onchange;
    }

    const headings = (): string[] => Array.from(target.querySelectorAll('h3')).map((h) => h.textContent?.trim() ?? '');
    const pick = (id: string, slug: string): void => {
      flushSync(() => (target.querySelector(`#enhance-${id}`) as HTMLButtonElement).click());
      flushSync(() => (document.querySelector(`[role="option"][data-slug="${slug}"]`) as HTMLButtonElement).click());
    };

    it('shows no Framing or Color without those axes, and orders them after Look otherwise', () => {
      mountOptions(BASE);
      expect(headings()).toEqual(['Look', 'Output']);
      expect(target.querySelector('section[aria-label="Framing"]')).toBeNull();
      expect(target.querySelector('section[aria-label="Color"]')).toBeNull();
    });

    it('lists Look, Framing, Color, then Focus on', () => {
      mountOptions(BASE, vi.fn(), { axes: [...cameraAxes] });
      expect(headings()).toEqual(['Look', 'Framing', 'Color', 'Output']);
      const labels = Array.from(target.querySelectorAll('.enhance-section')).map((section) => section.getAttribute('aria-label') ?? section.querySelector('.enhance-subtitle')?.textContent?.trim().split(' · ')[0]);
      expect(labels.slice(0, 4)).toEqual(['Look', 'Framing', 'Color', 'Focus on']);
    });

    it('shows the sections in video mode too', () => {
      mountOptions(BASE, vi.fn(), { axes: cameraAxes, mode: 'video' });
      expect(headings()).toEqual(expect.arrayContaining(['Framing', 'Color']));
    });

    it('drops the colours when another palette is chosen', () => {
      const onchange = mountHost({ ...BASE, palette: 'custom', colors: OWN });
      pick('palette', 'earth');
      expect(onchange).toHaveBeenLastCalledWith(expect.objectContaining({ palette: 'earth', colors: [] }));
    });

    it('drops the colours when the picker is cleared back to Keep', () => {
      const onchange = mountHost({ ...BASE, palette: 'custom', colors: OWN });
      flushSync(() => (target.querySelector('button[aria-label="Palette: back to Keep"]') as HTMLButtonElement).click());
      expect(onchange).toHaveBeenLastCalledWith(expect.objectContaining({ palette: 'keep', colors: [] }));
    });

    it('shows the colour row only while the palette is Custom', () => {
      mountHost({ ...BASE, palette: 'earth' });
      expect(target.querySelector('[data-color-add]')).toBeNull();
      pick('palette', 'custom');
      expect(target.querySelector('[data-color-add]')).not.toBeNull();
    });

    it('shows a hint beside the add control while Custom has no colours', () => {
      mountHost({ ...BASE, palette: 'custom' });
      expect(target.querySelector('[data-color-hint]')).not.toBeNull();
    });

    const swatchHexes = (): (string | undefined)[] => Array.from(target.querySelectorAll<HTMLElement>('#enhance-palette [data-swatch]')).map((node) => node.dataset.swatch);

    it('shows the Custom colours as the strip on the closed Palette picker', () => {
      mountHost({ ...BASE, palette: 'custom', colors: OWN });
      expect(swatchHexes()).toEqual(OWN);
    });

    it('shows no strip on the closed Palette picker for Custom without colours', () => {
      mountHost({ ...BASE, palette: 'custom' });
      expect(swatchHexes()).toEqual([]);
    });

    it('hides the add control at five colours', () => {
      const five = ['#' + '111111', '#' + '222222', '#' + '333333', '#' + '444444', '#' + '555555'];
      mountOptions({ ...BASE, palette: 'custom', colors: five }, vi.fn(), { axes: cameraAxes });
      expect(target.querySelector('[data-color-add]')).toBeNull();
    });

    it('keeps the add control enabled below five colours', () => {
      mountOptions({ ...BASE, palette: 'custom', colors: OWN }, vi.fn(), { axes: cameraAxes });
      expect((target.querySelector('[data-color-add] input') as HTMLInputElement).disabled).toBe(false);
    });

    it('keeps the same colour input while a colour is edited', () => {
      mountHost({ ...BASE, palette: 'custom', colors: OWN });
      const before = target.querySelector('[data-color-dot="0"] input') as HTMLInputElement;
      before.value = '#' + '778899';
      flushSync(() => before.dispatchEvent(new Event('input', { bubbles: true })));
      const after = target.querySelector('[data-color-dot="0"] input') as HTMLInputElement;
      expect(after).toBe(before);
      expect(after.value).toBe('#' + '778899');
    });
  });
  describe('audio mode', () => {
    const soundAxes = axesOf(AUDIO_MATRIX, 'sound');
    const SOUND: AudioEnhanceSettings = { ...AUDIO_MATRIX.defaults };
    const headings = (): string[] => Array.from(target.querySelectorAll('h3')).map((h) => h.textContent?.trim() ?? '');

    function mountSound(settings = SOUND, onchange = vi.fn()): typeof onchange {
      document.body.appendChild(target);
      app = flushSync(() => mount(EnhanceOptions, { target, props: { axes: soundAxes, settings, mode: 'audio', onchange, idPrefix: 't' } }));
      return onchange;
    }

    it("lists Sound's two sections with the axes passed in, and no image or video section", () => {
      mountSound();

      expect(headings()).toEqual(['Place', 'Motion']);
      expect(Array.from(target.querySelectorAll('.picker')).map((picker) => picker.id)).toEqual(['t-space', 't-distance', 't-over_time', 't-movement']);
      expect(target.querySelector('section[aria-label="Look"], section[aria-label="Output"]')).toBeNull();
    });

    it('shows only the sections whose axes it was given', () => {
      document.body.appendChild(target);
      app = flushSync(() => mount(EnhanceOptions, { target, props: { axes: soundAxes.filter((axis) => axis.key === 'space'), settings: SOUND, mode: 'audio', onchange: vi.fn(), idPrefix: 't' } }));

      expect(headings()).toEqual(['Place']);
      expect(target.querySelector('#t-distance')).toBeNull();
    });

    it('reports a choice with the other audio options as they were', () => {
      const onchange = mountSound({ ...SOUND, space: 'forest' });

      flushSync(() => (target.querySelector('#t-distance') as HTMLButtonElement).click());
      flushSync(() => (document.querySelector('[role="option"][data-slug="far"]') as HTMLButtonElement).click());

      expect(onchange).toHaveBeenLastCalledWith({ ...SOUND, space: 'forest', distance: 'far' });
    });

    it('reports a typed direction, asking for a sound\'s own kind of direction', () => {
      const onchange = mountSound();
      const input = target.querySelector('#t-note') as HTMLInputElement;
      expect(input.placeholder).toContain('thunder');

      input.value = 'far off';
      input.dispatchEvent(new Event('input', { bubbles: true }));

      expect(onchange).toHaveBeenLastCalledWith({ ...SOUND, note: 'far off' });
    });

    describe('a Music style', () => {
      const musicAxes = axesOf(AUDIO_MATRIX, 'style');
      const MUSIC: AudioEnhanceSettings = { ...AUDIO_MATRIX.defaults };
      const sections = (): string[] => Array.from(target.querySelectorAll('h3, legend .ui-area-label')).map((node) => node.textContent?.trim() ?? '');
      const chip = (axis: string, name: string) => Array.from(target.querySelectorAll(`#t-${axis}-label ~ .chip-list button`)).find((button) => button.textContent?.trim() === name) as HTMLButtonElement;

      function mountMusic(settings = MUSIC, onchange = vi.fn(), axesIn = musicAxes): typeof onchange {
        document.body.appendChild(target);
        app = flushSync(() => mount(EnhanceOptions, { target, props: { axes: axesIn, settings, mode: 'audio', onchange, idPrefix: 't' } }));
        return onchange;
      }

      it('lists the pickers, the two instrument lists and the singer, with a music direction', () => {
        mountMusic();

        expect(sections()).toEqual(['Sound', 'Instruments · pick any', 'Without · pick any', 'Voice']);
        expect(Array.from(target.querySelectorAll('.picker')).map((picker) => picker.id)).toEqual(['t-genre', 't-music_mood', 't-production', 't-singer']);
        expect((target.querySelector('#t-note') as HTMLInputElement).placeholder).toContain('guitar');
      });

      it('leaves out the Voice section when the singer is not offered', () => {
        mountMusic(MUSIC, vi.fn(), axesOf(AUDIO_MATRIX, 'style', false));

        expect(sections()).toEqual(['Sound', 'Instruments · pick any', 'Without · pick any']);
      });

      it('offers All on Instruments only, in the matrix order', () => {
        const onchange = mountMusic({ ...MUSIC, instruments: ['piano', 'guitar'] });

        expect(target.querySelector('[data-toggle-all="without"]')).toBeNull();
        flushSync(() => chip('instruments', 'Drums').click());

        expect(onchange).toHaveBeenLastCalledWith({ ...MUSIC, instruments: ['guitar', 'piano', 'drums'], without: [] });
      });

      it('drops an instrument from Without when it is asked for, and does not let Without take it back', () => {
        const onchange = mountMusic({ ...MUSIC, instruments: ['piano'], without: ['guitar', 'drums'] });

        expect(chip('without', 'Piano').disabled).toBe(true);
        expect(chip('without', 'Guitar').disabled).toBe(false);
        flushSync(() => chip('instruments', 'Guitar').click());

        expect(onchange).toHaveBeenLastCalledWith({ ...MUSIC, instruments: ['guitar', 'piano'], without: ['drums'] });
      });

      it('marks what is left out with a minus and a name that says so', () => {
        mountMusic({ ...MUSIC, without: ['drums'] });

        expect(chip('without', 'Drums').getAttribute('aria-pressed')).toBe('true');
        expect(chip('without', 'Drums').getAttribute('aria-label')).toBe('Without drums');
        expect(chip('without', 'Drums').querySelector('svg')).not.toBeNull();
        expect(chip('instruments', 'Drums').querySelector('svg')).toBeNull();
      });

      it('groups the genres', () => {
        mountMusic();
        flushSync(() => (target.querySelector('#t-genre') as HTMLButtonElement).click());

        const names = Array.from(document.querySelectorAll('[role="listbox"] .picker-group')).map((node) => node.textContent?.trim());
        expect(names).toContain('Pop & rock');
        expect(names).toContain('World');
      });
    });
  });
});
