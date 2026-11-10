import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';
import AssetViewer from './AssetViewer.svelte';
import { AUDIO_MODELS, LYRICS, makeAudioAsset, makeImportedAudio, makeSpeechAsset } from '../../../test-utils/audio';
import { mountComponent } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

// The touch layer follows the pointer; a test switches it on without a real media query.
const touchLayer = vi.hoisted(() => ({ coarse: false }));
vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  return { ...actual, layout: new Proxy(actual.layout, { get: (target, key) => (key === 'coarse' && touchLayer.coarse ? true : Reflect.get(target, key)) }) };
});

/** Dispatch a touch event with one finger at (x, y). */
function touch(node: Element, type: string, x: number, y: number): TouchEvent {
  const point = { clientX: x, clientY: y, target: node } as unknown as Touch;
  const event = new Event(type, { bubbles: true, cancelable: true }) as TouchEvent;
  Object.defineProperty(event, 'touches', { value: type === 'touchend' ? [] : [point] });
  Object.defineProperty(event, 'changedTouches', { value: [point] });
  node.dispatchEvent(event);
  return event;
}

describe('AssetViewer with audio', () => {
  let pause: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    localStorage.clear();
    pause = vi.fn();
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(pause as () => void);
  });
  afterEach(() => {
    vi.restoreAllMocks();
    touchLayer.coarse = false;
  });

  const song = makeAudioAsset();
  const talk = makeSpeechAsset();

  function mountViewer(props: Record<string, unknown> = {}): ReturnType<typeof mountComponent> {
    const assets = (props.assets as GalleryAsset[] | undefined) ?? [song, talk];
    return mountComponent(AssetViewer, { assets, currentId: assets[0].id, open: true, onclose: vi.fn(), onnavigate: vi.fn(), onreuse: vi.fn(), onuseaudio: vi.fn(), ondelete: vi.fn(), ...props });
  }

  const action = (name: string) => document.querySelector<HTMLElement>(`[data-action="${name}"]`);

  it('plays the take in its player with its words under it, and no image', () => {
    mountViewer();

    expect(document.querySelector('audio')?.getAttribute('src')).toBe('/media/out/song.flac');
    expect(document.querySelector('[data-testid="viewer-image"]')).toBeNull();
    const blocks = Array.from(document.querySelectorAll('[data-testid="viewer-words"] section')).map((section) => [section.querySelector('h4')?.textContent, section.querySelector('p')?.textContent]);
    expect(blocks).toEqual([['Style', 'Dreamy synth-pop, warm analog pads'], ['Lyrics', LYRICS]]);
  });

  describe('the take\'s player', () => {
    const player = () => document.querySelector<HTMLAudioElement>('audio')!;
    const song2 = makeAudioAsset({ id: 'out/song-2.flac', url: '/media/out/song-2.flac', filename: 'song-2.flac' });

    it('is the app\'s player with no native controls, and never autoplays', () => {
      mountViewer();

      expect(player().controls).toBe(false);
      expect(player().autoplay).toBe(false);
      expect(document.querySelector('[data-media-player]')?.getAttribute('data-kind')).toBe('audio');
      expect(document.querySelector('[data-media-player]')?.getAttribute('data-size')).toBe('viewer');
      expect(document.querySelector('[data-media-player]')?.getAttribute('aria-label')).toBe('song.flac');
    });

    it('pauses when another take is shown and when the viewer goes', async () => {
      const props = reactiveProps({ assets: [song, song2], currentId: song.id, open: true, onclose: vi.fn(), onnavigate: vi.fn() });
      const page = mountComponent(AssetViewer, props);
      flushSync();
      pause.mockClear();

      props.currentId = song2.id;
      flushSync();
      expect(pause).toHaveBeenCalledTimes(1);

      await page.unmount();
      expect(pause).toHaveBeenCalledTimes(2);
    });

    it('shows a stopped player, not the playing one, when another take is shown while one plays', () => {
      // A real element is paused or not; one that is playing when its source changes goes back to paused without a `pause` event.
      const playing = new WeakSet<HTMLMediaElement>();
      vi.spyOn(HTMLMediaElement.prototype, 'paused', 'get').mockImplementation(function (this: HTMLMediaElement) { return !playing.has(this); });
      vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) { playing.delete(this); });
      const props = reactiveProps({ assets: [song, song2], currentId: song.id, open: true, onclose: vi.fn(), onnavigate: vi.fn() });
      mountComponent(AssetViewer, props);
      flushSync();
      const first = player();
      playing.add(first);
      first.dispatchEvent(new Event('play'));

      props.currentId = song2.id;
      flushSync();

      expect(player().getAttribute('src')).toBe('/media/out/song-2.flac');
      expect(player()).not.toBe(first);
      expect(player().paused).toBe(true);
      expect(first.paused).toBe(true);
      expect(document.querySelectorAll('audio')).toHaveLength(1);
    });

    it('keeps the same player when a fresh copy of the same take arrives', () => {
      const props = reactiveProps({ assets: [song, talk], currentId: song.id, open: true, onclose: vi.fn(), onnavigate: vi.fn() });
      mountComponent(AssetViewer, props);
      flushSync();
      const first = player();
      pause.mockClear();

      props.assets = [{ ...song }, talk];
      flushSync();

      expect(player()).toBe(first);
      expect(pause).not.toHaveBeenCalled();
    });
  });

  it('shows a file without settings with a player and no words', () => {
    mountViewer({ assets: [makeImportedAudio()] });

    expect(document.querySelector('audio')).not.toBeNull();
    expect(document.querySelector('[data-testid="viewer-words"]')).toBeNull();
  });

  it('keeps the bar to Reuse, Use as, Download, Delete, Details, shortcuts and Close, with no Extend or Upscale for a null upscale and extend', () => {
    mountViewer();

    expect(['reuse', 'reference', 'download', 'delete', 'details', 'shortcuts'].map((name) => action(name) !== null)).toEqual([true, true, true, true, true, true]);
    expect(action('extend')).toBeNull();
    expect(action('upscale')).toBeNull();
  });

  it('lists no upscale shortcut for a take, and says what E does for one', () => {
    mountViewer();
    flushSync(() => action('shortcuts')!.click());

    const help = document.querySelector('#asset-viewer-shortcuts')!;
    expect(help.textContent).not.toContain('Upscale');
    expect(help.textContent).toContain('audio track or voice');
  });

  it('lists the audio track under Use as, and the voice and the saved voice for a speech take', () => {
    const onuseaudio = vi.fn();
    mountViewer({ assets: [talk], onuseaudio, useAs: { audio_track: { disabled: false, hint: 'Video tab, from 0 s' }, voice: { disabled: false, hint: 'Speech, text filled in' }, save_voice: { disabled: false, hint: 'Keep it in your voices' } } });

    flushSync(() => action('reference')!.click());
    const items = Array.from(document.querySelectorAll('[role="menu"] [role="menuitem"]')).map((item) => item.textContent?.replace(/\s+/g, ' ').trim());
    expect(items).toEqual(['Audio track Video tab, from 0 s', 'Voice Speech, text filled in', 'Saved voice Keep it in your voices']);

    flushSync(() => (document.querySelector('[role="menu"] [data-action="use-voice"]') as HTMLElement).click());
    expect(onuseaudio).toHaveBeenCalledWith(talk, 'voice');
  });

  it('reports Saved voice to the host', () => {
    const onuseaudio = vi.fn();
    mountViewer({ assets: [talk], onuseaudio, useAs: { voice: { disabled: false }, save_voice: { disabled: false }, voiceMaxSeconds: 30 } });
    flushSync(() => action('reference')!.click());
    flushSync(() => (document.querySelector('[role="menu"] [data-action="use-save-voice"]') as HTMLElement).click());
    expect(onuseaudio).toHaveBeenCalledWith(talk, 'save_voice');
  });

  it('greys Voice and Saved voice for a take over the limit, with the reason as the tooltip and none in the row', () => {
    const long = makeSpeechAsset({ duration: 44 });
    mountViewer({ assets: [long], useAs: { voice: { disabled: false }, save_voice: { disabled: false }, voiceMaxSeconds: 30 } });
    flushSync(() => action('reference')!.click());

    for (const id of ['use-voice', 'use-save-voice']) {
      const item = document.querySelector<HTMLElement>(`[role="menu"] [data-action="${id}"]`)!;
      expect(item.getAttribute('aria-disabled')).toBe('true');
      expect(item.title).toBe('Use a take of 30 s or less as a voice');
      expect(item.querySelector('.action-menu-hint')).toBeNull();
    }
  });

  it('shows the Generation and File facts of a take, with the cut at the cap', () => {
    mountViewer({ assets: [makeAudioAsset({ cut: true, duration: 360, sample_rate: 24000, channels: 1 })] });

    flushSync(() => action('details')!.click());
    const facts = Object.fromEntries(Array.from(document.querySelectorAll('.viewer-facts > div')).map((row) => [row.querySelector('dt')?.textContent, row.querySelector('dd')?.textContent]));
    expect(facts).toMatchObject({ Kind: 'Music', Model: 'yue2', Length: '6:00 (cut at the cap)', Seed: '4127', Format: 'FLAC · 24 kHz · mono', Size: '1 MB' });
    expect(Array.from(document.querySelectorAll('.viewer-h')).map((heading) => heading.textContent)).toEqual(expect.arrayContaining(['Generation', 'File']));
    expect(document.querySelector('[data-testid="viewer-enhance"]')).toBeNull();
  });

  it('shows how an enhanced take was written, with the user\'s own text, as it does for an image', () => {
    const enhance = { summary: 'Fast · Style · Folk', tier: 'fast' as const, prompt_input: 'sad folk', scene_input: null, settings: {}, model: 'org/fast', format: 'text' as const };
    mountViewer({ assets: [makeAudioAsset({ enhance })] });
    flushSync(() => action('details')!.click());

    const block = document.querySelector('[data-testid="viewer-enhance"]');
    expect(block?.textContent).toContain('Fast · Style · Folk');
    expect(block?.textContent).toContain('sad folk');
  });

  it('shows the notes of a take as one Note fact, and none for a take without', () => {
    mountViewer({ assets: [makeAudioAsset({ notes: ["YuE2 couldn't plan with this tempo and key.", 'The voice stays.'] })] });
    flushSync(() => action('details')!.click());
    const note = () => Array.from(document.querySelectorAll('.viewer-facts > div')).find((row) => row.querySelector('dt')?.textContent === 'Note');

    expect(note()?.querySelector('dd')?.textContent).toBe("YuE2 couldn't plan with this tempo and key. The voice stays.");
  });

  it('shows no Note fact for a take without notes', () => {
    mountViewer({ assets: [makeAudioAsset({ notes: [] })] });
    flushSync(() => action('details')!.click());

    expect(Array.from(document.querySelectorAll('.viewer-facts dt')).map((label) => label.textContent)).not.toContain('Note');
  });

  const voiceOf = (asset: ReturnType<typeof makeSpeechAsset>, audioModels = AUDIO_MODELS): string | null | undefined => {
    mountViewer({ assets: [asset], audioModels });
    flushSync(() => action('details')!.click());
    return Array.from(document.querySelectorAll('.viewer-facts > div')).find((row) => row.querySelector('dt')?.textContent === 'Voice')?.querySelector('dd')?.textContent;
  };

  it('names the voice of a cloned take by its recording', () => {
    const cloned = makeSpeechAsset({ has_reference_voice: true, reuse_workspace_url: '#/workspace?workflow=speech&voice_path=%2Fvoices%2Fmine.wav' });

    expect(voiceOf(cloned)).toBe('mine.wav');
  });

  it('names the built-in voice a take was read in, by its name', () => {
    const voiced = makeSpeechAsset({ reuse_workspace_url: '#/workspace?workflow=speech&model=qwen3_tts&speaker=vivian' });

    expect(voiceOf(voiced)).toBe('Vivian');
  });

  it('shows the id of a built-in voice this machine does not list', () => {
    const voiced = makeSpeechAsset({ reuse_workspace_url: '#/workspace?workflow=speech&model=qwen3_tts&speaker=vivian' });

    expect(voiceOf(voiced, [])).toBe('vivian');
  });

  it('shows a dash for a take that records no voice', () => {
    expect(voiceOf(talk)).toBe('—');
  });

  it('keeps playing when the list is replaced by a fresh copy holding the same take', async () => {
    const props = reactiveProps({ assets: [song, talk], currentId: talk.id, open: true, onclose: vi.fn(), onnavigate: vi.fn() });
    const page = mountComponent(AssetViewer, props);
    const element = document.querySelector('audio');
    pause.mockClear();

    props.assets = [makeAudioAsset({ id: 'out/new.flac', url: '/media/out/new.flac', filename: 'new.flac' }), { ...song }, { ...talk }];
    flushSync();

    expect(document.querySelector('audio')).toBe(element);
    expect(pause).not.toHaveBeenCalled();
    await page.unmount();
  });

  it('pauses the take when the viewer pages to the next one and when it closes', async () => {
    const props = reactiveProps({ assets: [song, talk], currentId: song.id, open: true, onclose: vi.fn(), onnavigate: vi.fn() });
    const page = mountComponent(AssetViewer, props);
    pause.mockClear();

    props.currentId = talk.id;
    flushSync();
    expect(pause).toHaveBeenCalled();
    expect(document.querySelector('audio')?.getAttribute('src')).toBe('/media/out/speech.flac');

    pause.mockClear();
    props.open = false;
    flushSync();
    expect(pause).toHaveBeenCalled();
    await page.unmount();
  });

  it('does not page to the next take when a finger drags sideways on the waveform', () => {
    touchLayer.coarse = true;
    const onnavigate = vi.fn();
    mountViewer({ onnavigate });
    const wave = document.querySelector('[data-testid="media-timeline"]')!;

    touch(wave, 'touchstart', 300, 200);
    touch(wave, 'touchmove', 220, 202);
    touch(wave, 'touchmove', 40, 204);
    touch(wave, 'touchend', 40, 204);

    expect(onnavigate).not.toHaveBeenCalled();
  });

  it('pages on the same drag anywhere else on the stage', () => {
    touchLayer.coarse = true;
    const onnavigate = vi.fn();
    mountViewer({ onnavigate });
    // The viewer settles after a swipe with an animation; a layout-less test settles it at once.
    const stage = document.querySelector('.viewer-stage')!;

    touch(stage, 'touchstart', 300, 200);
    touch(stage, 'touchmove', 220, 202);
    touch(stage, 'touchmove', 40, 204);
    touch(stage, 'touchend', 40, 204);

    expect(onnavigate).toHaveBeenCalledWith(1);
  });

  it('puts an audio take in the film strip as a waveform tile, not an image', () => {
    mountViewer();

    expect(document.querySelectorAll('.viewer-film .film-audio')).toHaveLength(2);
    expect(document.querySelector('.viewer-film img')).toBeNull();
  });
});
