import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';


import AudioPlayer from './AudioPlayer.svelte';
import { PEAKS } from '../../../test-utils/audio';
import { mountComponent } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

describe('AudioPlayer', () => {
  let target: HTMLElement;
  let pause: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    pause = vi.fn();
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(pause as () => void);
  });
  afterEach(() => vi.restoreAllMocks());

  function mountPlayer(props: Record<string, unknown> = {}): ReturnType<typeof mountComponent> {
    const page = mountComponent(AudioPlayer, { src: '/media/out/song.flac', peaks: PEAKS, filename: 'song.flac', duration: 100, ...props });
    target = page.target;
    // jsdom lays nothing out: the waveform is 400 px wide from x = 100, and the file is 100 s long.
    wave().getBoundingClientRect = () => ({ left: 100, width: 400, top: 0, right: 500, bottom: 140, height: 140, x: 100, y: 0, toJSON: () => ({}) });
    return page;
  }

  const wave = () => target.querySelector<HTMLElement>('[data-testid="audio-wave"]')!;
  const audio = () => target.querySelector<HTMLAudioElement>('audio')!;
  const playhead = () => target.querySelector<HTMLElement>('[data-testid="audio-playhead"]')!;

  function pointer(type: string, clientX: number): void {
    wave().dispatchEvent(new PointerEvent(type, { clientX, pointerType: 'touch', pointerId: 1, bubbles: true }));
    flushSync();
  }

  it('is one set of native controls under a waveform that only the pointer uses', () => {
    mountPlayer();

    expect(audio().controls).toBe(true);
    expect(audio().getAttribute('aria-label')).toBe('Play song.flac');
    expect(wave().getAttribute('aria-hidden')).toBe('true');
    expect(wave().tabIndex).toBeLessThan(0);
  });

  it('never autoplays', () => {
    mountPlayer();

    expect(audio().autoplay).toBe(false);
  });

  it('draws the played bars in the waveform up to the playhead', () => {
    mountPlayer();
    audio().currentTime = 25;
    audio().dispatchEvent(new Event('timeupdate'));
    flushSync();

    expect(target.querySelectorAll('rect.played')).toHaveLength(25);
    expect(playhead().style.left).toBe('25%');
  });

  it('seeks when the waveform is pressed, and follows a held drag', () => {
    mountPlayer();

    pointer('pointerdown', 300);
    expect(audio().currentTime).toBe(50);
    expect(playhead().style.left).toBe('50%');

    pointer('pointermove', 400);
    expect(audio().currentTime).toBe(75);

    pointer('pointerup', 400);
    pointer('pointermove', 200);
    expect(audio().currentTime).toBe(75);
  });

  it('pauses when another take is shown and when it leaves', async () => {
    const props = reactiveProps({ src: '/media/out/a.flac', peaks: PEAKS, filename: 'a.flac', duration: 100 });
    const page = mountComponent(AudioPlayer, props);
    target = page.target;
    flushSync();
    pause.mockClear();

    props.src = '/media/out/b.flac';
    flushSync();
    expect(pause).toHaveBeenCalledTimes(1);

    await page.unmount();
    expect(pause).toHaveBeenCalledTimes(2);
  });
});
