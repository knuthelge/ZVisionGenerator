import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PEAKS } from '../../../test-utils/audio';
import { mountComponent } from '../../../test-utils/mount';
import AudioPlayer from './AudioPlayer.svelte';
import VoicePlay from './VoicePlay.svelte';

describe('VoicePlay', () => {
  let target: HTMLElement;
  let pause: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    pause = vi.fn();
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(pause as () => void);
  });
  afterEach(() => vi.restoreAllMocks());

  function mountButton(src: string | null = '/api/voices/morgan/audio', label = 'Morgan'): ReturnType<typeof mountComponent> {
    const page = mountComponent(VoicePlay, { src, label });
    target = page.target;
    return page;
  }
  const audio = () => target.querySelector<HTMLAudioElement>('audio')!;
  const playButton = () => target.querySelector<HTMLButtonElement>('[data-testid="audio-row-play"]')!;

  it('is one toggle button with no waveform or native controls, and loads nothing until it is played', () => {
    mountButton();

    expect(target.querySelector('[data-testid="audio-wave"]')).toBeNull();
    expect(audio().controls).toBe(false);
    expect(audio().preload).toBe('none');
    expect(playButton().getAttribute('aria-label')).toBe('Play Morgan');
    expect(playButton().getAttribute('aria-pressed')).toBe('false');
  });

  it('plays on a press, shows it playing, and pauses on the next', () => {
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    mountButton();

    flushSync(() => playButton().click());
    expect(play).toHaveBeenCalledTimes(1);
    flushSync(() => audio().dispatchEvent(new Event('play')));
    expect(playButton().getAttribute('aria-pressed')).toBe('true');

    vi.spyOn(HTMLMediaElement.prototype, 'paused', 'get').mockReturnValue(false);
    flushSync(() => playButton().click());
    expect(pause).toHaveBeenCalled();
    flushSync(() => audio().dispatchEvent(new Event('pause')));
    expect(playButton().getAttribute('aria-pressed')).toBe('false');
  });

  it('is off, with no audio element, for a clip that is not there', () => {
    mountButton(null, 'Ryan');

    expect(playButton().disabled).toBe(true);
    expect(target.querySelector('audio')).toBeNull();
  });

  it('pauses another player or voice button when it starts, through the same claim as every player', () => {
    const paused: HTMLAudioElement[] = [];
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) { paused.push(this as HTMLAudioElement); });
    const button = mountButton();
    const other = mountComponent(AudioPlayer, { src: '/media/out/a.flac', peaks: PEAKS, filename: 'a.flac', duration: 100 });
    const otherAudio = other.target.querySelector<HTMLAudioElement>('audio')!;
    otherAudio.dispatchEvent(new Event('play'));
    paused.length = 0;

    const buttonAudio = button.target.querySelector<HTMLAudioElement>('audio')!;
    buttonAudio.dispatchEvent(new Event('play'));

    expect(paused).toEqual([otherAudio]);

    const sibling = mountButton('/api/voices/b/audio', 'B');
    paused.length = 0;
    sibling.target.querySelector('audio')!.dispatchEvent(new Event('play'));

    expect(paused).toEqual([buttonAudio]);
  });
});
