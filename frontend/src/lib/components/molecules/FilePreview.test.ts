import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { stepTarget } from '$lib/state/framePoint';
import { formatBytes, formatModifiedFull, previewFacts } from '$lib/state/pathBrowser';
import { videoFacts } from '$lib/state/videoFacts.svelte';
import type { BrowseEntry } from '$lib/types';

const browseApiMocks = vi.hoisted(() => ({
  browsePreviewUrl: vi.fn(),
  browseMediaUrl: vi.fn(),
  browseInfo: vi.fn(),
}));

vi.mock('$lib/api/browse', () => browseApiMocks);

import { installFakeMedia } from '../../../test-utils/media';
import { infoFor, stubBrowseMediaUrl, stubBrowsePreviewUrl } from '../../../test-utils/pathBrowser';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import FilePreview from './FilePreview.svelte';

const MODIFIED = new Date(2026, 10, 3, 14, 5, 9).getTime() / 1000;

function file(name: string, extra: Partial<BrowseEntry> = {}): BrowseEntry {
  return { name, path: `/media/${name}`, kind: 'file', size: 2048, modified: MODIFIED, selectable: true, ...extra };
}

describe('FilePreview', () => {
  let play: ReturnType<typeof vi.fn>;
  let pause: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    videoFacts.reset();
    for (const mock of Object.values(browseApiMocks)) mock.mockReset();
    browseApiMocks.browsePreviewUrl.mockImplementation(stubBrowsePreviewUrl);
    browseApiMocks.browseMediaUrl.mockImplementation(stubBrowseMediaUrl);
    browseApiMocks.browseInfo.mockImplementation(async (purpose: string) => infoFor(purpose === 'audio_track' ? 'audio' : purpose === 'source_video' ? 'video' : 'image'));
    play = vi.fn().mockResolvedValue(undefined);
    pause = vi.fn();
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(play as () => Promise<void>);
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(pause as () => void);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  const root = () => document.body.querySelector<HTMLElement>('[data-testid="file-preview"]')!;
  const name = () => root().querySelector('[data-testid="preview-name"]')?.textContent;
  /** The facts the preview should list: what ``previewFacts`` makes of the entry and the info (formatting is tested there). */
  const shown = (kind: 'image' | 'audio' | 'video', entry: BrowseEntry, info: ReturnType<typeof infoFor> | null) => Object.fromEntries(previewFacts(kind, entry, info).map((fact) => [fact.label, fact.value]));
  const facts = () => Object.fromEntries(Array.from(root().querySelectorAll('dt')).map((term) => [term.textContent, term.nextElementSibling?.textContent]));

  function mountPreview(purpose: 'reference_image' | 'audio_track' | 'source_video', entry: BrowseEntry | null) {
    const kind = { reference_image: 'image', audio_track: 'audio', source_video: 'video' }[purpose];
    const props = reactiveProps({ purpose, kind, entry });
    mountComponent(FilePreview, props);
    return props;
  }

  describe('an image', () => {
    it('shows the full name at once and asks for nothing until the selection settles', async () => {
      mountPreview('reference_image', file('fox-portrait_2026-11-05_21-38-51.png'));
      await settle(149);

      expect(name()).toBe('fox-portrait_2026-11-05_21-38-51.png');
      expect(root().querySelector('img')).toBeNull();
      expect(browseApiMocks.browseInfo).not.toHaveBeenCalled();

      await settle(1);

      expect(root().querySelector('img')?.getAttribute('src')).toBe('/preview/reference_image/media/fox-portrait_2026-11-05_21-38-51.png');
      expect(browseApiMocks.browseInfo).toHaveBeenCalledTimes(1);
      expect(browseApiMocks.browseInfo).toHaveBeenCalledWith('reference_image', '/media/fox-portrait_2026-11-05_21-38-51.png', expect.any(AbortSignal));
    });

    it('asks for the large picture', async () => {
      mountPreview('reference_image', file('fox.png'));
      await settle(150);

      expect(browseApiMocks.browsePreviewUrl).toHaveBeenCalledWith('reference_image', '/media/fox.png', { size: 'large' });
    });

    it('lists pixel size, file size and type, and the full date', async () => {
      mountPreview('reference_image', file('fox.png'));
      await settle(150);

      expect(facts()).toEqual(shown('image', file('fox.png'), infoFor('image')));
    });

    it('reads a dash for the pixel size while the facts load', async () => {
      browseApiMocks.browseInfo.mockImplementation(() => new Promise(() => {}));
      mountPreview('reference_image', file('fox.png'));
      await settle(150);

      expect(facts()).toEqual(shown('image', file('fox.png'), null));
    });

    it('keeps the facts it has when the info request fails', async () => {
      browseApiMocks.browseInfo.mockRejectedValue(new Error('Not found'));
      mountPreview('reference_image', file('fox.png'));
      await settle(150);

      expect(facts()).toEqual(shown('image', file('fox.png'), null));
      expect(root().querySelector('img')).not.toBeNull();
    });

    it('falls back to the icon when the picture cannot be read', async () => {
      mountPreview('reference_image', file('fox.png'));
      await settle(150);

      flushSync(() => root().querySelector('img')!.dispatchEvent(new Event('error')));

      expect(root().querySelector('img')).toBeNull();
      expect(root().querySelector('[data-testid="preview-icon"]')).not.toBeNull();
    });
  });

  describe('when the selection moves', () => {
    it('never asks about a file passed over before it settled', async () => {
      const props = mountPreview('reference_image', file('a.png'));
      await settle(100);

      props.entry = file('b.png');
      flushSync();
      await settle(100);
      expect(browseApiMocks.browseInfo).not.toHaveBeenCalled();

      await settle(50);

      expect(browseApiMocks.browseInfo).toHaveBeenCalledTimes(1);
      expect(browseApiMocks.browseInfo.mock.calls[0][1]).toBe('/media/b.png');
      expect(name()).toBe('b.png');
    });

    it('aborts the request for the file it leaves, and shows no facts of it', async () => {
      const signals: AbortSignal[] = [];
      browseApiMocks.browseInfo.mockImplementation((_purpose: string, _path: string, signal: AbortSignal) => {
        signals.push(signal);
        return new Promise(() => {});
      });
      const props = mountPreview('reference_image', file('a.png'));
      await settle(150);
      expect(signals[0].aborted).toBe(false);

      props.entry = file('b.png');
      flushSync();

      expect(signals[0].aborted).toBe(true);
      expect(facts()).toEqual(shown('image', file('b.png'), null));
    });

    it('mounts and requests nothing for the file it moved to before it has rested', async () => {
      const props = mountPreview('reference_image', file('a.png'));
      await settle(150);
      expect(root().querySelector('img')?.getAttribute('src')).toContain('a.png');
      browseApiMocks.browseInfo.mockClear();
      browseApiMocks.browsePreviewUrl.mockClear();

      props.entry = file('b.png');
      flushSync();
      await settle(149);

      expect(root().querySelector('img')).toBeNull();
      expect(browseApiMocks.browseInfo).not.toHaveBeenCalled();
      expect(browseApiMocks.browsePreviewUrl).not.toHaveBeenCalledWith('reference_image', '/media/b.png', { size: 'large' });

      await settle(1);

      expect(root().querySelector('img')?.getAttribute('src')).toContain('b.png');
      expect(browseApiMocks.browseInfo).toHaveBeenCalledTimes(1);
    });

    it('ignores a late answer for the file it left', async () => {
      let answer: (value: ReturnType<typeof infoFor>) => void = () => {};
      browseApiMocks.browseInfo.mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }));
      const props = mountPreview('reference_image', file('a.png'));
      await settle(150);

      props.entry = file('b.png');
      flushSync();
      answer(infoFor('image', { width: 1, height: 1 }));
      await settle();

      expect(facts()).toEqual(shown('image', file('b.png'), null));
    });
  });

  describe('the stand-in while the large picture loads', () => {
    const stage = () => root().querySelector<HTMLElement>('[data-testid="preview-stage"]')!;

    beforeEach(() => {
      browseApiMocks.browsePreviewUrl.mockImplementation((_purpose: string, path: string, options: { size?: string } = {}) => `/${options.size ?? 'small'}${path}`);
    });

    it('shows the small picture behind an image before it has settled', async () => {
      mountPreview('reference_image', file('fox.png'));
      await settle(10);

      expect(stage().style.backgroundImage).toContain('/small/media/fox.png');
      expect(root().querySelector('img')).toBeNull();
    });

    it('shows the small still behind a video', async () => {
      mountPreview('source_video', file('clip.mp4'));
      await settle(10);

      expect(stage().style.backgroundImage).toContain('/small/media/clip.mp4');
    });

    it('has none for audio', async () => {
      mountPreview('audio_track', file('take.flac'));
      await settle(10);

      expect(stage().style.backgroundImage).toBe('');
    });
  });

  describe('audio', () => {
    it('offers the app\'s player over the file, with its waveform and length, and never starts it', async () => {
      mountPreview('audio_track', file('take.flac'));
      await settle(150);

      const audio = root().querySelector<HTMLAudioElement>('audio')!;
      expect(audio.getAttribute('src')).toBe('/media/audio_track/media/take.flac');
      expect(audio.autoplay).toBe(false);
      expect(audio.controls).toBe(false);
      expect(root().querySelector('[data-media-player]')?.getAttribute('data-size')).toBe('pane');
      expect(root().querySelector('[role="slider"][aria-label="Playback position"]')).not.toBeNull();
      expect(play).not.toHaveBeenCalled();
    });

    it('lists length, format and file size', async () => {
      mountPreview('audio_track', file('take.flac'));
      await settle(150);

      expect(facts()).toEqual(shown('audio', file('take.flac'), infoFor('audio')));
    });

    it('does not load the audio until the selection settles', async () => {
      mountPreview('audio_track', file('take.flac'));
      await settle(100);

      expect(root().querySelector('audio')).toBeNull();
    });

    it('pauses when the selection moves to another file', async () => {
      const props = mountPreview('audio_track', file('a.flac'));
      await settle(150);
      const first = root().querySelector('audio')!;
      first.dispatchEvent(new Event('play'));
      pause.mockClear();

      props.entry = file('b.flac');
      flushSync();

      expect(pause.mock.contexts).toContain(first);
      expect(root().querySelector('audio')).toBeNull();
    });

    it('pauses when the preview goes', async () => {
      const page = mountComponent(FilePreview, { purpose: 'audio_track', kind: 'audio', entry: file('a.flac') });
      await settle(150);
      const audio = root().querySelector('audio')!;
      pause.mockClear();

      await page.unmount();

      expect(pause.mock.contexts).toContain(audio);
    });
  });

  describe('a video', () => {
    it('plays in the app\'s player over the file with the still as its poster, loading only its metadata', async () => {
      mountPreview('source_video', file('clip.mp4'));
      await settle(150);

      const video = root().querySelector<HTMLVideoElement>('video')!;
      expect(video.controls).toBe(false);
      expect(video.hasAttribute('playsinline')).toBe(true);
      expect(root().querySelector('[data-media-player]')?.getAttribute('data-size')).toBe('pane');
      expect(root().querySelector('button[aria-label="Fullscreen"]')).toBeNull();
      expect(root().querySelector('[data-volume]')).toBeNull();
      expect(video.getAttribute('preload')).toBe('metadata');
      expect(video.getAttribute('src')).toBe('/media/source_video/media/clip.mp4');
      expect(video.getAttribute('poster')).toBe('/preview/source_video/media/clip.mp4');
      expect(video.autoplay).toBe(false);
      expect(play).not.toHaveBeenCalled();
    });

    it('steps exact frames with the arrow keys, from the clip\'s own frame facts', async () => {
      const media = installFakeMedia();
      browseApiMocks.browseInfo.mockImplementation(async () => infoFor('video', { start: 0 }));
      mountPreview('source_video', file('clip.mp4'));
      await settle(150);
      const video = root().querySelector<HTMLVideoElement>('video')!;
      media.load(video, 10);
      flushSync();

      root().querySelector('[data-testid="media-timeline"]')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
      flushSync();

      expect(media.seeks(video)).toEqual([stepTarget(1, { fps: 24, frames: 240, start: 0 })]);
    });

    it('lists length, frames, frame rate, pixel size and file size', async () => {
      mountPreview('source_video', file('clip.mp4'));
      await settle(150);

      expect(facts()).toEqual(shown('video', file('clip.mp4'), infoFor('video')));
    });

    it('says the file cannot be played when the browser rejects it, under the still, without blaming the browser', async () => {
      mountPreview('source_video', file('clip.mkv'));
      await settle(150);

      flushSync(() => root().querySelector('video')!.dispatchEvent(new Event('error')));

      expect(root().querySelector('video')).toBeNull();
      expect(root().querySelector('img')?.getAttribute('src')).toBe('/preview/source_video/media/clip.mkv');
      expect(root().querySelector('[role="status"]')?.textContent).toBe('This file can’t be played.');
    });

    it('pauses the clip when the selection moves on, and when another thing starts playing', async () => {
      const props = mountPreview('source_video', file('a.mp4'));
      await settle(150);
      const first = root().querySelector('video')!;
      first.dispatchEvent(new Event('play'));
      pause.mockClear();

      props.entry = file('b.mp4');
      flushSync();

      expect(pause.mock.contexts).toContain(first);
    });

    it('takes over playback from the audio that was playing', async () => {
      const { claimPlayback } = await import('$lib/state/nowPlaying');
      const other = document.createElement('audio');
      claimPlayback(other);
      mountPreview('source_video', file('a.mp4'));
      await settle(150);
      pause.mockClear();

      root().querySelector('video')!.dispatchEvent(new Event('play'));

      expect(pause.mock.contexts).toContain(other);
    });
  });

  describe('what is not a previewable file', () => {
    it('shows a folder as its icon and asks for nothing', async () => {
      mountPreview('reference_image', file('archive', { kind: 'directory', size: null, selectable: false }));
      await settle(300);

      expect(name()).toBe('archive');
      expect(root().querySelector('[data-testid="preview-icon"]')).not.toBeNull();
      expect(root().getAttribute('data-kind')).toBe('folder');
      expect(facts()).toEqual({});
      expect(browseApiMocks.browseInfo).not.toHaveBeenCalled();
      expect(browseApiMocks.browsePreviewUrl).not.toHaveBeenCalled();
      expect(root().querySelector('[role="status"]')).toBeNull();
    });

    it('shows a file the picker does not take as its icon, size and date, and asks for nothing', async () => {
      mountPreview('reference_image', file('notes.txt', { selectable: false }));
      await settle(300);

      expect(root().querySelector('[data-testid="preview-icon"]')).not.toBeNull();
      expect(facts()).toEqual({ File: formatBytes(2048), Modified: formatModifiedFull(MODIFIED) });
      expect(browseApiMocks.browseInfo).not.toHaveBeenCalled();
      expect(browseApiMocks.browseMediaUrl).not.toHaveBeenCalled();
      expect(root().querySelector('[role="status"]')).toBeNull();
    });

    it('asks to select a file when nothing is selected', async () => {
      mountPreview('reference_image', null);
      await settle(300);

      expect(root().getAttribute('data-kind')).toBe('none');
      expect(browseApiMocks.browseInfo).not.toHaveBeenCalled();
    });
  });
});
