import { describe, expect, it } from 'vitest';

import { trackOf, trackSet } from '$lib/state/audioTrack';
import { workspaceTab } from '$lib/state/workspaceTab';

import { audioTrackView, type AudioTrackViewInput } from './audioTrackView';

const memo = new File(['x'], 'memo.m4a', { type: 'audio/mp4' });

/** Build a view; *shown* is the model's `showAudioTrack`, and the track is in effect when it is shown and the open video tab holds one. */
function view(state: Partial<AudioTrackViewInput['state']> = {}, input: Partial<Omit<AudioTrackViewInput, 'state' | 'caps'>> & { shown?: boolean } = {}) {
  const { shown = true, ...rest } = input;
  const full: AudioTrackViewInput['state'] = { workflow: 'txt2vid', audioTrackPath: null, audioTrackUpload: null, audioTrackStart: 0, extendAudioTrackPath: null, extendAudioTrackUpload: null, extendAudioTrackStart: 0, ...state };
  const tab = workspaceTab(full.workflow);
  return audioTrackView({
    state: full,
    caps: { showAudioTrack: shown, audioTrackInEffect: shown && (tab === 'video' || tab === 'extend') && trackSet(trackOf(full, tab)) },
    files: { audioTrack: null, extendAudioTrack: null },
    restored: true,
    lengthOf: () => null,
    ...rest,
  });
}

describe('audioTrackView', () => {
  it('sends nothing and reads None without a track', () => {
    expect(view()).toMatchObject({ tab: 'video', visible: true, set: false, sent: null, block: null, summary: { text: 'None', tone: 'muted' } });
  });

  it('sends a path with its Start at, and reads the name', () => {
    const v = view({ audioTrackPath: '/audio/song.mp3', audioTrackStart: 90 });

    expect(v.sent).toEqual({ path: '/audio/song.mp3', file: null, start: 90 });
    expect(v.summary).toEqual({ text: 'song.mp3 from 1:30', tone: 'active' });
  });

  it('reads just the name at 0', () => {
    expect(view({ audioTrackPath: '/audio/song.mp3' }).summary.text).toBe('song.mp3');
  });

  it('sends the held file over a path', () => {
    const v = view({ audioTrackUpload: 'memo.m4a', audioTrackStart: 2 }, { files: { audioTrack: memo, extendAudioTrack: null } });

    expect(v.sent).toEqual({ path: null, file: memo, start: 2 });
    expect(v.name).toBe('memo.m4a');
  });

  it('keeps the track of Extend Video apart from the Video track', () => {
    const extending = view({ workflow: 'extend', audioTrackPath: '/audio/video.mp3', extendAudioTrackPath: '/audio/extend.mp3', extendAudioTrackStart: 6.65 });

    expect(extending).toMatchObject({ tab: 'extend', sent: { path: '/audio/extend.mp3', file: null, start: 6.65 } });
  });

  it('sends a kept track on no model that cannot follow it: hidden, not sent', () => {
    const v = view({ audioTrackPath: '/audio/song.mp3' }, { shown: false });

    expect(v).toMatchObject({ visible: false, set: true, sent: null, block: null });
  });

  it('has no track in Image', () => {
    expect(view({ workflow: 'txt2img', audioTrackPath: '/audio/song.mp3' })).toMatchObject({ tab: null, visible: false, sent: null });
  });

  it('blocks Generate for an upload whose file did not come back, but not before the restore has settled', () => {
    const lost = view({ audioTrackUpload: 'memo.m4a' });
    expect(lost).toMatchObject({ gone: true, block: 'Choose the audio track again.', sent: null, summary: { tone: 'error' } });

    expect(view({ audioTrackUpload: 'memo.m4a' }, { restored: false })).toMatchObject({ gone: false, block: null });
    expect(view({ audioTrackUpload: 'memo.m4a' }, { files: { audioTrack: memo, extendAudioTrack: null } })).toMatchObject({ gone: false, block: null });
  });

  it('does not block for a lost upload on a model that cannot follow a track', () => {
    expect(view({ audioTrackUpload: 'memo.m4a' }, { shown: false }).block).toBeNull();
  });

  it('reads the length of the track in use', () => {
    const lengthOf = (key: string | null) => (key === 'path:/audio/song.mp3' ? 3.6 : null);

    expect(view({ audioTrackPath: '/audio/song.mp3', audioTrackStart: 4 }, { lengthOf }).length).toBe(3.6);
    expect(view({ audioTrackPath: '/audio/other.mp3' }, { lengthOf }).length).toBeNull();
  });

});
