import { slotSet, type SlotValue } from './imageSlots';
import { addToast } from './toasts.svelte';
import type { DraftState, WorkspaceTab } from '$lib/types';

/** The tabs that can hold an audio track: Video's new clips (with or without a first frame) and its Extend kind. Image holds none. */
export type TrackTab = Exclude<WorkspaceTab, 'image' | 'edit' | 'audio'>;

/** What a tab's track holds: a path on the host, the name of a file uploaded from this device, and where in the track the clip starts. */
export interface TrackFields {
  path: string | null;
  upload: string | null;
  start: number;
}

/** The audio types the server takes, by extension. The server's list is `AUDIO_TRACK_EXTENSIONS` in `zvisiongenerator/web/path_browser.py`. */
const AUDIO_TYPES: Record<string, string> = { '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4', '.flac': 'audio/flac', '.ogg': 'audio/ogg' };

/** The `accept` value of the track's file input: the extensions the server takes, which also cover a device that names no type. */
export const AUDIO_ACCEPT = [...Object.keys(AUDIO_TYPES), ...new Set(Object.values(AUDIO_TYPES))].join(',');

/** Whether the file looks like an audio track the server takes, by its extension; the job checks that it can be read. */
export function isSupportedAudio(file: File): boolean {
  const name = file.name.toLowerCase();
  return Object.keys(AUDIO_TYPES).some((extension) => name.endsWith(extension));
}

/** The toast for a single dropped file that is not an audio track. */
export const UNSUPPORTED_AUDIO_MESSAGE = "That file isn't a WAV, MP3, M4A, FLAC or OGG.";

/** Hand the first audio file of *files* to *onfile*; files that are not audio only raise the toast. */
export function takeAudioFiles(files: Iterable<File>, onfile: (file: File) => void): void {
  const all = [...files];
  const supported = all.find(isSupportedAudio);
  if (supported) onfile(supported);
  else if (all.length > 0) addToast(UNSUPPORTED_AUDIO_MESSAGE, 'info');
}

/** The draft fields that hold the tracks. */
export type TrackState = Pick<DraftState, 'audioTrackPath' | 'audioTrackUpload' | 'audioTrackStart' | 'extendAudioTrackPath' | 'extendAudioTrackUpload' | 'extendAudioTrackStart'>;

/** The draft's track fields of a tab. */
export function trackOf(state: TrackState, tab: TrackTab): TrackFields {
  return tab === 'video'
    ? { path: state.audioTrackPath, upload: state.audioTrackUpload, start: state.audioTrackStart }
    : { path: state.extendAudioTrackPath, upload: state.extendAudioTrackUpload, start: state.extendAudioTrackStart };
}

/** Whether the tab's track is set: it has a path or an upload marker (the same rule as an image slot). */
export function trackSet(fields: Pick<TrackFields, 'path' | 'upload'>): boolean {
  return slotSet(fields.path, fields.upload);
}

/**
 * Return the draft fields that make a value the tab's track: the path or the upload's file name, never both. A new value
 * starts at *start* (0 for a file the user chose); null clears the track and puts Start at back to 0.
 */
export function trackPatch(tab: TrackTab, value: SlotValue, start: number = 0): Partial<DraftState> {
  const path = value !== null && 'path' in value ? value.path : null;
  const upload = value !== null && 'file' in value ? value.file.name : null;
  const kept = value === null ? 0 : start;
  return tab === 'video'
    ? { audioTrackPath: path, audioTrackUpload: upload, audioTrackStart: kept }
    : { extendAudioTrackPath: path, extendAudioTrackUpload: upload, extendAudioTrackStart: kept };
}

/** Whether the Extend track is still exactly what a source's prefill put there: its path, no upload, and the prefilled Start at. */
export function isPrefilledTrack(state: TrackState, prefilled: { path: string; start: number } | null): boolean {
  const track = trackOf(state, 'extend');
  return prefilled !== null && track.path === prefilled.path && track.upload === null && track.start === prefilled.start;
}

/** Return the draft fields that hold the tab's Start at. */
export function trackStartPatch(tab: TrackTab, start: number): Partial<DraftState> {
  return tab === 'video' ? { audioTrackStart: start } : { extendAudioTrackStart: start };
}

/** Format seconds as a timestamp: 90 is "1:30", 5400 is "1:30:00" and 1.6 is "0:01.6". */
export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.round(seconds * 10) / 10);
  const whole = Math.floor(total);
  const tenths = Math.round((total - whole) * 10);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const rest = String(whole % 60).padStart(2, '0');
  const clock = hours > 0 ? `${hours}:${String(minutes).padStart(2, '0')}:${rest}` : `${minutes}:${rest}`;
  return tenths > 0 ? `${clock}.${tenths}` : clock;
}

/** Format seconds for the Start at field's note: "90 → 1:30". */
export function startNote(seconds: number): string {
  return `${Number(seconds.toFixed(2))} → ${formatTimestamp(seconds)}`;
}
