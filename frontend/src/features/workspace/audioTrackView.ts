import { fileName } from '$lib/components/molecules/assetDetails';
import { formatTimestamp, trackOf, trackSet, type TrackState, type TrackTab } from '$lib/state/audioTrack';
import { trackKey, trackLengths } from '$lib/state/audioTrackLength.svelte';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { hasPath } from '$lib/state/imageSlots';
import { workspaceTab } from '$lib/state/workspaceTab';
import type { DraftState, WorkspaceContext } from '$lib/types';
import { workspaceCapabilities, type WorkspaceCapabilities } from './capabilities';

/** How the Audio track section's header reads: muted with no track, active with one, error when an uploaded track did not come back. */
export type AudioTrackTone = 'muted' | 'active' | 'error';

/**
 * The one answer to whether a Generate follows an audio track. It is derived from the draft, the model's capabilities and
 * the held uploads; the section and its phone popover render it, the summary and the chip format it, Generate reads
 * `block` and the submit reads `sent`.
 */
export interface AudioTrackView {
  /** The open tab, or null in Image, which holds no track. */
  tab: TrackTab | null;
  /** The model follows a track, so the Audio track section is offered. */
  visible: boolean;
  /** A track is chosen, as a path or as an upload. It is kept while the model cannot use it. */
  set: boolean;
  path: string | null;
  /** The file held for the track, or null. */
  file: File | null;
  /** The name to show: the held file's, the path's file name, else the name an upload was stored under. */
  name: string;
  /** Seconds into the track where the clip's sound starts. */
  start: number;
  /** The track's length in seconds, once it is known; null while it is being read or cannot be. */
  length: number | null;
  /** The track is an upload whose file did not come back from a reload. */
  gone: boolean;
  /** Generate's reason when the track cannot be sent, or null. */
  block: string | null;
  /** What a submit sends: null unless the track is in effect and has a path or a file. */
  sent: { path: string | null; file: File | null; start: number } | null;
  /** The header summary: "None", "song.mp3" or "song.mp3 from 1:30". */
  summary: { text: string; tone: AudioTrackTone };
}

/** What `audioTrackView` reads from the draft and the screen. */
export interface AudioTrackViewInput {
  state: TrackState & Pick<DraftState, 'workflow'>;
  caps: Pick<WorkspaceCapabilities, 'showAudioTrack' | 'audioTrackInEffect'>;
  /** The held uploads (`heldUploads`): the track of Video and the track of Extend Video. */
  files: { audioTrack: File | null; extendAudioTrack: File | null };
  /** The held uploads have been read back, so an upload without a file is known to be lost. */
  restored: boolean;
  /** The length in seconds of the track under a key (`trackKey`), or null while unknown. */
  lengthOf: (key: string | null) => number | null;
}

/** Derive whether the open tab follows a track, what a submit sends and why Generate is off when the track is gone. */
export function audioTrackView({ state, caps, files, restored, lengthOf }: AudioTrackViewInput): AudioTrackView {
  const workspace = workspaceTab(state.workflow);
  const tab: TrackTab | null = workspace === 'image' ? null : workspace;
  if (tab === null) {
    return { tab, visible: false, set: false, path: null, file: null, name: '', start: 0, length: null, gone: false, block: null, sent: null, summary: { text: 'None', tone: 'muted' } };
  }
  const fields = trackOf(state, tab);
  const file = tab === 'video' ? files.audioTrack : files.extendAudioTrack;
  const set = trackSet(fields);
  const visible = caps.showAudioTrack;
  const effective = caps.audioTrackInEffect;
  const name = file?.name ?? (hasPath(fields.path) ? fileName(fields.path) : (fields.upload ?? ''));
  // An upload is lost when its marker stayed but its file did not come back; before the restore has settled it is not yet known to be.
  const gone = effective && file === null && !hasPath(fields.path) && fields.upload !== null && restored;
  const sentFile = file ?? null;
  const sentPath = hasPath(fields.path) ? fields.path : null;
  const sent = effective && (sentFile !== null || sentPath !== null) ? { path: sentFile === null ? sentPath : null, file: sentFile, start: fields.start } : null;
  const length = effective ? lengthOf(trackKey(sentPath, file)) : null;
  const text = !set ? 'None' : fields.start > 0 ? `${name} from ${formatTimestamp(fields.start)}` : name;
  return {
    tab,
    visible,
    set,
    path: fields.path,
    file,
    name,
    start: fields.start,
    length,
    gone,
    block: gone ? 'Choose the audio track again.' : null,
    sent,
    summary: { text, tone: gone ? 'error' : set ? 'active' : 'muted' },
  };
}

/** The view of the open tab, read from the draft and the held uploads. */
export function currentAudioTrackView(context: WorkspaceContext): AudioTrackView {
  const state = draft.state;
  return audioTrackView({
    state,
    caps: workspaceCapabilities(context, state),
    files: heldUploads,
    restored: heldUploads.restored,
    lengthOf: (key) => trackLengths.lengthOf(key),
  });
}
