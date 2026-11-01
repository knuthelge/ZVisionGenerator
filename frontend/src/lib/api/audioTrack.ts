import { api } from './client';

/** What the server reads from an audio file: its length and the name to show. */
export interface AudioTrackInfo {
  seconds: number;
  name: string;
}

/** Read the length of an audio file on the host running the server. Rejects when the file is missing or cannot be read. */
export function inspectAudioTrack(path: string): Promise<AudioTrackInfo> {
  return api.post<AudioTrackInfo>('/api/audio-track/inspect', { path });
}
