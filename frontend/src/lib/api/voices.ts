import { api } from './client';
import type { JobContext, JobDownloadPlan, SavedVoice, VoiceDesignState } from '$lib/types';

/** What the voice designer is asked: the speech model, how the voice sounds, and the line it reads. */
export interface VoiceDesignBody {
  model: string;
  description: string;
  text: string;
}

function voicePath(slug: string): string {
  return `/api/voices/${encodeURIComponent(slug)}`;
}

/** The address of a saved voice's recording, for a player. */
export function voiceAudioUrl(slug: string): string {
  return `${voicePath(slug)}/audio`;
}

/** The address of a design candidate's audio, for a player. */
export function candidateAudioUrl(candidateId: string): string {
  return `/api/voices/design/${encodeURIComponent(candidateId)}/audio`;
}

/** Return every saved voice. */
export async function listVoices(): Promise<SavedVoice[]> {
  const { voices } = await api.get<{ voices: SavedVoice[] }>('/api/voices');
  return voices;
}

/** Save a recording as a voice. The form carries `model`, `name`, `voice_file` or `voice_path`, `voice_text` and `voice_consent`. */
export function saveRecordingVoice(form: FormData): Promise<SavedVoice> {
  return api.post<SavedVoice>('/api/voices', form);
}

/** Give a voice a new name; its slug stays. */
export function renameVoice(slug: string, name: string): Promise<SavedVoice> {
  return api.post<SavedVoice>(`${voicePath(slug)}/rename`, { name });
}

export function deleteVoice(slug: string): Promise<{ status: string }> {
  return api.delete<{ status: string }>(voicePath(slug));
}

/** What the designer would download: the VoiceDesign files, with their size and licence. */
export function planVoiceDesign(body: VoiceDesignBody): Promise<JobDownloadPlan> {
  return api.post<JobDownloadPlan>('/api/voices/design/plan', body);
}

/** Queue the job that makes 3 candidate voices. */
export function submitVoiceDesign(body: VoiceDesignBody): Promise<JobContext> {
  return api.post<JobContext>('/api/voices/design', body);
}

/** Return every kept candidate, newest batch first, and how many of them wait to be picked. */
export function getDesignState(): Promise<VoiceDesignState> {
  return api.get<VoiceDesignState>('/api/voices/design');
}

/** Save the candidate *candidateId* as a voice named *name*: 404 when it is gone, 409 when the name is taken or the candidate already saved, 422 for an invalid name. */
export function saveCandidate(candidateId: string, name: string): Promise<SavedVoice> {
  return api.post<SavedVoice>(`/api/voices/design/${encodeURIComponent(candidateId)}/save`, { name });
}

/** Remove every kept candidate; refused (409) while a design job is queued or running. */
export function clearCandidates(): Promise<{ status: string }> {
  return api.delete<{ status: string }>('/api/voices/design/candidates');
}
