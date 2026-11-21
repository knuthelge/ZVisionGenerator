import { api } from './client';
import type { ExtendSourceInfo } from '$lib/types';

export interface ExtendSourceRequest {
  /** Path of the video on the host running the server. */
  path: string;
  /** The new frames the window is sized for; the server snaps it to a multiple of the model's step (a 2 s default when absent). */
  extend_frames?: number | null;
  /** The model that sets the step and the problems reported; the source's or the default one when absent. */
  model?: string | null;
}

/** Read a video chosen for extending: its facts, recorded text, and the problems that stop it. */
export function inspectExtendSource(body: ExtendSourceRequest): Promise<ExtendSourceInfo> {
  return api.post<ExtendSourceInfo>('/api/extend/source', body);
}
