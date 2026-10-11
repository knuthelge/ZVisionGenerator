import { api } from './client';
import type { ExtendSourceInfo } from '$lib/types';

export interface ExtendSourceRequest {
  /** Path of the video on the host running the server. */
  path: string;
  /** The new frames to size the memory estimate for; the server snaps it to a multiple of 8 (a 2 s default when absent). */
  extend_frames?: number | null;
  /** The model whose memory use is estimated; the source's or the default one when absent. */
  model?: string | null;
  low_memory?: boolean | null;
}

/** Read a video chosen for extending: its facts, recorded text, problems that stop it, and the memory the extension needs. */
export function inspectExtendSource(body: ExtendSourceRequest): Promise<ExtendSourceInfo> {
  return api.post<ExtendSourceInfo>('/api/extend/source', body);
}
