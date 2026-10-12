import { api } from './client';
import type { VideoMemoryFitInfo } from '$lib/types';
import type { ClipMemoryRequest } from '$lib/state/memoryFit';

/** Ask whether a Text to Video or Image to Video clip of this size and length fits this machine's memory. Nothing is queued. */
export function checkVideoMemoryFit(body: ClipMemoryRequest): Promise<VideoMemoryFitInfo> {
  return api.post<VideoMemoryFitInfo>('/api/video/memory-fit', body);
}
