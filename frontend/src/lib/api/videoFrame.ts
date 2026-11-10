import { api } from './client';

/** A video frame saved as a PNG the Workspace can use as an image. */
export interface SavedFrame {
  /** Where the PNG is on the host. */
  path: string;
}

/** Extract the frame of the clip at *path* that lies at *time* seconds, at full size, into the staging folder. */
export function grabFrame(path: string, time: number): Promise<SavedFrame> {
  return api.post<SavedFrame>('/api/video/frame', { path, time });
}
