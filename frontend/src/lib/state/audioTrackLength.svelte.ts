import { inspectAudioTrack } from '$lib/api/audioTrack';
import { readFileLength } from './audioLength';
import { hasPath } from './imageSlots';

/** The key a track's length is remembered by: the file held for it (name, size, age), else its path; null when neither. */
export function trackKey(path: string | null, file: File | null): string | null {
  if (file !== null) return `file:${file.name}:${file.size}:${file.lastModified}`;
  return hasPath(path) ? `path:${path}` : null;
}

// The lengths of the tracks seen this session. A track whose length could not be read is remembered as null, so it is not
// asked for again on every change; a reload asks anew.
class TrackLengths {
  #lengths = $state.raw<Record<string, number | null>>({});
  #pending = new Set<string>();

  /** The length in seconds of the track under *key*, or null while it is unknown, unreadable or *key* is null. */
  lengthOf(key: string | null): number | null {
    return key === null ? null : (this.#lengths[key] ?? null);
  }

  /** Look up the length of a track once: a file is read in the browser, a path is asked of the server. */
  ensure(path: string | null, file: File | null): void {
    const key = trackKey(path, file);
    if (key === null || key in this.#lengths || this.#pending.has(key)) return;
    this.#pending.add(key);
    const read = file !== null ? readFileLength(file) : inspectAudioTrack(path ?? '').then((info) => info.seconds);
    void read
      .catch(() => null)
      .then((seconds) => {
        this.#pending.delete(key);
        this.#lengths = { ...this.#lengths, [key]: seconds !== null && Number.isFinite(seconds) && seconds > 0 ? seconds : null };
      });
  }

  /** Forget every length. For tests. */
  reset(): void {
    this.#lengths = {};
    this.#pending.clear();
  }
}

export const trackLengths = new TrackLengths();
