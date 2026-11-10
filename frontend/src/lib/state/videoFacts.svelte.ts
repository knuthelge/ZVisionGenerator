import { browseInfo } from '$lib/api/browse';
import { videoFactsOf, type VideoFacts } from './framePoint';

// The frame facts of the videos seen this session, from `GET /api/browse/info`. A player asks once per file. An answer that
// carries no usable facts is remembered as having none; a request that fails is forgotten, so the next view asks again.
class VideoFactsStore {
  #fetched = $state.raw<Record<string, VideoFacts | null>>({});
  #wanted = new Set<string>();

  /** The facts of the video at *path*: `null` until they arrive, and for a video that has none. */
  of(path: string): VideoFacts | null {
    return this.#fetched[path] ?? null;
  }

  /** The length in seconds of the video at *path*, counted in frames: `null` until the facts arrive, and for a video that has none. */
  lengthOf(path: string): number | null {
    const facts = this.of(path);
    return facts ? facts.frames / facts.fps : null;
  }

  /** Ask the server for the facts of the video at *path*; once per path. */
  ensure(path: string): void {
    if (this.#wanted.has(path)) return;
    this.#wanted.add(path);
    browseInfo('source_video', path).then(
      (info) => {
        this.#fetched = { ...this.#fetched, [path]: videoFactsOf(info.fps, info.frames, info.start) };
      },
      () => {
        this.#wanted.delete(path);
      }
    );
  }

  /** Forget every video. For tests. */
  reset(): void {
    this.#fetched = {};
    this.#wanted.clear();
  }
}

export const videoFacts = new VideoFactsStore();
