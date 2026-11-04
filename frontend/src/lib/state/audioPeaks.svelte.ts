import { getAudioPeaks } from '$lib/api/gallery';
import type { GalleryAsset } from '$lib/types';

/** How many waveforms the browser asks the server for at once; each is one decode of an audio file. */
const MAX_IN_FLIGHT = 3;

type PeaksAsset = Pick<GalleryAsset, 'id' | 'media_type' | 'peaks'>;

// The waveforms of the takes seen this session. The server decodes a file the first time its waveform is asked for and keeps
// it, so a listing arrives without them (`peaks` null) and each tile or viewer asks once. A take whose waveform could not be
// read is remembered as empty (a flat line), so it is not asked for again on every change; a reload asks anew.
class AudioPeaks {
  #fetched = $state.raw<Record<string, readonly number[]>>({});
  #wanted = new Set<string>();
  #queue: string[] = [];
  #inFlight = 0;

  /** The peaks to draw for *asset*: the ones its listing carried, else the ones fetched; empty (a flat line) until they arrive. */
  of(asset: Pick<GalleryAsset, 'id' | 'peaks'>): readonly number[] {
    return asset.peaks ?? this.#fetched[asset.id] ?? [];
  }

  /** Ask the server for the peaks of an audio *asset* whose listing had none; once per asset, a few at a time. */
  ensure(asset: PeaksAsset): void {
    if (asset.media_type !== 'audio' || asset.peaks != null || asset.id in this.#fetched || this.#wanted.has(asset.id)) return;
    this.#wanted.add(asset.id);
    this.#queue.push(asset.id);
    this.#pump();
  }

  #pump(): void {
    while (this.#inFlight < MAX_IN_FLIGHT && this.#queue.length > 0) {
      const id = this.#queue.shift() as string;
      this.#inFlight += 1;
      void getAudioPeaks(id)
        .catch((): number[] => [])
        .then((peaks) => {
          this.#inFlight -= 1;
          this.#fetched = { ...this.#fetched, [id]: peaks };
          this.#pump();
        });
    }
  }

  /** Forget every waveform. For tests. */
  reset(): void {
    this.#fetched = {};
    this.#wanted.clear();
    this.#queue = [];
    this.#inFlight = 0;
  }
}

export const audioPeaks = new AudioPeaks();
