/** How long the browser gets to report an audio file's length from its header before decoding the file instead. */
const METADATA_TIMEOUT_MS = 4000;

/** Return a valid length in seconds, else null (a stream without a header reports `Infinity` or `NaN`). */
function usable(seconds: number): number | null {
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

/** Ask an `Audio` element for the file's length from its header. */
function readFromMetadata(file: File): Promise<number | null> {
  if (typeof URL.createObjectURL !== 'function' || typeof Audio === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    const timer = setTimeout(() => finish(null), METADATA_TIMEOUT_MS);
    function finish(seconds: number | null): void {
      clearTimeout(timer);
      audio.onloadedmetadata = null;
      audio.onerror = null;
      audio.removeAttribute('src');
      URL.revokeObjectURL(url);
      resolve(seconds);
    }
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => finish(usable(audio.duration));
    audio.onerror = () => finish(null);
    audio.src = url;
  });
}

/** Decode the whole file to find its length; slower and heavier than the header, but works where the header is silent (some phones). */
async function readFromDecode(file: File): Promise<number | null> {
  if (typeof OfflineAudioContext === 'undefined') return null;
  const context = new OfflineAudioContext(1, 1, 44100);
  const buffer = await context.decodeAudioData(await file.arrayBuffer());
  return usable(buffer.duration);
}

/**
 * Read the length in seconds of an audio file picked on this device, or null when the browser cannot tell. It reads the
 * file's header first and decodes the file only when that gives nothing, so a long song costs no memory in the usual case.
 */
export async function readFileLength(file: File): Promise<number | null> {
  try {
    const fromHeader = await readFromMetadata(file);
    if (fromHeader !== null) return fromHeader;
    return await readFromDecode(file);
  } catch {
    return null;
  }
}
