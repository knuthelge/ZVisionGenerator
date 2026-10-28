
/**
 * One reference image row of the Workspace. A row is either a path on the host or a file uploaded from this device;
 * an uploaded row has an empty `path` until it is sent. The row's position is its number in the prompt ("image 1").
 */
export interface ReferenceRow {
  path: string;
  file: File | null;
}

/** Pair the draft's paths with the pending uploads that stand in for them; a missing upload counts as none. */
export function referenceRows(paths: readonly string[], files: readonly (File | null | undefined)[]): ReferenceRow[] {
  return paths.map((path, index) => ({ path, file: files[index] ?? null }));
}

// The input image types by extension. The server's list is IMAGE_EXTENSIONS in zvisiongenerator/utils/input_images.py.
const IMAGE_TYPES: Record<string, string> = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' };

/** The `accept` value of every image file input: the types the server takes. */
export const IMAGE_ACCEPT = [...new Set(Object.values(IMAGE_TYPES))].join(',');

/** Whether the file looks like an input image the server takes, by its extension; the server checks the contents. */
function isSupportedImage(file: File): boolean {
  const name = file.name.toLowerCase();
  return Object.keys(IMAGE_TYPES).some((extension) => name.endsWith(extension));
}

/** The files of a drop or a choice that are supported images, and how many were not (a folder, a document, a GIF). */
export function splitImageFiles(files: Iterable<File>): { supported: File[]; unsupported: number } {
  const all = [...files];
  const supported = all.filter(isSupportedImage);
  return { supported, unsupported: all.length - supported.length };
}

/** The toast for a single dropped file that is not a supported image. */
export function unsupportedMessage(count: number): string {
  return count === 1 ? "That file isn't a PNG, JPEG or WebP." : `${count} files aren't PNG, JPEG or WebP.`;
}

/** The toast for an append where some files were not taken, saying why (no room, or not a supported type), or null when all were. */
export function skippedMessage(added: number, skipped: number, cap: number, unsupported = 0): string | null {
  const reasons = [
    ...(skipped > 0 ? [`${skipped} didn't fit (${cap} is the most)`] : []),
    ...(unsupported > 0 ? [unsupported === 1 ? "1 isn't a PNG, JPEG or WebP" : `${unsupported} aren't PNG, JPEG or WebP`] : []),
  ];
  if (reasons.length === 0) return null;
  return `Added ${added} image${added === 1 ? '' : 's'}; ${reasons.join('; ')}.`;
}

/**
 * Whether a row's image is gone: its path is flagged missing (Reuse, the Workspace's check, or a delete found it), or it
 * is an upload whose file did not survive leaving the Workspace or a reload (an empty path and no file). Either way the
 * row keeps its number.
 */
export function isMissingRow(row: ReferenceRow, missingPaths: readonly string[]): boolean {
  if (row.file !== null) return false;
  return row.path === '' || missingPaths.includes(row.path);
}

/** What to tell the user about a missing row: an upload lost to a reload is picked again, a moved path may also be removed. */
export function missingReason(row: ReferenceRow): string {
  return row.path === '' ? 'Choose this image again.' : 'This image has moved or been deleted. Choose it again or remove it.';
}

/**
 * Generate's reason when a reference is missing, or null when every row can be read. Dropping the row would renumber
 * the prompt's "image 2", so the row stays and the user chooses the file again or removes it.
 */
export function referenceGuard(rows: readonly ReferenceRow[], missingPaths: readonly string[]): string | null {
  const first = rows.findIndex((row) => isMissingRow(row, missingPaths));
  return first === -1 ? null : `Image ${first + 1} is missing.`;
}
