// The images picked from this device that wait for Generate: the start image, the last frame, and one entry per
// reference row (null for a row that holds a path). The Workspace is unmounted on another page, so they live here to
// survive a visit to the Gallery or the Prompts page; a reload loses them, and the slots then show only what the
// draft's paths say. Not draft state: a File cannot be stored with the draft.
class HeldUploads {
  /** The pending start image, or null. */
  image = $state.raw<File | null>(null);
  /** The pending last frame, or null. */
  endImage = $state.raw<File | null>(null);
  /** The pending reference uploads by row. */
  references = $state.raw<(File | null)[]>([]);

  /** Forget the start image. */
  clearStartImage(): void {
    this.image = null;
  }

  /** Forget the last frame. */
  clearLastFrame(): void {
    this.endImage = null;
  }

  /** Forget the start image and the last frame. */
  clearFrames(): void {
    this.clearStartImage();
    this.clearLastFrame();
  }

  /** Forget every pending reference upload. */
  clearReferences(): void {
    this.references = [];
  }

  /** Forget every pending upload. */
  clear(): void {
    this.clearFrames();
    this.clearReferences();
  }
}

export const heldUploads = new HeldUploads();
