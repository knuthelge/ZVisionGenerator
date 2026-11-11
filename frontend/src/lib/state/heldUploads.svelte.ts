import type { TrackTab } from './audioTrack';
import type { ImageTab } from './imageSlots';
import { defaultUploadStore, type TabUploads, type UploadStore, type UploadTab } from './uploadStore';

type FileField = 'startImage' | 'firstFrame' | 'lastFrame' | 'audioTrack' | 'extendAudioTrack' | 'voice' | 'editSource' | 'editMask';

// The files picked from this device that wait for Generate: per tab the start image (the first frame in Video), the
// last frame, and one entry per reference row (null for a row that holds a path), the audio track of Video and Extend
// Video, the recording of the Audio tab's voice, and in the Edit kind the image to edit with the PNG of its painted
// mask. The Workspace is unmounted on another page, so they live here to survive a visit to the Gallery or the Prompts
// page. Every write is mirrored into an upload store (IndexedDB), and `restore` reads it back after a reload. The draft
// keeps only the file's name as the upload marker.
class HeldUploads {
  /** The pending start image of the Image tab, or null. */
  startImage = $state.raw<File | null>(null);
  /** The pending first frame of the Video tab, or null. */
  firstFrame = $state.raw<File | null>(null);
  /** The pending last frame of the Video tab, or null. */
  lastFrame = $state.raw<File | null>(null);
  /** The pending audio track of the Video tab, or null. */
  audioTrack = $state.raw<File | null>(null);
  /** The pending audio track of the Extend kind, or null. */
  extendAudioTrack = $state.raw<File | null>(null);
  /** The pending voice recording of the Audio tab, or null. */
  voice = $state.raw<File | null>(null);
  /** The pending image to edit (the Edit kind), or null. */
  editSource = $state.raw<File | null>(null);
  /** The painted mask of the image to edit, a PNG, or null when nothing is painted. */
  editMask = $state.raw<File | null>(null);
  /** The pending reference uploads by row. */
  references = $state.raw<(File | null)[]>([]);
  /** Whether the stored uploads have been read back. True once `restore` settles, also when the store failed. */
  restored = $state(false);

  #store: UploadStore | null = null;
  #restoring: Promise<void> | null = null;
  // The fields written since the store was set, so a restore that finishes later does not overwrite them.
  #written = new Set<FileField | 'references'>();
  // The writes to the store run one after the other, in the order they were made.
  #queue: Promise<void> = Promise.resolve();
  // Bumped by useStore so work queued for the previous store is dropped.
  #generation = 0;

  /** The upload store: the default one (IndexedDB, else memory) until `useStore` sets another. */
  get store(): UploadStore {
    return (this.#store ??= defaultUploadStore());
  }

  /** Use another upload store and start over: empty fields, not restored. For tests, to simulate a reload over the same store. */
  useStore(store: UploadStore): void {
    this.#store = store;
    this.#restoring = null;
    this.#written.clear();
    this.#queue = Promise.resolve();
    this.#generation += 1;
    this.startImage = null;
    this.firstFrame = null;
    this.lastFrame = null;
    this.audioTrack = null;
    this.extendAudioTrack = null;
    this.voice = null;
    this.editSource = null;
    this.editMask = null;
    this.references = [];
    this.restored = false;
  }

  /** The pending start image of a tab (the first frame in Video), or null. */
  start(tab: ImageTab): File | null {
    return tab === 'image' ? this.startImage : this.firstFrame;
  }

  /** Hold the start image of a tab (the first frame in Video), or forget it with null. */
  setStart(tab: ImageTab, file: File | null): void {
    this.#setFile(tab === 'image' ? 'startImage' : 'firstFrame', file);
    this.#persist(tab);
  }

  /** Hold the last frame, or forget it with null. */
  setLastFrame(file: File | null): void {
    this.#setFile('lastFrame', file);
    this.#persist('video');
  }

  /** Hold the audio track of a tab, or forget it with null. */
  setTrack(tab: TrackTab, file: File | null): void {
    this.#setFile(tab === 'video' ? 'audioTrack' : 'extendAudioTrack', file);
    this.#persist(tab);
  }

  /** Hold the Audio tab's voice recording, or forget it with null. */
  setVoice(file: File | null): void {
    this.#setFile('voice', file);
    this.#persist('audio');
  }

  /** Hold the image to edit, or forget it with null. */
  setEditSource(file: File | null): void {
    this.#setFile('editSource', file);
    this.#persist('edit');
  }

  /** Hold the painted mask of the image to edit, or forget it with null. */
  setEditMask(file: File | null): void {
    this.#setFile('editMask', file);
    this.#persist('edit');
  }

  /** Replace the pending reference uploads. */
  setReferences(files: (File | null)[]): void {
    this.#setReferences(files);
    this.#persist('image');
  }

  /** Forget every pending upload of a tab: the start image and references in Image, the image to edit with its mask in Edit, the first and last frame and the audio track in Video, the audio track in the Extend kind, the voice recording in Audio. */
  clearTab(tab: UploadTab): void {
    this.#forget(tab);
    this.#persist(tab);
  }

  /** Forget every pending upload. */
  clear(): void {
    this.clearTab('image');
    this.clearTab('edit');
    this.clearTab('video');
    this.clearTab('extend');
    this.clearTab('audio');
  }

  /**
   * Read the stored uploads back into the fields. Runs once per store; later calls share it. A field written before the
   * read resolves keeps the written value. Always settles with `restored` true, also when the store fails.
   */
  restore(): Promise<void> {
    return (this.#restoring ??= this.#restore());
  }

  /** Resolve once the writes made so far have reached the store. */
  persisted(): Promise<void> {
    return this.#queue;
  }

  #setFile(field: FileField, file: File | null): void {
    this.#written.add(field);
    this[field] = file;
  }

  #setReferences(files: (File | null)[]): void {
    this.#written.add('references');
    this.references = files;
  }

  #forget(tab: UploadTab): void {
    if (tab === 'image') {
      this.#setFile('startImage', null);
      this.#setReferences([]);
    } else if (tab === 'edit') {
      this.#setFile('editSource', null);
      this.#setFile('editMask', null);
    } else if (tab === 'video') {
      this.#setFile('firstFrame', null);
      this.#setFile('lastFrame', null);
      this.#setFile('audioTrack', null);
    } else if (tab === 'extend') {
      this.#setFile('extendAudioTrack', null);
    } else {
      this.#setFile('voice', null);
    }
  }

  // The record a tab holds now, or null when it holds nothing.
  #record(tab: UploadTab): TabUploads | null {
    const record: TabUploads =
      tab === 'image'
        ? { start: this.startImage, end: null, references: this.references, track: null }
        : tab === 'edit'
          ? { start: this.editSource, end: this.editMask, references: [], track: null }
          : tab === 'video'
            ? { start: this.firstFrame, end: this.lastFrame, references: [], track: this.audioTrack }
            : tab === 'extend'
              ? { start: null, end: null, references: [], track: this.extendAudioTrack }
              : { start: null, end: null, references: [], track: this.voice };
    const empty = record.start === null && record.end === null && record.track === null && record.references.every((file) => file === null);
    return empty ? null : record;
  }

  // Queue the tab's current record for the store. It runs after the restore, started here if the Workspace has not yet, so
  // the record includes the stored fields the write did not change (a write from the Gallery page precedes any restore).
  #persist(tab: UploadTab): void {
    const generation = this.#generation;
    const store = this.store;
    this.#queue = this.#queue.then(async () => {
      try {
        if (generation !== this.#generation) return;
        await this.restore();
        if (generation !== this.#generation) return;
        const record = this.#record(tab);
        if (record === null) await store.delete(tab);
        else await store.put(tab, record);
      } catch {
        // A store that fails loses persistence, never the upload held in memory.
      }
    });
  }

  async #restore(): Promise<void> {
    const generation = this.#generation;
    try {
      const [image, edit, video, extend, audio] = await Promise.all([this.#read('image'), this.#read('edit'), this.#read('video'), this.#read('extend'), this.#read('audio')]);
      if (generation !== this.#generation) return;
      if (!this.#written.has('startImage')) this.startImage = image?.start ?? null;
      if (!this.#written.has('editSource')) this.editSource = edit?.start ?? null;
      if (!this.#written.has('editMask')) this.editMask = edit?.end ?? null;
      if (!this.#written.has('references')) this.references = image?.references ?? [];
      if (!this.#written.has('firstFrame')) this.firstFrame = video?.start ?? null;
      if (!this.#written.has('lastFrame')) this.lastFrame = video?.end ?? null;
      if (!this.#written.has('audioTrack')) this.audioTrack = video?.track ?? null;
      if (!this.#written.has('extendAudioTrack')) this.extendAudioTrack = extend?.track ?? null;
      if (!this.#written.has('voice')) this.voice = audio?.track ?? null;
    } finally {
      if (generation === this.#generation) this.restored = true;
    }
  }

  async #read(tab: UploadTab): Promise<TabUploads | null> {
    try {
      return await this.store.get(tab);
    } catch {
      return null;
    }
  }
}

export const heldUploads = new HeldUploads();
