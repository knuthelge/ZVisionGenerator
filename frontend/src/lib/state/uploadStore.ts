import type { ImageTab } from './imageSlots';

/** The tabs that hold uploads: Image and Video hold images, Video and its Extend kind an audio track, Audio the recording of a voice. */
export type UploadTab = ImageTab | 'extend' | 'audio';

/**
 * The files a tab holds from this device: its start image (the first frame in Video), the last frame, one entry per
 * reference row, and the audio track (Video and its Extend kind); the Audio tab's record keeps its voice recording in `track`.
 */
export interface TabUploads {
  start: File | null;
  end: File | null;
  references: (File | null)[];
  track: File | null;
}

/**
 * Where the uploads wait across a page reload, one record per tab. Every method settles: a store that cannot work answers
 * `get` with null and lets `put` and `delete` do nothing, so a caller never hangs or has to catch.
 */
export interface UploadStore {
  /** Read a tab's record, or null when there is none or the store cannot be read. */
  get(tab: UploadTab): Promise<TabUploads | null>;
  /** Replace a tab's record. */
  put(tab: UploadTab, uploads: TabUploads): Promise<void>;
  /** Drop a tab's record. */
  delete(tab: UploadTab): Promise<void>;
}

const DB_NAME = 'ziv-workspace-uploads';
const DB_VERSION = 1;
const OBJECT_STORE = 'tabs';

function _asFile(value: unknown): File | null {
  return value instanceof File ? value : null;
}

// A record read from disk is untrusted: keep only the files and drop the rest.
function _asTabUploads(value: unknown): TabUploads | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  const references = Array.isArray(record.references) ? record.references.map(_asFile) : [];
  return { start: _asFile(record.start), end: _asFile(record.end), references, track: _asFile(record.track) };
}

// Open the database, or resolve null when it cannot be opened or is blocked by another tab holding an older version.
function _open(factory: IDBFactory): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      const request = factory.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(OBJECT_STORE)) request.result.createObjectStore(OBJECT_STORE);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
      request.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

// Run one request in a transaction and resolve with its result, or null when anything fails.
function _run(db: IDBDatabase, mode: IDBTransactionMode, make: (objects: IDBObjectStore) => IDBRequest): Promise<unknown> {
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(OBJECT_STORE, mode);
      const request = make(tx.objectStore(OBJECT_STORE));
      let result: unknown = null;
      request.onsuccess = () => {
        result = request.result ?? null;
      };
      tx.oncomplete = () => resolve(result);
      tx.onerror = () => resolve(null);
      tx.onabort = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

/** Return an upload store backed by IndexedDB, which keeps File objects as they are. Opens on first use. */
export function indexedDbUploadStore(factory: IDBFactory = globalThis.indexedDB): UploadStore {
  let opened: Promise<IDBDatabase | null> | null = null;
  const open = (): Promise<IDBDatabase | null> => (opened ??= _open(factory));
  return {
    async get(tab) {
      const db = await open();
      return db === null ? null : _asTabUploads(await _run(db, 'readonly', (objects) => objects.get(tab)));
    },
    async put(tab, uploads) {
      const db = await open();
      if (db !== null) await _run(db, 'readwrite', (objects) => objects.put(uploads, tab));
    },
    async delete(tab) {
      const db = await open();
      if (db !== null) await _run(db, 'readwrite', (objects) => objects.delete(tab));
    },
  };
}

/** Return an upload store that keeps the records in memory: for tests, and where IndexedDB is missing. */
export function memoryUploadStore(): UploadStore {
  const records = new Map<UploadTab, TabUploads>();
  return {
    async get(tab) {
      return records.get(tab) ?? null;
    },
    async put(tab, uploads) {
      records.set(tab, uploads);
    },
    async delete(tab) {
      records.delete(tab);
    },
  };
}

/** Return the IndexedDB upload store when the browser has IndexedDB, else the memory one. */
export function defaultUploadStore(): UploadStore {
  return globalThis.indexedDB ? indexedDbUploadStore() : memoryUploadStore();
}
