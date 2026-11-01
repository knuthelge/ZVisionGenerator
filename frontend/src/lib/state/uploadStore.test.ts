import { afterEach, describe, expect, it, vi } from 'vitest';
import { defaultUploadStore, indexedDbUploadStore, memoryUploadStore, type TabUploads } from './uploadStore';

const file = (name: string): File => new File(['x'], name, { type: 'image/png' });

describe('uploadStore', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('memoryUploadStore round-trips and deletes a tab record', async () => {
    const store = memoryUploadStore();
    const record: TabUploads = { start: file('start.png'), end: null, references: [file('ref.png'), null], track: null };

    expect(await store.get('image')).toBeNull();
    await store.put('image', record);
    expect(await store.get('image')).toEqual(record);
    expect(await store.get('video')).toBeNull();

    await store.delete('image');
    expect(await store.get('image')).toBeNull();
  });

  it('defaultUploadStore falls back to memory without indexedDB', async () => {
    vi.stubGlobal('indexedDB', undefined);
    const store = defaultUploadStore();
    const record: TabUploads = { start: file('first.png'), end: file('last.png'), references: [], track: null };

    await store.put('video', record);
    expect(await store.get('video')).toEqual(record);
  });

  it.each(['onblocked', 'onerror'] as const)('a blocked or failed open resolves get to null (%s)', async (event) => {
    const factory = {
      open() {
        const request: Record<string, unknown> = {};
        queueMicrotask(() => (request[event] as () => void)());
        return request;
      },
    } as unknown as IDBFactory;
    const store = indexedDbUploadStore(factory);

    expect(await store.get('image')).toBeNull();
    await expect(store.put('image', { start: file('a.png'), end: null, references: [], track: null })).resolves.toBeUndefined();
    await expect(store.delete('image')).resolves.toBeUndefined();
  });

  it('a factory that throws resolves get to null', async () => {
    const factory = {
      open() {
        throw new Error('denied');
      },
    } as unknown as IDBFactory;

    expect(await indexedDbUploadStore(factory).get('video')).toBeNull();
  });
  // A just-enough IndexedDB: one object store in a Map, requests and transactions that fire their handlers in a microtask.
  function fakeFactory(records: Map<string, unknown>, options: { failTransaction?: boolean } = {}): IDBFactory {
    const stores = new Set<string>();
    const db = {
      objectStoreNames: { contains: (name: string) => stores.has(name) },
      createObjectStore: (name: string) => void stores.add(name),
      transaction() {
        const tx: Record<string, (() => void) | undefined | unknown> = {};
        tx.objectStore = () => {
          const request = (run: () => unknown) => {
            const req: Record<string, unknown> = { result: undefined };
            queueMicrotask(() => {
              if (options.failTransaction) {
                (tx.onabort as () => void)();
                return;
              }
              req.result = run();
              (req.onsuccess as () => void)();
              (tx.oncomplete as () => void)();
            });
            return req;
          };
          return {
            get: (key: string) => request(() => records.get(key)),
            put: (value: unknown, key: string) => request(() => void records.set(key, value)),
            delete: (key: string) => request(() => void records.delete(key)),
          };
        };
        return tx;
      },
    };
    return {
      open() {
        const request: Record<string, unknown> = { result: db };
        queueMicrotask(() => {
          (request.onupgradeneeded as () => void)();
          (request.onsuccess as () => void)();
        });
        return request;
      },
    } as unknown as IDBFactory;
  }

  it('indexedDbUploadStore round-trips and deletes a tab record through the database', async () => {
    const records = new Map<string, unknown>();
    const store = indexedDbUploadStore(fakeFactory(records));
    const record: TabUploads = { start: file('first.png'), end: file('last.png'), references: [], track: null };

    expect(await store.get('video')).toBeNull();
    await store.put('video', record);
    expect(records.has('video')).toBe(true);
    expect(await store.get('video')).toEqual(record);
    expect(await store.get('image')).toBeNull();

    await store.delete('video');
    expect(records.has('video')).toBe(false);
    expect(await store.get('video')).toBeNull();
  });

  it('keeps only files from a stored record', async () => {
    const records = new Map<string, unknown>([['image', { start: 'not a file', end: 3, references: [file('ref.png'), 'x', null] }]]);
    const stored = await indexedDbUploadStore(fakeFactory(records)).get('image');

    expect(stored).toEqual({ start: null, end: null, references: [file('ref.png'), null, null], track: null });
  });

  it('an aborted transaction resolves get to null and lets put and delete settle', async () => {
    const store = indexedDbUploadStore(fakeFactory(new Map(), { failTransaction: true }));

    expect(await store.get('image')).toBeNull();
    await expect(store.put('image', { start: file('a.png'), end: null, references: [], track: null })).resolves.toBeUndefined();
    await expect(store.delete('image')).resolves.toBeUndefined();
  });
});
