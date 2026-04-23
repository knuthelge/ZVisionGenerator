// Mock EventSource for vitest (jsdom does not implement EventSource)
class MockEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;

  // Tracks the most recently constructed instance — used by SSE tests
  static lastInstance: MockEventSource | null = null;

  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readonly CLOSED = 2;

  readyState = MockEventSource.CONNECTING;
  url: string;
  withCredentials = false;

  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  private listeners: Map<string, Set<EventListener>> = new Map();

  constructor(url: string) {
    this.url = url;
    MockEventSource.lastInstance = this;
    setTimeout(() => {
      this.readyState = MockEventSource.OPEN;
      this.onopen?.(new Event('open'));
    }, 0);
  }

  addEventListener(type: string, listener: EventListener): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(listener);
  }

  removeEventListener(type: string, listener: EventListener): void {
    this.listeners.get(type)?.delete(listener);
  }

  close(): void {
    this.readyState = MockEventSource.CLOSED;
  }

  // Test helper: emit an event programmatically
  emit(type: string, data: unknown): void {
    const event = new MessageEvent(type, { data: JSON.stringify(data) });
    this.listeners.get(type)?.forEach(l => l(event));
    if (type === 'message') this.onmessage?.(event);
  }
}

globalThis.EventSource = MockEventSource as unknown as typeof EventSource;

// Provide a fully-implemented in-memory localStorage for jsdom environments
// that do not implement all Storage methods (e.g. clear, key).
const _localStorageStore: Record<string, string> = {};
const localStorageMock: Storage = {
  getItem(key: string): string | null { return Object.prototype.hasOwnProperty.call(_localStorageStore, key) ? _localStorageStore[key] : null; },
  setItem(key: string, value: string): void { _localStorageStore[key] = String(value); },
  removeItem(key: string): void { delete _localStorageStore[key]; },
  clear(): void { Object.keys(_localStorageStore).forEach(k => { delete _localStorageStore[k]; }); },
  get length(): number { return Object.keys(_localStorageStore).length; },
  key(index: number): string | null { return Object.keys(_localStorageStore)[index] ?? null; }
};
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock, writable: true });
