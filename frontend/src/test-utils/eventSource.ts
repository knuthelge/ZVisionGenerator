/** The transport the page opened last: the `MockEventSource` that `test-setup.ts` installs. */
export interface MockEventSourceHandle {
  emit: (type: string, data: unknown) => void;
  emitError: () => void;
  emitFatalError: () => void;
  close: () => void;
  closeCalls: number;
  readyState: number;
  url: string;
}

/** Return the most recently opened mock EventSource. */
export function lastEventSource(): MockEventSourceHandle {
  return (globalThis.EventSource as unknown as { lastInstance: MockEventSourceHandle }).lastInstance;
}
