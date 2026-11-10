import { vi } from 'vitest';

/** What a fake media element reports. */
interface MediaState {
  paused: boolean;
  ended: boolean;
  time: number;
  duration: number;
  ready: number;
  seeking: boolean;
  volume: number;
  muted: boolean;
}

/** Control over the fake media elements of one test. */
export interface FakeMedia {
  /** Every value `currentTime` was set to on *element*, in order. */
  seeks(element: HTMLMediaElement): number[];
  /** Give *element* a length and say its metadata has loaded, as the browser does after it fetches the file. */
  load(element: HTMLMediaElement, duration?: number): void;
  /** Finish the seek that is in flight on *element*: the picture has landed. */
  land(element: HTMLMediaElement): void;
  /** Make *element*'s clock read *time* without a seek, as playback does. */
  tick(element: HTMLMediaElement, time: number): void;
  /** Play *element* through to its end. */
  finish(element: HTMLMediaElement): void;
  /** Show a picture on *element* whose time is *mediaTime*, as the browser does: the video's frame callbacks run with it. Needs `{ frameCallbacks: true }`. */
  present(element: HTMLMediaElement, mediaTime: number): void;
  /** The elements `play` and `pause` were called on. */
  played: HTMLMediaElement[];
  paused: HTMLMediaElement[];
  /** The elements the browser was told to load again (`element.load()`), which with no `src` left is how one is let go. */
  reloaded: HTMLMediaElement[];
}

/**
 * Make jsdom's media elements behave like a browser's, as far as the player reads them: a clock that moves when it is set,
 * `seeking` until a seek lands, and `play` and `pause` that change `paused` and fire their events. The spies on the elements'
 * properties are undone by `vi.restoreAllMocks`; `requestVideoFrameCallback` and `cancelVideoFrameCallback` on the video prototype
 * are set or deleted outright by each call, and stay as the last call left them.
 * The events are plain dispatches: a test calls `flushSync` afterwards to let the player react.
 * With `frameCallbacks`, videos also have `requestVideoFrameCallback`, which a test answers with `present`; without it they have none, as in a browser that lacks it.
 */
export function installFakeMedia(options: { frameCallbacks?: boolean } = {}): FakeMedia {
  const states = new WeakMap<HTMLMediaElement, MediaState>();
  const seeks = new WeakMap<HTMLMediaElement, number[]>();
  const stateOf = (element: HTMLMediaElement): MediaState => {
    let state = states.get(element);
    if (!state) {
      state = { paused: true, ended: false, time: 0, duration: Number.NaN, ready: 0, seeking: false, volume: 1, muted: false };
      states.set(element, state);
      seeks.set(element, []);
    }
    return state;
  };
  const proto = HTMLMediaElement.prototype;
  const videoProto = HTMLVideoElement.prototype as Partial<Record<'requestVideoFrameCallback' | 'cancelVideoFrameCallback', unknown>>;
  delete videoProto.requestVideoFrameCallback;
  delete videoProto.cancelVideoFrameCallback;
  if (options.frameCallbacks) {
    Object.defineProperty(videoProto, 'requestVideoFrameCallback', {
      configurable: true,
      writable: true,
      value(this: HTMLVideoElement, callback: VideoFrameRequestCallback): number {
        const waiting = callbacks.get(this) ?? new Map<number, VideoFrameRequestCallback>();
        callbacks.set(this, waiting);
        waiting.set(nextHandle, callback);
        return nextHandle++;
      },
    });
    Object.defineProperty(videoProto, 'cancelVideoFrameCallback', {
      configurable: true,
      writable: true,
      value(this: HTMLVideoElement, handle: number): void {
        callbacks.get(this)?.delete(handle);
      },
    });
  }
  const fire = (element: HTMLMediaElement, type: string): void => { element.dispatchEvent(new Event(type)); };
  const callbacks = new WeakMap<HTMLMediaElement, Map<number, VideoFrameRequestCallback>>();
  let nextHandle = 1;
  const fake: FakeMedia = {
    present(element, mediaTime) {
      const waiting = callbacks.get(element);
      if (!waiting) return;
      const run = [...waiting.values()];
      waiting.clear();
      for (const callback of run) callback(0, { mediaTime } as VideoFrameCallbackMetadata);
    },
    played: [],
    paused: [],
    reloaded: [],
    seeks: (element) => { stateOf(element); return seeks.get(element)!; },
    load(element, duration = 6) {
      const state = stateOf(element);
      state.duration = duration;
      state.ready = 4;
      fire(element, 'loadedmetadata');
    },
    land(element) {
      stateOf(element).seeking = false;
      fire(element, 'seeked');
    },
    tick(element, time) {
      stateOf(element).time = time;
      fire(element, 'timeupdate');
    },
    finish(element) {
      const state = stateOf(element);
      state.time = state.duration;
      state.paused = true;
      state.ended = true;
      fire(element, 'pause');
      fire(element, 'ended');
    },
  };
  vi.spyOn(proto, 'paused', 'get').mockImplementation(function (this: HTMLMediaElement) { return stateOf(this).paused; });
  vi.spyOn(proto, 'ended', 'get').mockImplementation(function (this: HTMLMediaElement) { return stateOf(this).ended; });
  vi.spyOn(proto, 'duration', 'get').mockImplementation(function (this: HTMLMediaElement) { return stateOf(this).duration; });
  vi.spyOn(proto, 'readyState', 'get').mockImplementation(function (this: HTMLMediaElement) { return stateOf(this).ready; });
  vi.spyOn(proto, 'seeking', 'get').mockImplementation(function (this: HTMLMediaElement) { return stateOf(this).seeking; });
  vi.spyOn(proto, 'currentTime', 'get').mockImplementation(function (this: HTMLMediaElement) { return stateOf(this).time; });
  vi.spyOn(proto, 'currentTime', 'set').mockImplementation(function (this: HTMLMediaElement, value: number) {
    const state = stateOf(this);
    state.time = value;
    state.ended = false;
    state.seeking = true;
    seeks.get(this)!.push(value);
  });
  vi.spyOn(proto, 'volume', 'get').mockImplementation(function (this: HTMLMediaElement) { return stateOf(this).volume; });
  vi.spyOn(proto, 'volume', 'set').mockImplementation(function (this: HTMLMediaElement, value: number) { stateOf(this).volume = value; });
  vi.spyOn(proto, 'muted', 'get').mockImplementation(function (this: HTMLMediaElement) { return stateOf(this).muted; });
  vi.spyOn(proto, 'muted', 'set').mockImplementation(function (this: HTMLMediaElement, value: boolean) { stateOf(this).muted = value; });
  vi.spyOn(proto, 'play').mockImplementation(function (this: HTMLMediaElement) {
    const state = stateOf(this);
    if (state.ended) { state.ended = false; state.time = 0; }
    state.paused = false;
    fake.played.push(this);
    fire(this, 'play');
    return Promise.resolve();
  });
  vi.spyOn(proto, 'load').mockImplementation(function (this: HTMLMediaElement) { fake.reloaded.push(this); });
  vi.spyOn(proto, 'pause').mockImplementation(function (this: HTMLMediaElement) {
    const state = stateOf(this);
    if (state.paused) return;
    state.paused = true;
    fake.paused.push(this);
    fire(this, 'pause');
  });
  return fake;
}
