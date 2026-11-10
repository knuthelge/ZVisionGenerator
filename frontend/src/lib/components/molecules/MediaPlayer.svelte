<script lang="ts" module>
  type FullscreenVideo = HTMLVideoElement & { webkitEnterFullscreen?: () => void };

  /** How long the picture, or the player's chrome, stays quiet before the chrome fades while a video plays. */
  const IDLE_MS = 2600;
  /** How long a load or a stall goes on before a spinner shows. */
  const SPINNER_MS = 400;
  /** How long a seek is in flight before the loupe shows what the playhead is going to. */
  const SLOW_SEEK_MS = 150;
  /** How much of each new picture the ambient light takes in while a video plays, so its colours glide rather than step. */
  const AMBIENT_BLEND = 0.3;
  /** The ambient light's width in pixels; its height follows the pane's shape. */
  const AMBIENT_COLUMNS = 32;
  /** How far the ambient light reaches past the pane, in CSS pixels, so its soft edge is never in view. */
  const AMBIENT_REACH = 104;
  /** How far the ambient light spreads past the picture, at most, in CSS pixels. */
  const GLOW_REACH = 160;
  /** The softening of the ambient light, in its own pixels (one is about a thirty-second of the pane's width). */
  const AMBIENT_SOFTEN = 1.4;
  /** How long a viewer's video stays on its metadata before the browser may load the rest, so a clip paged past never downloads. */
  const WARM_MS = 300;
  /** The big Play's sizes, largest first, in CSS pixels; a picture with no room for any gets the smallest. */
  const CENTRE_SIZES = [72, 52, 44] as const;
  const CENTRE_LEAST = 36;
  /** What one step is worth to a video whose frame rate is not known yet. */
  const FALLBACK_STEP_SECONDS = 1 / 30;

  /** The player last used (pressed, tapped or played): Space with no focus in a player or a field plays and pauses it. */
  let engagedPlayer: symbol | null = null;

  const TOKEN_NAMES = {
    rest: '--color-hairline',
    played: '--color-primary-main',
    tick: '--color-hairline-tick',
    shade: '--color-hairline-shade',
    playhead: '--color-text-primary',
    bloomRest: '--color-bloom-rest',
    waveRest: '--color-wave-rest',
  } as const;
</script>

<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { Icon, Spinner } from '$lib/components/atoms';
  import ActionMenu, { type ActionMenuEntry, type ActionMenuItem } from './ActionMenu.svelte';
  import { seekFraction } from '$lib/state/audioAsset';
  import { frameAt, framePoint, pointLabel, stepTarget, type FramePoint, type MediaSnapshot } from '$lib/state/framePoint';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { mediaVolume } from '$lib/state/mediaVolume.svelte';
  import { claimPlayback, releasePlayback } from '$lib/state/nowPlaying';
  import { videoFacts } from '$lib/state/videoFacts.svelte';
  import { spriteTile, videoSprite } from '$lib/state/videoSprite.svelte';
  import { ambientRegion, glowMask, glowShows, glowSpread, leavesBands, shownPicture, type Box } from './mediaAmbient';
  import { FRAME_BUSY_LABEL } from './frameMenu';
  import { navigatesByKey, playerKeyCommand, type KeyTarget, type PlayerCommand } from './mediaKeys';
  import { hasOpenModal, isPlainKey, isTyping, isUnfocusedTarget } from '$lib/keyboard';
  import { loupeCentre, loupeImage } from './mediaLoupe';
  import { TINY_PLAYER } from './playerSize';
  import { canSetVolume } from './mediaVolumeProbe';
  import { moveScrub, startScrub, type Scrub } from './mediaScrub';
  import { readsAsClock, secondsLabel, timeLabel } from './mediaTime';
  import { drawVideoTimeline, drawWaveform, easeBloom, wavePlayedBars, type TimelineColors } from './mediaTimeline';
  import { COMPACT_PICTURE, TIGHT_PICTURE, deckFor, deckMetrics } from './mediaDeck';

  interface Props {
    /**
     * Where the browser loads the clip or take from. A caller keys the player on it (`{#key src}`): a source change on a playing
     * element goes back to paused without a `pause` event, which can leave its button on Pause, and a fresh player starts on Play.
     */
    src: string;
    kind: 'video' | 'audio';
    /** The file's name, which names the player for a screen reader. */
    label: string;
    /** The still shown before the first play. */
    poster?: string;
    /** The host path of the video, which the frame facts and the contact sheet are fetched by. Without it the player works in plain seconds. */
    filePath?: string;
    /** A take's peaks (0 to 255 each), drawn as its waveform. */
    peaks?: readonly number[];
    /** The length in seconds, used until the browser has read it from the file. */
    duration?: number | null;
    /** Start muted whatever the session's volume is: the stage plays each take silent first. */
    muted?: boolean;
    /** `viewer`: the asset viewer's. `stage`: the Workspace's preview. `pane`: the file browser's, with no volume, fullscreen or loupe. */
    size: 'viewer' | 'stage' | 'pane';
    /** The rows of the paused frame's menu, built by the caller; without any the time stays a plain readout. A row marked `busy` puts a spinner on the time. */
    actions?: readonly ActionMenuItem[];
    /** The pane the viewer's player sits in, padding included, which its ambient light fills; without it the light fills the player. */
    pane?: Box;
    /** The clip's own size, from the gallery's data: the picture and its controls are laid out from the first paint, not when the file has been read. */
    naturalSize?: { width: number; height: number } | null;
    /** Called with the frame a paused video is on, or `null` when nothing is selected. */
    onpoint?: (point: FramePoint | null) => void;
  }

  let { src, kind, label, poster, filePath, peaks = [], duration = null, muted, size, actions = [], pane, naturalSize = null, onpoint }: Props = $props();

  const isVideo = $derived(kind === 'video');
  // Only a video outside the file browser's pane fades its chrome; the pane and a take's panel are plain rows.
  const fades = $derived(isVideo && size !== 'pane');
  const viewer = $derived(size === 'viewer');

  let root = $state<HTMLDivElement | null>(null);
  let video = $state<HTMLVideoElement | null>(null);
  let audio = $state<HTMLAudioElement | null>(null);
  let ambient = $state<HTMLCanvasElement | null>(null);
  let timeline = $state<HTMLDivElement | null>(null);
  let timeEl = $state<HTMLElement | null>(null);
  let canvas = $state<HTMLCanvasElement | null>(null);
  let trackWidth = $state(0);
  let trackHeight = $state(0);
  // The player's own size and the video's picture, which say whether the picture leaves bands for the ambient light.
  let stageWidth = $state(0);
  let stageHeight = $state(0);
  let pictureWidth = $state(0);
  let pictureHeight = $state(0);
  // The source the viewer's video has been let load in full; until then it holds to its metadata.
  let warmed = $state<string | null>(null);
  const media = $derived<HTMLMediaElement | null>(video ?? audio);

  // What the element reports; `sync` copies it, on the element's events and once a frame while it plays.
  let elPaused = $state(true);
  let ended = $state(false);
  let clock = $state(0);
  // The time of the frame on screen, from the browser's own report of each picture it shows; `null` until one is reported, after a seek and where the browser has no such report.
  // `clock` runs ahead of the picture while a video plays and after it pauses, by a frame or three.
  let presented = $state<number | null>(null);
  let mediaDuration = $state(Number.NaN);
  let ready = $state(0);
  let failed = $state(false);
  let stalled = $state(false);
  let loadingSlow = $state(false);

  // The playhead's target while a seek is on its way: the picture, the time and the actions all read it, so none waits for the browser.
  let target = $state<number | null>(null);
  let scrubbing = $state(false);
  // A drag while playing pauses the picture and lets it go again on release; to the user it never stopped.
  let resumeAfter = $state(false);
  let fine = $state(false);
  let slow = $state(false);

  const loaded = $derived(ready >= 1 && Number.isFinite(mediaDuration) && mediaDuration > 0);
  const total = $derived(loaded ? mediaDuration : (duration ?? 0));
  const position = $derived(target ?? presented ?? clock);
  const paused = $derived(resumeAfter ? false : elPaused);
  const playing = $derived(!paused && !ended);

  const facts = $derived(isVideo && filePath ? videoFacts.of(filePath) : null);
  const lastFrame = $derived(facts ? facts.frames - 1 : null);
  const frameNow = $derived(facts ? frameAt(position, facts) : null);
  const fraction = $derived.by(() => {
    if (facts && lastFrame !== null && lastFrame > 0 && frameNow !== null) return frameNow / lastFrame;
    return total > 0 ? Math.min(1, Math.max(0, position / total)) : 0;
  });

  const shownSeconds = $derived(facts && frameNow !== null ? frameNow / facts.fps : position);
  const totalSeconds = $derived(facts ? facts.frames / facts.fps : total);
  const asClock = $derived(readsAsClock(kind, totalSeconds));
  const nowText = $derived(timeLabel(shownSeconds, asClock));
  const totalText = $derived(totalSeconds > 0 ? timeLabel(totalSeconds, asClock) : '');

  const snapshot = $derived<MediaSnapshot>({ paused, ended: ended && target === null, time: position, error: failed, duration: loaded ? mediaDuration : 0 });
  const point = $derived(isVideo ? framePoint(snapshot, facts) : null);
  const atEnd = $derived(point?.atEnd === true || (ended && target === null));
  // The time readout of a paused video that has a selection and something to offer is the menu's button; a scrub or fullscreen puts it away.
  let fullscreen = $state(false);
  let menuOpen = $state(false);
  // A tiny player has room for no menu button (the tile's ⋯ menu offers the same rows).
  const tiny = $derived(isVideo && stageHeight > 0 && stageHeight < TINY_PLAYER);
  const hasMenu = $derived(isVideo && point !== null && !scrubbing && !fullscreen && !failed && !tiny && actions.length > 0);
  const busyNow = $derived(actions.some((action) => action.busy === true));
  // The frame's menu is the app's own: a heading with the frame, then a row per action; a row that can't be used shows why.
  // On a phone the menu is a sheet whose title already names the frame, so the heading is left out there.
  const frameTitle = $derived(point ? (point.atEnd ? `${pointLabel(point)}, the last frame` : `${pointLabel(point)}, this frame`) : 'Use this frame');
  const frameMenu = $derived<ActionMenuEntry[]>(point
    ? [
        ...(layout.compact ? [] : [{ kind: 'heading', label: frameTitle } as ActionMenuEntry]),
        // A phone's sheet keeps to the rows, so it opens under the picture; a row that can't be used still says why.
        ...actions.map((action): ActionMenuEntry => (layout.compact ? { ...action, description: undefined } : action)),
      ]
    : []);

  const coarse = $derived(layout.coarse);
  let chromeOn = $state(true);
  // Whether the last input on the page was keyboard navigation or a pointer: focus rings, and anything a focused control brings out, are for navigation only.
  // Space and Enter act on a control and do not count, so playing with Space never shows a ring.
  // Browsers differ on whether a focus moved by script after a tap shows its ring (Safari does), so the player decides for itself.
  let input = $state<'keys' | 'pointer'>('pointer');
  let reduced = $state(false);
  let flash = $state(0);
  let flashShape = $state<'play' | 'pause'>('play');
  let status = $state('');

  let mutedNow = $state(untrack(() => muted ?? mediaVolume.muted));
  const silent = $derived(mutedNow || mediaVolume.volume === 0);

  // The pointer over the timeline: how far along it is (0 to 1), and where the timeline and player sit on the page.
  let hovering = $state(false);
  let hoverFraction = $state(0);
  let trackLeft = $state(0);
  let trackSpan = $state(0);
  let trackLift = $state(0);
  let playerWidth = $state(0);

  let colors: TimelineColors | null = null;

  // --- The element ---

  function sync(): void {
    const el = media;
    if (!el) return;
    elPaused = el.paused;
    ended = el.ended;
    clock = el.currentTime;
    mediaDuration = el.duration;
    ready = el.readyState;
    pictureWidth = video?.videoWidth ?? 0;
    pictureHeight = video?.videoHeight ?? 0;
  }

  /** Let the viewer's video load in full: it is being played or scrubbed. */
  function warm(): void {
    warmed = src;
  }

  // One seek is in flight at a time, and a newer target replaces one that is still waiting: the latest wins, nothing queues.
  let want: number | null = null;
  let inFlight = false;
  let slowTimer: ReturnType<typeof setTimeout> | undefined;
  let keyHeld = false;

  function gestureActive(): boolean {
    return scrubbing || keyHeld;
  }

  function settleSlow(): void {
    if (!gestureActive() && !inFlight && want === null) slow = false;
  }

  function pump(): void {
    const el = media;
    if (inFlight || want === null || !el) return;
    const next = want;
    want = null;
    inFlight = true;
    clearTimeout(slowTimer);
    slowTimer = setTimeout(() => {
      if (inFlight && gestureActive()) {
        measure();
        slow = true;
      }
    }, SLOW_SEEK_MS);
    el.currentTime = next;
  }

  /** Move the playhead to *time* seconds. The picture follows when the browser has it. */
  function seekTo(time: number): void {
    if (!loaded) return;
    warm();
    const clamped = Math.min(Math.max(time, 0), total);
    want = clamped;
    target = clamped;
    presented = null;
    pump();
  }

  function landed(): void {
    inFlight = false;
    sync();
    if (want !== null) {
      pump();
      return;
    }
    target = null;
    settleSlow();
    paintAmbient();
  }

  function timeForFraction(at: number): number {
    if (facts) return stepTarget(Math.round(at * (facts.frames - 1)), facts);
    return at * total;
  }

  function startOfClip(): number {
    return facts ? stepTarget(0, facts) : 0;
  }

  function endOfClip(): number {
    return facts ? stepTarget(facts.frames - 1, facts) : total;
  }

  /** Play or pause. A clip that has run out, or sits on its last frame, plays again from the start. */
  function togglePlay(): void {
    const el = media;
    if (!el || failed) return;
    wake();
    if (!paused) {
      el.pause();
      return;
    }
    if (atEnd) seekTo(startOfClip());
    void el.play()?.catch(() => undefined);
  }

  /** The row's Play: a tap leaves no focus on a button on an iPad, so the timeline takes it and Space works at once. */
  function onRowPlay(event: MouseEvent): void {
    togglePlay();
    if (event.detail !== 0 && lastPointer !== 'mouse') focusTimeline();
  }

  /** Mute or unmute. */
  function toggleMute(): void {
    wake();
    if (silent && mediaVolume.volume === 0) {
      setVolume(1);
      return;
    }
    mutedNow = !silent;
    mediaVolume.setMuted(mutedNow);
  }

  function setVolume(value: number): void {
    mediaVolume.setVolume(value);
    mutedNow = mediaVolume.muted;
    wake();
  }

  /** Enter or leave fullscreen; only a video outside the file browser's pane has it. */
  function toggleFullscreen(): void {
    if (!isVideo || size === 'pane' || !root) return;
    if (document.fullscreenElement === root) {
      void document.exitFullscreen();
    } else if (typeof root.requestFullscreen === 'function') {
      void root.requestFullscreen().catch(() => undefined);
    } else {
      // iPhone Safari cannot fullscreen an element, only the video itself.
      (video as FullscreenVideo | null)?.webkitEnterFullscreen?.();
    }
  }

  /** Put keyboard focus on the timeline, so the keys work at once. */
  function focusTimeline(): void {
    if (timeline && !failed) timeline.focus({ preventScroll: true });
  }

  /** Play or pause from outside the player, and give the timeline the focus as a click on the picture does. */
  export function engage(): void {
    focusTimeline();
    togglePlay();
  }

  // The element's events arrive after the player is gone (pausing it on the way out queues one), when its state is no longer readable.
  let gone = false;
  const handlers = {
    onplay() {
      const el = media;
      if (el) claimPlayback(el);
      engagedPlayer = me;
      warm();
      sync();
      // The clip was only held still for a scrub: to the user it never stopped, so nothing flashes and the menu button stays away.
      if (resumeAfter) {
        resumeAfter = false;
        wake();
        return;
      }
      if (!scrubbing) {
        flashFor('play');
        status = 'Playing';
      }
      wake();
    },
    onpause() {
      sync();
      if (resumeAfter || ended) return;
      flashFor('pause');
      status = `Paused at ${nowText}`;
      clearWaiting();
    },
    onended() {
      sync();
      wake();
    },
    onseeking() {
      presented = null;
      sync();
    },
    onseeked: landed,
    ontimeupdate: sync,
    onloadedmetadata() {
      sync();
      paintAmbient();
    },
    ondurationchange: sync,
    onloadeddata: sync,
    onresize: sync,
    oncanplay() {
      sync();
      paintAmbient();
    },
    onplaying() {
      sync();
      clearWaiting();
    },
    onwaiting() {
      if (paused) return;
      clearTimeout(waitingTimer);
      waitingTimer = setTimeout(() => { stalled = true; }, SPINNER_MS);
    },
    onemptied() {
      presented = null;
      inFlight = false;
      want = null;
      target = null;
      sync();
    },
    onerror() {
      failed = true;
      inFlight = false;
      want = null;
      target = null;
    },
  };
  const mediaEvents = Object.fromEntries(Object.entries(handlers).map(([name, handler]) => [name, (): void => { if (!gone) handler(); }])) as typeof handlers;

  let waitingTimer: ReturnType<typeof setTimeout> | undefined;
  function clearWaiting(): void {
    clearTimeout(waitingTimer);
    stalled = false;
  }

  function flashFor(shape: 'play' | 'pause'): void {
    if (reduced) return;
    flashShape = shape;
    flash += 1;
  }

  // --- The chrome ---

  let idleTimer: ReturnType<typeof setTimeout> | undefined;

  function restartIdle(): void {
    clearTimeout(idleTimer);
    if (fades && playing && !scrubbing) idleTimer = setTimeout(() => { chromeOn = false; }, IDLE_MS);
  }

  /** Show the chrome and start its countdown again. */
  function wake(): void {
    chromeOn = true;
    restartIdle();
  }

  $effect(() => {
    void playing;
    void fades;
    restartIdle();
    if (!playing) chromeOn = true;
    return () => clearTimeout(idleTimer);
  });

  // --- Pointer on the picture and the player ---

  let lastPointer = 'mouse';
  // When a finger last touched the player. A phone sends mouse events after a tap; those are not a mouse hovering, and must never bring the loupe up.
  let touchedAt = -Infinity;
  const TOUCH_ECHO_MS = 1200;

  const me = Symbol('player');

  function onRootPointerDown(event: PointerEvent): void {
    engagedPlayer = me;
    if (event.pointerType !== 'mouse') touchedAt = performance.now();
    else if (performance.now() - touchedAt < TOUCH_ECHO_MS) return;
    lastPointer = event.pointerType;
  }

  function onRootTouch(): void {
    touchedAt = performance.now();
    if (!scrubbing) stopHover();
  }

  /** Whether a pointer event is a real mouse: not a finger, and not the mouse events a phone sends after a tap. */
  function realMouse(event: PointerEvent): boolean {
    return event.pointerType === 'mouse' && !coarse && performance.now() - touchedAt >= TOUCH_ECHO_MS;
  }

  /** Put the hover away: the loupe and the frames over the line. */
  function stopHover(): void {
    hovering = false;
    if (!scrubbing) bloomTarget = 0;
    requestDraw();
  }

  function onRootPointerMove(event: PointerEvent): void {
    if (!realMouse(event) || !root) return;
    if (fades) wake();
  }

  // Where a press on the picture began: a press that moved is a swipe (the viewer pages on it), not a tap.
  let pressFrom: { x: number; y: number } | null = null;
  const TAP_SLOP = 10;

  function onPicturePointerDown(event: PointerEvent): void {
    pressFrom = { x: event.clientX, y: event.clientY };
  }

  function onPictureClick(event: MouseEvent): void {
    if (failed) return;
    const moved = pressFrom !== null && Math.hypot(event.clientX - pressFrom.x, event.clientY - pressFrom.y) > TAP_SLOP;
    pressFrom = null;
    if (moved) return;
    // A click or a tap on the picture plays or pauses, on every screen; the keys then work at once.
    focusTimeline();
    if (menuOpen) {
      menuOpen = false;
      return;
    }
    togglePlay();
  }

  // --- The timeline ---

  let scrub: Scrub | null = null;

  /** Read where the timeline and the player sit, for the loupe. */
  function measure(): void {
    if (!root || !timeline) return;
    const box = root.getBoundingClientRect();
    const track = timeline.getBoundingClientRect();
    trackLeft = track.left - box.left;
    trackSpan = track.width;
    trackLift = box.bottom - track.top;
    playerWidth = box.width;
  }

  // What is scrubbing: a finger's scrub shows the loupe only while it drags.
  let scrubPointer = $state('mouse');

  function onTrackDown(event: PointerEvent): void {
    if (!timeline || !loaded || failed || (event.pointerType === 'mouse' && event.button !== 0)) return;
    if (event.pointerType === 'mouse' && !realMouse(event)) return;
    scrubPointer = event.pointerType;
    if (event.pointerType !== 'mouse') {
      touchedAt = performance.now();
      hovering = false;
    }
    scrub = startScrub(event.clientX, event.clientY, seekFraction(event.clientX, timeline.getBoundingClientRect()));
    scrubbing = true;
    fine = false;
    if (!elPaused) {
      resumeAfter = true;
      media?.pause();
    }
    timeline.setPointerCapture?.(event.pointerId);
    bloomTarget = 1;
    measure();
    seekTo(timeForFraction(scrub.fraction));
    wake();
  }

  function onTrackMove(event: PointerEvent): void {
    if (!timeline) return;
    const box = timeline.getBoundingClientRect();
    if (scrubbing && scrub) {
      scrub = moveScrub(scrub, event.clientX, event.clientY, { width: box.width, lastFrame });
      fine = scrub.fine;
      seekTo(timeForFraction(scrub.fraction));
      return;
    }
    if (!realMouse(event)) return;
    hovering = true;
    hoverFraction = seekFraction(event.clientX, box);
    bloomTarget = 1;
    measure();
    requestDraw();
  }

  function endScrub(event: PointerEvent): void {
    if (event.pointerType !== 'mouse') {
      touchedAt = performance.now();
      stopHover();
    }
    if (!scrubbing) return;
    scrubbing = false;
    fine = false;
    scrub = null;
    if (timeline?.hasPointerCapture?.(event.pointerId)) timeline.releasePointerCapture(event.pointerId);
    bloomTarget = hovering ? 1 : 0;
    // The clip counts as playing until the element says it is, so the play glyph and the menu button do not flicker in between.
    if (resumeAfter) {
      void media?.play()?.catch(() => {
        resumeAfter = false;
        sync();
      });
    }
    settleSlow();
    restartIdle();
    requestDraw();
  }

  function onTrackLeave(): void {
    stopHover();
  }

  // --- Keys ---

  function keyTarget(element: Element): KeyTarget {
    if (element.closest('[data-volume]')) return 'volume';
    return element.closest('button') ? 'button' : 'timeline';
  }

  /** A step, a seek or a jump from the keyboard moves the playhead like a scrub: the frames rise out of the line around it. */
  function keyScrub(): void {
    keyHeld = true;
    bloomTarget = 1;
    requestDraw();
  }

  function run(command: PlayerCommand): void {
    switch (command.type) {
      case 'toggle':
        togglePlay();
        break;
      case 'volume':
        setVolume(mediaVolume.volume + command.delta);
        showVolumeLevel();
        break;
      case 'step':
        keyScrub();
        if (facts && frameNow !== null) seekTo(stepTarget(Math.min(Math.max(frameNow + command.frames, 0), facts.frames - 1), facts));
        else seekTo(position + command.frames * FALLBACK_STEP_SECONDS);
        break;
      case 'seek':
        keyScrub();
        seekTo(position + command.seconds);
        break;
      case 'start':
        keyScrub();
        seekTo(startOfClip());
        break;
      case 'end':
        keyScrub();
        seekTo(endOfClip());
        break;
      case 'jump':
        keyScrub();
        seekTo(timeForFraction(command.fraction));
        break;
    }
  }

  function onKeydown(event: KeyboardEvent): void {
    if (event.defaultPrevented || !(event.target instanceof Element)) return;
    const command = playerKeyCommand(event, { kind, paused, target: keyTarget(event.target), volume: showVolume });
    if (!command) return;
    // A digit in a field is the field's.
    if (command.type === 'jump' && isTyping(event.target)) return;
    event.preventDefault();
    wake();
    run(command);
  }

  /** Step or seek from the open frame menu: the menu sits in the page's body, so its keys never reach the player's own handler. The menu stays open and follows. */
  function onMenuKeydown(event: KeyboardEvent): void {
    const command = playerKeyCommand(event, { kind, paused, target: 'timeline', volume: showVolume });
    if (command?.type !== 'step' && command?.type !== 'seek') return;
    event.preventDefault();
    // The key's release happens in the menu too, which the player's own `onkeyup` cannot hear.
    window.addEventListener('keyup', onKeyup, { once: true });
    wake();
    run(command);
  }

  function onKeyup(): void {
    keyHeld = false;
    if (!scrubbing) bloomTarget = hovering ? 1 : 0;
    settleSlow();
    requestDraw();
  }

  // --- Drawing ---

  let bloom = 0;
  let bloomTarget = 0;
  let frameId = 0;

  function requestDraw(): void {
    if (frameId || gone || typeof requestAnimationFrame !== 'function') return;
    frameId = requestAnimationFrame(frame);
  }

  function frame(): void {
    frameId = 0;
    if (playing) sync();
    bloom = easeBloom(bloom, bloomTarget, reduced);
    paint();
    if (playing || bloom !== bloomTarget) requestDraw();
  }

  // What the last paint was drawn from, down to the device pixel; a frame that would draw the same picture again is skipped.
  let painted: unknown[] | null = null;

  function paintKey(ratio: number, focus: number): unknown[] {
    const shared = [canvas, colors, ratio, trackWidth, trackHeight, failed];
    if (!isVideo) return [...shared, peaks, wavePlayedBars(trackWidth, fraction)];
    return [...shared, Math.round(fraction * trackWidth * ratio), scrubbing, bloom, bloom > 0 ? Math.round(focus * ratio) : 0, totalSeconds, facts?.frames ?? 0];
  }

  function paint(): void {
    if (!canvas || !colors || trackWidth <= 0 || trackHeight <= 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const focus = (hovering && !scrubbing ? hoverFraction : fraction) * trackWidth;
    const key = paintKey(ratio, focus);
    if (painted?.length === key.length && key.every((value, index) => Object.is(value, painted?.[index]))) return;
    painted = key;
    const width = Math.round(trackWidth * ratio);
    const height = Math.round(trackHeight * ratio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const view = { width: trackWidth, height: trackHeight, fraction, duration: totalSeconds, frames: facts?.frames ?? 0, scrubbing, bloom, focus };
    if (isVideo) drawVideoTimeline(ctx, view, colors, !failed);
    else drawWaveform(ctx, view, peaks, colors);
  }

  /** Return the colour a design token resolves to, which a canvas cannot read from a custom property by itself. */
  function resolveColor(host: HTMLElement, token: string): string {
    const probe = document.createElement('span');
    probe.style.color = `var(${token})`;
    host.appendChild(probe);
    const color = getComputedStyle(probe).color;
    probe.remove();
    return color;
  }

  $effect(() => {
    if (!root) return;
    const host = root;
    colors = Object.fromEntries(Object.entries(TOKEN_NAMES).map(([key, token]) => [key, resolveColor(host, token)])) as unknown as TimelineColors;
    requestDraw();
  });

  // Anything the canvas draws from changes: redraw once, on the next animation frame.
  $effect(() => {
    void [fraction, trackWidth, trackHeight, scrubbing, failed, peaks, totalSeconds, facts, hoverFraction, hovering];
    requestDraw();
  });

  $effect(() => {
    if (playing) requestDraw();
  });

  // --- Ambient light: the picture's colours, a few pixels across, softened behind it. ---

  // The light fills the viewer's pane, padding and all, and is drawn only where the picture leaves a band; with none, it is neither drawn nor shown.
  const lightPane = $derived<Box>(!pane || fullscreen ? { width: stageWidth, height: stageHeight } : pane);
  // A clip shows no larger than its own size, except in fullscreen and in the file browser's pane, which fill what they have.
  // The clip's size: the file's once read, the gallery's until then, so nothing moves when the file arrives.
  const knownWidth = $derived(pictureWidth || naturalSize?.width || 0);
  const knownHeight = $derived(pictureHeight || naturalSize?.height || 0);
  const shown = $derived(shownPicture({ width: stageWidth, height: stageHeight }, { width: knownWidth, height: knownHeight }, fullscreen || size === 'pane'));
  const banded = $derived(viewer && isVideo && leavesBands(lightPane, shown));

  // The light is laid out like the pane with the picture in it, and its bands are the picture's own edges drawn outward.
  const region = $derived(ambientRegion(lightPane, shown, AMBIENT_REACH, AMBIENT_COLUMNS));
  // Where the pane lies, from the player's own corner: the padding is the same on both sides.
  const glowBox = $derived({ left: (stageWidth - lightPane.width) / 2, top: (stageHeight - lightPane.height) / 2, width: lightPane.width, height: lightPane.height });
  // How far the light spreads past the picture, and the soft mask that shapes it.
  const spread = $derived(glowSpread(lightPane, shown, GLOW_REACH));
  const lit = $derived(banded && glowShows(spread));
  const glowMaskImage = $derived(lit ? glowMask(lightPane, shown, spread) : 'none');

  // The light is built on a scratch canvas of the same tiny size, then laid over the last light: at once after a seek or a pause, a share at a time while the video plays.
  // The picture is read once per paint, already shrunk to the light's few pixels (off the main thread where the browser can), and its edges are drawn outward from that copy, never from the full frame.
  let scratch: HTMLCanvasElement | null = null;
  let reading = false;

  function lightFrom(source: CanvasImageSource, share: number): void {
    if (!ambient || !region) return;
    const ctx = ambient.getContext('2d');
    if (!ctx) return;
    const { columns, rows, picture } = region;
    scratch ??= document.createElement('canvas');
    if (scratch.width !== columns || scratch.height !== rows) {
      scratch.width = columns;
      scratch.height = rows;
    }
    const work = scratch.getContext('2d');
    if (!work) return;
    const right = picture.x + picture.width;
    const bottom = picture.y + picture.height;
    work.drawImage(source, picture.x, picture.y, picture.width, picture.height);
    work.drawImage(scratch, picture.x, picture.y, 1, picture.height, 0, picture.y, picture.x, picture.height);
    work.drawImage(scratch, right - 1, picture.y, 1, picture.height, right, picture.y, columns - right, picture.height);
    work.drawImage(scratch, 0, picture.y, columns, 1, 0, 0, columns, picture.y);
    work.drawImage(scratch, 0, bottom - 1, columns, 1, 0, bottom, columns, rows - bottom);
    // The softness is drawn into the light's few pixels and the browser's smooth upscaling does the rest: a CSS blur over a layer this size costs a playing video frames.
    ctx.globalAlpha = share;
    ctx.filter = `blur(${AMBIENT_SOFTEN}px)`;
    ctx.drawImage(scratch, 0, 0);
    ctx.filter = 'none';
    ctx.globalAlpha = 1;
  }

  function paintAmbient(share = 1): void {
    const el = video;
    const at = region;
    if (!lit || !ambient || !el || el.readyState < 2 || !at || reading) return;
    const width = Math.max(1, at.picture.width);
    const height = Math.max(1, at.picture.height);
    if (typeof createImageBitmap !== 'function') {
      try { lightFrom(el, share); } catch { /* A frame that cannot be read leaves the last light. */ }
      return;
    }
    reading = true;
    createImageBitmap(el, { resizeWidth: width, resizeHeight: height, resizeQuality: 'low' })
      .then((bitmap) => {
        if (!gone) lightFrom(bitmap, share);
        bitmap.close();
      })
      .catch(() => { /* A frame that cannot be read leaves the last light. */ })
      .finally(() => { reading = false; });
  }

  // While the video plays the light follows each picture it shows, eased in over a few frames; a browser without the per-picture report follows the screen's frames.
  $effect(() => {
    const el = video;
    if (!lit || reduced || !playing || !el) return;
    let handle = 0;
    let frameHandle = 0;
    const perPicture = typeof el.requestVideoFrameCallback === 'function';
    const step = (): void => {
      if (document.visibilityState === 'visible') paintAmbient(AMBIENT_BLEND);
      schedule();
    };
    const schedule = (): void => {
      if (perPicture) handle = el.requestVideoFrameCallback(step);
      else frameHandle = requestAnimationFrame(step);
    };
    schedule();
    return () => {
      if (perPicture) el.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(frameHandle);
    };
  });

  // The light is out of date when it comes back after a resize or fullscreen: paint it once as it shows.
  $effect(() => {
    void region;
    if (lit) paintAmbient();
  });

  // --- Facts, contact sheet, volume, screen ---

  $effect(() => {
    if (isVideo && filePath) videoFacts.ensure(filePath);
  });

  // The sheet is asked for once per player: one that failed to build is not asked for again on every hover, each ask holding a worker on the
  // server for as long as the build takes. The next time the clip is opened asks anew.
  let spriteAskedFor: string | null = null;
  $effect(() => {
    if (!isVideo || !filePath || size === 'pane' || !(hovering || slow || scrubbing) || spriteAskedFor === filePath) return;
    spriteAskedFor = filePath;
    videoSprite.ensure(filePath);
  });

  $effect(() => {
    const el = media;
    if (!el) return;
    el.volume = mediaVolume.volume;
    el.muted = mutedNow;
  });

  $effect(() => {
    const el = media;
    if (el) sync();
  });

  // Each picture the video shows reports its own time, which is the frame the viewer sees; a seek or a pause leaves `currentTime` ahead of it.
  $effect(() => {
    const el = video;
    if (!el || typeof el.requestVideoFrameCallback !== 'function') return;
    let handle = 0;
    const onFrame = (_now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata): void => {
      handle = el.requestVideoFrameCallback(onFrame);
      // A picture shown on the way to a seek is not where the seek lands.
      if (gone || el.seeking || target !== null) return;
      presented = metadata.mediaTime;
    };
    handle = el.requestVideoFrameCallback(onFrame);
    return () => el.cancelVideoFrameCallback(handle);
  });

  $effect(() => () => {
    gone = true;
    if (frameId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(frameId);
  });

  // A player never keeps playing when it goes: paging, closing and another take all remove it.
  $effect(() => {
    const el = media;
    return () => {
      clearTimeout(slowTimer);
      clearTimeout(waitingTimer);
      if (!el) return;
      el.pause();
      releasePlayback(el);
      // A clip left behind frees its buffers, decoder and connection now rather than when the browser collects the element.
      el.removeAttribute('src');
      el.load();
    };
  });

  // A video holds to its metadata until the page has settled, so one paged past never downloads; playing or scrubbing does not wait.
  // Once loaded it shows its own first frame in place of the poster, which is a still from further in. The file browser's pane stays on the poster.
  $effect(() => {
    if (size === 'pane' || !isVideo) return;
    const current = src;
    const timer = setTimeout(() => { warmed = current; }, WARM_MS);
    return () => clearTimeout(timer);
  });

  // Space plays and pauses the player last used when nothing that takes the keyboard has the focus: a tap leaves the focus on the page's body on an iPad,
  // and a big Play that is pressed goes away as the clip starts. A tab stop (button, link, switch, checkbox, slider, a player's own timeline) keeps its Space,
  // and so does a field being typed in. A modal the player is not in keeps its keys.
  $effect(() => {
    const onSpace = (event: KeyboardEvent): void => {
      if (event.key !== ' ' || event.defaultPrevented || !isPlainKey(event) || engagedPlayer !== me || failed || !root?.isConnected) return;
      if (!isUnfocusedTarget(event.target)) return;
      if (hasOpenModal() && !root.closest('[aria-modal="true"]')) return;
      event.preventDefault();
      wake();
      togglePlay();
    };
    window.addEventListener('keydown', onSpace);
    return () => window.removeEventListener('keydown', onSpace);
  });

  $effect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (navigatesByKey(event)) input = 'keys';
    };
    const onPointer = (): void => { input = 'pointer'; };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('pointerdown', onPointer, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('pointerdown', onPointer, true);
    };
  });

  $effect(() => {
    const motion = typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : null;
    reduced = motion?.matches ?? false;
    const onMotion = (event: MediaQueryListEvent): void => { reduced = event.matches; };
    motion?.addEventListener?.('change', onMotion);
    const onFullscreen = (): void => { fullscreen = document.fullscreenElement !== null && document.fullscreenElement === root; };
    document.addEventListener('fullscreenchange', onFullscreen);
    return () => {
      motion?.removeEventListener?.('change', onMotion);
      document.removeEventListener('fullscreenchange', onFullscreen);
    };
  });

  // A file that is slow to open, or a picture that stalls while it plays, shows a spinner in place of the centre glyph.
  $effect(() => {
    if (loaded || failed) {
      loadingSlow = false;
      return;
    }
    const timer = setTimeout(() => { loadingSlow = true; }, SPINNER_MS);
    return () => clearTimeout(timer);
  });

  $effect(() => {
    onpoint?.(point);
  });
  $effect(() => () => onpoint?.(null));

  // The menu closes with its button: when the video plays, scrubs or loses its selection.
  $effect(() => {
    if (hasMenu || !menuOpen) return;
    menuOpen = false;
    // A key that stepped to the clip's start took the menu's button with it: the keys carry on from the timeline.
    if (document.activeElement?.closest('[role="menu"]')) focusTimeline();
  });

  $effect(() => {
    if (busyNow) status = FRAME_BUSY_LABEL;
  });

  // --- What the template reads ---

  const spinning = $derived(!failed && ((!loaded && loadingSlow) || (stalled && playing)));
  // The controls belong to the picture: their deck is exactly the picture's width and sits on its bottom edge; a narrow picture gets less padding, never a wider deck.
  const pictured = $derived(isVideo && shown.width > 0 && shown.height > 0);
  const compact = $derived(pictured && shown.height < COMPACT_PICTURE);
  const tight = $derived(pictured && shown.height < TIGHT_PICTURE);
  const deckSize = $derived(tight ? 'tight' : compact ? 'compact' : 'normal');
  const metrics = $derived(deckMetrics(coarse, deckSize));
  // The space above the picture in the stage; the deck, the big Play and the picture's place all read it.
  const pictureTop = $derived(isVideo ? (stageHeight - shown.height) / 2 : 0);
  const pictureGap = $derived(pictured ? Math.max(0, pictureTop) : 0);
  // Whether the picture is laid out by the player itself (absolutely, at its own size) rather than filling the stage.
  const placed = $derived(pictured && !fullscreen && size !== 'pane');
  const deck = $derived(deckFor(metrics, pictureGap, deckSize));
  const deckWidth = $derived(pictured ? Math.min(stageWidth, shown.width) : stageWidth);
  const deckLeft = $derived(isVideo ? Math.max(0, (stageWidth - deckWidth) / 2) : 0);
  const deckPad = $derived(Math.round(Math.min(32, Math.max(6, deckWidth * 0.03))));
  // A deck too narrow for the time and two buttons side by side drops the clip's length from the readout.
  const narrowDeck = $derived(isVideo && deckWidth > 0 && deckWidth < 300);
  // The big Play is in the very middle of the picture: the largest size that stays clear of the controls, else the smallest (the row's Play is always there).
  const pictureMiddle = $derived(stageHeight / 2);
  const centreSize = $derived<number>(CENTRE_SIZES.find((size) => pictureMiddle + size / 2 + 4 <= stageHeight - deck.bottom - deck.reach && pictureMiddle - size / 2 >= pictureTop + 4) ?? CENTRE_LEAST);
  // The open menu is what the user is working with: the big Play steps aside until it closes.
  const centreShown = $derived(!failed && !menuOpen && (paused || spinning) && (chromeOn || !fades));
  // Waiting with the chrome hidden: a spinner alone, not a button.
  const spinnerShown = $derived(spinning && !centreShown);
  const canFullscreen = $derived(isVideo && size !== 'pane');
  // A volume slider only where the browser lets a page set the volume: iPhone and iPad keep Mute alone.
  const showVolume = $derived(size !== 'pane' && canSetVolume());
  // The volume keys bring the slider up for a moment, so its level shows while they change it.
  let volumeShown = $state(false);
  let volumeTimer: ReturnType<typeof setTimeout> | undefined;
  function showVolumeLevel(): void {
    if (!showVolume) return;
    volumeShown = true;
    clearTimeout(volumeTimer);
    volumeTimer = setTimeout(() => { volumeShown = false; }, 1400);
  }
  // A mouse sees the loupe while it hovers the line, or while a seek is slow; a finger sees it only while it drags.
  const loupeOn = $derived(isVideo && size !== 'pane' && loaded && !failed && (scrubbing ? scrubPointer !== 'mouse' || slow : hovering || (slow && scrubPointer === 'mouse')));
  const loupeAt = $derived(slow || scrubbing ? fraction : hoverFraction);
  const loupeFrame = $derived(facts && lastFrame !== null ? Math.round(loupeAt * lastFrame) : null);
  const loupeSeconds = $derived(facts && loupeFrame !== null ? loupeFrame / facts.fps : loupeAt * total);
  const sprite = $derived(filePath ? videoSprite.of(filePath) : null);
  const loupePicture = $derived(sprite && facts && loupeFrame !== null && lastFrame !== null ? loupeImage(sprite, spriteTile(loupeFrame, facts.frames)) : null);
  const loupeX = $derived(loupeCentre(trackLeft + loupeAt * trackSpan, playerWidth));
  const loupeLift = $derived(`${trackLift + 18}px`);
  const valueNow = $derived(Math.round(shownSeconds * 10) / 10);
</script>

{#snippet track()}
  <!-- The timeline is the one slider of the player: its keys, its name and its value are the screen reader's way in. -->
  <div
    bind:this={timeline}
    bind:clientWidth={trackWidth}
    bind:clientHeight={trackHeight}
    class="timeline"
    role="slider"
    tabindex={failed ? -1 : 0}
    aria-label="Playback position"
    aria-orientation="horizontal"
    aria-valuemin={0}
    aria-valuemax={Math.round(totalSeconds * 10) / 10}
    aria-valuenow={valueNow}
    aria-valuetext={nowText}
    aria-disabled={!loaded || failed}
    data-own-drag={viewer ? '' : undefined}
    data-testid="media-timeline"
    onpointerdown={onTrackDown}
    onpointermove={onTrackMove}
    onpointerup={endScrub}
    onpointercancel={endScrub}
    onlostpointercapture={endScrub}
    onpointerleave={onTrackLeave}
  >
    <canvas bind:this={canvas} aria-hidden="true"></canvas>
    <span class="knob-ring" style:left="{fraction * 100}%" aria-hidden="true"></span>
  </div>
{/snippet}

{#snippet timeText()}
  <span class="now">{nowText}</span>{#if totalText && !narrowDeck}<span class="total"> / {totalText}</span>{/if}
{/snippet}

{#snippet centreGlyph()}
  {#if spinning}
    <Spinner size="lg" />
  {:else}
    <Icon name={paused ? 'play' : 'pause'} size={size === 'pane' || centreSize <= CENTRE_LEAST ? 18 : centreSize < CENTRE_SIZES[1] ? 22 : 30} solid={paused} />
  {/if}
{/snippet}

<!-- svelte-ignore a11y_no_noninteractive_element_interactions because the group only collects the keys and the pointer of the controls inside it -->
<div
  bind:this={root}
  bind:clientWidth={stageWidth}
  bind:clientHeight={stageHeight}
  class="player"
  role="group"
  aria-label={label}
  data-media-player
  data-kind={kind}
  data-size={size}
  data-fades={fades ? '' : undefined}
  data-chrome={chromeOn || !fades ? 'shown' : 'hidden'}
  data-state={failed ? 'error' : playing ? 'playing' : 'paused'}
  data-fullscreen={fullscreen}
  data-input={input}
  data-volume-shown={volumeShown ? '' : undefined}
  data-tiny={tiny ? '' : undefined}
  data-reduced={reduced}
  data-own-drag={size === 'viewer' ? undefined : ''}
  style:--centre-size="{centreSize}px"
  style:--centre-top={isVideo ? `${pictureMiddle}px` : undefined}
  style:--pic-left={placed ? `${(stageWidth - shown.width) / 2}px` : undefined}
  style:--pic-top={placed ? `${pictureTop}px` : undefined}
  style:--pic-w={placed ? `${shown.width}px` : undefined}
  style:--pic-h={placed ? `${shown.height}px` : undefined}
  data-sized={pictured ? '' : undefined}
  data-placed={placed ? '' : undefined}
  style:--track-h={isVideo ? `${metrics.track}px` : undefined}
  style:--ctl-h={isVideo ? `${metrics.control}px` : undefined}
  onpointerdown={onRootPointerDown}
  ontouchstart={onRootTouch}
  ontouchend={onRootTouch}
  ontouchcancel={onRootTouch}
  onpointermove={onRootPointerMove}
  onkeydown={onKeydown}
  onkeyup={onKeyup}
>
  {#if isVideo}
    {#if viewer}
      <div class="glow" hidden={!lit} style:left="{glowBox.left}px" style:top="{glowBox.top}px" style:width="{glowBox.width}px" style:height="{glowBox.height}px" style:mask-image={glowMaskImage} style:-webkit-mask-image={glowMaskImage} aria-hidden="true">
        <canvas bind:this={ambient} class="ambient" width={region?.columns ?? AMBIENT_COLUMNS} height={region?.rows ?? 18} style:--ambient-reach="{AMBIENT_REACH}px"></canvas>
      </div>
    {/if}
    <!-- svelte-ignore a11y_media_has_caption because generated videos have no caption tracks -->
    <!-- svelte-ignore a11y_click_events_have_key_events because the keys belong to the timeline, which has the focus -->
    <video
      bind:this={video}
      class="picture"
      {src}
      poster={ready >= 2 ? undefined : poster}
      playsinline
      disablePictureInPicture
      preload={size !== 'pane' && warmed === src ? 'auto' : 'metadata'}
      aria-hidden="true"
      onclick={onPictureClick}
      onpointerdown={onPicturePointerDown}
      {...mediaEvents}
    ></video>
  {:else}
    <audio bind:this={audio} {src} preload="metadata" {...mediaEvents}></audio>
  {/if}

  {#if flash > 0 && !reduced}
    {#key flash}
      <span class="flash" aria-hidden="true"><Icon name={flashShape} size={44} solid={flashShape === 'play'} /></span>
    {/key}
  {/if}

  {#if failed}
    <div class="error" role="alert"><Icon name="alert" size={20} />This file can’t be played.</div>
  {/if}

  <!-- The tab order follows the document: the big Play, the timeline, then the row. -->
  {#if centreShown}
    <button type="button" class="centre ui-media-action" aria-label={paused ? 'Play' : 'Pause'} data-testid="media-big-play" data-own-drag={viewer ? '' : undefined} onclick={engage}>
      {@render centreGlyph()}
    </button>
  {:else if spinnerShown}
    <span class="centre waiting" aria-hidden="true"><Spinner size="lg" /></span>
  {/if}

  <div
    class="bottom"
    style:--deck-left="{deckLeft}px"
    style:--deck-w="{deckWidth}px"
    style:--deck-bottom="{deck.bottom}px"
    style:--deck-h="{deck.height}px"
    style:--deck-pad="{deckPad}px"
    style:--lane-b="{deck.laneBottom}px"
    style:--row-b="{deck.rowBottom}px"
  >
    <!-- The timeline comes first in the document, so Tab reaches it, then the row. A broken file has none. -->
    {#if !failed}
      {@render track()}
      <div class="row">
        {#if hasMenu && point}
          <!-- The time readout is the frame's menu button while paused: the chip keeps its left edge and its time, and grows by the chevron on the right. -->
          <button
            bind:this={timeEl}
            type="button"
            class="time-chip ui-media-action"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-busy={busyNow}
            aria-label="Use this frame, {pointLabel(point)}"
            data-testid="frame-menu"
            data-own-drag
            onclick={() => { menuOpen = !menuOpen; }}
          >
            {@render timeText()}<span class="chev" aria-hidden="true">{#if busyNow}<Spinner size="sm" />{:else}<Icon name="chevup" size={12} />{/if}</span>
          </button>
          <ActionMenu open={menuOpen} anchor={timeEl} items={frameMenu} label={frameTitle} align="start" side="above" returnFocus={timeline} onkeydown={onMenuKeydown} onclose={() => { menuOpen = false; }} />
        {:else}
          <div class="time-chip">{@render timeText()}</div>
        {/if}
        <div class="cluster" role="group" aria-label="Playback controls" data-own-drag={viewer ? '' : undefined}>
          <button type="button" class="tool ui-media-action" aria-label={paused ? 'Play' : 'Pause'} onclick={onRowPlay}>
            <Icon name={paused ? 'play' : 'pause'} size={18} solid={paused} />
          </button>
          <div class="mute-group">
            <button type="button" class="tool ui-media-action" aria-label={silent ? 'Unmute' : 'Mute'} onclick={toggleMute}>
              <Icon name={silent ? 'volume-off' : 'volume'} size={18} />
            </button>
            {#if showVolume}
              <!-- The volume opens as a layer beside the mute button, so nothing in the row moves. -->
              <div class="volume-pop ui-overlay">
                <input
                  class="volume-range ui-range"
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  aria-label="Volume"
                  aria-valuetext="{Math.round((mutedNow ? 0 : mediaVolume.volume) * 100)}%"
                  value={Math.round((mutedNow ? 0 : mediaVolume.volume) * 100)}
                  style:--fill="{(mutedNow ? 0 : mediaVolume.volume) * 100}%"
                  data-volume
                  oninput={(event) => setVolume(Number(event.currentTarget.value) / 100)}
                >
              </div>
            {/if}
          </div>
          {#if canFullscreen}
            <button type="button" class="tool ui-media-action" aria-label={fullscreen ? 'Exit fullscreen' : 'Fullscreen'} onclick={toggleFullscreen}>
              <Icon name={fullscreen ? 'fullscreen-exit' : 'fullscreen'} size={18} />
            </button>
          {/if}
        </div>
      </div>
    {/if}
    {#if fine && scrubbing}
      <span class="fine media-tag">{isVideo ? 'Fine scrub: one frame at a time' : 'Fine scrub'}</span>
    {/if}
  </div>

  {#if loupeOn}
    <div class="loupe" style:left="{loupeX}px" style:bottom={loupeLift} data-testid="media-loupe" aria-hidden="true">
      {#if loupePicture}
        <span
          class="loupe-picture"
          style:width="{loupePicture.width}px"
          style:height="{loupePicture.height}px"
          style:background-image="url('{sprite?.url}')"
          style:background-size="{loupePicture.sheetWidth}px {loupePicture.sheetHeight}px"
          style:background-position="{loupePicture.offsetX}px {loupePicture.offsetY}px"
        ></span>
      {/if}
      <span class="loupe-time media-tag">{secondsLabel(loupeSeconds)}</span>
    </div>
  {/if}

  <span class="sr-only" role="status">{status}</span>
</div>

<style>
  .player { --safe-bottom: 0px; --row-gap: 8px; --ctl-h: 36px; position: relative; width: 100%; height: 100%; min-height: 0; overflow: hidden; color: var(--color-text-primary); -webkit-user-select: none; user-select: none; -webkit-touch-callout: none; }
  @media (pointer: coarse) { .player { --ctl-h: 48px; } }
  .player[data-kind='video'][data-size='viewer'] { position: absolute; inset: 0; }
  /* The viewer's stage clips the light, so the player lets it out. */
  .player[data-kind='video'][data-size='viewer']:not(:fullscreen) { overflow: visible; }
  .player:fullscreen { --safe-bottom: env(safe-area-inset-bottom, 0px); background: var(--color-viewer-void); }

  /* The picture. */
  /* The light is the picture's edges drawn outward behind the picture and the controls. Its mask is a softly blurred rounded box round the picture, so it is strongest at the picture's edge, fades out with no line where it ends, and rounds the corners. */
  .glow { position: absolute; z-index: 0; overflow: hidden; pointer-events: none; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; -webkit-mask-size: 100% 100%; mask-size: 100% 100%; }
  /* The canvas reaches past the pane by twice the blur, so the blur's fading edge is outside it. */
  .ambient { position: absolute; z-index: 0; top: calc(var(--ambient-reach) * -1); left: calc(var(--ambient-reach) * -1); width: calc(100% + var(--ambient-reach) * 2); height: calc(100% + var(--ambient-reach) * 2); opacity: 0.36; pointer-events: none; }
  /* A clip smaller than its player stays at its own size; the file browser's pane and fullscreen fill what they have. */
  .picture { position: relative; z-index: 1; display: block; width: 100%; height: 100%; object-fit: scale-down; cursor: pointer; }
  /* With the clip's size known (from the gallery before the file is read) the picture is laid out where it will be from the first paint; without it, it fades in once read. */
  .player[data-kind='video']:not([data-sized]) .picture { opacity: 0; }
  .player[data-kind='video'] .picture { transition: opacity 160ms var(--ease-standard); }
  .player[data-placed] .picture { position: absolute; left: var(--pic-left); top: var(--pic-top); width: var(--pic-w); height: var(--pic-h); object-fit: contain; }
  .player[data-size='pane'] .picture, .player:fullscreen .picture { object-fit: contain; }
  .player[data-size='pane'] .picture { max-width: 100%; max-height: 100%; }

  /* A take is a panel, the same in the viewer and on the stage: its waveform is its timeline, with the time and the controls under it. */
  .player[data-kind='audio'] { --wave-h: clamp(96px, 24vh, 160px); --pad-t: 12px; --pad-b: 12px; --pad-x: 16px; --centre-top: calc(var(--pad-t) + var(--wave-h) / 2); display: flex; flex: none; height: auto; max-width: 960px; flex-direction: column; margin: 0 auto; padding: var(--pad-t) var(--pad-x) var(--pad-b); border-radius: var(--radius-md); background: var(--color-bg-surface); overflow: visible; }
  /* The stage tile's corner actions lie along the top of its panel. */
  .player[data-kind='audio'][data-size='stage'] { --pad-t: 44px; }
  .player[data-kind='audio'][data-size='pane'] { --wave-h: 88px; --pad-t: 10px; --pad-b: 8px; --pad-x: 10px; }
  .player[data-kind='audio'][data-state='error'] { min-height: 96px; }
  @media (max-width: 639px), (max-height: 499px) {
    .player[data-kind='audio'][data-size='stage'] { --pad-t: 52px; --pad-x: 12px; --pad-b: 10px; }
  }

  /* The timeline: a hairline for a clip, the waveform for a take. */
  .timeline { position: relative; cursor: pointer; touch-action: none; outline: none; }
  .player[data-kind='audio'] .timeline { height: var(--wave-h); touch-action: pan-y; }
  .timeline canvas { display: block; width: 100%; height: 100%; }
  .player[data-kind='video'] .timeline { height: calc(var(--track-h) + 2px); }
  /* The timeline's keyboard focus is a ring round the playhead's knob, not a box round the whole lane. */
  .timeline:focus-visible { outline: none; }
  .knob-ring { position: absolute; bottom: 2px; display: none; width: 14px; height: 24px; margin-left: -7px; border-radius: 7px; box-shadow: 0 0 0 2px var(--color-primary-main); pointer-events: none; }
  .player[data-kind='audio'] .knob-ring { top: 0; bottom: 0; height: auto; width: 6px; margin-left: -3px; border-radius: 3px; }
  .player[data-input='keys'] .timeline:focus .knob-ring { display: block; }
  .timeline[aria-disabled='true'] { cursor: default; }

  /* The deck: the timeline and the row of the time and the controls, bound to the picture. Over a picture it is a box as wide as the picture on its bottom edge; in a take's panel it is the panel. */
  .bottom { position: absolute; z-index: 4; left: var(--deck-left); bottom: calc(var(--deck-bottom) + var(--safe-bottom)); width: var(--deck-w); height: var(--deck-h); pointer-events: none; }
  .bottom::before { content: ''; position: absolute; z-index: -1; inset: -40px 0 0; background: linear-gradient(to top, var(--color-viewer-fade-low) 0%, var(--color-viewer-fade-mid) 38%, var(--color-viewer-fade-soft) 68%, var(--color-viewer-fade-faint) 88%, transparent 100%); opacity: 1; transition: opacity 320ms var(--ease-standard); pointer-events: none; }
  .bottom > * { pointer-events: auto; }
  .player[data-kind='video'] .timeline { position: absolute; right: var(--deck-pad); bottom: var(--lane-b); left: var(--deck-pad); }
  .player[data-kind='video'] .row { position: absolute; right: var(--deck-pad); bottom: var(--row-b); left: var(--deck-pad); height: var(--ctl-h); }
  /* A take: the controls under its waveform, as under a picture. */
  .player[data-kind='audio'] .bottom { position: static; display: flex; width: auto; height: auto; flex-direction: column; padding: 0; }
  .player[data-kind='audio'] .row { padding: var(--row-gap) 2px 0; }
  /* The fade lies over the picture only: not before the picture's size is known, not on a take, not on a broken file. */
  .player[data-kind='audio'] .bottom::before, .player[data-kind='video']:not([data-sized]) .bottom::before, .player[data-state='error'] .bottom::before { display: none; }
  .row { display: flex; min-height: var(--ctl-h); align-items: center; justify-content: space-between; gap: 12px; pointer-events: none; }
  .row > * { pointer-events: auto; }
  /* A tiny stage keeps the timeline and the big Play, and leaves the time and the controls out. */
  .player[data-tiny] .row { display: none; }

  /* Hidden chrome is all of it at once: the time, the buttons and the line go together, and come back together. A control with the keyboard focus keeps it all in view. */
  .bottom { transition: opacity 320ms var(--ease-standard); }
  .player[data-kind='video'][data-chrome='hidden'] .bottom, .player[data-chrome='hidden'] .centre { opacity: 0; }
  .player[data-kind='video'][data-chrome='hidden'] .bottom, .player[data-kind='video'][data-chrome='hidden'] .bottom *, .player[data-chrome='hidden'] .centre { pointer-events: none; }
  .player[data-input='keys'][data-kind='video'][data-chrome='hidden'] .bottom:has(:focus-visible) { opacity: 1; }
  .player[data-input='keys'][data-kind='video'][data-chrome='hidden'] .bottom:has(:focus-visible) > * { pointer-events: auto; }
  /* After a tap or a click no ring shows, whatever the browser thinks of a focus the player moved; the next key brings them back. Rings are outlines, so they never move anything. */
  .player[data-input='pointer'] :global(:focus-visible) { outline: none; }

  /* The time is a media label in the app's media-action fill, the same for a clip and a take. For a clip it is the frame menu's button while paused: it keeps its left edge and its time, and grows by the chevron on the right. */
  .time-chip { display: inline-flex; height: var(--spacing-control); box-sizing: border-box; align-items: center; gap: 6px; padding: 0 10px; border: 0; border-radius: var(--radius-sm); font-family: var(--font-mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
  div.time-chip { background-color: color-mix(in srgb, var(--color-bg-base) 90%, transparent); color: var(--color-text-primary); }
  .time-chip .now { font-family: var(--font-mono); font-size: var(--text-content); font-weight: 600; line-height: 1; }
  .time-chip .total { font-size: var(--text-ui); color: var(--color-text-muted); }
  .time-chip .chev { display: inline-grid; width: 12px; place-items: center; color: var(--color-text-muted); }
  button.time-chip { padding-right: 8px; cursor: pointer; }
  /* Open, it is a menu button that is open: the app's teal wash. */
  button.time-chip[aria-expanded='true'], button.time-chip[aria-expanded='true'] .total, button.time-chip[aria-expanded='true'] .chev { background-color: color-mix(in srgb, var(--color-primary-main) 16%, var(--color-bg-base)); color: var(--color-primary-main); }
  @media (hover: hover) { button.time-chip:hover .chev { color: var(--color-text-primary); } }

  .cluster { display: flex; flex: none; align-items: center; gap: 6px; }
  .tool { display: grid; width: var(--spacing-control); height: var(--spacing-control); place-items: center; border-radius: var(--radius-sm); }

  /* The volume opens beside the mute button (above it in a take's panel) as a layer, so nothing in the row moves, and is left out on a screen without hover. */
  .mute-group { position: relative; display: flex; align-items: center; }
  .volume-pop { position: absolute; z-index: 2; bottom: 100%; left: 50%; padding: 8px 10px; translate: -50% 0; visibility: hidden; opacity: 0; transition: opacity 160ms var(--ease-standard), visibility 0s linear 160ms; }
  .player[data-kind='video'] .volume-pop { bottom: 50%; left: auto; right: calc(100% + 6px); translate: 0 50%; }
  .volume-range { display: block; width: 96px; }
  @media (hover: hover) {
    .mute-group:hover .volume-pop { visibility: visible; opacity: 1; transition-delay: 0s; }
  }
  /* Without a hover the slider is reached from the keys: it opens while the mute button or the slider has the keyboard focus, and while ↑ or ↓ change the volume. */
  .player[data-input='keys'] .mute-group:has(:focus-visible) .volume-pop, .player[data-volume-shown] .volume-pop { visibility: visible; opacity: 1; transition-delay: 0s; }

  /* The big Play and the flash of the last press: in the middle of a picture, and over the middle of a take's waveform. */
  .centre { position: absolute; z-index: 5; top: var(--centre-top, 50%); left: 50%; display: grid; width: var(--centre-size, 72px); height: var(--centre-size, 72px); translate: -50% -50%; place-items: center; border-radius: var(--radius-md); color: var(--color-text-primary); transition: opacity 320ms var(--ease-standard), background-color 200ms; }
  .player[data-size='pane'] .centre { width: 44px; height: 44px; }
  .centre.waiting { pointer-events: none; }
  .flash { position: absolute; z-index: 2; top: var(--centre-top, 50%); left: 50%; translate: -50% -50%; color: var(--color-text-primary); pointer-events: none; animation: player-flash 480ms var(--ease-soft-out) both; }
  @keyframes player-flash { from { opacity: 0.9; scale: 0.7; } to { opacity: 0; scale: 1.6; } }

  .error { position: absolute; z-index: 3; top: 50%; left: 50%; display: flex; max-width: calc(100% - 28px); align-items: center; gap: 10px; padding: 10px 14px; translate: -50% -50%; border-radius: var(--radius-md); background: var(--color-glass-strong); -webkit-backdrop-filter: blur(var(--glass-blur)); backdrop-filter: blur(var(--glass-blur)); font-size: var(--text-content); color: var(--color-text-primary); }
  .error :global(svg) { color: var(--color-warning); }

  /* The slowed scrub's label and the loupe. */
  .player[data-kind='audio'] .fine { bottom: calc(var(--pad-b) + var(--wave-h) + 8px); }
  .fine { position: absolute; right: 0; bottom: calc(var(--deck-h) + 8px); left: 0; width: fit-content; margin: 0 auto; height: auto; padding: 3px 8px; font-family: var(--font-heading); font-size: var(--text-ui); font-weight: 800; line-height: 1.4; color: var(--color-primary-main); pointer-events: none; }
  .loupe { position: absolute; z-index: 7; display: flex; flex-direction: column; align-items: center; gap: 4px; translate: -50% 0; pointer-events: none; }
  .loupe-picture { display: block; background-repeat: no-repeat; background-color: var(--color-bg-surface-hover); border-radius: var(--radius-md); box-shadow: 0 0 0 1px color-mix(in srgb, var(--color-text-primary) 6%, transparent); }
  .loupe-time { height: auto; padding: 2px 8px; font-family: var(--font-heading); font-size: var(--text-content); font-weight: 800; line-height: 1.4; }

  @media (prefers-reduced-motion: reduce) { .flash { display: none; } }
</style>
