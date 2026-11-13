// Pure view model of the launcher window: one snapshot in, everything the window shows out. The words live here (Rust
// sends codes and raw values); `LauncherApp` only lays the result out. `launcherControls` mirrors
// `snapshot::controls()` in Rust, and both are tested against the same fixture.
import type { IconName } from '$lib/components/atoms/Icon.svelte';
import type { MascotMood } from '$lib/components/atoms/Mascot.svelte';
import type {
  Controls,
  ErrorCode,
  FailedAction,
  LinkId,
  Packaging,
  Phase,
  Primary,
  Progress,
  Snapshot,
  StepId,
  Unsupported
} from './types';

export type HaloKind = 'rest' | 'busy' | 'progress' | 'live' | 'error';

export interface HaloView {
  kind: HaloKind;
  /** 0 to 1; set for `progress` and for a frozen `error`. */
  fraction?: number;
  label: string;
  valuetext?: string;
}

export type PoolStrength = 'live' | 'work' | 'rest' | 'dim';

export interface HeroView {
  mood: MascotMood;
  /** Share of the install the Mascot's canvas follows while it creates. */
  mascotProgress?: number;
  halo: HaloView;
  /** Shown as the stamp on the ring. */
  percent?: number;
  /** The brand light dims: something is wrong or off. */
  fault: boolean;
  /** The amber dot: an update is offered. */
  update: boolean;
  /** Strength of the brand light behind the ring. */
  pool: PoolStrength;
}

export interface SubView {
  /** A bold first sentence: a failure's cause or an uninstall instruction. */
  lead: string | null;
  /** The lead is a failure's cause, announced at once. */
  leadAlert: boolean;
  text: string;
}

export interface PrimaryView {
  id: Primary;
  label: string;
  icon: IconName | null;
  variant: 'primary' | 'secondary';
  /** The inert "Starting…" button. */
  loading: boolean;
}

export interface LinkView {
  id: LinkId;
  label: string;
}

export type NoticeView =
  | { kind: 'reason'; name: string; found: string | null; needed: string; link: LinkView | null }
  | { kind: 'note'; icon: IconName; lead: string | null; text: string }
  | { kind: 'update'; version: string; lead: string; text: string; link: LinkView; button: string }
  | { kind: 'newer'; lead: string; text: string; link: LinkView }
  | { kind: 'log'; label: string; sheetTitle: string }
  | { kind: 'kept'; entries: { path: string; what: string }[]; help: string; link: LinkView };

export interface CaptionView {
  label: string;
  line: string | null;
}

export interface MenuItemView {
  id: Controls['menu'][number];
  /** Null for a checkbox item, which shows its check instead. */
  icon: IconName | null;
  label: string;
  description: string;
  danger: boolean;
  /** Set for a checkbox item: whether it is on. */
  checked?: boolean;
}

export interface LauncherView {
  controls: Controls;
  hero: HeroView;
  kicker: string;
  /** The kicker is teal: the step is running. */
  kickerAccent: boolean;
  headline: string;
  sub: SubView | null;
  notice: NoticeView | null;
  caption: CaptionView | null;
  primary: PrimaryView | null;
  /** Entries of the More menu, in order; empty when More is hidden. */
  menu: MenuItemView[];
}

export interface ViewOptions {
  /** Formats `stoppedAt` (Unix ms) as a clock time; the browser's locale by default. */
  formatTime?: (ms: number) => string;
}

const MENU_ALL: Controls['menu'] = ['repair', 'uninstall', 'beta'];
const MENU_WITHOUT_REPAIR: Controls['menu'] = ['uninstall', 'beta'];

/** Derive which controls the snapshot's state offers (the Phase x control table of the design). */
export function launcherControls(s: Snapshot): Controls {
  const c: Controls = { primary: null, update: false, log: false, bar: true, autostart: false, menu: [], locked: false };
  const installed = (menu: Controls['menu']): void => {
    c.autostart = true;
    c.menu = [...menu];
  };
  switch (s.phase) {
    case 'unsupported':
      if (s.unsupported) c.primary = isFixable(s.unsupported) ? 'check_again' : 'see_requirements';
      break;
    case 'needs_install':
      c.primary = 'install';
      c.menu = ['beta'];
      break;
    case 'starting':
      c.primary = 'starting';
      c.locked = true;
      installed(MENU_ALL);
      break;
    case 'running':
      c.primary = 'open';
      c.update = s.updateVersion !== null;
      installed(MENU_ALL);
      break;
    case 'updating':
    case 'repairing':
      c.locked = true;
      installed(MENU_ALL);
      break;
    case 'stopped':
      c.primary = 'open';
      c.log = true;
      installed(MENU_ALL);
      break;
    case 'failed':
      c.log = true;
      if (s.failedAction === 'install') {
        c.primary = 'retry';
        // A failed install leaves a partial install behind; Windows' own uninstaller cleans it.
        c.menu = s.packaging === 'windows_nsis' ? ['beta'] : [...MENU_WITHOUT_REPAIR];
      } else if (s.failedAction === 'repair') {
        c.primary = 'retry';
        installed(MENU_WITHOUT_REPAIR);
      } else if (s.failedAction === 'update') {
        c.primary = 'repair';
        installed(MENU_WITHOUT_REPAIR);
      } else {
        c.primary = 'open';
        installed(MENU_ALL);
      }
      break;
    case 'uninstalled':
      c.primary = 'quit';
      c.bar = false;
      break;
    case 'checking':
    case 'move_to_applications':
    case 'installing':
    case 'uninstalling':
      break;
  }
  return c;
}

function isFixable(u: Unsupported): boolean {
  return u.code === 'driver_too_old' || u.code === 'low_disk';
}

const NAME = 'Z-Vision Generator';

/** "4.2 GB", "850 MB": decimal units, as disks are sold. */
export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${trimZero((bytes / 1e9).toFixed(1))} GB`;
  return `${Math.max(0, Math.round(bytes / 1e6))} MB`;
}

function trimZero(text: string): string {
  return text.endsWith('.0') ? text.slice(0, -2) : text;
}

function bytesOf(value: string | null): number | null {
  if (value === null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function stepName(step: StepId, s: Snapshot): string {
  switch (step) {
    case 'download_tools':
      return 'Downloading uv and ffmpeg';
    case 'clean_cache':
      return 'Clearing the download cache';
    case 'install_app':
      return s.phase === 'repairing' || s.failedAction === 'repair' ? `Reinstalling ${NAME}` : `Installing ${NAME}`;
    case 'update_app':
      return s.updateVersion ? `Installing ${s.updateVersion}` : 'Installing the update';
    case 'finish':
      return 'Finishing';
  }
}

function currentStep(s: Snapshot): string | null {
  const p = s.progress;
  const step = p?.steps[p.index];
  return step ? stepName(step, s) : null;
}

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1);
}

function leftoverText(s: Snapshot): string {
  return s.error?.code === 'uninstall_incomplete' ? 'Some files could not be removed.' : '';
}

/** How to bring the window back after closing it: Linux desktops may show no tray icon. */
const findAgain: Record<Packaging, string> = {
  macos_app: 'Find it again in the menu bar.',
  windows_nsis: 'Find it again in the notification area.',
  appimage: 'Open the launcher again to bring it back.',
  linux_package: 'Open the launcher again to bring it back.'
};

const finishLead: Record<Packaging, string> = {
  macos_app: `Move ${NAME} to the Trash to finish.`,
  windows_nsis: `${NAME} is removed.`,
  appimage: 'Delete the AppImage to finish.',
  linux_package: `Remove the ${NAME} package to finish.`
};

/** The cause of a failed install or repair, and what to do next. */
const causes: Record<ErrorCode, { lead: string; text: string }> = {
  download_failed: {
    lead: 'The download didn’t finish.',
    text: 'Check the internet connection, then retry. What’s already downloaded is kept.'
  },
  download_unavailable: {
    lead: 'A file is no longer available to download.',
    text: 'A newer launcher fixes this.'
  },
  checksum_mismatch: {
    lead: 'A downloaded file was not the one expected.',
    text: 'It was deleted. Retry downloads it again.'
  },
  tool_broken: {
    lead: 'A downloaded tool does not run on this computer.',
    text: 'Retry downloads it again.'
  },
  low_disk: {
    lead: 'There isn’t enough free disk space.',
    text: 'Free up some space, then retry. What’s already downloaded is kept.'
  },
  uv_failed: {
    lead: 'The installer stopped with an error.',
    text: 'Show the log for details, then retry. What’s already downloaded is kept.'
  },
  start_exited: { lead: 'It stopped while starting.', text: '' },
  start_timeout: { lead: 'It didn’t answer in time.', text: '' },
  uninstall_incomplete: { lead: '', text: '' }
};

function causeOf(s: Snapshot): { lead: string; text: string } {
  return causes[s.error?.code ?? 'uv_failed'];
}

/** Whether a version string is a development, alpha, beta or release-candidate version (`0.15.0b2`, `1.0rc1`). */
export function isPrerelease(version: string): boolean {
  return /^v?\d+(\.\d+)*[._-]?(dev|a|alpha|b|beta|rc|c|pre|preview)[._-]?\d*(\+.*)?$/i.test(version.trim());
}

function stepKicker(p: Progress): string {
  return `Step ${p.index + 1} of ${p.steps.length}`;
}

function kickerOf(s: Snapshot): string {
  if (s.progress) return stepKicker(s.progress);
  if (s.phase === 'uninstalled') return '';
  if (!s.installedVersion) return 'Setup';
  return isPrerelease(s.installedVersion) ? `Version ${s.installedVersion} · beta` : `Version ${s.installedVersion}`;
}

const MOODS: Record<Phase, MascotMood> = {
  checking: 'thinking',
  move_to_applications: 'curious',
  unsupported: 'sad',
  needs_install: 'nodding',
  installing: 'creating',
  starting: 'thinking',
  running: 'idle',
  updating: 'creating',
  repairing: 'creating',
  stopped: 'sad',
  failed: 'sad',
  uninstalling: 'thinking',
  uninstalled: 'waving'
};

const BUSY_LABELS: Partial<Record<Phase, string>> = {
  checking: 'Checking this computer',
  starting: 'Starting',
  uninstalling: 'Uninstalling',
  updating: 'Update progress',
  repairing: 'Repair progress',
  installing: 'Install progress'
};

function haloOf(s: Snapshot): { halo: HaloView; percent?: number } {
  const fraction = s.progress?.fraction ?? null;
  const working = s.phase === 'installing' || s.phase === 'updating' || s.phase === 'repairing';
  const label = BUSY_LABELS[s.phase] ?? `${NAME} progress`;
  if (working && fraction !== null) {
    const percent = Math.round(fraction * 100);
    const step = currentStep(s);
    return {
      halo: { kind: 'progress', fraction, label, valuetext: step ? `${percent} percent, ${lowerFirst(step)}` : `${percent} percent` },
      percent
    };
  }
  if (working || s.phase === 'checking' || s.phase === 'starting' || s.phase === 'uninstalling') {
    return { halo: { kind: 'busy', label } };
  }
  if (s.phase === 'running') return { halo: { kind: 'live', label } };
  if (s.phase === 'failed' && (s.failedAction === 'install' || s.failedAction === 'repair')) {
    return { halo: { kind: 'error', label, ...(fraction !== null ? { fraction } : {}) } };
  }
  return { halo: { kind: 'rest', label } };
}

/** The brand light: bright while running, working while it works, dim when something is off, quiet otherwise. */
function poolOf(s: Snapshot, fault: boolean): PoolStrength {
  if (s.phase === 'running') return 'live';
  if (fault) return 'dim';
  if (s.phase === 'installing' || s.phase === 'updating' || s.phase === 'repairing' || s.phase === 'starting') return 'work';
  return 'rest';
}

function heroOf(s: Snapshot, controls: Controls): HeroView {
  const { halo, percent } = haloOf(s);
  const fault = s.phase === 'unsupported' || s.phase === 'stopped' || s.phase === 'failed';
  const fraction = s.progress?.fraction ?? null;
  return {
    mood: MOODS[s.phase],
    ...(s.phase === 'installing' || s.phase === 'repairing' ? (fraction !== null ? { mascotProgress: fraction } : {}) : {}),
    halo,
    ...(percent !== undefined ? { percent } : {}),
    fault,
    update: controls.update,
    pool: poolOf(s, fault)
  };
}

/** "NVIDIA GeForce GTX 1080" becomes "GTX 1080": the card's own name, without the maker the row already names. */
function shortGpuName(name: string): string {
  return name.replace(/^NVIDIA\s+/i, '').replace(/^GeForce\s+/i, '');
}

function unsupportedView(u: Unsupported): {
  headline: string;
  sub: SubView;
  notice: NoticeView;
} {
  const fixable = isFixable(u);
  const link: LinkView = { id: 'requirements', label: 'See the requirements' };
  let notice: NoticeView;
  switch (u.code) {
    case 'driver_too_old':
      notice = { kind: 'reason', name: 'NVIDIA driver', found: u.found, needed: `Needs version ${u.needed} or newer.`, link };
      break;
    case 'low_disk': {
      const found = bytesOf(u.found);
      const needed = bytesOf(u.needed);
      notice = {
        kind: 'reason',
        name: 'Free disk space',
        found: found === null ? null : formatBytes(found),
        needed: needed === null ? 'Needs more free space.' : `Needs about ${formatBytes(needed)}.`,
        link
      };
      break;
    }
    case 'no_nvidia_gpu':
      notice = { kind: 'reason', name: 'NVIDIA graphics card', found: null, needed: 'None found.', link: null };
      break;
    case 'gpu_too_old':
      notice = { kind: 'reason', name: 'NVIDIA graphics card', found: u.found === null ? null : shortGpuName(u.found), needed: 'Needs an RTX 20-series card or newer.', link: null };
      break;
  }
  return fixable
    ? {
        headline: 'Can’t install yet',
        sub: { lead: null, leadAlert: false, text: 'Nothing was downloaded. Fix this, then check again.' },
        notice
      }
    : {
        headline: 'Can’t run on this computer',
        sub: {
          lead: null,
          leadAlert: false,
          text: 'It needs an NVIDIA RTX 20-series graphics card or newer. Nothing was downloaded.'
        },
        notice
      };
}

const LOG_TITLES: Record<FailedAction, string> = {
  install: 'End of the install log',
  repair: 'End of the log',
  update: 'End of the update log',
  start: 'End of the log'
};

function logNotice(s: Snapshot): NoticeView {
  return {
    kind: 'log',
    label: 'Show the log',
    sheetTitle: s.phase === 'failed' && s.failedAction ? LOG_TITLES[s.failedAction] : 'End of the log'
  };
}

function failedView(s: Snapshot): { headline: string; sub: SubView } {
  const cause = causeOf(s);
  switch (s.failedAction) {
    case 'install':
    case 'repair':
      return {
        headline: s.failedAction === 'install' ? 'Install didn’t finish' : 'Repair didn’t finish',
        sub: { lead: cause.lead, leadAlert: true, text: cause.text }
      };
    case 'update':
      return {
        headline: 'Update didn’t finish',
        sub: { lead: null, leadAlert: false, text: 'Repair reinstalls the newest version. Your outputs, settings and models stay.' }
      };
    default:
      return s.error?.code === 'start_exited'
        ? { headline: 'Didn’t start', sub: { lead: null, leadAlert: false, text: 'It stopped while starting. Open tries again.' } }
        : { headline: 'Didn’t start', sub: { lead: null, leadAlert: false, text: 'It didn’t answer in time. Open tries again.' } };
  }
}

const PRIMARIES: Record<Primary, Omit<PrimaryView, 'id'>> = {
  check_again: { label: 'Check again', icon: 'reset', variant: 'primary', loading: false },
  see_requirements: { label: 'See the requirements', icon: null, variant: 'secondary', loading: false },
  install: { label: 'Install', icon: 'download', variant: 'primary', loading: false },
  retry: { label: 'Retry', icon: 'reset', variant: 'primary', loading: false },
  starting: { label: 'Starting…', icon: null, variant: 'primary', loading: true },
  open: { label: `Open ${NAME}`, icon: null, variant: 'primary', loading: false },
  repair: { label: 'Repair', icon: 'reset', variant: 'primary', loading: false },
  quit: { label: 'Quit', icon: null, variant: 'primary', loading: false }
};

const MENU_ITEMS: Record<Controls['menu'][number], MenuItemView> = {
  repair: {
    id: 'repair',
    icon: 'reset',
    label: 'Repair',
    description: 'Downloads its tools again and reinstalls. Your outputs, settings and models stay.',
    danger: false
  },
  uninstall: {
    id: 'uninstall',
    icon: 'trash',
    label: 'Uninstall…',
    description: 'Asks first. Your outputs, settings and models stay.',
    danger: true
  },
  beta: {
    id: 'beta',
    icon: null,
    label: 'Get beta versions',
    description: 'Offers pre-release versions. Turning it off keeps the version you have.',
    danger: false
  }
};

/** The words of the uninstall confirmation (macOS and Linux; Windows starts its own uninstaller). */
export const UNINSTALL_DIALOG = {
  question: `Uninstall ${NAME}?`,
  info: 'It stops and removes the app, Python, uv, ffmpeg and the start-at-login entry.',
  confirm: 'Uninstall',
  staysBefore: 'Your outputs, settings and models stay in',
  staysAnd: 'and',
  paths: ['~/.ziv', '~/.cache/huggingface'],
  link: { id: 'data_folder', label: 'Where your data lives' } satisfies LinkView
} as const;

function defaultFormatTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Everything the window shows for `s`. */
export function launcherView(s: Snapshot, options: ViewOptions = {}): LauncherView {
  const formatTime = options.formatTime ?? defaultFormatTime;
  const controls = launcherControls(s);
  const plain = (text: string): SubView => ({ lead: null, leadAlert: false, text });
  let headline = NAME;
  let sub: SubView | null = null;
  let notice: NoticeView | null = null;

  switch (s.phase) {
    case 'checking':
      headline = 'Checking this computer';
      sub = plain('Before anything is downloaded.');
      break;
    case 'unsupported':
      if (s.unsupported) {
        ({ headline, sub, notice } = unsupportedView(s.unsupported));
      } else {
        headline = 'Can’t install yet';
        sub = plain('Nothing was downloaded.');
      }
      break;
    case 'move_to_applications':
      headline = 'Move to Applications';
      sub = plain('It’s running from the disk image. It needs a fixed place to start at login and to update.');
      notice = { kind: 'note', icon: 'folder', lead: null, text: 'Drag it into Applications, then open it from there.' };
      break;
    case 'needs_install':
      headline = 'Ready to install';
      sub = plain(`This computer can run ${NAME}.`);
      notice = {
        kind: 'note',
        icon: 'download',
        lead: 'About 4 GB to download.',
        text: 'It can take half an hour on a slow connection. Models come later, in your browser.'
      };
      break;
    case 'installing':
      headline = 'Installing';
      sub = plain('You can close this window. Installing carries on, and your browser opens when it’s ready.');
      break;
    case 'starting':
      headline = 'Starting';
      sub = plain('The first start can take a few minutes. Your browser opens when it’s ready.');
      break;
    case 'running':
      headline = 'Running';
      sub = plain(`Closing this window keeps it running. ${findAgain[s.packaging]}`);
      if (s.updateVersion) {
        notice = {
          kind: 'update',
          version: s.updateVersion,
          lead: isPrerelease(s.updateVersion) ? `Beta ${s.updateVersion} is out.` : `Version ${s.updateVersion} is out.`,
          text: 'It restarts to update.',
          link: { id: 'whats_new', label: 'What’s new' },
          button: 'Update'
        };
      } else if (s.needsNewerLauncher) {
        notice = {
          kind: 'newer',
          lead: `Version ${s.needsNewerLauncher} needs a newer launcher.`,
          text: 'Your app, settings and models stay.',
          link: { id: 'install_guide', label: 'Open the install guide' }
        };
      }
      break;
    case 'updating':
      headline = s.updateVersion ? `Updating to ${s.updateVersion}` : 'Updating';
      sub = plain('It’s stopped while it updates, and starts again when it’s done.');
      break;
    case 'repairing':
      headline = 'Repairing';
      sub = plain('It downloads its tools again and reinstalls. Your outputs, settings and models stay.');
      break;
    case 'stopped':
      headline = 'Stopped';
      sub = plain(
        s.stoppedAt !== null
          ? `It stopped on its own at ${formatTime(s.stoppedAt)}. Open starts it again.`
          : 'It stopped on its own. Open starts it again.'
      );
      notice = logNotice(s);
      break;
    case 'failed':
      ({ headline, sub } = failedView(s));
      notice = logNotice(s);
      break;
    case 'uninstalling':
      headline = 'Uninstalling';
      break;
    case 'uninstalled':
      headline = 'Uninstalled';
      sub = {
        lead: finishLead[s.packaging],
        leadAlert: false,
        text: leftoverText(s)
      };
      notice = {
        kind: 'kept',
        entries: [
          { path: '~/.ziv', what: 'Outputs, settings and prompts' },
          { path: '~/.cache/huggingface', what: 'Models' }
        ],
        help: 'Delete them yourself if you want the space back.',
        link: { id: 'data_folder', label: 'Where your data lives' }
      };
      break;
  }

  const working = s.phase === 'installing' || s.phase === 'updating' || s.phase === 'repairing';
  const step = working ? currentStep(s) : null;
  return {
    controls,
    hero: heroOf(s, controls),
    kicker: kickerOf(s),
    kickerAccent: working,
    headline,
    sub,
    notice,
    caption: step ? { label: step, line: s.progress?.line ?? null } : null,
    primary: controls.primary ? { id: controls.primary, ...PRIMARIES[controls.primary] } : null,
    menu: controls.menu.map((id) => (id === 'beta' ? { ...MENU_ITEMS.beta, checked: s.beta } : MENU_ITEMS[id]))
  };
}
