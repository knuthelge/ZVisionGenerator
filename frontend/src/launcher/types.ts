// The Rust-to-window contract. Mirrors `launcher/src-tauri/src/snapshot.rs`; the fixture
// `launcher/contract/snapshot_cases.json` holds Rust's own snapshots, which the window's tests render. Words live in
// the window; Rust sends codes and raw values.

export type Phase =
  | 'checking'
  | 'move_to_applications'
  | 'unsupported'
  | 'needs_install'
  | 'installing'
  | 'starting'
  | 'running'
  | 'updating'
  | 'repairing'
  | 'stopped'
  | 'failed'
  | 'uninstalling'
  | 'uninstalled';

export type StepId = 'download_tools' | 'clean_cache' | 'install_app' | 'update_app' | 'finish';

export type ErrorCode =
  | 'download_failed'
  | 'download_unavailable'
  | 'checksum_mismatch'
  | 'tool_broken'
  | 'low_disk'
  | 'uv_failed'
  | 'start_exited'
  | 'start_timeout'
  | 'uninstall_incomplete';

export type LinkId = 'requirements' | 'install_guide' | 'data_folder' | 'whats_new';

export type Packaging = 'macos_app' | 'windows_nsis' | 'appimage' | 'linux_package';

export type UnsupportedCode = 'no_nvidia_gpu' | 'driver_too_old' | 'gpu_too_old' | 'low_disk';

export type FailedAction = 'install' | 'repair' | 'update' | 'start';

export interface Unsupported {
  code: UnsupportedCode;
  /** Raw value: the driver string, the best GPU's name, or free bytes as a decimal string; null when absent. */
  found: string | null;
  /** Raw value: "580", "7.5" or "10000000000". */
  needed: string;
}

export interface Progress {
  steps: StepId[];
  /** 0-based position in `steps`. */
  index: number;
  /** 0..1 during the app install step of install and repair, else null (a busy ring). */
  fraction: number | null;
  line: string | null;
}

export interface Snapshot {
  phase: Phase;
  installedVersion: string | null;
  /** Set while an update is offered, kept through `updating` as the target; cleared on success or needs-newer. */
  updateVersion: string | null;
  /** Never set together with `updateVersion`. */
  needsNewerLauncher: string | null;
  autostart: boolean;
  /** The user opted into pre-release versions of the app. */
  beta: boolean;
  packaging: Packaging;
  /** Phase `unsupported` only. */
  unsupported: Unsupported | null;
  progress: Progress | null;
  failedAction: FailedAction | null;
  error: ErrorCode | null;
  /** Root, which an incomplete uninstall could not delete; set only with error `uninstall_incomplete`. */
  leftoverFolder: string | null;
  /** `failed` and `stopped` only; [] otherwise. */
  logTail: string[];
  /** Unix milliseconds; `stopped` only. */
  stoppedAt: number | null;
  /** What the window shows and allows in this state; derived in Rust. */
  controls: Controls;
}

export type Primary =
  | 'check_again'
  | 'see_requirements'
  | 'install'
  | 'retry'
  | 'starting'
  | 'open'
  | 'repair'
  | 'quit';

export interface Controls {
  primary: Primary | null;
  update: boolean;
  log: boolean;
  bar: boolean;
  autostart: boolean;
  /** More is shown iff non-empty; in this order, `beta` last after a separator. */
  menu: ('repair' | 'uninstall' | 'beta')[];
  /** Toggle and More are shown but disabled. */
  locked: boolean;
}

/** One case of `launcher/contract/snapshot_cases.json`. */
export interface SnapshotCase {
  name: string;
  snapshot: Snapshot;
}
