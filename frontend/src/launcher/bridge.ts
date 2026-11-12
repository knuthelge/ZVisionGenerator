// The only module that talks to Tauri: command and event names of the Rust-to-window contract live here and nowhere
// else (see "Command and event contract" in the launcher design). Everything else takes a `Bridge`, so the window is
// tested and previewed without a Tauri runtime.
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import type { LinkId, Snapshot } from './types';

/** Event carrying the full snapshot after every state change. */
export const SNAPSHOT_EVENT = 'launcher:snapshot';

export interface Bridge {
  getSnapshot(): Promise<Snapshot>;
  /** Call `handler` with every snapshot Rust emits; resolves to the function that stops listening. */
  onSnapshot(handler: (snapshot: Snapshot) => void): Promise<UnlistenFn>;
  checkAgain(): Promise<void>;
  install(): Promise<void>;
  retry(): Promise<void>;
  openApp(): Promise<void>;
  updateApp(): Promise<void>;
  repairApp(): Promise<void>;
  uninstallApp(): Promise<void>;
  /** Resolves to the new state of start at login. */
  setAutostart(enabled: boolean): Promise<boolean>;
  /** Resolves to the stored value of Get beta versions; rejects with "busy" while an action runs. */
  setBeta(enabled: boolean): Promise<boolean>;
  openLink(link: LinkId): Promise<void>;
  quitLauncher(): Promise<void>;
  /** Close (hide) the window; Rust turns the close request into a hide. */
  closeWindow(): Promise<void>;
}

export const bridge: Bridge = {
  getSnapshot: () => invoke<Snapshot>('get_snapshot'),
  onSnapshot: (handler) => listen<Snapshot>(SNAPSHOT_EVENT, (event) => handler(event.payload)),
  checkAgain: () => invoke('check_again'),
  install: () => invoke('install'),
  retry: () => invoke('retry'),
  openApp: () => invoke('open_app'),
  updateApp: () => invoke('update_app'),
  repairApp: () => invoke('repair_app'),
  uninstallApp: () => invoke('uninstall_app'),
  setAutostart: (enabled) => invoke<boolean>('set_autostart', { enabled }),
  setBeta: (enabled) => invoke<boolean>('set_beta', { enabled }),
  openLink: (link) => invoke('open_link', { link }),
  quitLauncher: () => invoke('quit_launcher'),
  closeWindow: () => getCurrentWindow().close()
};
