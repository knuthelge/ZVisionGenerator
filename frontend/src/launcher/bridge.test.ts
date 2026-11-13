import { clearMocks, mockIPC } from '@tauri-apps/api/mocks';
import { emit } from '@tauri-apps/api/event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SNAPSHOT_EVENT, bridge } from './bridge';
import type { Snapshot } from './types';

const SNAPSHOT = { phase: 'running', installedVersion: '0.14.0' } as unknown as Snapshot;

function mock(results: Record<string, unknown> = {}) {
  const calls: [string, unknown][] = [];
  mockIPC(
    (cmd, args) => {
      calls.push([cmd, args]);
      return results[cmd];
    },
    { shouldMockEvents: true }
  );
  return calls;
}

afterEach(() => clearMocks());

describe('bridge', () => {
  it('names the event the snapshots arrive on', () => {
    expect(SNAPSHOT_EVENT).toBe('launcher:snapshot');
  });

  it('asks for the snapshot with get_snapshot', async () => {
    const calls = mock({ get_snapshot: SNAPSHOT });

    await expect(bridge.getSnapshot()).resolves.toEqual(SNAPSHOT);
    expect(calls.map(([cmd]) => cmd)).toEqual(['get_snapshot']);
  });

  it('sends each action as its own command with no arguments', async () => {
    const calls = mock();

    await bridge.checkAgain();
    await bridge.install();
    await bridge.retry();
    await bridge.openApp();
    await bridge.updateApp();
    await bridge.repairApp();
    await bridge.uninstallApp();
    await bridge.quitLauncher();

    expect(calls.map(([cmd]) => cmd)).toEqual([
      'check_again',
      'install',
      'retry',
      'open_app',
      'update_app',
      'repair_app',
      'uninstall_app',
      'quit_launcher'
    ]);
  });

  it('sends start at login and Get beta versions with an enabled flag and returns the stored state', async () => {
    const calls = mock({ set_autostart: true, set_beta: false });

    await expect(bridge.setAutostart(true)).resolves.toBe(true);
    await expect(bridge.setBeta(true)).resolves.toBe(false);
    expect(calls).toEqual([
      ['set_autostart', { enabled: true }],
      ['set_beta', { enabled: true }]
    ]);
  });

  it('opens a link by its id', async () => {
    const calls = mock();

    await bridge.openLink('whats_new');

    expect(calls).toEqual([['open_link', { link: 'whats_new' }]]);
  });

  it('hands each launcher:snapshot payload to the listener, until it stops', async () => {
    mock();
    const handler = vi.fn();

    const stop = await bridge.onSnapshot(handler);
    await emit(SNAPSHOT_EVENT, SNAPSHOT);
    expect(handler).toHaveBeenCalledExactlyOnceWith(SNAPSHOT);

    stop();
    await emit(SNAPSHOT_EVENT, SNAPSHOT);
    expect(handler).toHaveBeenCalledTimes(1);
  });
});
