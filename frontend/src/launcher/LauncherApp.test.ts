import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { mountComponent, settle } from '../test-utils/mount';

import type { Bridge } from './bridge';
import LauncherApp from './LauncherApp.svelte';
import type { Snapshot, SnapshotCase } from './types';

const CASES = JSON.parse(readFileSync(resolve(process.cwd(), '..', 'launcher/contract/snapshot_cases.json'), 'utf8')) as SnapshotCase[];

function snapshot(name: string): Snapshot {
  const found = CASES.find((c) => c.name === name);
  if (!found) throw new Error(`No fixture case "${name}"`);
  return structuredClone(found.snapshot);
}

interface Fake {
  bridge: Bridge;
  emit: (snapshot: Snapshot) => void;
  unlisten: ReturnType<typeof vi.fn>;
}

function fakeBridge(first: Snapshot): Fake {
  let handler: ((snapshot: Snapshot) => void) | undefined;
  const unlisten = vi.fn();
  const done = () => Promise.resolve();
  const bridge: Bridge = {
    getSnapshot: vi.fn(() => Promise.resolve(first)),
    onSnapshot: vi.fn((next) => {
      handler = next;
      return Promise.resolve(unlisten);
    }),
    checkAgain: vi.fn(done),
    install: vi.fn(done),
    retry: vi.fn(done),
    openApp: vi.fn(done),
    updateApp: vi.fn(done),
    repairApp: vi.fn(done),
    uninstallApp: vi.fn(done),
    setAutostart: vi.fn((enabled: boolean) => Promise.resolve(enabled)),
    setBeta: vi.fn((enabled: boolean) => Promise.resolve(enabled)),
    openLink: vi.fn(done),
    quitLauncher: vi.fn(done)
  };
  return { bridge, emit: (next) => { flushSync(() => handler?.(next)); }, unlisten };
}

async function open(name: string, change?: (s: Snapshot) => void): Promise<Fake & { target: HTMLElement }> {
  const first = snapshot(name);
  change?.(first);
  const fake = fakeBridge(first);
  const { target } = mountComponent(LauncherApp, { bridge: fake.bridge });
  await settle();
  return { ...fake, target };
}

/** The phase the window shows; null while it draws nothing. */
const shown = (target: HTMLElement) => target.querySelector('[data-testid="launcher"]')?.getAttribute('data-phase') ?? null;
const primary = (target: HTMLElement) => target.querySelector('[data-testid="primary"]') as HTMLButtonElement | null;
const notice = (target: HTMLElement) => target.querySelector('[data-testid="notice"]') as HTMLElement | null;
const link = (target: HTMLElement, id: string) => target.querySelector(`[data-link="${id}"]`) as HTMLButtonElement;
const quitButton = (target: HTMLElement) => target.querySelector('[data-testid="quiet-bar"] [data-quit]') as HTMLButtonElement;
const moreButton = (target: HTMLElement) => target.querySelector('[data-testid="more-button"]') as HTMLButtonElement | null;
const mood = (target: HTMLElement) => target.querySelector('svg[data-mood]')?.getAttribute('data-mood');

beforeEach(() => {
  // Reduced motion: transitions take no time (jsdom has no Web Animations API) and Blob holds still frames.
  window.matchMedia = vi.fn((query: string) => ({
    matches: query.includes('prefers-reduced-motion'),
    media: query,
    addEventListener() {},
    removeEventListener() {}
  })) as never;
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('LauncherApp snapshots', () => {
  it('draws nothing until the first snapshot arrives, then the state', async () => {
    const fake = fakeBridge(snapshot('6 running'));
    let resolveFirst: (s: Snapshot) => void = () => {};
    fake.bridge.getSnapshot = () => new Promise((resolve) => { resolveFirst = resolve; });
    const { target } = mountComponent(LauncherApp, { bridge: fake.bridge });
    await settle();
    expect(shown(target)).toBeNull();

    resolveFirst(snapshot('6 running'));
    await settle();
    expect(shown(target)).toBe('running');
  });

  it('follows the snapshots Rust emits', async () => {
    const { target, emit } = await open('4a needs_install');
    expect(shown(target)).toBe('needs_install');

    emit(snapshot('4b installing'));
    await settle();

    expect(shown(target)).toBe('installing');
    expect(target.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('41');
    expect(target.querySelector('[data-testid="halo-stamp"]')?.textContent).toBe('41%');
  });

  it('prefers an emitted snapshot over a slower answer to get_snapshot', async () => {
    const fake = fakeBridge(snapshot('4a needs_install'));
    let resolveFirst: (s: Snapshot) => void = () => {};
    fake.bridge.getSnapshot = () => new Promise((resolve) => { resolveFirst = resolve; });
    const { target } = mountComponent(LauncherApp, { bridge: fake.bridge });
    await settle();

    fake.emit(snapshot('6 running'));
    resolveFirst(snapshot('4a needs_install'));
    await settle();

    expect(shown(target)).toBe('running');
  });

  it('stops listening when it is torn down', async () => {
    const fake = fakeBridge(snapshot('6 running'));
    const mounted = mountComponent(LauncherApp, { bridge: fake.bridge });
    await settle();

    await mounted.unmount();
    await settle();

    expect(fake.unlisten).toHaveBeenCalledOnce();
  });

  it('waits before drawing the first check, so a quick one never flashes', async () => {
    vi.useFakeTimers();
    const fake = fakeBridge(snapshot('1 checking'));
    const { target } = mountComponent(LauncherApp, { bridge: fake.bridge });
    await settle(100);
    expect(shown(target)).toBeNull();

    await settle(250);
    expect(shown(target)).toBe('checking');
  });

  it('skips the check state when the answer comes in time', async () => {
    vi.useFakeTimers();
    const fake = fakeBridge(snapshot('1 checking'));
    const { target } = mountComponent(LauncherApp, { bridge: fake.bridge });
    await settle(100);

    fake.emit(snapshot('4a needs_install'));
    await settle(500);

    expect(shown(target)).toBe('needs_install');
  });

  it('cheers once when the app comes up, not when the window opens on it', async () => {
    vi.useFakeTimers();
    const opened = await open('6 running');
    expect(mood(opened.target)).toBe('idle');
    opened.emit(snapshot('5 starting'));
    await settle(10);
    opened.emit(snapshot('6 running'));
    await settle(10);
    expect(mood(opened.target)).toBe('cheerful');

    await settle(2500);
    expect(mood(opened.target)).toBe('idle');
  });
});

describe('LauncherApp actions', () => {
  it.each([
    ['2a unsupported driver_too_old', 'check_again', 'checkAgain'],
    ['2a unsupported driver_not_working', 'check_again', 'checkAgain'],
    ['4a needs_install', 'install', 'install'],
    ['4c failed install', 'retry', 'retry'],
    ['6 running', 'open', 'openApp'],
    ['9 stopped', 'open', 'openApp'],
    ['10a failed update', 'open', 'openApp'],
    ['R failed repair', 'open', 'openApp'],
    ['11b uninstalled', 'quit', 'quitLauncher']
  ] as const)('%s: the main button is %s and calls %s', async (name, id, method) => {
    const { target, bridge } = await open(name);

    expect(primary(target)?.dataset.primary).toBe(id);
    primary(target)?.click();

    expect(bridge[method]).toHaveBeenCalledOnce();
  });

  it('shows the main button the snapshot’s controls name, not one it works out itself', async () => {
    const { target, bridge } = await open('6 running', (s) => { s.controls.primary = 'retry'; });

    primary(target)?.click();

    expect(bridge.retry).toHaveBeenCalledOnce();
    expect(bridge.openApp).not.toHaveBeenCalled();
  });

  it('shows no main button where the controls have none', async () => {
    const { target } = await open('4b installing');
    expect(primary(target)).toBeNull();
    expect(target.querySelector('[data-testid="caption"]')).not.toBeNull();
  });

  it('opens the requirements page from See the requirements', async () => {
    const { target, bridge } = await open('2b unsupported no_nvidia_gpu');

    expect(primary(target)?.dataset.primary).toBe('see_requirements');
    primary(target)?.click();

    expect(bridge.openLink).toHaveBeenCalledWith('requirements');
  });

  it('keeps Starting inert', async () => {
    const { target, bridge } = await open('5 starting');

    const button = primary(target) as HTMLButtonElement;
    expect(button.dataset.primary).toBe('starting');
    expect(button.disabled).toBe(true);
    button.click();

    expect(bridge.openApp).not.toHaveBeenCalled();
  });

  it('updates and opens the release notes from the update notice', async () => {
    const { target, bridge } = await open('7 running with update');

    (notice(target)?.querySelector('[data-testid="update-button"]') as HTMLButtonElement).click();
    link(target, 'whats_new').click();

    expect(bridge.updateApp).toHaveBeenCalledOnce();
    expect(bridge.openLink).toHaveBeenCalledWith('whats_new');
  });

  it('offers no update where the controls offer none', async () => {
    const { target } = await open('6 running');
    expect(notice(target)).toBeNull();
    expect(target.querySelector('[data-testid="update-dot"]')).toBeNull();
  });

  it('offers the update again once a failed update has been opened and the app runs', async () => {
    const failed = await open('10a failed update');
    expect(failed.target.querySelector('[data-testid="update-button"]')).toBeNull();

    failed.emit(snapshot('7 running with update'));
    await settle();

    expect(failed.target.querySelector('[data-testid="update-button"]')).not.toBeNull();
    expect(failed.target.querySelector('[data-testid="update-dot"]')).not.toBeNull();
  });

  it('shows the folder an incomplete uninstall left behind', async () => {
    const { target } = await open('11b uninstalled incomplete');
    const folder = target.querySelector('[data-testid="leftover-folder"]');
    expect(folder?.textContent).toBe(snapshot('11b uninstalled incomplete').leftoverFolder);

    document.body.innerHTML = '';
    const clean = await open('11b uninstalled');
    expect(clean.target.querySelector('[data-testid="leftover-folder"]')).toBeNull();
  });

  it('ignores a command Rust refuses as busy', async () => {
    const { target, bridge } = await open('4a needs_install');
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    bridge.install = vi.fn(() => Promise.reject('busy'));

    primary(target)?.click();
    await settle();

    expect(error).not.toHaveBeenCalled();
  });

  it('puts focus on the primary action', async () => {
    const { target } = await open('6 running');
    expect(document.activeElement).toBe(primary(target));
  });

  it('puts focus on Quit when there is no primary action', async () => {
    const { target } = await open('4b installing');
    expect(document.activeElement).toBe(quitButton(target));
  });
});

describe('LauncherApp quiet bar', () => {
  it('offers Quit and More, but no Start at login, before install', async () => {
    const before = await open('4a needs_install');
    expect(before.target.querySelector('[role="switch"]')).toBeNull();
    expect(moreButton(before.target)).not.toBeNull();
    expect(quitButton(before.target)).toBeTruthy();
  });

  it('offers Quit alone while installing', async () => {
    const { target } = await open('4b installing');
    expect(target.querySelector('[role="switch"]')).toBeNull();
    expect(moreButton(target)).toBeNull();
  });

  it('quits from the bar', async () => {
    const { target, bridge } = await open('6 running');

    quitButton(target).click();

    expect(bridge.quitLauncher).toHaveBeenCalledOnce();
  });

  it('drops the bar once uninstalled', async () => {
    const { target } = await open('11b uninstalled');
    expect(target.querySelector('[data-testid="quiet-bar"]')).toBeNull();
  });

  it('sets start at login and shows the state Rust answers', async () => {
    const { target, bridge } = await open('6 running');
    const toggle = target.querySelector('[role="switch"]') as HTMLInputElement;
    expect(toggle.checked).toBe(true);

    toggle.click();
    await settle();

    expect(bridge.setAutostart).toHaveBeenCalledWith(false);
    expect(toggle.checked).toBe(false);
  });

  it('puts the switch back when Rust refuses', async () => {
    const { target, bridge } = await open('6 running');
    bridge.setAutostart = vi.fn(() => Promise.reject('busy'));
    const toggle = target.querySelector('[role="switch"]') as HTMLInputElement;

    toggle.click();
    await settle();

    expect(toggle.checked).toBe(true);
  });

  it('describes the switch for screen readers', async () => {
    const { target } = await open('6 running');
    const toggle = target.querySelector('[role="switch"]') as HTMLInputElement;

    const hint = document.getElementById(toggle.getAttribute('aria-describedby') ?? '');
    expect(hint?.textContent).toBeTruthy();
  });

  it('disables Start at login and More while updating', async () => {
    const { target } = await open('8a updating');

    expect((target.querySelector('[role="switch"]') as HTMLInputElement).disabled).toBe(true);
    expect((moreButton(target) as HTMLButtonElement).disabled).toBe(true);
  });

  it('lists Repair and Uninstall in More, and repairs from it', async () => {
    const { target, bridge } = await open('6 running');

    (moreButton(target) as HTMLButtonElement).click();
    await settle();
    const items = Array.from(document.querySelectorAll('[role^="menuitem"]')).map((el) => el.getAttribute('data-action'));
    expect(items).toEqual(['repair', 'uninstall', 'beta']);
    expect(document.querySelectorAll('[role="separator"]')).toHaveLength(1);

    (document.querySelector('[data-action="repair"]') as HTMLButtonElement).click();
    await settle();
    expect(bridge.repairApp).toHaveBeenCalledOnce();
  });

  it('asks before uninstalling, then uninstalls', async () => {
    const { target, bridge } = await open('6 running');
    (moreButton(target) as HTMLButtonElement).click();
    await settle();

    (document.querySelector('[data-action="uninstall"]') as HTMLButtonElement).click();
    await settle();
    expect(bridge.uninstallApp).not.toHaveBeenCalled();
    expect(document.querySelector('[data-testid="confirm-dialog"]')).not.toBeNull();

    (document.querySelector('[data-action="confirm"]') as HTMLButtonElement).click();
    await settle();
    expect(bridge.uninstallApp).toHaveBeenCalledOnce();
  });

  it('starts the Windows uninstaller without asking', async () => {
    const { target, bridge } = await open('6 running', (s) => { s.packaging = 'windows_nsis'; });
    (moreButton(target) as HTMLButtonElement).click();
    await settle();

    (document.querySelector('[data-action="uninstall"]') as HTMLButtonElement).click();
    await settle();

    expect(bridge.uninstallApp).toHaveBeenCalledOnce();
    expect(document.querySelector('[data-testid="confirm-dialog"]')).toBeNull();
  });
});

describe('LauncherApp beta', () => {
  const openMore = async (target: HTMLElement) => {
    (moreButton(target) as HTMLButtonElement).click();
    await settle();
    return document.querySelector('[data-action="beta"]') as HTMLButtonElement;
  };

  it('offers Get beta versions as the only More entry before install, unchecked by default', async () => {
    const { target } = await open('4a needs_install');
    const item = await openMore(target);

    expect(Array.from(document.querySelectorAll('[role^="menuitem"]'))).toHaveLength(1);
    expect(document.querySelectorAll('[role="separator"]')).toHaveLength(0);
    expect(item.getAttribute('aria-checked')).toBe('false');
    expect(item.getAttribute('role')).toBe('menuitemcheckbox');
  });

  it('shows the stored choice as checked', async () => {
    const { target } = await open('4a needs_install with beta');
    expect((await openMore(target)).getAttribute('aria-checked')).toBe('true');
  });

  it('turns beta on through set_beta and shows what Rust stored', async () => {
    const { target, bridge } = await open('6 running');
    (await openMore(target)).click();
    await settle();

    expect(bridge.setBeta).toHaveBeenCalledExactlyOnceWith(true);
    expect((await openMore(target)).getAttribute('aria-checked')).toBe('true');
  });

  it('turns beta off again from a checked item', async () => {
    const { target, bridge } = await open('6 running a beta version');
    (await openMore(target)).click();
    await settle();

    expect(bridge.setBeta).toHaveBeenCalledExactlyOnceWith(false);
  });

  it('puts the checkbox back when Rust refuses', async () => {
    const { target, bridge } = await open('6 running');
    bridge.setBeta = vi.fn(() => Promise.reject('busy'));
    (await openMore(target)).click();
    await settle();

    expect(bridge.setBeta).toHaveBeenCalledOnce();
    expect((await openMore(target)).getAttribute('aria-checked')).toBe('false');
  });

  it('follows a snapshot that changes the stored choice', async () => {
    const { target, emit } = await open('6 running');
    const next = snapshot('6 running');
    next.beta = true;
    emit(next);
    await settle();

    expect((await openMore(target)).getAttribute('aria-checked')).toBe('true');
  });
});

describe('LauncherApp log', () => {
  it('opens the end of the log in a sheet', async () => {
    const { target } = await open('4c failed install', (s) => { s.logTail = ['first line', 'last line']; });

    (target.querySelector('[data-testid="show-log"]') as HTMLButtonElement).click();
    await settle();

    const sheet = document.querySelector('[data-testid="sheet"] [role="dialog"]') as HTMLElement;
    expect(sheet.getAttribute('aria-label')).toBeTruthy();
    expect(sheet.querySelector('pre')?.textContent).toBe('first line\nlast line');
  });

  it('copies the log and acknowledges it', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    Object.defineProperty(window, 'isSecureContext', { value: true, configurable: true });
    const { target } = await open('9 stopped', (s) => { s.logTail = ['a', 'b']; });
    (target.querySelector('[data-testid="show-log"]') as HTMLButtonElement).click();
    await settle();
    const copy = document.querySelector('[data-testid="copy-log"]') as HTMLButtonElement;
    const before = copy.textContent;

    copy.click();
    await settle();

    expect(writeText).toHaveBeenCalledWith('a\nb');
    expect(copy.textContent).not.toBe(before);
  });
});
