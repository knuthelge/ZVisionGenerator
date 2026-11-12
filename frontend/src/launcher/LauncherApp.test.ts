import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { buttonByText, mountComponent, settle } from '../test-utils/mount';

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
    quitLauncher: vi.fn(done),
    closeWindow: vi.fn(done)
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

const headline = (target: HTMLElement) => target.querySelector('h1')?.textContent;

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
    expect(target.querySelector('h1')).toBeNull();

    resolveFirst(snapshot('6 running'));
    await settle();
    expect(headline(target)).toBe('Running');
  });

  it('follows the snapshots Rust emits', async () => {
    const { target, emit } = await open('4a needs_install');
    expect(headline(target)).toBe('Ready to install');

    emit(snapshot('4b installing'));
    await settle();

    expect(headline(target)).toBe('Installing');
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

    expect(headline(target)).toBe('Running');
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
    expect(target.querySelector('h1')).toBeNull();

    await settle(250);
    expect(headline(target)).toBe('Checking this computer');
  });

  it('skips the check state when the answer comes in time', async () => {
    vi.useFakeTimers();
    const fake = fakeBridge(snapshot('1 checking'));
    const { target } = mountComponent(LauncherApp, { bridge: fake.bridge });
    await settle(100);

    fake.emit(snapshot('4a needs_install'));
    await settle(500);

    expect(headline(target)).toBe('Ready to install');
  });

  it('cheers once when the app comes up, not when the window opens on it', async () => {
    vi.useFakeTimers();
    const opened = await open('6 running');
    expect(opened.target.querySelector('svg[data-mood]')?.getAttribute('data-mood')).toBe('idle');
    opened.emit(snapshot('5 starting'));
    await settle(10);
    opened.emit(snapshot('6 running'));
    await settle(10);
    expect(opened.target.querySelector('svg[data-mood]')?.getAttribute('data-mood')).toBe('cheerful');

    await settle(2500);
    expect(opened.target.querySelector('svg[data-mood]')?.getAttribute('data-mood')).toBe('idle');
  });
});

describe('LauncherApp actions', () => {
  it.each([
    ['2a unsupported driver_too_old', 'Check again', 'checkAgain'],
    ['4a needs_install', 'Install', 'install'],
    ['4c failed install', 'Retry', 'retry'],
    ['6 running', 'Open Z-Vision Generator', 'openApp'],
    ['10a failed update', 'Repair', 'repairApp'],
    ['11b uninstalled', 'Quit', 'quitLauncher']
  ] as const)('%s: %s calls %s', async (name, label, method) => {
    const { target, bridge } = await open(name);

    buttonByText(label, target.querySelector('.action-zone') as HTMLElement).click();

    expect(bridge[method]).toHaveBeenCalledOnce();
  });

  it('opens the requirements page from See the requirements', async () => {
    const { target, bridge } = await open('2b unsupported no_nvidia_gpu');

    buttonByText('See the requirements', target).click();

    expect(bridge.openLink).toHaveBeenCalledWith('requirements');
  });

  it('keeps Starting inert', async () => {
    const { target, bridge } = await open('5 starting');

    const button = buttonByText('Starting', target.querySelector('.action-zone') as HTMLElement);
    expect(button.disabled).toBe(true);
    button.click();

    expect(bridge.openApp).not.toHaveBeenCalled();
  });

  it('updates and opens the release notes from the update notice', async () => {
    const { target, bridge } = await open('7 running with update');

    buttonByText('Update', target.querySelector('.notice') as HTMLElement).click();
    buttonByText('What’s new', target).click();

    expect(bridge.updateApp).toHaveBeenCalledOnce();
    expect(bridge.openLink).toHaveBeenCalledWith('whats_new');
  });

  it('opens the install guide from the newer-launcher warning', async () => {
    const { target, bridge } = await open('10b running needs newer launcher');

    buttonByText('Open the install guide', target).click();

    expect(bridge.openLink).toHaveBeenCalledWith('install_guide');
  });

  it('ignores a command Rust refuses as busy', async () => {
    const { target, bridge } = await open('4a needs_install');
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    bridge.install = vi.fn(() => Promise.reject('busy'));

    buttonByText('Install', target.querySelector('.action-zone') as HTMLElement).click();
    await settle();

    expect(error).not.toHaveBeenCalled();
  });

  it('hides the window on Ctrl+W and Cmd+W', async () => {
    const { bridge } = await open('6 running');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w', ctrlKey: true, cancelable: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w', metaKey: true, cancelable: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w', cancelable: true }));

    expect(bridge.closeWindow).toHaveBeenCalledTimes(2);
  });

  it('puts focus on the primary action', async () => {
    await open('6 running');
    expect(document.activeElement?.textContent?.trim()).toBe('Open Z-Vision Generator');
  });

  it('puts focus on Quit when there is no primary action', async () => {
    await open('4b installing');
    expect(document.activeElement?.textContent?.trim()).toBe('Quit');
  });
});

describe('LauncherApp quiet bar', () => {
  it('offers Quit and More, but no Start at login, before install', async () => {
    const before = await open('4a needs_install');
    expect(before.target.querySelector('[role="switch"]')).toBeNull();
    expect(before.target.querySelector('[aria-label="More actions"]')).not.toBeNull();
    expect(buttonByText('Quit', before.target.querySelector('footer') as HTMLElement)).toBeTruthy();
  });

  it('offers Quit alone while installing', async () => {
    const { target } = await open('4b installing');
    expect(target.querySelector('[role="switch"]')).toBeNull();
    expect(target.querySelector('[aria-label="More actions"]')).toBeNull();
  });

  it('quits from the bar', async () => {
    const { target, bridge } = await open('6 running');

    buttonByText('Quit', target.querySelector('footer') as HTMLElement).click();

    expect(bridge.quitLauncher).toHaveBeenCalledOnce();
  });

  it('drops the bar once uninstalled', async () => {
    const { target } = await open('11b uninstalled');
    expect(target.querySelector('footer')).toBeNull();
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
    expect(hint?.textContent).toBe('Starts in the background at login.');
  });

  it('disables Start at login and More while updating', async () => {
    const { target } = await open('8a updating');

    expect((target.querySelector('[role="switch"]') as HTMLInputElement).disabled).toBe(true);
    expect((target.querySelector('[aria-label="More actions"]') as HTMLButtonElement).disabled).toBe(true);
  });

  it('lists Repair and Uninstall in More, and repairs from it', async () => {
    const { target, bridge } = await open('6 running');

    (target.querySelector('[aria-label="More actions"]') as HTMLButtonElement).click();
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
    (target.querySelector('[aria-label="More actions"]') as HTMLButtonElement).click();
    await settle();

    (document.querySelector('[data-action="uninstall"]') as HTMLButtonElement).click();
    await settle();
    expect(bridge.uninstallApp).not.toHaveBeenCalled();
    expect(document.querySelector('[data-testid="confirm-dialog"]')?.textContent).toContain('Uninstall Z-Vision Generator?');

    (document.querySelector('[data-action="confirm"]') as HTMLButtonElement).click();
    await settle();
    expect(bridge.uninstallApp).toHaveBeenCalledOnce();
  });

  it('starts the Windows uninstaller without asking', async () => {
    const { target, bridge } = await open('6 running', (s) => { s.packaging = 'windows_nsis'; });
    (target.querySelector('[aria-label="More actions"]') as HTMLButtonElement).click();
    await settle();

    (document.querySelector('[data-action="uninstall"]') as HTMLButtonElement).click();
    await settle();

    expect(bridge.uninstallApp).toHaveBeenCalledOnce();
    expect(document.querySelector('[data-testid="confirm-dialog"]')).toBeNull();
  });
});

describe('LauncherApp beta', () => {
  const openMore = async (target: HTMLElement) => {
    (target.querySelector('[aria-label="More actions"]') as HTMLButtonElement).click();
    await settle();
    return document.querySelector('[data-action="beta"]') as HTMLButtonElement;
  };

  it('offers Get beta versions as the only More entry before install, unchecked by default', async () => {
    const { target } = await open('4a needs_install');
    const item = await openMore(target);

    expect(Array.from(document.querySelectorAll('[role^="menuitem"]'))).toHaveLength(1);
    expect(document.querySelectorAll('[role="separator"]')).toHaveLength(0);
    expect(item.textContent).toContain('Get beta versions');
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

  it('marks a beta install in the kicker and a beta update in the notice', async () => {
    const running = await open('6 running a beta version');
    expect(running.target.querySelector('.kicker')?.textContent).toBe('Version 0.15.0b2 · beta');
    document.body.innerHTML = '';
    const update = await open('7 running with a beta update');
    expect(update.target.textContent).toContain('Beta 0.15.0b3 is out.');
  });
});

describe('LauncherApp log', () => {
  it('opens the end of the log in a sheet titled by what failed', async () => {
    const { target } = await open('4c failed install', (s) => { s.logTail = ['first line', 'last line']; });

    buttonByText('Show the log', target).click();
    await settle();

    const sheet = document.querySelector('[data-testid="sheet"] [role="dialog"]') as HTMLElement;
    expect(sheet.getAttribute('aria-label')).toBe('End of the install log');
    expect(sheet.querySelector('pre')?.textContent).toBe('first line\nlast line');
  });

  it('copies the log and says so', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    Object.defineProperty(window, 'isSecureContext', { value: true, configurable: true });
    const { target } = await open('9 stopped', (s) => { s.logTail = ['a', 'b']; });
    buttonByText('Show the log', target).click();
    await settle();

    (document.querySelector('[aria-label="Copy the end of the log"]') as HTMLButtonElement).click();
    await settle();

    expect(writeText).toHaveBeenCalledWith('a\nb');
    expect(document.querySelector('[aria-label="Copy the end of the log"]')?.textContent).toContain('Copied');
  });
});
