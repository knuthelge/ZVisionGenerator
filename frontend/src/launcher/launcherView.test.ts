import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { formatBytes, isPrerelease, launcherControls, launcherView } from './launcherView';
import type { Snapshot, SnapshotCase } from './types';

// The cases `snapshot::controls()` in Rust is tested against: both sides must produce the same controls.
const CASES = JSON.parse(readFileSync(resolve(process.cwd(), '..', 'launcher/contract/snapshot_cases.json'), 'utf8')) as SnapshotCase[];

function snapshot(name: string): Snapshot {
  const found = CASES.find((c) => c.name === name);
  if (!found) throw new Error(`No fixture case "${name}"`);
  return found.snapshot;
}

const view = (name: string) => launcherView(snapshot(name), { formatTime: () => '14:32' });

describe('controls', () => {
  it.each(CASES.map((c) => [c.name, c] as const))('match the fixture for %s', (_name, c) => {
    expect(launcherControls(c.snapshot)).toEqual(c.controls);
    expect(launcherView(c.snapshot).controls).toEqual(c.controls);
  });

  it('covers every phase of the contract', () => {
    const phases = new Set(CASES.map((c) => c.snapshot.phase));
    expect(phases.size).toBe(13);
  });
});

describe('hero', () => {
  it('draws the ring from the progress fraction while installing', () => {
    const { hero, kicker, kickerAccent } = view('4b installing');
    expect(hero.halo).toMatchObject({ kind: 'progress', fraction: 0.41 });
    expect(hero.halo.valuetext).toBe('41 percent, installing Z-Vision Generator');
    expect(hero.percent).toBe(41);
    expect(hero.mood).toBe('creating');
    expect(hero.mascotProgress).toBe(0.41);
    expect(hero.pool).toBe('work');
    expect(kicker).toBe('Step 2 of 3');
    expect(kickerAccent).toBe(true);
  });

  it('shows a busy ring when the work has no fraction', () => {
    for (const name of ['8a updating', '1 checking', '5 starting', 'U uninstalling']) {
      const { hero } = view(name);
      expect(hero.halo.kind, name).toBe('busy');
      expect(hero.percent, name).toBeUndefined();
    }
  });

  it('keeps the brand light quiet while checking and uninstalling, working while starting', () => {
    expect(view('1 checking').hero.pool).toBe('rest');
    expect(view('U uninstalling').hero.pool).toBe('rest');
    expect(view('5 starting').hero.pool).toBe('work');
  });

  it('freezes the ring in red where an install stopped, with no stamp', () => {
    const { hero, kicker, kickerAccent } = view('4c failed install');
    expect(hero.halo).toMatchObject({ kind: 'error', fraction: 0.41 });
    expect(hero.percent).toBeUndefined();
    expect(hero.fault).toBe(true);
    expect(hero.pool).toBe('dim');
    expect(kicker).toBe('Step 2 of 3');
    expect(kickerAccent).toBe(false);
  });

  it('draws no arc for a failure that had no fraction', () => {
    expect(view('4c failed install (windows)').hero.halo.fraction).toBeUndefined();
  });

  it('closes the ring and lights up while running; an update adds the dot', () => {
    expect(view('6 running').hero).toMatchObject({ mood: 'idle', halo: { kind: 'live' }, pool: 'live', update: false });
    expect(view('7 running with update').hero.update).toBe(true);
    expect(view('10b running needs newer launcher').hero.update).toBe(false);
  });

  it('opens the ring back to its track when it stopped or did not start', () => {
    for (const name of ['9 stopped', '9b failed start', '10a failed update']) {
      const { hero } = view(name);
      expect(hero.halo.kind, name).toBe('rest');
      expect(hero.mood, name).toBe('sad');
      expect(hero.pool, name).toBe('dim');
    }
  });

  it('gives each unsupported and setup state its mood', () => {
    expect(view('2a unsupported driver_too_old').hero.mood).toBe('sad');
    expect(view('3 move_to_applications').hero.mood).toBe('curious');
    expect(view('4a needs_install').hero.mood).toBe('nodding');
    expect(view('11b uninstalled').hero.mood).toBe('waving');
  });
});

describe('kicker', () => {
  it('says Setup before install, the version once installed, and nothing when uninstalled', () => {
    expect(view('1 checking').kicker).toBe('Setup');
    expect(view('4a needs_install').kicker).toBe('Setup');
    expect(view('6 running').kicker).toBe('Version 0.14.0');
    expect(view('9 stopped').kicker).toBe('Version 0.14.0');
    expect(view('11b uninstalled').kicker).toBe('');
  });

  it('counts the steps of the work in progress', () => {
    expect(view('8a updating').kicker).toBe('Step 1 of 2');
    expect(view('8b repairing').kicker).toBe('Step 3 of 4');
    expect(view('R failed repair').kicker).toBe('Step 1 of 4');
  });
});

describe('headline and sub', () => {
  it('names the update target while updating', () => {
    expect(view('8a updating with updateVersion').headline).toBe('Updating to 0.14.1');
    expect(view('8a updating').headline).toBe('Updating');
  });

  it('says where to find the app by platform', () => {
    expect(view('6 running').sub?.text).toContain('the menu bar');
    expect(view('7 running with update').sub?.text).toContain('the top bar');
    const windows = { ...snapshot('6 running'), packaging: 'windows_nsis' as const };
    expect(launcherView(windows).sub?.text).toContain('the notification area');
  });

  it('gives a failed install or repair its cause as an alert lead', () => {
    const install = view('4c failed install');
    expect(install.headline).toBe('Install didn’t finish');
    expect(install.sub).toMatchObject({ lead: 'The download didn’t finish.', leadAlert: true });
    const repair = view('R failed repair');
    expect(repair.headline).toBe('Repair didn’t finish');
    expect(repair.sub?.lead).toContain('free disk space');
    expect(view('4c failed install (windows)').sub?.lead).toBe('The installer stopped with an error.');
  });

  it('tells a failed update to repair, and a failed start to open again', () => {
    expect(view('10a failed update').headline).toBe('Update didn’t finish');
    expect(view('10a failed update').sub?.text).toContain('Repair reinstalls');
    expect(view('9b failed start').headline).toBe('Didn’t start');
  });

  it('reports when it stopped', () => {
    expect(view('9 stopped').sub?.text).toBe('It stopped on its own at 14:32. Open starts it again.');
  });

  it('has no sub while uninstalling, and an instruction once uninstalled', () => {
    expect(view('U uninstalling').sub).toBeNull();
    expect(view('11b uninstalled').sub).toMatchObject({ lead: 'Delete the AppImage to finish.', leadAlert: false });
    expect(view('11b uninstalled incomplete').sub).toMatchObject({
      lead: 'Move Z-Vision Generator to the Trash to finish.',
      text: 'Some files could not be removed.'
    });
  });
});

describe('notice', () => {
  it('explains an unsupported machine with one reason', () => {
    expect(view('2a unsupported driver_too_old').notice).toMatchObject({
      kind: 'reason',
      name: 'NVIDIA driver',
      found: '575.51.03',
      needed: 'Needs version 580 or newer.',
      link: { id: 'requirements' }
    });
    expect(view('2a unsupported low_disk (macos)').notice).toMatchObject({
      name: 'Free disk space',
      found: '7.3 GB',
      needed: 'Needs about 10 GB.'
    });
    expect(view('2b unsupported no_nvidia_gpu').notice).toMatchObject({ found: null, needed: 'None found.', link: null });
    expect(view('2b unsupported gpu_too_old').notice).toMatchObject({ found: 'GTX 1080', link: null });
  });

  it('offers the update and the newer-launcher warning as alerts', () => {
    expect(view('7 running with update').notice).toMatchObject({ kind: 'update', version: '0.14.1', link: { id: 'whats_new' } });
    expect(view('10b running needs newer launcher').notice).toMatchObject({ kind: 'newer', link: { id: 'install_guide' } });
    expect(view('6 running').notice).toBeNull();
  });

  it('offers the log where the controls do, titled by what failed', () => {
    for (const c of CASES) {
      const notice = launcherView(c.snapshot).notice;
      expect(notice?.kind === 'log', c.name).toBe(c.controls.log);
    }
    expect(view('4c failed install').notice).toMatchObject({ sheetTitle: 'End of the install log' });
    expect(view('10a failed update').notice).toMatchObject({ sheetTitle: 'End of the update log' });
    expect(view('9 stopped').notice).toMatchObject({ sheetTitle: 'End of the log' });
  });

  it('shows what stays once uninstalled', () => {
    const notice = view('11b uninstalled').notice;
    expect(notice).toMatchObject({ kind: 'kept', link: { id: 'data_folder' } });
  });

  it('only notes the move to Applications; there is no button for it', () => {
    expect(view('3 move_to_applications').notice?.kind).toBe('note');
    expect(view('3 move_to_applications').primary).toBeNull();
  });
});

describe('action zone', () => {
  it('names the primary action of each state', () => {
    expect(view('4a needs_install').primary).toMatchObject({ id: 'install', label: 'Install', icon: 'download' });
    expect(view('2b unsupported no_nvidia_gpu').primary).toMatchObject({ id: 'see_requirements', variant: 'secondary' });
    expect(view('5 starting').primary).toMatchObject({ id: 'starting', label: 'Starting…', loading: true });
    expect(view('6 running').primary?.label).toBe('Open Z-Vision Generator');
    expect(view('10a failed update').primary).toMatchObject({ id: 'repair', label: 'Repair' });
    expect(view('11b uninstalled').primary).toMatchObject({ id: 'quit', label: 'Quit' });
  });

  it('has a primary exactly when the controls do', () => {
    for (const c of CASES) expect(launcherView(c.snapshot).primary?.id ?? null, c.name).toBe(c.controls.primary);
  });

  it('shows the step in progress as a caption while working, and not otherwise', () => {
    expect(view('4b installing').caption).toEqual({ label: 'Installing Z-Vision Generator', line: 'Downloading torch (2.1GiB)' });
    expect(view('8b repairing').caption?.label).toBe('Reinstalling Z-Vision Generator');
    expect(view('8a updating with updateVersion').caption).toEqual({ label: 'Installing 0.14.1', line: null });
    expect(view('4c failed install').caption).toBeNull();
    expect(view('6 running').caption).toBeNull();
  });

  it('lists More entries in order, with Repair left out where it is the primary', () => {
    expect(view('6 running').menu.map((m) => m.id)).toEqual(['repair', 'uninstall', 'beta']);
    expect(view('10a failed update').menu.map((m) => m.id)).toEqual(['uninstall', 'beta']);
    expect(view('4a needs_install').menu.map((m) => m.id)).toEqual(['beta']);
    expect(view('4c failed install (windows)').menu.map((m) => m.id)).toEqual(['beta']);
    expect(view('4c failed install').menu.map((m) => m.id)).toEqual(['uninstall', 'beta']);
    expect(view('1 checking').menu).toEqual([]);
  });

  it('carries the stored beta choice on the beta entry only', () => {
    const entries = (name: string) => view(name).menu;
    expect(entries('6 running').map((m) => m.checked)).toEqual([undefined, undefined, false]);
    expect(entries('4a needs_install with beta')[0]).toMatchObject({ id: 'beta', label: 'Get beta versions', checked: true });
  });
});

describe('beta wording', () => {
  it('adds a beta tag to the kicker of a pre-release install only', () => {
    expect(view('6 running').kicker).toBe('Version 0.14.0');
    expect(view('6 running a beta version').kicker).toBe('Version 0.15.0b2 · beta');
  });

  it('calls a pre-release update a beta, and a stable one a version', () => {
    expect(view('7 running with update').notice).toMatchObject({ lead: 'Version 0.14.1 is out.' });
    expect(view('7 running with a beta update').notice).toMatchObject({ kind: 'update', version: '0.15.0b3', lead: 'Beta 0.15.0b3 is out.' });
  });

  it('knows pre-release version spellings', () => {
    for (const v of ['0.15.0b2', '1.0rc1', '1.0.0a1', '1.0.dev3', '1.0-beta.2', 'v2.0rc1', '1.0b1+local']) expect(isPrerelease(v), v).toBe(true);
    for (const v of ['0.14.1', '1.0', '1.0.post1', 'garbage', '']) expect(isPrerelease(v), v).toBe(false);
  });
});

describe('copy', () => {
  it('words a failed or unavailable download without a dead end', () => {
    const failed = launcherView({ ...snapshot('4c failed install'), error: { code: 'download_failed', detail: null } });
    expect(failed.sub?.lead).toBe('The download didn’t finish.');
    const gone = launcherView({ ...snapshot('4c failed install'), error: { code: 'download_unavailable', detail: 'x' } });
    expect(gone.sub?.text).toBe('A newer launcher fixes this.');
  });
});

describe('formatBytes', () => {
  it('uses decimal units and drops a trailing .0', () => {
    expect(formatBytes(4_200_000_000)).toBe('4.2 GB');
    expect(formatBytes(10_000_000_000)).toBe('10 GB');
    expect(formatBytes(850_000_000)).toBe('850 MB');
  });
});
