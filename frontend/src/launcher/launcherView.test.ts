import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { isPrerelease, launcherView } from './launcherView';
import type { ErrorCode, Snapshot, SnapshotCase } from './types';

// Rust's own snapshots, controls included: the same cases `snapshot::controls()` is tested against.
const CASES = JSON.parse(readFileSync(resolve(process.cwd(), '..', 'launcher/contract/snapshot_cases.json'), 'utf8')) as SnapshotCase[];

function snapshot(name: string): Snapshot {
  const found = CASES.find((c) => c.name === name);
  if (!found) throw new Error(`No fixture case "${name}"`);
  return found.snapshot;
}

const view = (name: string) => launcherView(snapshot(name));

describe('fixture', () => {
  it('covers every phase of the contract', () => {
    const phases = new Set(CASES.map((c) => c.snapshot.phase));
    expect(phases.size).toBe(13);
  });

  it('renders every case', () => {
    for (const c of CASES) expect(() => launcherView(c.snapshot), c.name).not.toThrow();
  });
});

describe('controls', () => {
  it('are the snapshot’s own: the view derives none', () => {
    const running = snapshot('6 running');
    const changed = { ...running, controls: { ...running.controls, primary: 'quit' as const, menu: [], update: true } };
    const shown = launcherView(changed);
    expect(shown.controls).toBe(changed.controls);
    expect(shown.primary?.id).toBe('quit');
    expect(shown.menu).toEqual([]);
    expect(shown.hero.update).toBe(true);
  });
});

describe('hero', () => {
  it('draws the ring from the progress fraction while installing', () => {
    const { hero, kicker, kickerAccent } = view('4b installing');
    expect(hero.halo).toMatchObject({ kind: 'progress', fraction: 0.41 });
    expect(hero.halo.valuetext).toContain('41');
    expect(hero.percent).toBe(41);
    expect(kicker).toMatch(/2.*3/);
    expect(kickerAccent).toBe(true);
  });

  it('shows a busy ring when the work has no fraction', () => {
    for (const name of ['8a updating', '1 checking', '5 starting', 'U uninstalling']) {
      const { hero } = view(name);
      expect(hero.halo.kind, name).toBe('busy');
      expect(hero.percent, name).toBeUndefined();
    }
  });

  it('freezes the ring in red where an install stopped, with no stamp', () => {
    const { hero, kickerAccent } = view('4c failed install');
    expect(hero.halo).toMatchObject({ kind: 'error', fraction: 0.41 });
    expect(hero.percent).toBeUndefined();
    expect(kickerAccent).toBe(false);
  });

  it('draws no arc for a failure that had no fraction', () => {
    expect(view('4c failed install (windows)').hero.halo.fraction).toBeUndefined();
  });

  it('closes the ring while running; an update adds the dot', () => {
    expect(view('6 running').hero).toMatchObject({ halo: { kind: 'live' }, update: false });
    expect(view('7 running with update').hero.update).toBe(true);
  });

  it('opens the ring back to its track when it stopped or did not start', () => {
    for (const name of ['9 stopped', '9b failed start', '10a failed update']) {
      const { hero } = view(name);
      expect(hero.halo.kind, name).toBe('rest');
    }
  });
});

describe('kicker', () => {
  it('names the installed version, and nothing when uninstalled', () => {
    expect(view('6 running').kicker).toContain('0.14.0');
    expect(view('9 stopped').kicker).toContain('0.14.0');
    expect(view('1 checking').kicker).not.toContain('0.14.0');
    expect(view('11b uninstalled').kicker).toBe('');
  });

  it('counts the steps of the work in progress', () => {
    expect(view('8a updating').kicker).toMatch(/1.*2/);
    expect(view('8b repairing').kicker).toMatch(/3.*4/);
    expect(view('4c failed install').kicker).toMatch(/2.*3/);
    expect(view('R failed repair').kicker).toContain('0.14.0');
  });
});

describe('headline and sub', () => {
  it('names the update target while updating', () => {
    expect(view('8a updating with updateVersion').headline).toContain('0.14.1');
    expect(view('8a updating').headline).not.toContain('0.14.1');
  });

  it('says where to find the app by platform', () => {
    const sub = (packaging: Snapshot['packaging']) => launcherView({ ...snapshot('6 running'), packaging }).sub?.text;
    expect(new Set([sub('macos_app'), sub('windows_nsis'), sub('appimage')]).size).toBe(3);
    expect(sub('appimage')).toBe(sub('linux_package'));
  });

  it('announces the download size by platform', () => {
    const lead = (packaging: Snapshot['packaging']) => {
      const notice = launcherView({ ...snapshot('4a needs_install'), packaging }).notice;
      return notice?.kind === 'note' ? notice.lead : null;
    };
    expect(new Set([lead('macos_app'), lead('windows_nsis'), lead('appimage')]).size).toBe(3);
    expect(lead('appimage')).toBe(lead('linux_package'));
  });

  it('gives a failed install, repair or update its cause as an alert lead, one per error code', () => {
    const codes: ErrorCode[] = ['download_failed', 'download_unavailable', 'checksum_mismatch', 'tool_broken', 'low_disk', 'uv_failed'];
    for (const name of ['4c failed install', 'R failed repair', '10a failed update']) {
      const leads = codes.map((error) => {
        const sub = launcherView({ ...snapshot(name), error }).sub;
        expect(sub?.leadAlert, `${name} ${error}`).toBe(true);
        return sub?.lead;
      });
      expect(new Set(leads).size, name).toBe(codes.length);
    }
  });

  it('tells a failed update and a failed start apart from a failed install', () => {
    const headlines = ['4c failed install', 'R failed repair', '10a failed update', '9b failed start'].map((name) => view(name).headline);
    expect(new Set(headlines).size).toBe(4);
  });

  it('offers Open and keeps Repair in the menu after a failed repair, where a failed install freezes the ring in red', () => {
    const repair = view('R failed repair');
    expect(repair.hero.halo.kind).toBe('rest');
    expect(repair.primary?.id).toBe('open');
    expect(repair.menu.map((item) => item.id)).toContain('repair');
    expect(view('4c failed install').hero.halo.kind).toBe('error');
  });

  it('names the folder an incomplete uninstall left, and only then', () => {
    const incomplete = snapshot('11b uninstalled incomplete');
    expect(incomplete.leftoverFolder).toBeTruthy();
    expect(view('11b uninstalled incomplete').sub?.folder).toBe(incomplete.leftoverFolder);
    expect(view('11b uninstalled').sub?.folder).toBeUndefined();
  });
});

describe('notice', () => {
  it('explains an unsupported machine with one reason', () => {
    expect(view('2a unsupported driver_too_old').notice).toMatchObject({
      kind: 'reason',
      found: '575.51.03',
      link: { id: 'requirements' }
    });
    expect(view('2a unsupported low_disk (macos)').notice).toMatchObject({ kind: 'reason', found: '7.3 GB' });
    expect(view('2b unsupported no_nvidia_gpu').notice).toMatchObject({ found: null, link: null });
    expect(view('2a unsupported driver_not_working').notice).toMatchObject({
      kind: 'reason',
      name: 'NVIDIA driver',
      found: null,
      link: null
    });
    expect(view('2b unsupported gpu_too_old').notice).toMatchObject({ found: 'GTX 1080', link: null });
  });

  it('lets a driver that does not work be checked again, and says a missing GPU is permanent', () => {
    expect(view('2a unsupported driver_not_working').primary?.id).toBe('check_again');
    expect(view('2b unsupported no_nvidia_gpu').headline).not.toBe(view('2a unsupported driver_not_working').headline);
  });

  it('offers the update as an alert', () => {
    expect(view('7 running with update').notice).toMatchObject({ kind: 'update', version: '0.14.1', link: { id: 'whats_new' } });
    expect(view('6 running').notice).toBeNull();
  });

  it('offers the log when the app stopped or an action failed', () => {
    for (const c of CASES) {
      const notice = launcherView(c.snapshot).notice;
      expect(notice?.kind === 'log', c.name).toBe(c.snapshot.phase === 'stopped' || c.snapshot.phase === 'failed');
    }
  });

  it('shows what stays once uninstalled', () => {
    expect(view('11b uninstalled').notice).toMatchObject({ kind: 'kept', link: { id: 'data_folder' } });
  });

  it('notes the move to Applications, and Quit is the one button', () => {
    expect(view('3 move_to_applications').notice?.kind).toBe('note');
    expect(view('3 move_to_applications').primary?.id).toBe('quit');
  });
});

describe('action zone', () => {
  it('has a primary exactly when the controls do', () => {
    for (const c of CASES) expect(launcherView(c.snapshot).primary?.id ?? null, c.name).toBe(c.snapshot.controls.primary);
  });

  it('shows the step in progress as a caption while working, and not otherwise', () => {
    expect(view('4b installing').caption?.line).toBe('Downloading torch (2.1GiB)');
    expect(view('4b installing').caption?.label).toBeTruthy();
    expect(view('8a updating with updateVersion').caption).toMatchObject({ line: null });
    expect(view('8a updating with updateVersion').caption?.label).toContain('0.14.1');
    expect(view('4c failed install').caption).toBeNull();
    expect(view('6 running').caption).toBeNull();
  });

  it('lists More entries in the snapshot’s order', () => {
    for (const c of CASES) expect(launcherView(c.snapshot).menu.map((m) => m.id), c.name).toEqual(c.snapshot.controls.menu);
  });
});

describe('beta wording', () => {
  it('marks a pre-release install in the kicker only', () => {
    expect(view('6 running a beta version').kicker).toContain('0.15.0b2');
    expect(view('6 running a beta version').kicker).toMatch(/beta/i);
    expect(view('6 running').kicker).not.toMatch(/beta/i);
  });

  it('calls a pre-release update a beta, and a stable one a version', () => {
    const lead = (name: string) => {
      const notice = view(name).notice;
      return notice?.kind === 'update' ? notice.lead : '';
    };
    expect(lead('7 running with update')).toContain('0.14.1');
    expect(lead('7 running with update')).not.toMatch(/beta/i);
    expect(lead('7 running with a beta update')).toContain('0.15.0b3');
    expect(lead('7 running with a beta update')).toMatch(/beta/i);
  });

  it('knows pre-release version spellings', () => {
    for (const v of ['0.15.0b2', '1.0.0rc1', '1.0.0a1']) expect(isPrerelease(v), v).toBe(true);
    for (const v of ['0.14.1', '1.0.0.post1', '1.0.0.dev3', '1.0.0b2.dev1', '1.0', 'v1.0.0', '1.0.0-beta.2', 'garbage', '']) expect(isPrerelease(v), v).toBe(false);
  });
});
