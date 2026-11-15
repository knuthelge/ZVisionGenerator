import { beforeEach, describe, expect, it, vi } from 'vitest';

type Versions = typeof import('./fileVersions.svelte').fileVersions;

let versions: Versions;

beforeEach(async () => {
  vi.resetModules();
  versions = (await import('./fileVersions.svelte')).fileVersions;
});

describe('fileVersions', () => {
  it('reads 0 for a file that was never rewritten', () => {
    expect(versions.of('/out/a.png')).toBe(0);
  });

  it('counts the rewrites of each path on its own', () => {
    versions.bump('/out/a.png');
    versions.bump('/out/a.png');
    versions.bump('/out/b.png');
    expect(versions.of('/out/a.png')).toBe(2);
    expect(versions.of('/out/b.png')).toBe(1);
    expect(versions.of('/out/c.png')).toBe(0);
  });
});
