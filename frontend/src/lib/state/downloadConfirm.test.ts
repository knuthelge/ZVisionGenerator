import { describe, expect, it } from 'vitest';

import type { DownloadPlan } from '$lib/types';
import { downloadConfirmText } from './downloadConfirm';

function plan(overrides: Partial<DownloadPlan> = {}): DownloadPlan {
  return {
    kind: 'model',
    name: 'klein9b',
    label: 'klein9b',
    parts: [{ repo: 'org/a', files: 3, bytes: 30e9 }, { repo: 'org/b', files: 1, bytes: 4.7e9 }],
    total_bytes: 34.7e9,
    space: { path: '/hub', needed_bytes: 34.7e9, free_bytes: 512e9, enough: true },
    ...overrides,
  };
}

describe('downloadConfirmText', () => {
  it('confirms with Download when the space suffices and names size, parts and free space', () => {
    const text = downloadConfirmText(plan());
    expect(text.confirmLabel).toBe('Download');
    for (const part of ['34.7 GB', '512.0 GB', '/hub']) expect(text.info).toContain(part);
  });

  it('leaves out the part count for one part', () => {
    const single = downloadConfirmText(plan({ parts: [{ repo: 'org/a', files: 1, bytes: 34.7e9 }] })).info;
    expect(single).not.toContain('parts');
    expect(downloadConfirmText(plan()).info).toContain('2 parts');
  });

  it('offers Download anyway only when the space is short', () => {
    const text = downloadConfirmText(plan({ space: { path: '/hub', needed_bytes: 34.7e9, free_bytes: 21e9, enough: false } }));
    expect(text.confirmLabel).toBe('Download anyway');
    for (const part of ['34.7 GB', '21.0 GB', '/hub']) expect(text.info).toContain(part);
  });

  it('says the size is unknown and skips the space line when it cannot be checked', () => {
    const text = downloadConfirmText(plan({ total_bytes: null, space: null }));
    expect(text.confirmLabel).toBe('Download');
    expect(text.info).not.toContain('/hub');
    expect(text.info).not.toMatch(/\d GB/);
  });

  it('keeps the free space of an unknown-size download', () => {
    const info = downloadConfirmText(plan({ total_bytes: null })).info;
    expect(info).toContain('512.0 GB');
    expect(info).toContain('/hub');
    expect(info).not.toContain('34.7');
  });
});
