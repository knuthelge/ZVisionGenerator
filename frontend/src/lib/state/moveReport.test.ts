import { describe, expect, it } from 'vitest';

import type { GalleryAsset, MoveResult } from '$lib/types';

import { moveOutcome } from './moveReport';

const moved = (id: string) => ({ from: `out/${id}`, from_path: `/o/out/${id}`, asset: { id } as GalleryAsset });
const failure = (id: string, detail: string) => ({ id, detail });

describe('moveOutcome', () => {
  it('is a success naming the destination when every file moved', () => {
    const result: MoveResult = { moved: [moved('a.png')], failed: [] };
    expect(moveOutcome(result, 'Night Bus')).toMatchObject({ tone: 'success', message: expect.stringContaining('Night Bus') });
    expect(moveOutcome({ moved: [moved('a.png'), moved('b.png')], failed: [] }, null)).toMatchObject({ tone: 'success', message: expect.stringContaining('No project') });
  });

  it('is an error that gives the first reason when nothing moved and a file failed', () => {
    const result: MoveResult = { moved: [], failed: [failure('a.png', 'Could not move the file: Permission denied'), failure('b.png', 'The file is missing.')] };
    const { tone, message } = moveOutcome(result, 'Night Bus');
    expect(tone).toBe('error');
    expect(message).toContain('Could not move the file: Permission denied');
    expect(message).not.toContain('missing');
  });

  it('is a warning with the count of failures and the first reason when some moved', () => {
    const result: MoveResult = { moved: [moved('a.png'), moved('b.png')], failed: [failure('c.png', 'The file is missing.')] };
    const { tone, message } = moveOutcome(result, 'Night Bus');
    expect(tone).toBe('warning');
    expect(message).toContain('2');
    expect(message).toContain('1');
    expect(message).toContain('The file is missing');
  });

  it('is not an error when nothing moved and nothing failed: the files were already there', () => {
    const { tone } = moveOutcome({ moved: [], failed: [] }, 'Night Bus');
    expect(tone).not.toBe('error');
    expect(tone).not.toBe('warning');
  });
});
