import { beforeEach, describe, expect, it } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { memoryUploadStore } from '$lib/state/uploadStore';
import type { ReferenceList } from '$lib/state/references';
import { referenceEditor, referenceWording } from './referenceEditor';

const file = (name: string): File => new File(['x'], name, { type: 'image/png' });

describe('referenceWording', () => {
  it('names the Video rows by the tag the video model reads', () => {
    const wording = referenceWording('video');

    expect([wording.label(1), wording.label(2)]).toEqual(['<Picture 1>', '<Picture 2>']);
  });
});

describe('referenceEditor', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
  });

  // `other` is every path the list does not own.
  const lists: Record<ReferenceList, { paths: () => string[]; held: () => readonly (File | null)[]; other: () => string[] }> = {
    image: { paths: () => draft.state.referenceImagePaths, held: () => heldUploads.references, other: () => [...draft.state.videoReferencePaths, ...draft.state.extendReferencePaths] },
    video: { paths: () => draft.state.videoReferencePaths, held: () => heldUploads.videoReferences, other: () => [...draft.state.referenceImagePaths, ...draft.state.extendReferencePaths] },
    extend: { paths: () => draft.state.extendReferencePaths, held: () => heldUploads.extendReferences, other: () => [...draft.state.referenceImagePaths, ...draft.state.videoReferencePaths] },
  };

  it.each(['image', 'video', 'extend'] as const)('edits the %s list and its held uploads together, and leaves the other list', (list) => {
    const { paths, held, other } = lists[list];
    const editor = referenceEditor(() => 2, () => list);

    editor.append([{ path: '/in/a.png', file: null }]);
    editor.appendFiles([file('b.png')]);

    expect(paths()).toEqual(['/in/a.png', '']);
    expect(held().map((entry) => entry?.name ?? null)).toEqual([null, 'b.png']);
    expect(other()).toEqual([]);

    editor.replaceWithPath(1, '/in/c.png');
    expect([paths(), held()]).toEqual([['/in/a.png', '/in/c.png'], [null, null]]);

    editor.replaceWithFile(0, file('d.png'));
    expect(paths()).toEqual(['', '/in/c.png']);
    expect(held()[0]?.name).toBe('d.png');

    editor.remove(0);
    expect(paths()).toEqual(['/in/c.png']);
    expect(other()).toEqual([]);
  });

  it.each(['image', 'video', 'extend'] as const)('takes as many as the cap leaves room for on the %s list', (list) => {
    const { paths } = lists[list];
    const editor = referenceEditor(() => 2, () => list);

    editor.append([{ path: '/in/a.png', file: null }, { path: '/in/b.png', file: null }, { path: '/in/c.png', file: null }]);

    expect(paths()).toEqual(['/in/a.png', '/in/b.png']);
  });

  it('stops flagging the path that was replaced or removed', () => {
    draft.patch({ videoReferencePaths: ['/in/a.png', '/in/b.png'], missingReferencePaths: ['/in/a.png', '/in/b.png'] });
    const editor = referenceEditor(() => 2, () => 'video');

    editor.replaceWithPath(0, '/in/c.png');
    editor.remove(1);

    expect(draft.state.missingReferencePaths).toEqual([]);
  });

  it('follows the list it is asked for each time, so one editor serves whichever tab is open', () => {
    let list: ReferenceList = 'image';
    const editor = referenceEditor(() => 2, () => list);

    editor.append([{ path: '/in/a.png', file: null }]);
    list = 'video';
    editor.append([{ path: '/in/b.png', file: null }]);
    list = 'extend';
    editor.append([{ path: '/in/c.png', file: null }]);

    expect([draft.state.referenceImagePaths, draft.state.videoReferencePaths, draft.state.extendReferencePaths]).toEqual([['/in/a.png'], ['/in/b.png'], ['/in/c.png']]);
  });
});
