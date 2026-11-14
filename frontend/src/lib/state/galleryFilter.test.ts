import { describe, expect, it } from 'vitest';

import { filterAdmits, viewAdmits } from './galleryFilter';

describe('filterAdmits', () => {
  const image = { media_type: 'image' as const, project: 'A' };
  const topVideo = { media_type: 'video' as const, project: null };

  it('admits everything with all media', () => {
    expect([filterAdmits('all', image), filterAdmits('all', topVideo)]).toEqual([true, true]);
  });

  it('admits one kind of media', () => {
    expect([filterAdmits('image', image), filterAdmits('image', topVideo)]).toEqual([true, false]);
  });

  it('admits only the files in no project with Not in a project, whatever the media', () => {
    expect([filterAdmits('no_project', image), filterAdmits('no_project', topVideo)]).toEqual([false, true]);
  });
});

describe('viewAdmits', () => {
  const starredImage = { media_type: 'image' as const, project: null, starred: true };
  const plainImage = { media_type: 'image' as const, project: null };
  const starredVideo = { media_type: 'video' as const, project: 'A', starred: true };

  it.each([
    ['image', false, starredImage, true],
    ['image', false, plainImage, true],
    ['image', false, starredVideo, false],
    ['all', true, starredImage, true],
    ['all', true, plainImage, false],
    ['all', true, { ...plainImage, starred: false }, false],
    ['image', true, starredVideo, false],
    ['video', true, starredVideo, true],
    ['no_project', true, starredImage, true],
    ['no_project', true, starredVideo, false],
    ['no_project', true, plainImage, false],
  ] as const)('with the filter %s and the star toggle %s admits %j: %s', (filter, starred, asset, admitted) => {
    expect(viewAdmits(filter, starred, asset)).toBe(admitted);
  });
});
