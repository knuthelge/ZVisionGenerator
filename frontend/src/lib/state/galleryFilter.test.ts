import { describe, expect, it } from 'vitest';

import { filterAdmits } from './galleryFilter';

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
