import { beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { makeContext } from '../../test-utils/workspace';
import { mountComponent, settle } from '../../test-utils/mount';
import SourceVideoSection from './SourceVideoSection.svelte';

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);
vi.mock('$lib/api/extend', () => ({ inspectExtendSource: vi.fn(async () => { throw new Error('not inspected here'); }) }));

describe('SourceVideoSection', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  function show(): HTMLElement {
    mountComponent(SourceVideoSection, { context: makeContext(), busy: false, guard: { kind: 'none', blocked: true, note: null } });
    return document.body.querySelector('[data-testid="slot-thumb"]') as HTMLElement;
  }

  it('shows the server\'s last frame of a clip outside the output folder, with no hidden video element', async () => {
    draft.setExtendSourcePath('/mnt/clips/walk.mp4');
    const thumb = show();
    await settle();

    expect(thumb.querySelector('img')?.getAttribute('src')).toBe('/preview/source_video/mnt/clips/walk.mp4?frame=last');
    expect(thumb.querySelector('video')).toBeNull();
    expect(thumb.querySelector('[data-testid="play-badge"]')).toBeNull();
  });

  it('keeps the film icon while no clip is chosen', async () => {
    const thumb = show();
    await settle();

    expect(thumb.querySelector('img')).toBeNull();
    expect(thumb.querySelector('svg')).not.toBeNull();
  });
});
