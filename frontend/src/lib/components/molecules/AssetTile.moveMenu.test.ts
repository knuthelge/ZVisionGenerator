import { flushSync } from 'svelte';
import { describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';

import AssetTile from './AssetTile.svelte';
import { mountComponent, settle } from '../../../test-utils/mount';

// The Move list is the one piece a tile can leave out until it is wanted: a grid holds hundreds of tiles.
const built = vi.hoisted(() => vi.fn());
vi.mock('./MoveMenu.svelte', () => ({ default: (...args: unknown[]) => { built(...args); } }));

const asset: GalleryAsset = {
  id: 'a.png',
  url: '/media/a.png',
  thumbnail_url: '/thumbs/a.png',
  filename: 'a.png',
  created_at: '2026-10-03T12:00:00Z',
  project: null,
  workflow: 'txt2img',
  prompt: 'a fox',
  model: 'zit',
  media_type: 'image',
  file_path: '/outputs/a.png',
  reuse_workspace_url: '#/workspace?workflow=txt2img',
};

describe('AssetTile Move list', () => {
  it('is built only once Move is chosen from the menu', async () => {
    built.mockClear();
    const { target } = mountComponent(AssetTile, { asset, onmove: vi.fn() });
    (target.querySelector('button[aria-label="More actions for a.png"]') as HTMLButtonElement).click();
    flushSync();
    await settle();
    expect(built).not.toHaveBeenCalled();

    document.querySelector<HTMLElement>('[data-action="move"]')!.click();
    await settle();
    expect(built).toHaveBeenCalledOnce();
  });
});
