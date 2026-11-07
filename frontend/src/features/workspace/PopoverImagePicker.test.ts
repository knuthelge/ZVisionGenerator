import { beforeEach, describe, expect, it, vi } from 'vitest';

import { mountComponent } from '../../test-utils/mount';
import PopoverImagePicker from './PopoverImagePicker.svelte';

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

describe('PopoverImagePicker', () => {
  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => 'blob:upload');
    URL.revokeObjectURL = vi.fn();
  });

  function show(props: Record<string, unknown> = {}): HTMLElement {
    return mountComponent(PopoverImagePicker, {
      title: 'Start image', name: '', path: null, file: null, emptyText: 'No start image yet', clearLabel: 'Clear start image',
      onfile: vi.fn(), onpath: vi.fn(), onclear: vi.fn(), ...props,
    }).target;
  }

  const thumb = (root: HTMLElement) => root.querySelector('[data-testid="slot-thumb"]') as HTMLElement;

  it('shows the picture of a path outside the output folder through the server', () => {
    const root = show({ name: 'fox.png', path: '/mnt/photos/fox.png' });

    expect(thumb(root).querySelector('img')?.getAttribute('src')).toBe('/preview/reference_image/mnt/photos/fox.png');
    expect(thumb(root).querySelector('[role="img"]')?.getAttribute('aria-label')).toBe('Start image preview');
  });

  it('shows the alert icon, not the picture, for a missing file', () => {
    const root = show({ name: 'gone.png', path: '/mnt/photos/gone.png', missing: true });

    expect(thumb(root).getAttribute('data-missing')).toBe('true');
    expect(thumb(root).querySelector('img')).toBeNull();
    expect(thumb(root).querySelector('svg')).not.toBeNull();
  });
});
