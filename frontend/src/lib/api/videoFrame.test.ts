import { describe, expect, it, vi } from 'vitest';

import { grabFrame } from './videoFrame';

const mocks = vi.hoisted(() => ({ post: vi.fn() }));
vi.mock('./client', () => ({ api: { post: mocks.post } }));

describe('grabFrame', () => {
  it('posts the clip and the time to the frame route and returns what it saved', async () => {
    const saved = { path: '/up/clip_3.0s.png' };
    mocks.post.mockResolvedValue(saved);

    await expect(grabFrame('/outputs/clip.mp4', 3.02)).resolves.toBe(saved);

    expect(mocks.post).toHaveBeenCalledWith('/api/video/frame', { path: '/outputs/clip.mp4', time: 3.02 });
  });
});
