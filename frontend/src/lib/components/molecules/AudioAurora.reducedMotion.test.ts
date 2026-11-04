import { afterEach, describe, expect, it, vi } from 'vitest';

import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import AudioAurora from './AudioAurora.svelte';

const painted = vi.hoisted(() => ({ times: [] as number[] }));

vi.mock('./audioAuroraDraw', () => ({
  createAuroraPainter: () => ({
    paint: (_ctx: unknown, _width: number, _height: number, input: { time: number }) => { painted.times.push(input.time); },
    reset: () => undefined,
  }),
}));

afterEach(() => {
  vi.unstubAllGlobals();
  painted.times.length = 0;
});

describe('AudioAurora with reduced motion', () => {
  it('redraws on progress without moving the sky\'s drift', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: query.includes('reduce'), addEventListener() {}, removeEventListener() {} }));
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as CanvasRenderingContext2D);
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(600);
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(150);
    const props = reactiveProps({ kind: 'music', progress: { phase: 'Song tokens', step: 10, total: null, part: 1, parts: 1 } });
    mountComponent(AudioAurora, props);
    await settle();

    props.progress = { phase: 'Song tokens', step: 400, total: null, part: 1, parts: 1 };
    await settle();
    props.progress = { phase: 'Song tokens', step: 900, total: null, part: 1, parts: 1 };
    await settle();

    expect(painted.times.length).toBeGreaterThan(1);
    expect(new Set(painted.times).size).toBe(1);
  });
});
