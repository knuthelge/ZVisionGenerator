import { afterEach, describe, expect, it, vi } from 'vitest';

import { NEUTRAL_LIGHT, previewTables } from '$lib/state/adjustMath';
import { fakeGl } from '../../../../test-utils/fakeGl';
import { createRenderer, packTable, previewSize, PREVIEW_MAX_PIXELS, TABLE_WIDTH, TEXTURE_STEP, textureSize } from './adjustGl';

afterEach(() => vi.restoreAllMocks());

describe('packTable', () => {
  it('lays a table out in rows of 1024 so entry i is at column i % 1024 of row floor(i / 1024)', () => {
    const table = Float32Array.from({ length: 4097 }, (_, i) => i / 4096);
    const packed = packTable(table);
    expect(packed.width).toBe(TABLE_WIDTH);
    expect(packed.height).toBe(5);
    expect(packed.data).toHaveLength(5 * 1024);
    for (const i of [0, 1, 1023, 1024, 2047, 4095, 4096]) {
      expect(packed.data[Math.floor(i / 1024) * 1024 + (i % 1024)]).toBe(table[i]);
    }
    expect(packed.data[4097]).toBe(0);
  });

  it('gives the 256-entry decode table one row and the 1025-entry shift table two', () => {
    expect(packTable(new Float32Array(256)).height).toBe(1);
    expect(packTable(new Float32Array(1025)).height).toBe(2);
  });
});

describe('previewSize', () => {
  it('keeps a picture within 4 MP and 4096 px as it is', () => {
    expect(previewSize(2000, 1500)).toEqual({ width: 2000, height: 1500 });
    expect(previewSize(4096, 976)).toEqual({ width: 4096, height: 976 });
  });

  it('scales a bigger picture down to at most 4 MP, keeping its shape', () => {
    const size = previewSize(5000, 4000);
    expect(size.width * size.height).toBeLessThanOrEqual(PREVIEW_MAX_PIXELS);
    expect(size.width / size.height).toBeCloseTo(1.25, 2);
  });

  it('scales a picture with a side over 4096 px down to 4096', () => {
    expect(previewSize(8192, 256)).toEqual({ width: 4096, height: 128 });
  });
});

describe('textureSize', () => {
  const natural = { width: 1152, height: 1728 };

  it('matches the canvas: one texture pixel per canvas pixel, rounded up to a step', () => {
    const size = textureSize(natural, natural, 493);
    expect(size.width).toBe(Math.ceil(493 / TEXTURE_STEP) * TEXTURE_STEP);
    expect(size.height).toBe(Math.round((1728 * size.width) / 1152));
  });

  it('never exceeds the picture, however large the canvas', () => {
    expect(textureSize(natural, natural, 4000)).toEqual(natural);
  });

  it('follows the region on show: a small crop needs more texture than the whole picture', () => {
    const whole = textureSize(natural, natural, 600);
    const crop = textureSize(natural, { width: 300, height: 450 }, 600);
    expect(crop.width).toBeGreaterThan(whole.width);
    expect(crop.width).toBeLessThanOrEqual(natural.width);
  });

  it('keeps a huge picture within the preview limits', () => {
    const size = textureSize({ width: 6000, height: 4000 }, { width: 6000, height: 4000 }, 6000);
    expect(size.width * size.height).toBeLessThanOrEqual(PREVIEW_MAX_PIXELS);
  });

  it('keeps at least a pixel for a degenerate region or canvas', () => {
    expect(textureSize(natural, { width: 0, height: 0 }, 0).width).toBeGreaterThanOrEqual(1);
    expect(textureSize({ width: 1, height: 1 }, { width: 1, height: 1 }, 1)).toEqual({ width: 1, height: 1 });
  });
});

describe('createRenderer', () => {
  it('is null without WebGL2', () => {
    const canvas = document.createElement('canvas');
    vi.spyOn(canvas, 'getContext').mockReturnValue(null);
    expect(createRenderer(canvas)).toBeNull();
  });

  it('keeps the drawing buffer, so WebKit presents the first draw after a resize', () => {
    const canvas = document.createElement('canvas');
    const { gl } = fakeGl();
    const getContext = vi.spyOn(canvas, 'getContext').mockReturnValue(gl);
    createRenderer(canvas);
    expect(getContext).toHaveBeenCalledWith('webgl2', expect.objectContaining({ preserveDrawingBuffer: true }));
  });

  it('is null when the browser refuses the context', () => {
    const canvas = document.createElement('canvas');
    vi.spyOn(canvas, 'getContext').mockImplementation(() => {
      throw new Error('no');
    });
    expect(createRenderer(canvas)).toBeNull();
  });

  it('is null when the shaders do not build', () => {
    const canvas = document.createElement('canvas');
    const { gl } = fakeGl();
    vi.spyOn(canvas, 'getContext').mockReturnValue(new Proxy(gl, { get: (target, name) => (name === 'getShaderParameter' ? () => false : Reflect.get(target, name)) }));
    expect(createRenderer(canvas)).toBeNull();
  });

  it('uploads the picture premultiplied, the tables once per change, and draws a four-vertex strip', () => {
    const canvas = document.createElement('canvas');
    const { gl, calls } = fakeGl();
    vi.spyOn(canvas, 'getContext').mockReturnValue(gl);
    const renderer = createRenderer(canvas)!;
    expect(renderer).not.toBeNull();
    const source = document.createElement('canvas');
    renderer.setImage(source);
    const premultiply = calls('pixelStorei').find((call) => call[1] === true);
    expect(premultiply?.[0]).toBe('UNPACK_PREMULTIPLY_ALPHA_WEBGL'.length);
    // Plain bilinear sampling of a picture at its shown size: no mip chain is built or sampled.
    expect(calls('generateMipmap')).toHaveLength(0);
    const filters = calls('texParameteri').filter((call) => call[1] === 'TEXTURE_MIN_FILTER'.length).map((call) => call[2]);
    expect(filters.at(-1)).toBe('LINEAR'.length);
    expect(filters).not.toContain('LINEAR_MIPMAP_LINEAR'.length);

    const tables = previewTables(NEUTRAL_LIGHT);
    const corners = [[0, 0], [1, 0], [1, 1], [0, 1]] as const;
    renderer.draw(tables, corners, 320, 240);
    expect(canvas.width).toBe(320);
    expect(canvas.height).toBe(240);
    expect(calls('drawArrays')).toHaveLength(1);
    expect(calls('texImage2D').filter((call) => call.length === 9)).toHaveLength(3);
    const strip = calls('bufferData')[0][1] as Float32Array;
    expect([...strip]).toEqual([-1, 1, 0, 0, 1, 1, 1, 0, -1, -1, 0, 1, 1, -1, 1, 1]);

    renderer.draw(tables, corners, 320, 240);
    expect(calls('texImage2D').filter((call) => call.length === 9)).toHaveLength(3);
    renderer.draw(previewTables({ ...NEUTRAL_LIGHT, exposure: 20 }), corners, 320, 240);
    expect(calls('texImage2D').filter((call) => call.length === 9)).toHaveLength(6);
    expect(calls('drawArrays')).toHaveLength(3);
  });

  it('sends the saturation and the vibrance strength as uniforms on every draw', () => {
    const canvas = document.createElement('canvas');
    const { gl, calls } = fakeGl();
    vi.spyOn(canvas, 'getContext').mockReturnValue(gl);
    const renderer = createRenderer(canvas)!;
    renderer.setImage(document.createElement('canvas'));
    const corners = [[0, 0], [1, 0], [1, 1], [0, 1]] as const;
    renderer.draw(previewTables({ ...NEUTRAL_LIGHT, saturation: 20, vibrance: 50 }), corners, 320, 240);
    expect(calls('getUniformLocation').map((call) => call[1])).toContain('uVib');
    expect(calls('uniform1f').map((call) => call[1])).toEqual([1.2, 1]);
  });

  it('sends the eight colour-mixer amounts and whether any is set on every draw', () => {
    const canvas = document.createElement('canvas');
    const { gl, calls } = fakeGl();
    vi.spyOn(canvas, 'getContext').mockReturnValue(gl);
    const renderer = createRenderer(canvas)!;
    renderer.setImage(document.createElement('canvas'));
    const corners = [[0, 0], [1, 0], [1, 1], [0, 1]] as const;
    renderer.draw(previewTables(NEUTRAL_LIGHT), corners, 320, 240);
    renderer.draw(previewTables({ ...NEUTRAL_LIGHT, mix_aqua: 50 }), corners, 320, 240);
    expect(calls('getUniformLocation').map((call) => call[1])).toEqual(expect.arrayContaining(['uMix', 'uMixOn']));
    expect(calls('uniform1fv').map((call) => [...(call[1] as number[])])).toEqual([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0.5, 0, 0, 0]]);
    expect(calls('uniform1i').map((call) => call[1]).filter((v) => v === 0 || v === 1)).toEqual(expect.arrayContaining([0, 1]));
  });

  it('releases the GL context when disposed', () => {
    const canvas = document.createElement('canvas');
    const { gl, calls } = fakeGl();
    vi.spyOn(canvas, 'getContext').mockReturnValue(gl);
    createRenderer(canvas)!.dispose();
    expect(calls('loseContext')).toHaveLength(1);
  });

  it('does not lose a context that is already lost', () => {
    const canvas = document.createElement('canvas');
    const { gl, calls } = fakeGl();
    vi.spyOn(canvas, 'getContext').mockReturnValue(new Proxy(gl, { get: (target, name) => (name === 'isContextLost' ? () => true : Reflect.get(target, name)) }));
    createRenderer(canvas)!.dispose();
    expect(calls('loseContext')).toHaveLength(0);
  });

  it('draws nothing before a picture is set', () => {
    const canvas = document.createElement('canvas');
    const { gl, calls } = fakeGl();
    vi.spyOn(canvas, 'getContext').mockReturnValue(gl);
    const renderer = createRenderer(canvas)!;
    renderer.draw(previewTables(NEUTRAL_LIGHT), [[0, 0], [1, 0], [1, 1], [0, 1]], 100, 100);
    expect(calls('drawArrays')).toHaveLength(0);
  });
});
