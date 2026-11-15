import type { Point } from '$lib/state/adjustGeometry';
import {
  KB,
  KG,
  KR,
  MIX_BOOST_MUTED,
  MIX_BOOST_VIVID,
  MIX_CENTRES,
  MIX_FADE_END,
  MIX_FADE_START,
  MIX_FLAT,
  MIX_GAMUT_SOFTNESS,
  MIX_MIN_CHROMA,
  MIX_MUTE,
  MIX_VIVID_CHROMA,
  SHIFT_INTERVALS,
  TONE_INTERVALS,
  type PreviewTables,
} from '$lib/state/adjustMath';

/** A number as a GLSL float literal: a whole number gets a decimal point. */
const glsl = (n: number): string => (Number.isInteger(n) ? n.toFixed(1) : String(n));

/** The width of every table texture: an entry `i` sits at column `i % 1024` of row `floor(i / 1024)`. */
export const TABLE_WIDTH = 1024;

/** The picture's corners in source UVs: top-left, top-right, bottom-right, bottom-left (`quadCorners`). */
export type Quad = readonly [Point, Point, Point, Point];

/** A table laid out for upload as a `TABLE_WIDTH`-wide float texture, padded with zeros to whole rows. */
export interface PackedTable {
  data: Float32Array;
  width: number;
  height: number;
}

/** Return the table padded to whole rows of `TABLE_WIDTH` entries. */
export function packTable(table: Float32Array): PackedTable {
  const height = Math.max(1, Math.ceil(table.length / TABLE_WIDTH));
  const data = new Float32Array(TABLE_WIDTH * height);
  data.set(table);
  return { data, width: TABLE_WIDTH, height };
}

/** The most pixels, and the longest side, of the picture the preview uploads. */
export const PREVIEW_MAX_PIXELS = 4_000_000;
export const PREVIEW_MAX_SIDE = 4096;

/** Return the size the preview texture gets: the picture itself, or scaled down to at most 4 MP and 4096 px a side. */
export function previewSize(width: number, height: number): { width: number; height: number } {
  const factor = Math.min(1, Math.sqrt(PREVIEW_MAX_PIXELS / Math.max(1, width * height)), PREVIEW_MAX_SIDE / Math.max(1, width, height));
  if (factor >= 1) return { width, height };
  return { width: Math.max(1, Math.floor(width * factor)), height: Math.max(1, Math.floor(height * factor)) };
}

/** The picture's texture width is rounded up to a multiple of this, so a window resize does not upload a new texture on every frame. */
export const TEXTURE_STEP = 64;

/**
 * Return the size of the texture to upload: the picture scaled so the region on show maps to about one texture pixel
 * per canvas pixel, never larger than the picture, within the `previewSize` limits. The shader then samples it with
 * plain bilinear filtering and needs no mip chain.
 *
 * @param natural The picture's size in pixels.
 * @param shown The size of the region on show, in source pixels.
 * @param canvasWidth The canvas's width in device pixels.
 */
export function textureSize(natural: { width: number; height: number }, shown: { width: number; height: number }, canvasWidth: number): { width: number; height: number } {
  const perSource = shown.width > 0 && canvasWidth > 0 ? canvasWidth / shown.width : 1;
  const wanted = Math.min(natural.width, Math.ceil((natural.width * perSource) / TEXTURE_STEP) * TEXTURE_STEP);
  const width = Math.max(1, wanted);
  return previewSize(width, Math.max(1, Math.round((natural.height * width) / Math.max(1, natural.width))));
}

/** What the renderer draws with: the live GL context of one canvas. */
export interface AdjustRenderer {
  /** Upload the picture (premultiplied by alpha, bilinear, no mip chain) and keep it for every draw. */
  setImage(source: TexImageSource): void;
  /** Draw the picture's quad with the tables, filling a canvas of `width × height` device pixels. */
  draw(tables: PreviewTables, corners: Quad, width: number, height: number): void;
  /** Free the program, textures and buffer, and release the GL context; the canvas is not drawn on again. */
  dispose(): void;
}

const VERTEX = `#version 300 es
in vec2 aPos;
in vec2 aUv;
out vec2 vUv;
void main() {
  vUv = aUv;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

// The shader does table lookups, multiplies, dot products, mix and clamp, in the order `previewPixel` uses: decode and gains,
// saturation, vibrance and the colour mixer as one mix around the luminance, the encode and contrast table, then the tone shift.
// The colour mixer's hue and chroma are computed per pixel (Oklab, 8 bands, flat on each centre with a raised cosine between), and only when a band is set.
const FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uImage;
uniform sampler2D uDecode;
uniform sampler2D uTone;
uniform sampler2D uShift;
uniform vec3 uGains;
uniform float uSat;
uniform float uVib;
uniform float uMix[8];
uniform bool uMixOn;
out vec4 outColor;
const vec3 K = vec3(${KR}, ${KG}, ${KB});
const mat3 OKLAB_LMS = mat3(0.4122214708, 0.2119034982, 0.0883024619, 0.5363325363, 0.6806995451, 0.2817188376, 0.0514459929, 0.1073969566, 0.6299787005);
const float CENTRE[9] = float[9](${MIX_CENTRES.join('.0, ')}.0, ${MIX_CENTRES[0] + 360}.0);
const int TONE_N = ${TONE_INTERVALS};
const int SHIFT_N = ${SHIFT_INTERVALS};

float entry(sampler2D table, int i) {
  return texelFetch(table, ivec2(i & ${TABLE_WIDTH - 1}, i >> 10), 0).r;
}

// A manual lerp between neighbouring entries; the coordinate is clamped like a clamped sampler's.
float lerpTable(sampler2D table, float x, int n) {
  float f = clamp(x, 0.0, 1.0) * float(n);
  int i = min(int(floor(f)), n - 1);
  float a = entry(table, i);
  return a + (entry(table, i + 1) - a) * (f - float(i));
}

// The largest saturation factor at which no channel of linear RGB \`l\` with luminance \`y\` clips; a huge number for a grey.
float gamutFactor(vec3 l, float y) {
  float limit = 1e30;
  for (int k = 0; k < 3; k++) {
    float d = l[k] - y;
    if (d > 1e-9) limit = min(limit, (1.0 - y) / d);
    else if (d < -1e-9) limit = min(limit, y / -d);
  }
  return limit;
}

// The saturation factor once the colour mixer has changed \`base\`, for linear RGB \`l\` of luminance \`y\`: see \`mixFactor\` in adjustMath.ts.
float mixFactor(vec3 l, float y, float base) {
  vec3 q = pow(max(OKLAB_LMS * l, 0.0), vec3(1.0 / 3.0));
  float A = dot(vec3(1.9779984951, -2.428592205, 0.4505937099), q);
  float B = dot(vec3(0.0259040371, 0.7827717662, -0.808675766), q);
  float C = length(vec2(A, B));
  if (C <= ${glsl(MIX_MIN_CHROMA)}) return base;
  float h = mod(degrees(atan(B, A)), 360.0);
  if (h < CENTRE[0]) h += 360.0;
  int i = 0;
  for (int k = 1; k < 8; k++) if (h >= CENTRE[k]) i = k;
  float t = (h - CENTRE[i]) / (CENTRE[i + 1] - CENTRE[i]);
  float w = 0.5 + 0.5 * cos(3.14159265 * clamp((t - ${glsl(MIX_FLAT)}) / ${glsl(1 - 2 * MIX_FLAT)}, 0.0, 1.0));
  float m = w * uMix[i] + (1.0 - w) * uMix[(i + 1) % 8];
  if (m <= 0.0) return base * (1.0 + ${glsl(MIX_MUTE)} * m);
  float x = clamp(C / ${glsl(MIX_VIVID_CHROMA)}, 0.0, 1.0);
  float gain = ${glsl(MIX_BOOST_VIVID)} + ${glsl(MIX_BOOST_MUTED - MIX_BOOST_VIVID)} * (1.0 - x) * (1.0 - x);
  float e = m * gain * smoothstep(${glsl(MIX_FADE_START)}, ${glsl(MIX_FADE_END)}, C) * base;
  float room = max(gamutFactor(l, y) - base, 0.0);
  if (e <= 0.0 || room <= 0.0) return base;
  // The soft minimum e * room / (e^p + room^p)^(1/p), written as e / (1 + (e / room)^p)^(1/p) to stay finite when room is huge or tiny.
  float r = min(e / room, 1e6);
  return base + e * pow(1.0 + pow(r, ${glsl(MIX_GAMUT_SOFTNESS)}), -1.0 / ${glsl(MIX_GAMUT_SOFTNESS)});
}

void main() {
  vec4 t = texture(uImage, vUv);
  float a = t.a;
  vec3 c = clamp(t.rgb / max(a, 1.0 / 255.0), 0.0, 1.0);
  ivec3 x = ivec3(floor(c * 255.0 + 0.5));
  vec3 l = clamp(vec3(entry(uDecode, x.r), entry(uDecode, x.g), entry(uDecode, x.b)) * uGains, 0.0, 1.0);
  float y = dot(K, l);
  float mx = max(l.r, max(l.g, l.b));
  float mn = min(l.r, min(l.g, l.b));
  float s = mx > 0.0 ? (mx - mn) / mx : 0.0;
  float w = (1.0 - s) * (1.0 - s);
  float base = uSat * (1.0 + uVib * w);
  float f = uMixOn ? mixFactor(l, y, base) : base;
  l = clamp(mix(vec3(y), l, f), 0.0, 1.0);
  vec3 p = vec3(lerpTable(uTone, l.r, TONE_N), lerpTable(uTone, l.g, TONE_N), lerpTable(uTone, l.b, TONE_N));
  p = clamp(p + lerpTable(uShift, dot(K, p), SHIFT_N), 0.0, 1.0);
  outColor = vec4(p * a, a);
}
`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return shader;
  gl.deleteShader(shader);
  return null;
}

function buildProgram(gl: WebGL2RenderingContext): WebGLProgram | null {
  const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT);
  const program = gl.createProgram();
  if (!vertex || !fragment || !program) return null;
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (gl.getProgramParameter(program, gl.LINK_STATUS)) return program;
  gl.deleteProgram(program);
  return null;
}

/** Create a float table texture that `texelFetch` reads (nearest, no filtering). */
function tableTexture(gl: WebGL2RenderingContext): WebGLTexture | null {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return texture;
}

function uploadTable(gl: WebGL2RenderingContext, texture: WebGLTexture | null, table: Float32Array): void {
  const packed = packTable(table);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, packed.width, packed.height, 0, gl.RED, gl.FLOAT, packed.data);
}

/**
 * Create the WebGL2 renderer of Adjust's preview on a canvas, or return null when the browser has no WebGL2 or the
 * shaders do not build. The canvas keeps the same context after a loss and restore, so a restore calls this again.
 */
export function createRenderer(canvas: HTMLCanvasElement): AdjustRenderer | null {
  let gl: WebGL2RenderingContext | null = null;
  try {
    // preserveDrawingBuffer: WebKit does not present the first draw after the canvas's backing store is resized unless the buffer is kept,
    // so the picture would stay blank after the crop frame is shown or hidden or the device turns, until the next draw.
    gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false, preserveDrawingBuffer: true });
  } catch {
    gl = null;
  }
  if (!gl) return null;
  const context = gl;
  const program = buildProgram(context);
  if (!program) return null;

  const image = context.createTexture();
  const decode = tableTexture(context);
  const tone = tableTexture(context);
  const shift = tableTexture(context);
  const buffer = context.createBuffer();
  const vao = context.createVertexArray();
  context.bindVertexArray(vao);
  context.bindBuffer(context.ARRAY_BUFFER, buffer);
  const aPos = context.getAttribLocation(program, 'aPos');
  const aUv = context.getAttribLocation(program, 'aUv');
  context.enableVertexAttribArray(aPos);
  context.vertexAttribPointer(aPos, 2, context.FLOAT, false, 16, 0);
  context.enableVertexAttribArray(aUv);
  context.vertexAttribPointer(aUv, 2, context.FLOAT, false, 16, 8);
  const uniform = (name: string): WebGLUniformLocation | null => context.getUniformLocation(program, name);
  const uImage = uniform('uImage');
  const uDecode = uniform('uDecode');
  const uTone = uniform('uTone');
  const uShift = uniform('uShift');
  const uGains = uniform('uGains');
  const uSat = uniform('uSat');
  const uVib = uniform('uVib');
  const uMix = uniform('uMix');
  const uMixOn = uniform('uMixOn');

  let uploaded: PreviewTables | null = null;
  let hasImage = false;

  return {
    setImage(source): void {
      context.bindTexture(context.TEXTURE_2D, image);
      // Premultiplied, so a soft edge keeps its colour when it is filtered; the shader divides alpha out again.
      context.pixelStorei(context.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      context.pixelStorei(context.UNPACK_FLIP_Y_WEBGL, false);
      context.texImage2D(context.TEXTURE_2D, 0, context.RGBA, context.RGBA, context.UNSIGNED_BYTE, source);
      context.pixelStorei(context.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      // The picture arrives at about the size it is shown at, so it is never minified much: no mip chain to build.
      context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MIN_FILTER, context.LINEAR);
      context.texParameteri(context.TEXTURE_2D, context.TEXTURE_MAG_FILTER, context.LINEAR);
      context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_S, context.CLAMP_TO_EDGE);
      context.texParameteri(context.TEXTURE_2D, context.TEXTURE_WRAP_T, context.CLAMP_TO_EDGE);
      hasImage = true;
    },

    draw(tables, corners, width, height): void {
      if (!hasImage || width < 1 || height < 1) return;
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      context.viewport(0, 0, width, height);
      context.clearColor(0, 0, 0, 0);
      context.clear(context.COLOR_BUFFER_BIT);
      context.useProgram(program);
      context.bindVertexArray(vao);
      if (uploaded !== tables) {
        uploadTable(context, decode, tables.decode);
        uploadTable(context, tone, tables.tone);
        uploadTable(context, shift, tables.shift);
        uploaded = tables;
      }
      const [tl, tr, br, bl] = corners;
      // A strip of two triangles: top-left, top-right, bottom-left, bottom-right.
      const vertices = new Float32Array([-1, 1, tl[0], tl[1], 1, 1, tr[0], tr[1], -1, -1, bl[0], bl[1], 1, -1, br[0], br[1]]);
      context.bindBuffer(context.ARRAY_BUFFER, buffer);
      context.bufferData(context.ARRAY_BUFFER, vertices, context.DYNAMIC_DRAW);
      const bind = (unit: number, texture: WebGLTexture | null, location: WebGLUniformLocation | null): void => {
        context.activeTexture(context.TEXTURE0 + unit);
        context.bindTexture(context.TEXTURE_2D, texture);
        context.uniform1i(location, unit);
      };
      bind(0, image, uImage);
      bind(1, decode, uDecode);
      bind(2, tone, uTone);
      bind(3, shift, uShift);
      context.uniform3f(uGains, tables.gains[0], tables.gains[1], tables.gains[2]);
      context.uniform1f(uSat, tables.sat);
      context.uniform1f(uVib, tables.vib);
      context.uniform1fv(uMix, tables.mix);
      context.uniform1i(uMixOn, tables.mixOn ? 1 : 0);
      context.drawArrays(context.TRIANGLE_STRIP, 0, 4);
    },

    dispose(): void {
      context.deleteTexture(image);
      context.deleteTexture(decode);
      context.deleteTexture(tone);
      context.deleteTexture(shift);
      context.deleteBuffer(buffer);
      context.deleteVertexArray(vao);
      context.deleteProgram(program);
      // Without this the context lives until the canvas is collected, and the browser caps how many may be live.
      if (!context.isContextLost()) context.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
}
