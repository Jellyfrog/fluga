// FlyEye: render a photo "the way a fly's eye would see it" with one WebGL2 fragment shader.
//
// Two effects, crossfaded by `mix` (0 = normal human photo, 1 = fly view):
//
// 1. Mosaic. A Drosophila compound eye has ~800 ommatidia, each reporting roughly one
//    averaged colour. We tile the canvas with ~800 regular hexagons and fill each with the
//    mean colour of the image patch beneath it, darkening softly toward the facet rim so the
//    honeycomb is visible.
//    Averaging: we sample the mipmapped source texture at the level of detail whose texel
//    size is ~half a facet, at 7 taps (centre + 6 around a ring). Each mip texel is itself a
//    box average, so this approximates the patch mean cheaply without square mip artefacts.
//
// 2. Colour. Fly photoreceptors respond to UV, blue and green but barely to red. See
//    FLY_MATRIX / UV_TINT in the shader below for the exact mapping.

/** One-sentence legend for the UI. */
export const FLY_EYE_LEGEND =
  '≈800 facets · green & blue kept, red mostly dropped · violet = estimated UV (false color)';

/** Target number of hex facets across the whole canvas (one compound eye ≈ 800 ommatidia). */
const FACETS = 800;

const VERT = `#version 300 es
// Full-screen triangle; no vertex buffers needed.
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

uniform sampler2D uTex;
uniform vec2  uCanvas;   // backing-store size in pixels
uniform vec2  uUvScale;  // cover-fit: canvas uv -> texture uv scale (<= 1 on the cropped axis)
uniform float uSpacing;  // hex centre-to-centre distance, canvas pixels
uniform float uLod;      // mip level for per-facet averaging
uniform float uMix;      // 0 = human, 1 = fly

out vec4 outColor;

// Colour mapping (camera RGB -> displayed "fly" RGB), in one sentence:
// red is mostly dropped, green and blue are kept, and an estimated UV channel
// (blue minus half the green, boosted) is painted on top as a violet false colour.
// GLSL mat3 is column-major: each vec3 below is the contribution of R, G, B respectively.
const mat3 FLY_MATRIX = mat3(
  0.08, 0.05, 0.00,   // camera R -> (out r, out g, out b): almost nothing
  0.15, 0.65, 0.10,   // camera G -> mostly out g (a little into r to mute the green)
  0.00, 0.05, 0.60);  // camera B -> mostly out b
const vec3 UV_TINT = vec3(0.70, 0.0, 0.30);  // estimated UV shown as violet false colour
// Resulting landmarks: a neutral grey v -> about (0.75, 0.75, 0.93)·v (pale lavender),
// tomato red -> dark (only its weak green survives), leaf green -> dark green, sky blue -> violet.

// Canvas pixel (origin bottom-left) -> texture uv, cover-fitted and centre-cropped.
vec2 toTex(vec2 px) {
  vec2 uv = px / uCanvas;
  uv = (uv - 0.5) * uUvScale + 0.5;
  return vec2(uv.x, 1.0 - uv.y);  // image rows are uploaded top-first
}

vec3 flyColor(vec3 c) {
  float uv = clamp((c.b - 0.5 * c.g) * 1.5, 0.0, 1.0);  // UV estimate
  return clamp(FLY_MATRIX * c + UV_TINT * uv, 0.0, 1.0);
}

void main() {
  vec2 px = gl_FragCoord.xy;
  vec3 human = texture(uTex, toTex(px)).rgb;

  // Nearest hex centre: union of two offset rectangular lattices (pointy-top hexes).
  vec2 g = vec2(1.0, 1.7320508) * uSpacing;
  vec2 a = mod(px, g) - 0.5 * g;
  vec2 b = mod(px - 0.5 * g, g) - 0.5 * g;
  vec2 local = dot(a, a) < dot(b, b) ? a : b;   // offset from facet centre
  vec2 centre = px - local;

  // Per-facet average: 7 mip taps over the facet (centre + ring at ~half the inradius).
  vec3 sum = textureLod(uTex, toTex(centre), uLod).rgb;
  float r = 0.28 * uSpacing;
  for (int i = 0; i < 6; i++) {
    float ang = float(i) * 1.0471976 + 0.5235988;
    sum += textureLod(uTex, toTex(centre + r * vec2(cos(ang), sin(ang))), uLod).rgb;
  }
  vec3 facet = flyColor(sum / 7.0);

  // Hex "radius" in [0,1] (1 = on an edge): max projection onto the 3 edge normals.
  vec2 q = abs(local);
  float d = max(q.x, dot(q, vec2(0.5, 0.8660254))) / (0.5 * uSpacing);
  // Soft lens shading: gentle dome, then a darker soft rim.
  float shade = (1.0 - 0.12 * d * d) * (1.0 - 0.55 * smoothstep(0.72, 1.0, d));
  vec3 fly = facet * shade;

  outColor = vec4(mix(human, fly, clamp(uMix, 0.0, 1.0)), 1.0);
}`;

/** True if this browser can create a WebGL2 context. */
export function flyEyeSupported(): boolean {
  try {
    return !!document.createElement('canvas').getContext('webgl2');
  } catch {
    return false;
  }
}

type Source = ImageBitmap | HTMLImageElement | HTMLCanvasElement;

export class FlyEye {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private tex: WebGLTexture;
  private vao: WebGLVertexArrayObject;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private imgW = 0;
  private imgH = 0;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error('FlyEye: WebGL2 is not available');
    this.gl = gl;
    this.prog = link(gl, VERT, FRAG);
    for (const n of ['uTex', 'uCanvas', 'uUvScale', 'uSpacing', 'uLod', 'uMix']) {
      this.u[n] = gl.getUniformLocation(this.prog, n);
    }
    this.vao = gl.createVertexArray()!;
    this.tex = gl.createTexture()!;
  }

  /** Upload a new photo (any size; downscaled if it exceeds the GPU texture limit). */
  setImage(src: Source): void {
    const gl = this.gl;
    let w = src instanceof HTMLImageElement ? src.naturalWidth : src.width;
    let h = src instanceof HTMLImageElement ? src.naturalHeight : src.height;
    let data: Source = src;
    const max = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (w > max || h > max) {
      const s = max / Math.max(w, h);
      const c = document.createElement('canvas');
      c.width = Math.floor(w * s);
      c.height = Math.floor(h * s);
      c.getContext('2d')!.drawImage(src, 0, 0, c.width, c.height);
      data = c;
      w = c.width;
      h = c.height;
    }
    this.imgW = w;
    this.imgH = h;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.generateMipmap(gl.TEXTURE_2D); // WebGL2 supports NPOT mipmaps
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** Draw the current image. mix: 0 = human view, 1 = fly view. */
  render(mix: number): void {
    const gl = this.gl;
    const c = this.canvas;
    // Size the backing store to the CSS size × devicePixelRatio for crisp facet edges.
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.max(1, Math.round(c.clientWidth * dpr));
    const bh = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== bw || c.height !== bh) {
      c.width = bw;
      c.height = bh;
    }
    gl.viewport(0, 0, bw, bh);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.imgW) return;

    // Cover fit: the axis where the image is relatively longer gets cropped (scale < 1).
    const ca = bw / bh;
    const ia = this.imgW / this.imgH;
    const sx = ca < ia ? ca / ia : 1;
    const sy = ca < ia ? 1 : ia / ca;

    // Hex with centre spacing D has area (√3/2)·D²; pick D so FACETS hexes cover the canvas.
    const spacing = Math.sqrt((2 * bw * bh) / (Math.sqrt(3) * FACETS));
    // Texture pixels per canvas pixel (after cover fit), then the mip level whose texels are
    // about half a facet wide (the 7 taps cover the rest of the facet).
    const texPerPx = (sx * this.imgW) / bw;
    const lod = Math.max(0, Math.log2(spacing * texPerPx * 0.5));

    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this.u.uTex, 0);
    gl.uniform2f(this.u.uCanvas, bw, bh);
    gl.uniform2f(this.u.uUvScale, sx, sy);
    gl.uniform1f(this.u.uSpacing, spacing);
    gl.uniform1f(this.u.uLod, lod);
    gl.uniform1f(this.u.uMix, mix);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose(): void {
    const gl = this.gl;
    gl.deleteTexture(this.tex);
    gl.deleteProgram(this.prog);
    gl.deleteVertexArray(this.vao);
    this.imgW = this.imgH = 0;
  }
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('FlyEye shader: ' + gl.getShaderInfoLog(s));
    }
    gl.attachShader(p, s);
    gl.deleteShader(s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('FlyEye link: ' + gl.getProgramInfoLog(p));
  }
  return p;
}
