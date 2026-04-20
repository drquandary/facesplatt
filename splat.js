import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { detectFaceLandmarks } from './face-depth.js';

// Soft Gaussian sprite — proper exp(-r²) falloff, computed pixel-by-pixel so splats
// blend into a continuous surface instead of reading as discrete dots.
let _spriteTex = null;
function getSpriteTexture() {
  if (_spriteTex) return _spriteTex;
  const size = 64;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(size, size);
  const sigma2 = 0.22;
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const dx = i / (size - 1) * 2 - 1;
      const dy = j / (size - 1) * 2 - 1;
      const r2 = dx * dx + dy * dy;
      const a = Math.exp(-r2 / sigma2) * 255;
      const k = (j * size + i) * 4;
      img.data[k] = 255; img.data[k + 1] = 255; img.data[k + 2] = 255;
      img.data[k + 3] = Math.min(255, Math.max(0, a));
    }
  }
  ctx.putImageData(img, 0, 0);
  _spriteTex = new THREE.CanvasTexture(c);
  _spriteTex.colorSpace = THREE.SRGBColorSpace;
  return _spriteTex;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
}

function imageToCanvas(img, maxDim) {
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return { canvas: c, ctx, w, h };
}

// IDW-interpolated depth at (px, py) from 478 face landmarks (in pixel coords).
// power=2 gives smooth interpolation inside the face hull and graceful extrapolation outside.
function depthFromLandmarks(px, py, lmsP) {
  let num = 0, den = 0;
  for (let i = 0; i < lmsP.length; i++) {
    const dx = lmsP[i].x - px;
    const dy = lmsP[i].y - py;
    const d2 = dx * dx + dy * dy + 1;
    const w = 1 / (d2 * d2); // power 4 — more localized depth detail
    num += lmsP[i].z * w;
    den += w;
  }
  return num / den;
}

/**
 * Build a 3D splat from a 2D portrait.
 * With useLandmarks=true (default): MediaPipe 3D face mesh → per-pixel depth via IDW.
 * With useLandmarks=false or detection fail: fall back to radial-bulge heuristic.
 * Returns THREE.Points with userData.rgbColors + userData.depthColors for view-mode swap.
 */
export async function buildFaceSplat(url, opts = {}) {
  const {
    maxDim = 200,
    splatSize = 0.010,
    depthScale = 0.25,
    bgThreshold = 12,
    stride = 1,
    useLandmarks = true,
  } = opts;

  const img = await loadImage(url);
  const { canvas, ctx, w, h } = imageToCanvas(img, maxDim);
  const data = ctx.getImageData(0, 0, w, h).data;

  let lmsP = null;
  let zMin = 0, zMax = 1;
  if (useLandmarks) {
    try {
      const landmarks = await detectFaceLandmarks(canvas);
      if (landmarks) {
        lmsP = landmarks.map(p => ({ x: p.x * w, y: p.y * h, z: p.z }));
        zMin = Infinity; zMax = -Infinity;
        for (const p of lmsP) { if (p.z < zMin) zMin = p.z; if (p.z > zMax) zMax = p.z; }
        if (zMax - zMin < 1e-6) lmsP = null;
      }
    } catch (e) {
      console.warn('Landmark detection failed for', url, e);
    }
  }

  const positions = [];
  const rgbColors = [];
  const depthColors = [];
  const aspect = w / h;
  const cx = w * 0.5, cy = h * 0.5;
  const radMax = Math.hypot(cx, cy);
  const zSpan = Math.max(1e-6, zMax - zMin);

  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const i = (y * w + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
      if (a < 16) continue;
      const lum = r * 0.299 + g * 0.587 + b * 0.114;
      if (lum < bgThreshold) continue;

      const nx = (x / w - 0.5) * aspect;
      const ny = -(y / h - 0.5);

      let z01;
      if (lmsP) {
        // MediaPipe z: smaller = closer to camera. Flip so closer → larger z01.
        // Gamma 1.6 stretches the mid-face range so nose/brow/cheeks actually separate.
        const zRaw = depthFromLandmarks(x, y, lmsP);
        let t = 1 - (zRaw - zMin) / zSpan;
        t = Math.max(0, Math.min(1, t));
        z01 = Math.pow(t, 1.6);
      } else {
        // Fallback: radial bulge peaking at image center.
        const dx = (x - cx) / radMax;
        const dy = (y - cy) / radMax;
        const rad = Math.sqrt(dx * dx + dy * dy);
        const bulge = Math.max(0, 1 - rad * rad * 1.2);
        const lumBump = (lum / 255) * 0.2;
        z01 = Math.min(1, bulge + lumBump);
      }
      const z = (z01 - 0.5) * depthScale; // center depth at 0 so face plane sits at local origin

      positions.push(nx, ny, z);
      rgbColors.push(r / 255, g / 255, b / 255);
      depthColors.push(z01, z01, z01);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(rgbColors, 3));
  geom.computeBoundingSphere();

  const mat = new THREE.PointsMaterial({
    size: splatSize,
    map: getSpriteTexture(),
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
    sizeAttenuation: true,
    alphaTest: 0.0,
  });

  const points = new THREE.Points(geom, mat);
  points.userData = {
    url,
    pointCount: positions.length / 3,
    aspect,
    usedLandmarks: !!lmsP,
    rgbColors: new Float32Array(rgbColors),
    depthColors: new Float32Array(depthColors),
  };
  return points;
}

// Swap the displayed color attribute between RGB (photo) and depth (grayscale).
export function setDepthView(points, on) {
  const src = on ? points.userData.depthColors : points.userData.rgbColors;
  const attr = points.geometry.attributes.color;
  attr.array.set(src);
  attr.needsUpdate = true;
  points.userData.depthView = on;
}
