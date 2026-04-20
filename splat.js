import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

// Soft circular sprite used as the point texture — cached once.
let _spriteTex = null;
function getSpriteTexture() {
  if (_spriteTex) return _spriteTex;
  const size = 64;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0.0, 'rgba(255,255,255,1.0)');
  g.addColorStop(0.4, 'rgba(255,255,255,0.8)');
  g.addColorStop(1.0, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  _spriteTex = new THREE.CanvasTexture(c);
  _spriteTex.colorSpace = THREE.SRGBColorSpace;
  return _spriteTex;
}

// Load image → offscreen canvas → pixel data.
function loadImagePixels(url, maxDim = 180) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      const c = document.createElement('canvas');
      c.width = w;
      c.height = h;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, w, h);
      const data = ctx.getImageData(0, 0, w, h).data;
      resolve({ data, w, h });
    };
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
}

// Build a THREE.Points splat from an image URL.
// Depth heuristic: radial bulge (closer to center = more forward) + brightness bias.
// This approximates face volume from a flat portrait — reads as 3D under head-tracked parallax.
export async function buildFaceSplat(url, opts = {}) {
  const {
    maxDim = 180,
    splatSize = 0.018,
    depthScale = 0.35,
    bgThreshold = 12, // drop dark/near-transparent background pixels
    stride = 1,
  } = opts;

  const { data, w, h } = await loadImagePixels(url, maxDim);

  const positions = [];
  const colors = [];
  const aspect = w / h;
  const cx = w * 0.5;
  const cy = h * 0.5;
  const radMax = Math.hypot(cx, cy);

  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const i = (y * w + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
      if (a < 16) continue;
      const lum = (r * 0.299 + g * 0.587 + b * 0.114);
      if (lum < bgThreshold) continue; // skip background

      // Normalized coords: x∈[-aspect/2, aspect/2], y∈[-0.5, 0.5]
      const nx = (x / w - 0.5) * aspect;
      const ny = -(y / h - 0.5);

      // Depth: radial bulge (peak at face center) + luminance highlight bump.
      const dx = (x - cx) / radMax;
      const dy = (y - cy) / radMax;
      const rad = Math.sqrt(dx * dx + dy * dy);
      const bulge = Math.max(0, 1 - rad * rad * 1.2);
      const lumBump = (lum / 255) * 0.25;
      const z = (bulge + lumBump) * depthScale;

      positions.push(nx, ny, z);
      colors.push(r / 255, g / 255, b / 255);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geom.computeBoundingSphere();

  const mat = new THREE.PointsMaterial({
    size: splatSize,
    map: getSpriteTexture(),
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
    sizeAttenuation: true,
    alphaTest: 0.05,
  });

  const points = new THREE.Points(geom, mat);
  points.userData = { url, pointCount: positions.length / 3, aspect };
  return points;
}
