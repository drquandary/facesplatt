import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { buildFaceSplat, setDepthView } from './splat.js';

// Gallery mode: single face centered, carousel navigation.
// Concert mode: all faces arranged as a crowd — each billboards to camera, subtle idle motion.

export class Gallery {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = 'gallery-group';
    scene.add(this.group);

    this.manifest = [];
    this.splats = new Map(); // `${file}|${useLandmarks}` → THREE.Points
    this.mode = null;
    this.index = 0;
    this.concertGroup = null;
    this._loading = null;
    this.depthView = false;
  }

  toggleDepthView() {
    this.depthView = !this.depthView;
    this.group.traverse(o => { if (o.isPoints) setDepthView(o, this.depthView); });
    return this.depthView;
  }

  async load(manifestUrl = './manifest.json') {
    const res = await fetch(manifestUrl);
    this.manifest = await res.json();
  }

  async setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    this._clearGroup();
    if (mode === 'gallery') {
      await this._showGalleryIndex(this.index);
    } else if (mode === 'concert') {
      await this._buildConcert();
    }
  }

  async next()     { this.index = (this.index + 1) % this.manifest.length; await this._showGalleryIndex(this.index); }
  async prev()     { this.index = (this.index - 1 + this.manifest.length) % this.manifest.length; await this._showGalleryIndex(this.index); }
  current()        { return this.manifest[this.index]; }

  _clearGroup() {
    while (this.group.children.length) {
      const c = this.group.children.pop();
      this.group.remove(c);
    }
    this.concertGroup = null;
  }

  async _getSplat(entry, useLandmarks) {
    const key = `${entry.file}|${useLandmarks ? 'lm' : 'rb'}`;
    if (this.splats.has(key)) return this.splats.get(key);
    const pts = await buildFaceSplat(`./cfad/${entry.file}`, {
      maxDim: useLandmarks ? 220 : 160,
      splatSize: useLandmarks ? 0.010 : 0.004,
      depthScale: useLandmarks ? 0.38 : 0.18,
      useLandmarks,
    });
    pts.userData.entry = entry;
    this.splats.set(key, pts);
    return pts;
  }

  async _showGalleryIndex(i) {
    this._clearGroup();
    const entry = this.manifest[i];
    if (!entry) return;
    const pts = await this._getSplat(entry, /*useLandmarks=*/true);
    // Face local geom ~1 unit tall; world screen ~0.30m → scale 0.28 and sit just behind plane.
    pts.position.set(0, 0, -0.08);
    pts.rotation.set(0, 0, 0);
    pts.scale.setScalar(0.28);
    pts.material.size = 0.012;
    setDepthView(pts, this.depthView);
    this.group.add(pts);
    this._currentEntry = entry;
  }

  async _buildConcert() {
    const g = new THREE.Group();
    g.name = 'concert-group';
    this.concertGroup = g;
    this.group.add(g);

    // Arrange faces as a stadium crowd — rows curving back, rising slightly.
    // Units tuned for 0.3m screen height; faces are ~1 unit tall locally.
    const N = this.manifest.length;
    const rows = 6;
    const perRow = Math.ceil(N / rows);
    const radius0 = 0.45;
    const radiusStep = 0.18;
    const rowHeight = 0.09;
    const arc = Math.PI * 0.75; // ~135° arc
    const scale = 0.10;

    // Load in small batches to avoid blocking too long.
    let idx = 0;
    for (let r = 0; r < rows; r++) {
      const radius = radius0 + r * radiusStep;
      const yBase = r * rowHeight - 0.4;
      const countThisRow = Math.min(perRow, N - idx);
      const batch = [];
      for (let k = 0; k < countThisRow; k++, idx++) {
        const entry = this.manifest[idx];
        const t = (k + 0.5) / countThisRow;
        const theta = -arc / 2 + t * arc;
        const jitter = (r % 2) * (arc / countThisRow) * 0.5;
        const xs = Math.sin(theta + jitter) * radius;
        const zs = -Math.cos(theta + jitter) * radius + 0.15;
        batch.push(this._getSplat(entry, /*useLandmarks=*/false).then(pts => {
          pts.position.set(xs, yBase, zs);
          pts.scale.setScalar(scale);
          pts.material.size = 0.004;
          pts.lookAt(0, yBase, 0.3);
          pts.userData.idlePhase = Math.random() * Math.PI * 2;
          pts.userData.idleAmp = 0.008 + Math.random() * 0.008;
          setDepthView(pts, this.depthView);
          g.add(pts);
        }));
      }
      await Promise.all(batch);
    }
  }

  update(dt, t) {
    if (this.mode === 'concert' && this.concertGroup) {
      // Subtle idle sway — concert crowd breathing motion.
      for (const pts of this.concertGroup.children) {
        const ph = pts.userData.idlePhase || 0;
        const amp = pts.userData.idleAmp || 0.008;
        pts.position.y += Math.sin(t * 0.0008 + ph) * amp * dt * 0.0005;
      }
    }
  }
}
