// facesplatt — holographic face splat gallery with webcam head tracking.
// Architecture:
//   - splat.js: 2D photo → 3D point-cloud "splat"
//   - tracker.js: MediaPipe FaceLandmarker → normalized eye position
//   - gallery.js: gallery / concert modes
//   - main.js (this file): scene, off-axis frustum projection, orchestration

import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { HeadTracker } from './tracker.js';
import { Gallery } from './gallery.js';

const canvas = document.getElementById('stage');
const videoEl = document.getElementById('webcam');
const statusEl = document.getElementById('status');
const hudEl = document.getElementById('hud');

const renderer = new THREE.WebGLRenderer({
  canvas, antialias: true, alpha: false, preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x05060a, 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05060a);

// Camera is a PerspectiveCamera but we override its projection each frame
// with an off-axis frustum based on viewer head position. This is the SharpGlass /
// "Fishbowl" effect — the scene appears locked behind the screen as the viewer moves.
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
camera.position.set(0, 0, 0); // camera sits AT the screen plane; frustum is pushed back by head z

// Physical screen model in world units (meters-ish):
//   screenHeight = 0.30 m (tune to your display), screenWidth from aspect.
//   The viewer's eye is in front of the screen at (ex, ey, ez).
const SCREEN_HEIGHT = 0.30;

// Ambient / fill — points use vertex colors but ambient helps any non-point meshes.
scene.add(new THREE.AmbientLight(0xffffff, 1.0));

const gallery = new Gallery(scene);
const tracker = new HeadTracker();

let eye = { x: 0, y: 0, z: 0.6 }; // default viewer position (m in front of screen)
let headActive = false;

function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
}
window.addEventListener('resize', onResize);
onResize();

// Off-axis projection. Reference: Robert Kooima, "Generalized Perspective Projection".
// Simpler form sufficient for axis-aligned screen: we compute frustum edges from the
// eye position relative to the screen rectangle.
function updateOffAxisProjection() {
  const aspect = renderer.domElement.width / renderer.domElement.height;
  const sh = SCREEN_HEIGHT;
  const sw = sh * aspect;

  const ex = eye.x * (sw * 0.5) * 1.2; // eye x in screen units, extra gain for drama
  const ey = eye.y * (sh * 0.5) * 1.2;
  const ez = Math.max(0.12, eye.z);    // distance from eye to screen plane

  const near = 0.05;
  const far  = 100;
  const n_ez = near / ez;
  const left   = (-sw / 2 - ex) * n_ez;
  const right  = ( sw / 2 - ex) * n_ez;
  const bottom = (-sh / 2 - ey) * n_ez;
  const top    = ( sh / 2 - ey) * n_ez;

  camera.projectionMatrix.makePerspective(left, right, top, bottom, near, far);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();

  // Camera sits at eye position; looking toward -Z (the scene lives behind the screen).
  camera.position.set(ex, ey, ez);
  camera.quaternion.identity();
}

// Fallback mouse tracking (when webcam not enabled yet) — same effect, toy version.
let mouseEye = { x: 0, y: 0 };
window.addEventListener('mousemove', (e) => {
  mouseEye.x = (e.clientX / window.innerWidth  - 0.5) * 2;
  mouseEye.y = -(e.clientY / window.innerHeight - 0.5) * 2;
});

tracker.onUpdate = (p) => {
  eye.x = p.x;
  eye.y = p.y;
  eye.z = p.z;
  headActive = true;
};

// Keyboard controls.
window.addEventListener('keydown', async (e) => {
  if (e.key === 'ArrowRight') { await gallery.next(); updateHUD(); }
  else if (e.key === 'ArrowLeft')  { await gallery.prev(); updateHUD(); }
  else if (e.key === 'g') { await gallery.setMode('gallery'); updateHUD(); }
  else if (e.key === 'c') { await gallery.setMode('concert'); updateHUD(); }
  else if (e.key === 'h') { document.getElementById('help').classList.toggle('hidden'); }
  else if (e.key === 'w') { await enableWebcam(); }
  else if (e.key === 'd') { const on = gallery.toggleDepthView(); setStatus(on ? 'Depth mask view — white = forward, black = back' : 'RGB view'); }
});

async function enableWebcam() {
  if (tracker.active) return;
  setStatus('Requesting webcam…');
  try {
    await tracker.start(videoEl);
    setStatus('Head tracking active');
    document.getElementById('webcam-wrap').classList.remove('hidden');
  } catch (err) {
    console.error(err);
    setStatus('Webcam failed: ' + err.message);
  }
}

function setStatus(s) { statusEl.textContent = s; }
function updateHUD() {
  const e = gallery.current();
  if (!e) { hudEl.textContent = ''; return; }
  hudEl.textContent =
    `${gallery.mode.toUpperCase()}  ·  ${gallery.index + 1}/${gallery.manifest.length}  ·  ` +
    (gallery.mode === 'gallery' ? `${e.id} (${e.race} ${e.sex} · ${e.pose})` : `${gallery.manifest.length} faces`);
}

// Animate.
let last = performance.now();
function tick(now) {
  const dt = now - last;
  last = now;

  // If webcam not active, fall back to mouse for the parallax effect.
  if (!headActive) {
    eye.x = mouseEye.x * 0.6;
    eye.y = mouseEye.y * 0.4;
    eye.z = 0.5;
  }

  updateOffAxisProjection();
  gallery.update(dt, now);

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

window.__fp = { THREE, scene, camera, renderer, gallery, tracker, eye };

(async () => {
  setStatus('Loading manifest…');
  await gallery.load();
  setStatus(`Loaded ${gallery.manifest.length} faces. Building first splat…`);
  await gallery.setMode('gallery');
  setStatus('Ready. Press W to enable webcam head tracking. H for help.');
  updateHUD();
  requestAnimationFrame(tick);
})().catch(err => {
  console.error(err);
  setStatus('Error: ' + err.message);
});
