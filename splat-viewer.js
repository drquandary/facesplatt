// Gaussian Splat viewer using @mkkellogg/gaussian-splats-3d, with optional
// trompe-l'œil head-tracked camera parallax (webcam + MediaPipe FaceLandmarker).

import * as GaussianSplats3D from './vendor/gaussian-splats-3d.module.js';
import * as THREE from 'three';
import { HeadTracker } from './tracker.js';

let viewer = null;
let currentEntry = null;
let tracker = null;
let headTrackActive = false;
let headTrackRAF = 0;
// Baseline camera pose captured when head-tracking is enabled.
let baseCam = null;

function setStatus(msg, show = true) {
  const el = document.getElementById('viewer-status');
  el.textContent = msg || '';
  el.classList.toggle('hidden', !show);
}

export async function openSplatViewer(entry) {
  currentEntry = entry;
  const id = entry.file.replace(/\.png$/, '');
  document.getElementById('viewer-id').textContent = id;
  document.getElementById('viewer-meta').textContent =
    `${entry.race} ${entry.sex} · ${entry.pose}`;
  document.getElementById('viewer').classList.remove('hidden');
  setStatus('loading splat…');

  const rootEl = document.getElementById('viewer-canvas');
  if (viewer) {
    try { viewer.dispose(); } catch (e) {}
    viewer = null;
    rootEl.innerHTML = '';
  }

  try {
    viewer = new GaussianSplats3D.Viewer({
      rootElement: rootEl,
      cameraUp: [0, 1, 0],
      initialCameraPosition: [0, 0, 3],
      initialCameraLookAt: [0, 0, 0],
      sharedMemoryForWorkers: false,
      selfDrivenMode: true,
      useBuiltInControls: true,
      sphericalHarmonicsDegree: 0,
      renderMode: GaussianSplats3D.RenderMode.OnChange,
    });
    await viewer.addSplatScene(`./splats/${id}.ply`, {
      splatAlphaRemovalThreshold: 5,
      showLoadingUI: false,
      progressiveLoad: true,
      // -90° around X: flips FaceLift's OpenCV convention so the face ends up
      // looking along +Z with Y-up — camera at [0,0,3] sees it face-forward.
      rotation: [-0.7071067811865475, 0, 0, 0.7071067811865476],
    });
    viewer.start();
    setStatus('', false);

    // Reset head-tracking state for new subject
    if (headTrackActive) disableHeadTracking();
  } catch (err) {
    console.error(err);
    setStatus(`failed: ${err.message || err}`);
  }
}

export function closeSplatViewer() {
  if (headTrackActive) disableHeadTracking();
  document.getElementById('viewer').classList.add('hidden');
  if (viewer) {
    try { viewer.dispose(); } catch (e) {}
    viewer = null;
  }
  document.getElementById('viewer-canvas').innerHTML = '';
  currentEntry = null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Trompe-l'œil head tracking
// Approach (per vivien000/trompeloeil): drive the camera on a sphere around
// the target based on eye position from webcam. Keeps camera always looking
// at the subject; near-field parallax sells the 3D illusion without needing
// off-axis frustum projection.
// ─────────────────────────────────────────────────────────────────────────────

export async function toggleHeadTracking() {
  if (!viewer) return;
  if (headTrackActive) {
    disableHeadTracking();
  } else {
    await enableHeadTracking();
  }
}

async function enableHeadTracking() {
  const btn = document.getElementById('viewer-headtrack');
  const video = document.getElementById('headtrack-video');
  btn.textContent = '⦿ starting…';
  btn.disabled = true;

  if (!tracker) tracker = new HeadTracker();
  try {
    if (!tracker.ready) await tracker.start(video);
  } catch (e) {
    console.error(e);
    btn.textContent = '⦿ Head track';
    btn.disabled = false;
    setStatus('webcam failed: ' + (e.message || e));
    setTimeout(() => setStatus('', false), 3000);
    return;
  }

  // Disable mkkellogg's orbit controls and capture the current camera pose
  // as the "neutral" pose. Motion is relative to that.
  try { viewer.controls.enabled = false; } catch (e) {}
  const cam = viewer.camera;
  const lookAt = new THREE.Vector3(0, 0, 0);
  const dir = cam.position.clone().sub(lookAt);
  baseCam = {
    distance: dir.length(),
    up: cam.up.clone(),
    target: lookAt.clone(),
  };

  video.style.display = 'block';
  btn.textContent = '⦿ tracking';
  btn.classList.add('active');
  btn.disabled = false;
  headTrackActive = true;
  headTrackRAF = requestAnimationFrame(trackLoop);
}

function disableHeadTracking() {
  headTrackActive = false;
  if (headTrackRAF) cancelAnimationFrame(headTrackRAF);
  headTrackRAF = 0;
  if (tracker) tracker.stop();
  tracker = null;
  const btn = document.getElementById('viewer-headtrack');
  const video = document.getElementById('headtrack-video');
  btn.textContent = '⦿ Head track';
  btn.classList.remove('active');
  video.style.display = 'none';
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  try { if (viewer) viewer.controls.enabled = true; } catch (e) {}
}

function trackLoop() {
  if (!headTrackActive || !viewer || !baseCam || !tracker) return;
  const p = tracker.pos; // smoothed {x, y, z}; x,y ∈ [-1,1]; z ∈ [0.25, 1.6]m

  // Convert eye normalized position → azimuth/elevation around target.
  // Gain chosen so small head movement yields noticeable parallax without
  // looking jittery. Clamp to ±35° so we never see the back of the head.
  const maxAzim = Math.PI * 35 / 180;
  const maxElev = Math.PI * 25 / 180;
  const azim = -p.x * maxAzim;
  const elev = p.y * maxElev;

  const d = baseCam.distance;
  const x = d * Math.sin(azim) * Math.cos(elev);
  const y = d * Math.sin(elev);
  const z = d * Math.cos(azim) * Math.cos(elev);

  const cam = viewer.camera;
  cam.position.set(x, y, z);
  cam.up.copy(baseCam.up);
  cam.lookAt(baseCam.target);
  cam.updateMatrixWorld();

  headTrackRAF = requestAnimationFrame(trackLoop);
}
