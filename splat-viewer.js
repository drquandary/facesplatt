// Opens a .ply Gaussian Splat in a full-screen viewer using
// @mkkellogg/gaussian-splats-3d. Mouse drag rotates, wheel zooms.

import * as GaussianSplats3D from './vendor/gaussian-splats-3d.module.js';

let viewer = null;
let currentEntry = null;

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
  // Clear any previous viewer instance
  if (viewer) {
    try { viewer.dispose(); } catch (e) {}
    viewer = null;
    rootEl.innerHTML = '';
  }

  try {
    // FaceLift splats use OpenCV convention (+Y down, face looks along +Z).
    // We rotate the scene 180° around X to flip to Y-up so standard camera
    // orbiting feels natural, and position the camera on the -Z side so we
    // look into the face front-on.
    // FaceLift's training camera convention puts the head's long axis along Z
    // (top of head at +Z, chin at -Z), with the face looking along +Y in
    // splat-local space. We rotate the scene -90° around X so the face ends
    // up looking along +Z (toward a camera on the +Z side), with world-Y
    // being up. That makes the splat load face-forward, right-side-up.
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
      // quaternion [x, y, z, w] = -90° around X axis:
      //   [sin(-45°), 0, 0, cos(-45°)] = [-0.7071, 0, 0, 0.7071]
      rotation: [-0.7071067811865475, 0, 0, 0.7071067811865476],
    });
    viewer.start();
    setStatus('', false);
  } catch (err) {
    console.error(err);
    setStatus(`failed: ${err.message || err}`);
  }
}

export function closeSplatViewer() {
  document.getElementById('viewer').classList.add('hidden');
  if (viewer) {
    try { viewer.dispose(); } catch (e) {}
    viewer = null;
  }
  document.getElementById('viewer-canvas').innerHTML = '';
  currentEntry = null;
}
