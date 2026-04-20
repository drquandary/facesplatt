// Singleton MediaPipe FaceLandmarker in IMAGE mode.
// Used by splat.js to recover per-face 3D landmark geometry from static portraits.

import {
  FaceLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/vision_bundle.mjs';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm';

let _lm = null;
let _pending = null;

async function getLandmarker() {
  if (_lm) return _lm;
  if (_pending) return _pending;
  _pending = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
    _lm = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'IMAGE',
      numFaces: 1,
    });
    return _lm;
  })();
  return _pending;
}

// Returns [{x,y,z}, ...478] in MediaPipe normalized coords:
//   x,y ∈ [0,1] relative to image; z ≈ image-width units, smaller = closer to camera.
// Returns null if no face found.
export async function detectFaceLandmarks(imageOrCanvas) {
  const lm = await getLandmarker();
  const res = lm.detect(imageOrCanvas);
  if (!res?.faceLandmarks?.length) return null;
  return res.faceLandmarks[0];
}
