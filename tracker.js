// Head tracker: MediaPipe FaceLandmarker → normalized viewer eye position.
// Output is consumed by the off-axis frustum camera to create holographic parallax.

import {
  FaceLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/vision_bundle.mjs';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm';

export class HeadTracker {
  constructor() {
    this.ready = false;
    this.active = false;
    this.landmarker = null;
    this.video = null;
    this.pos = { x: 0, y: 0, z: 0.6 }; // smoothed, normalized screen-space eye pos
    this.raw = { x: 0, y: 0, z: 0.6 };
    this.smoothing = 0.35; // balance: responsive but not twitchy
    this.lastStamp = -1;
    this.onUpdate = null;
    this.lost = true;
  }

  async start(videoEl) {
    this.video = videoEl;
    const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: false,
    });

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    this.ready = true;
    this.active = true;
    this._loop();
  }

  stop() {
    this.active = false;
    if (this.video?.srcObject) {
      this.video.srcObject.getTracks().forEach(t => t.stop());
    }
  }

  _loop = () => {
    if (!this.active) return;
    const v = this.video;
    if (v && v.readyState >= 2) {
      const ts = performance.now();
      if (ts !== this.lastStamp) {
        this.lastStamp = ts;
        const res = this.landmarker.detectForVideo(v, ts);
        if (res?.faceLandmarks?.length) {
          this._updateFromLandmarks(res.faceLandmarks[0]);
          this.lost = false;
        } else {
          this.lost = true;
        }
      }
    }
    requestAnimationFrame(this._loop);
  };

  _updateFromLandmarks(lm) {
    // MediaPipe landmarks: normalized [0,1] image coords.
    // Use eye midpoint (L: 33, R: 263) for x,y; interocular distance for z.
    const L = lm[33], R = lm[263];
    const cxN = (L.x + R.x) * 0.5;
    const cyN = (L.y + R.y) * 0.5;
    const iod = Math.hypot(R.x - L.x, R.y - L.y); // 0..~0.15

    // Mirror X because webcam is mirrored in UI — moving head right should shift view right.
    const x = (0.5 - cxN) * 2.0;        // [-1, 1]
    const y = (0.5 - cyN) * 2.0;        // [-1, 1]
    // Map IOD to z distance in meters-ish. Closer face → larger IOD → smaller z.
    // 0.16 (very close) → 0.3m, 0.04 (far) → 1.2m.
    const z = THREE_clamp(0.06 / Math.max(0.02, iod), 0.25, 1.6);

    this.raw.x = x;
    this.raw.y = y;
    this.raw.z = z;

    const s = this.smoothing;
    this.pos.x += (x - this.pos.x) * s;
    this.pos.y += (y - this.pos.y) * s;
    this.pos.z += (z - this.pos.z) * s;

    if (this.onUpdate) this.onUpdate(this.pos);
  }
}

function THREE_clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
