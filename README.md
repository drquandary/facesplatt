# facesplatt

Head-tracked holographic face splat gallery. Each 2D portrait is converted in-browser to a 3D point-cloud "splat", rendered with an off-axis frustum camera so the scene feels locked behind the screen as you move your head (SharpGlass-style "window into a world" parallax, [TrondW/SharpGlass](https://github.com/TrondW/SharpGlass)).

## Modes

- **Gallery** (`G`) — single face splat, navigate with `←` / `→`.
- **Concert** (`C`) — all 120 faces arranged as a stadium crowd.
- **Webcam** (`W`) — MediaPipe FaceLandmarker drives the off-axis frustum from your head position (fallback: mouse).

## Run

```bash
./serve.sh       # python3 -m http.server 8000
open http://localhost:8000/
```

Webcam APIs require a secure context — `localhost` counts.

## Architecture

| file | role |
| --- | --- |
| `index.html` | viewport, HUD, help overlay |
| `main.js` | scene, off-axis projection, eye fallback |
| `splat.js` | image → `THREE.Points` with depth heuristic |
| `tracker.js` | MediaPipe head tracker |
| `gallery.js` | gallery + concert modes |
| `manifest.json` | CFAD file list w/ demographic metadata |
| `cfad/` | 120 face PNGs |

## How the splat is built

`buildFaceSplat` loads each image to an offscreen canvas, samples every pixel, and emits one point per non-background pixel. The depth for each point is a radial-bulge heuristic — pixels near image center sit forward (nose/brow), edges recede (ears/jaw). Luminance adds a small additional forward bump. It's an approximation of portrait volume from a single photo: not a true 3DGS reconstruction, but sufficient for head-tracked parallax to read as 3D.

## How the holography works

`main.js::updateOffAxisProjection` computes the camera frustum from the viewer's eye position relative to a virtual 0.30m-tall screen. Instead of rotating the camera with the viewer, the projection matrix itself is skewed — so geometry appears anchored in world space behind the screen, creating the "window" illusion.

## Keys

| key | action |
| --- | --- |
| `←` `→` | prev / next face |
| `G` | gallery mode |
| `C` | concert mode |
| `W` | enable webcam head tracking |
| `H` | toggle help |

## Dataset

CFAD (Chicago Face Database-style) portraits, organized by prefix:

- `AF` Asian Female · `AM` Asian Male
- `BF` Black Female · `BM` Black Male
- `LF` Latino Female · `LM` Latino Male
- `WF` White Female · `WM` White Male

Pose code `111` = frontal, `211/212` = profile.
