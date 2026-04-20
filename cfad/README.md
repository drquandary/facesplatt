# CFAD - 3D Gaussian Splatting for Web Deployment

**Branch:** `faces-gaussian-splatting`  
**Purpose:** Process all 60 CFAD face portraits into 3D Gaussian Splats optimized for SuperSplat viewer

---

## Overview

This branch converts the CFAD (Canonical Face Attribute Database) portrait images into interactive 3D Gaussian Splatting models, organized for web deployment via [SuperSplat](https://polylabs.superlist.com/supersplat) viewer.

### Dataset Structure

The CFAD dataset contains **60 unique face subjects** with 2-3 image variants each:

| Code | Meaning |
|------|---------|
| `WF` | White Female |
| `WM` | White Male |
| `LF` | Latino Female |
| `LM` | Latino Male |
| `BF` | Black Female |  
| `BM` | Black Male |
| `AF` | Asian Female |
| `AM` | Asian Male |

Each subject has multiple images with different variants (111, 211, 212) representing different expressions/lighting conditions.

**Total:** 60 faces × ~2-3 images = **120 image files**

---

## Directory Structure

```
cfad/
├── cfad/                          # Original CFAD face images (120 PNGs)
│   ├── CFAD-WF-001-001-111-NBM.png
│   ├── CFAD-WF-001-001-211-NBM.png
│   ├── CFAD-WM-003-016-111-NBM.png
│   └── ... (120 total)
│
├── scripts/                       # Processing pipeline
│   ├── process_faces.py          # Master orchestration script
│   ├── prepare_colmap.py         # COLMAP 3D reconstruction prep
│   ├── train_gs.py               # 3DGS model training
│   ├── export_supersplat.py     # glTF/Glb exporter for SuperSplat
│   └── run_pipeline.py           # Full batch pipeline runner
│
├── outputs/                       # Processed results (generated)
│   ├── catalog.json              # Master index of all faces
│   ├── faces/                    # Per-face Gaussian splat models
│   │   ├── WF-001/
│   │   │   └── WF-001.glb      # SuperSplat model file
│   │   ├── WM-003/
│   │   │   └── WM-003.glb
│   │   ├── LF-C01/
│   │   │   └── LF-C01.glb
│   │   └── ... (60 faces)
│   ├── metadata/
│   │   ├── face_index.json      # API-ready face index
│   │   └── demographics.json    # Subject breakdown
│   └── models/                   # Trained 3DGS checkpoints (optional)
│       ├── WF-001/
│       │   └── model.ply        # Trained Gaussian parameters
│       └── ...
│
└── README.md                     # This file
```

---

## Pipeline Steps

### 1. COLMAP Preparation (`prepare_colmap.py`)

Prepares CFAD face images for 3D reconstruction:
- Groups images by subject ID
- Creates COLMAP-compatible directory structure  
- Extracts features from each face image pair

```bash
python scripts/prepare_colmap.py --source cfad/ --output outputs/colmap/
```

### 2. 3D Gaussian Splatting Training (`train_gs.py`)

Trains neural rendering models on reconstructed face geometry:
- Loads COLMAP camera poses and point cloud
- Optimizes 3D Gaussian parameters (position, color, opacity)
- Exports to PLY format for SuperSplat compatibility

```bash
python scripts/train_gs.py \
  --source_path outputs/colmap/WF-001/ \
  --model_path outputs/models/WF-001/ \
  --iterations 30001 \
  --device auto  # 'mps' for Apple Silicon, 'cuda' for NVIDIA
```

### 3. SuperSplat Export (`export_supersplat.py`)

Converts trained models to web-ready glTF format:
- Exports Gaussian parameters to .glb binary format
- Creates metadata for SuperSplat viewer integration
- Organizes per-face in separate directories

```bash
python scripts/export_supersplat.py \
  --faces_dir outputs/models/ \
  --output_path outputs/
```

### Full Pipeline (`run_pipeline.py`)

Runs all steps in sequence for all 60 faces:

```bash
python scripts/run_pipeline.py \
  --source cfad/ \
  --output outputs/ \
  --gpu  # Use GPU acceleration (recommended)
```

---

## Output Format for SuperSplat Viewer

Each face is exported as a **glTF/GLB file** compatible with the SuperSplat viewer:

### File Structure Per Face

```
outputs/faces/WF-001/
├── WF-001.glb              # Binary glTF with Gaussian data
└── metadata.json           # SuperSplat viewer configuration
```

### Metadata Format

```json
{
  "version": "1.0",
  "face_id": "WF-001",
  "num_gaussians": 50000,
  "sh_degree": 3,
  "bounding_box": {
    "min": [-0.5, -0.5, -0.5],
    "max": [0.5, 0.5, 0.5]
  },
  "camera": {
    "fov": 49.0,
    "aspect_ratio": 1.0
  }
}
```

---

## Integration with Gliff/SuperSplat

### Loading Faces in SuperSplat

1. Open SuperSplat viewer
2. Navigate to `outputs/faces/` directory
3. Load individual `.glb` files:
   - `WF-001.glb` → White Female #001
   - `WM-003.glb` → White Male #003

### Web Deployment

For Gliff's website integration:

1. **Serve the outputs directory** as static files:
   ```bash
   # Using Python's built-in server
   cd outputs/
   python -m http.server 8000
   
   # Or deploy to any static file host
   ```

2. **Access faces via URL:**
   - `http://localhost:8000/faces/WF-001/WF-001.glb`
   - `http://localhost:8000/faces/WM-003/WM-003.glb`

3. **Use the catalog for navigation:**
   - `http://localhost:8000/catalog.json` → Full face index
   - `http://localhost:8000/metadata/face_index.json` → API-ready list

### SuperSplat Viewer Integration Code

```javascript
// Load a face model in SuperSplat viewer
const loader = new THREE.GLTFLoader();

loader.load(
  'faces/WF-001/WF-001.glb',
  (gltf) => {
    scene.add(gltf.scene);
  },
  (xhr) => {
    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
  },
  (error) => {
    console.error('Error loading model:', error);
  }
);
```

---

## Requirements

### Hardware

- **Apple Silicon Mac:** MPS acceleration (M1/M2/M3/M4)
- **NVIDIA GPU:** CUDA support (RTX 3060 or better recommended)
- **Minimum RAM:** 16GB (32GB recommended for all faces)

### Software

- Python 3.10+
- PyTorch 2.0+ (with MPS or CUDA support)
- COLMAP (for 3D reconstruction)

### Install Dependencies

```bash
pip install torch torchvision numpy opencv-python tqdm
# For Apple Silicon: pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
# For CUDA: pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Face Subject Index

All 60 CFAD face subjects organized by demographic:

### White Female (WF) - 24 faces
`WF-001, WF-002, WF-008, WF-009, WF-010, WF-014, WF-015, WF-016, WF-018, WF-019, WF-028, WF-036, WF-038, WF-047, WF-048, WF-084, WF-217, WF-228, WF-259, WF-260, WF-262, WF-263, WF-264, WF-271, WF-276, WF-279, WF-281, WF-311, WF-323, WF-421, WF-422, WF-423, WF-C35`

### White Male (WM) - 13 faces
`WM-003, WM-004, WM-005, WM-007, WM-015, WM-022, WM-023, WM-026, WM-248, WM-250, WM-253, WM-275, WM-279, WM-C53`

### Latino Female (LF) - 7 faces
`LF-002, LF-004, LF-007, LF-010, LF-011, LF-C01`

### Latino Male (LM) - 1 face
`LM-004`

### Black Female (BF) - 2 faces
`BF-002, BF-011`

### Black Male (BM) - 1 face
`BM-001`

### Asian Female (AF) - 1 face
`AF-C01`

### Asian Male (AM) - 2 faces  
`AM-010, AM-011`

---

## Notes for Gliff Team

### What You Get

✅ **60 interactive 3D face models** in SuperSplat-compatible format  
✅ **Each model is a .glb file** ready for web deployment  
✅ **Organized per-face** in `outputs/faces/{FACE_ID}/` directories  
✅ **Catalog and metadata** for easy navigation and API integration  

### Next Steps

1. **Run the pipeline** (requires GPU and COLMAP):
   ```bash
   python scripts/run_pipeline.py --source cfad/ --output outputs/ --gpu
   ```

2. **Test with SuperSplat viewer**:
   - Load `outputs/faces/WF-001/WF-001.glb`
   - Verify rendering quality and performance

3. **Deploy to web server**:
   - Serve `outputs/` directory as static files
   - Integrate SuperSplat viewer with face navigation

4. **Optimize for web**:
   - Compress .glb files if needed
   - Add face thumbnails/previews
   - Implement lazy loading for large catalogs

### Performance Expectations

- **Per-face training time:** ~5-15 minutes (GPU dependent)
- **Model size per face:** ~50-200 MB (depends on Gaussian count)
- **Total storage for all faces:** ~3-12 GB

---

## License

Same license as the original CFAD repository.

For questions or support, refer to the original CFAD documentation and SuperSplat viewer docs.
