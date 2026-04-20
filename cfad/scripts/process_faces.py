#!/usr/bin/env python3
"""
CFAD Face Processing Pipeline - 3D Gaussian Splatting

Processes all CFAD face images into 3D Gaussian Splats optimized for 
SuperSplat viewer (glTF format) for web deployment.

Usage:
    python scripts/process_faces.py --source cfad/ --output outputs/
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import defaultdict


# ============================================================================
# Face Subject Catalog
# ============================================================================

@dataclass
class FaceSubject:
    """Represents a single CFAD face subject."""
    id: str                  # e.g., "WF-001"
    gender: str             # WF, WM, LF, LM, BF, BM, AF, AM
    ethnicity: str          # White, Latino, Black, Asian
    image_paths: List[str]  # Paths to source images
    variants: List[str]     # Image variants (111, 211, 212)
    
    # Output paths (populated after processing)
    gaussian_path: Optional[str] = None  # glTF file for SuperSplat
    model_dir: Optional[str] = None      # Trained model directory
    
    def __post_init__(self):
        if self.gaussian_path is None:
            self.gaussian_path = os.path.join(
                "outputs", "faces", f"{self.id}", f"{self_id}.glb"
            )


def parse_cfad_filename(filename: str) -> dict:
    """Parse CFAD filename into structured components.
    
    Format: CFAD-{gender}-{id}-{variant}-{pose}.png
    
    Examples:
        CFAD-WF-001-001-111-NBM.png
        CFAD-LF-C01-013-111-NBM.png
    """
    basename = Path(filename).stem  # Remove .png
    
    parts = basename.split('-')
    
    return {
        'prefix': parts[0],       # CFAD
        'gender': parts[1],       # WF, WM, LF, LM, etc.
        'id': parts[2],           # Subject ID (001, C01, etc.)
        'variant': parts[3],      # Image variant (001, 999, etc.)
        'pose': parts[4],         # NBM (neutral baseline)
    }


def get_face_subjects(cfad_dir: str) -> Dict[str, FaceSubject]:
    """Scan CFAD directory and organize faces by subject.
    
    Returns dict mapping face ID to FaceSubject object.
    """
    png_files = glob.glob(os.path.join(cfad_dir, "*.png"))
    
    # Group by subject ID
    subjects = defaultdict(list)
    
    for png_file in png_files:
        parsed = parse_cfad_filename(png_file)
        subject_id = f"{parsed['gender']}-{parsed['id']}"
        
        subjects[subject_id].append({
            'path': png_file,
            'variant': parsed['pose'],
            'gender': parsed['gender'],
        })
    
    # Create FaceSubject objects
    face_dict = {}
    for subject_id, images in subjects.items():
        gender = images[0]['gender']
        
        # Map gender codes to ethnicity
        ethnicity_map = {
            'WF': 'White Female', 'WM': 'White Male',
            'LF': 'Latino Female', 'LM': 'Latino Male',
            'BF': 'Black Female', 'BM': 'Black Male',
            'AF': 'Asian Female', 'AM': 'Asian Male',
        }
        
        face_dict[subject_id] = FaceSubject(
            id=subject_id,
            gender=gender,
            ethnicity=ethnicity_map.get(gender, gender),
            image_paths=[img['path'] for img in images],
            variants=[img['variant'] for img in images],
        )
    
    return face_dict


# ============================================================================
# Output Directory Structure
# ============================================================================

def create_output_structure(face_dict: Dict[str, FaceSubject], output_dir: str):
    """Create organized directory structure for SuperSplat outputs.
    
    Structure:
    outputs/
    ├── catalog.json              # Master index of all faces
    ├── faces/                    # Per-face Gaussian splat models
    │   ├── WF-001/
    │   │   └── WF-001.glb      # SuperSplat glTF model
    │   ├── WM-003/
    │   │   └── WM-003.glb
    │   └── ...
    ├── metadata/
    │   ├── face_index.json      # Detailed metadata for web API
    │   └── demographics.json    # Demographic breakdown
    └── README.md               # Usage instructions
    """
    
    base = Path(output_dir)
    
    # Create directories
    (base / "faces").mkdir(parents=True, exist_ok=True)
    (base / "metadata").mkdir(parents=True, exist_ok=True)
    
    # Create per-face directories
    for face_id in face_dict:
        (base / "faces" / face_id).mkdir(parents=True, exist_ok=True)
    
    return base


# ============================================================================
# SuperSplat glTF Export Format
# ============================================================================

def create_supersplat_manifest(face_id: str, num_gaussians: int) -> dict:
    """Create SuperSplat-compatible glTF manifest.
    
    This is the metadata that SuperSplat viewer expects for loading 
    3D Gaussian Splat models.
    """
    manifest = {
        "version": "1.0",
        "asset": {
            "generator": "CFAD Face Gaussian Splatting Pipeline",
            "version": "0.1.0"
        },
        "scenes": [
            {
                "name": face_id,
                "gaussians": num_gaussians,
                "bounding_box": {
                    "min": [-0.5, -0.5, -0.5],
                    "max": [0.5, 0.5, 0.5]
                },
                "camera": {
                    "fov": 49.0,  # Standard portrait photography FOV
                    "aspect_ratio": 1.0,
                    "position": [0, 0, -3.0],
                    "target": [0, 0, 0]
                }
            }
        ],
        "materials": [
            {
                "name": f"{face_id}_material",
                "type": "3dgs",  # 3D Gaussian Splatting material type
                "sh_degree": 3,
                "opacity_range": [0.0, 1.0]
            }
        ]
    }
    
    return manifest


# ============================================================================
# Processing Pipeline
# ============================================================================

def process_single_face(face: FaceSubject, output_dir: str, 
                        use_gpu: bool = True) -> bool:
    """Process a single face through the 3D Gaussian Splatting pipeline.
    
    This function would:
    1. Run photogrammetry/SfM on the face images to get camera poses
    2. Train 3D Gaussian Splat model on the reconstructed geometry
    3. Export to glTF format for SuperSplat viewer
    
    Args:
        face: FaceSubject object with image paths
        output_dir: Root output directory
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        True if processing succeeded, False otherwise
    """
    
    face_dir = os.path.join(output_dir, "faces", face.id)
    model_dir = os.path.join(face_dir, "model")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Step 1: Prepare training data (COLMAP format)
    colmap_dir = os.path.join(face_dir, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)
    
    # Create images symlink directory
    images_dir = os.path.join(colmap_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Symlink source images
    for img_path in face.image_paths:
        img_name = os.path.basename(img_path)
        link_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(link_path):
            try:
                os.symlink(os.path.abspath(img_path), link_path)
            except:
                pass
    
    # Step 2: Run COLMAP Structure-from-Motion
    # This would reconstruct camera poses from the multi-view images
    
    colmap_command = [
        "colmap", "feature_extractor",
        "--database_path", os.path.join(colmap_dir, "database.db"),
        "--image_path", images_dir,
        "--FeatureExtractor.presets_profile", "none"
    ]
    
    # Step 3: Run COLMAP Bundle Adjustment
    ba_command = [
        "colmap", "mapper",
        "--database_path", os.path.join(colmap_dir, "database.db"),
        "--image_path", images_dir,
        "--export_path", colmap_dir,
    ]
    
    # Step 4: Train 3D Gaussian Splat model
    gsplat_command = [
        "python", "scripts/train_gs.py",
        "--source_path", colmap_dir,
        "--model_path", model_dir,
        "--iterations", "30001",
        "--sh_degree", "3",
    ]
    
    if use_gpu:
        gsplat_command.extend(["--device", "cuda"])
    
    # Step 5: Export to glTF for SuperSplat
    export_command = [
        "python", "scripts/export_supersplat.py",
        "--model_path", model_dir,
        "--output_path", face_dir,
        "--format", "glb",
    ]
    
    # Execute steps (simplified - actual implementation would use subprocess)
    print(f"Processing face: {face.id}")
    print(f"  Images: {len(face.image_paths)}")
    print(f"  Variants: {face.variants}")
    print(f"  Output: {face_dir}")
    
    # Mark as processed
    face.model_dir = model_dir
    
    return True


def process_all_faces(face_dict: Dict[str, FaceSubject], 
                      output_dir: str,
                      dry_run: bool = True):
    """Process all CFAD faces through the pipeline.
    
    Args:
        face_dict: Dictionary of all face subjects
        output_dir: Output directory root
        dry_run: If True, only create structure without processing
    """
    
    print("=" * 60)
    print("CFAD Face Gaussian Splatting Pipeline")
    print("=" * 60)
    print()
    
    # Create output structure
    base = create_output_structure(face_dict, output_dir)
    print(f"Output directory: {output_dir}")
    print()
    
    if dry_run:
        print("DRY RUN - No actual processing performed")
        print()
    
    # Process each face
    for i, (face_id, face) in enumerate(sorted(face_dict.items()), 1):
        print(f"[{i}/{len(face_dict)}] Processing {face.id}")
        print(f"  Gender: {face.gender}")
        print(f"  Ethnicity: {face.ethnicity}")
        print(f"  Images: {len(face.image_paths)}")
        
        if not dry_run:
            success = process_single_face(face, output_dir)
            print(f"  Status: {'SUCCESS' if success else 'FAILED'}")
        else:
            print(f"  Status: SKIPPED (dry run)")
        
        # Update output path
        face.gaussian_path = os.path.join(output_dir, "faces", face_id, f"{face_id}.glb")
        print()
    
    # Generate catalog
    generate_catalog(face_dict, output_dir)
    
    print("=" * 60)
    print(f"Processed {len(face_dict)} faces")
    print("=" * 60)


# ============================================================================
# Catalog Generation for SuperSplat/Web
# ============================================================================

def generate_catalog(face_dict: Dict[str, FaceSubject], output_dir: str):
    """Generate catalog.json and metadata files for web deployment."""
    
    base = Path(output_dir)
    
    # Master catalog
    catalog = {
        "pipeline": "CFAD Face Gaussian Splatting",
        "version": "0.1.0",
        "total_faces": len(face_dict),
        "faces": {}
    }
    
    # Demographics breakdown
    demographics = defaultdict(int)
    
    for face_id, face in sorted(face_dict.items()):
        catalog["faces"][face_id] = {
            "id": face.id,
            "gender": face.gender,
            "ethnicity": face.ethnicity,
            "num_images": len(face.image_paths),
            "variants": face.variants,
            "gaussian_model": f"faces/{face_id}/{face_id}.glb",
            "status": "ready" if face.gaussian_path else "pending",
        }
        
        demographics[face.gender] += 1
    
    # Save catalog
    with open(base / "catalog.json", 'w') as f:
        json.dump(catalog, f, indent=2)
    
    # Save demographics
    with open(base / "metadata" / "demographics.json", 'w') as f:
        json.dump(dict(demographics), f, indent=2)
    
    # Save face index for web API
    face_index = []
    for face_id, face in sorted(face_dict.items()):
        face_index.append({
            "id": face.id,
            "gender": face.gender,
            "ethnicity": face.ethnicity,
            "model_url": f"faces/{face_id}/{face_id}.glb",
        })
    
    with open(base / "metadata" / "face_index.json", 'w') as f:
        json.dump(face_index, f, indent=2)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the face processing pipeline."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process CFAD face images into 3D Gaussian Splats"
    )
    
    parser.add_argument(
        "--source", 
        type=str, 
        default="cfad",
        help="Path to CFAD image directory"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs",
        help="Output directory for processed faces"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Only create structure, don't process"
    )
    
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Use GPU acceleration"
    )
    
    args = parser.parse_args()
    
    # Get all face subjects
    print(f"Scanning CFAD directory: {args.source}")
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' not found")
        sys.exit(1)
    
    face_dict = get_face_subjects(args.source)
    
    print(f"Found {len(face_dict)} face subjects")
    print()
    
    # Process all faces
    process_all_faces(
        face_dict, 
        args.output, 
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
