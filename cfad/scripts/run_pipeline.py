#!/usr/bin/env python3
"""
Complete CFAD Face Processing Pipeline - Batch Mode

Processes all 60 CFAD faces through the complete pipeline:
1. COLMAP preparation (3D reconstruction from multi-view images)
2. 3D Gaussian Splatting training  
3. SuperSplat glTF export for web deployment

Usage:
    python scripts/run_pipeline.py --source cfad/ --output outputs/
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def run_command(cmd: list, cwd: str = None):
    """Run a shell command and return success status."""
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd, 
        cwd=cwd,
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return False
    
    return True


def main():
    """Run the complete pipeline for all CFAD faces."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete CFAD Face Gaussian Splatting Pipeline"
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
        help="Output directory root"
    )
    
    parser.add_argument(
        "--skip-colmap", 
        action="store_true",
        help="Skip COLMAP preparation (use existing data)"
    )
    
    parser.add_argument(
        "--skip-train", 
        action="store_true",
        help="Skip training (use existing models)"
    )
    
    parser.add_argument(
        "--skip-export", 
        action="store_true",
        help="Skip SuperSplat export"
    )
    
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Use GPU for training"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CFAD FACE GAUSSIAN SPLATTING PIPELINE")
    print("=" * 70)
    print()
    
    # Step 1: COLMAP Preparation
    if not args.skip_colmap:
        print("STEP 1/3: COLMAP Preparation")
        print("-" * 70)
        
        success = run_command([
            sys.executable, "scripts/prepare_colmap.py",
            "--source", args.source,
            "--output", os.path.join(args.output, "colmap")
        ])
        
        if not success:
            print("Warning: COLMAP preparation failed, continuing...")
        
        print()
    
    # Step 2: Train 3DGS Models
    if not args.skip_train:
        print("STEP 2/3: 3D Gaussian Splatting Training")
        print("-" * 70)
        
        # Get list of face directories (from COLMAP output or source)
        colmap_dir = os.path.join(args.output, "colmap")
        
        if os.path.exists(colmap_dir):
            face_ids = [d for d in os.listdir(colmap_dir) 
                       if os.path.isdir(os.path.join(colmap_dir, d))]
        else:
            # Fallback: extract from source directory filenames
            import glob
            png_files = glob.glob(os.path.join(args.source, "*.png"))
            
            face_ids = set()
            for png in png_files:
                basename = os.path.basename(png)
                parts = basename.split('-')
                if len(parts) >= 3:
                    face_ids.add(f"{parts[1]}-{parts[2]}")
            
            face_ids = sorted(face_ids)
        
        print(f"Processing {len(face_ids)} faces...")
        print()
        
        for i, face_id in enumerate(sorted(face_ids), 1):
            print(f"[{i}/{len(face_ids)}] Training {face_id}")
            
            colmap_path = os.path.join(colmap_dir, face_id) if os.path.exists(colmap_dir) else None
            model_path = os.path.join(args.output, "models", face_id)
            
            if colmap_path and os.path.exists(colmap_path):
                success = run_command([
                    sys.executable, "scripts/train_gs.py",
                    "--source_path", colmap_path,
                    "--model_path", model_path,
                    "--iterations", "30001",
                ] + (["--device", "cuda"] if args.gpu else ["--device", "auto"]))
            else:
                print(f"  Skipping (no COLMAP data for {face_id})")
                success = False
            
            print()
    
    # Step 3: Export to SuperSplat Format
    if not args.skip_export:
        print("STEP 3/3: SuperSplat glTF Export")
        print("-" * 70)
        
        models_dir = os.path.join(args.output, "models")
        
        if os.path.exists(models_dir):
            success = run_command([
                sys.executable, "scripts/export_supersplat.py",
                "--faces_dir", models_dir,
                "--output_path", args.output,
            ])
        else:
            print("Skipping export (no models found)")
        
        print()
    
    # Generate final catalog
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    # Create catalog for Gliff/SuperSplat integration
    output_catalog = {
        "pipeline": "CFAD Face Gaussian Splatting",
        "version": "0.1.0",
        "output_dir": args.output,
        "instructions": {
            "super_splat_viewer": "Load .glb files from outputs/faces/ directory",
            "web_integration": "Serve outputs/ directory via static file server",
        }
    }
    
    catalog_path = os.path.join(args.output, "pipeline_catalog.json")
    
    with open(catalog_path, 'w') as f:
        json.dump(output_catalog, f, indent=2)
    
    print(f"Catalog saved to: {catalog_path}")
    print()


if __name__ == "__main__":
    main()
