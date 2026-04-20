#!/usr/bin/env python3
"""
COLMAP Preparation Script for CFAD Faces

Prepares CFAD face images into COLMAP-compatible structure for 
3D reconstruction and Gaussian Splatting training.

Creates:
- COLMAP database with image features
- Sparse reconstruction with camera poses  
- Dense point cloud for initialization
"""

import os
import sys
import glob
import subprocess
import shutil
from pathlib import Path


def prepare_colmap_for_face(face_id: str, cfad_dir: str, output_dir: str):
    """Prepare COLMAP structure for a single face subject.
    
    Args:
        face_id: Face identifier (e.g., "WF-001")
        cfad_dir: Path to CFAD image directory
        output_dir: Output directory for COLMAP data
        
    Returns:
        Path to prepared COLMAP directory
    """
    
    # Find all images for this face
    png_files = glob.glob(os.path.join(cfad_dir, f"*{face_id}*.png"))
    
    if not png_files:
        print(f"Warning: No images found for {face_id}")
        return None
    
    # Create COLMAP directory structure
    colmap_dir = os.path.join(output_dir, "colmap", face_id)
    images_dir = os.path.join(colmap_dir, "images")
    
    os.makedirs(images_dir, exist_ok=True)
    
    # Create symlinks to source images (COLMAP expects specific structure)
    for i, img_path in enumerate(png_files):
        img_name = os.path.basename(img_path)
        
        # Rename to COLMAP-friendly format (0000.png, 0001.png, etc.)
        new_name = f"{i:04d}.png"
        link_path = os.path.join(images_dir, new_name)
        
        if not os.path.exists(link_path):
            try:
                os.symlink(os.path.abspath(img_path), link_path)
            except Exception as e:
                # If symlink fails, copy the file
                shutil.copy(img_path, link_path)
    
    print(f"Prepared {len(png_files)} images for {face_id}")
    return colmap_dir


def run_colmap_feature_extraction(images_dir: str, db_path: str):
    """Run COLMAP feature extraction on prepared images.
    
    Args:
        images_dir: Directory containing image symlinks/copies
        db_path: Path to COLMAP database file
        
    Returns:
        True if successful, False otherwise
    """
    
    command = [
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--FeatureExtractor.presets_profile", "none",
        "--ImageReader.single_camera", "1",
    ]
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout per face
        )
        
        if result.returncode == 0:
            print(f"Feature extraction completed for {db_path}")
            return True
        else:
            print(f"Feature extraction failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Feature extraction timed out for {db_path}")
        return False


def run_colmap_paired_matcher(images_dir: str, db_path: str):
    """Run COLMAP paired matching for stereo image pairs.
    
    Args:
        images_dir: Directory containing images  
        db_path: Path to COLMAP database
        
    Returns:
        True if successful, False otherwise
    """
    
    command = [
        "colmap", "paired_matcher",
        "--database_path", db_path,
        "--image_path", images_dir,
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"Paired matching completed")
            return True
        else:
            print(f"Paired matching failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Paired matching timed out")
        return False


def run_colmap_mapper(colmap_dir: str, images_dir: str):
    """Run COLMAP mapper (SfM) to reconstruct 3D structure.
    
    Args:
        colmap_dir: COLMAP output directory  
        images_dir: Directory containing images
        
    Returns:
        True if successful, False otherwise
    """
    
    command = [
        "colmap", "mapper",
        "--database_path", os.path.join(colmap_dir, "database.db"),
        "--image_path", images_dir,
        "--export_path", colmap_dir,
        "--Mapper.num_threads", "4",
        "--Mapper.init_min_tri_angle", "15.0",
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"Mapper completed for {colmap_dir}")
            
            # Check if reconstruction succeeded
            sparse_model = os.path.join(colmap_dir, "sparse", "0")
            if os.path.exists(sparse_model):
                cameras_file = os.path.join(sparse_model, "cameras.bin")
                images_file = os.path.join(sparse_model, "images.bin")
                
                if os.path.exists(cameras_file) and os.path.exists(images_file):
                    print(f"  Sparse reconstruction saved to {sparse_model}")
                    return True
        
        print(f"Mapper failed or no output for {colmap_dir}")
        return False
            
    except subprocess.TimeoutExpired:
        print(f"Mapper timed out for {colmap_dir}")
        return False


def prepare_all_faces(cfad_dir: str, output_dir: str):
    """Prepare COLMAP structure for all CFAD faces.
    
    Args:
        cfad_dir: Path to CFAD image directory
        output_dir: Output directory for COLMAP data
        
    Returns:
        Dict mapping face_id to colmap_dir path
    """
    
    print("=" * 60)
    print("COLMAP Preparation for CFAD Faces")
    print("=" * 60)
    print()
    
    # Get all PNG files
    png_files = glob.glob(os.path.join(cfad_dir, "*.png"))
    
    if not png_files:
        print(f"Error: No PNG files found in {cfad_dir}")
        return {}
    
    # Extract unique face IDs
    face_ids = set()
    for png_file in png_files:
        basename = os.path.basename(png_file)
        # Parse CFAD-{gender}-{id}-{variant}-{pose}.png
        parts = basename.split('-')
        if len(parts) >= 3:
            face_id = f"{parts[1]}-{parts[2]}"
            face_ids.add(face_id)
    
    print(f"Found {len(face_ids)} unique faces")
    print()
    
    # Prepare each face
    colmap_dirs = {}
    
    for i, face_id in enumerate(sorted(face_ids), 1):
        print(f"[{i}/{len(face_ids)}] Preparing {face_id}")
        
        colmap_dir = prepare_colmap_for_face(face_id, cfad_dir, output_dir)
        
        if colmap_dir:
            colmap_dirs[face_id] = colmap_dir
            
            # Run feature extraction
            db_path = os.path.join(colmap_dir, "database.db")
            
            if not os.path.exists(db_path):
                success = run_colmap_feature_extraction(
                    os.path.join(colmap_dir, "images"), 
                    db_path
                )
                
                if success:
                    # Run paired matching
                    run_colmap_paired_matcher(
                        os.path.join(colmap_dir, "images"), 
                        db_path
                    )
                    
                    # Run mapper for 3D reconstruction
                    run_colmap_mapper(colmap_dir, os.path.join(colmap_dir, "images"))
        
        print()
    
    print("=" * 60)
    print(f"Prepared {len(colmap_dirs)} faces for COLMAP")
    print("=" * 60)
    
    return colmap_dirs


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare CFAD faces for COLMAP 3D reconstruction"
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
        default="outputs/colmap",
        help="Output directory for COLMAP data"
    )
    
    args = parser.parse_args()
    
    prepare_all_faces(args.source, args.output)


if __name__ == "__main__":
    main()
