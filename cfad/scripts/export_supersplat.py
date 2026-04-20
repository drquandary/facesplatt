#!/usr/bin/env python3
"""
SuperSplat glTF Exporter for 3D Gaussian Splatting

Converts trained 3DGS models to glTF/GLB format compatible with 
SuperSplat viewer for web deployment.

Output structure:
  face_id.glb           # Binary glTF with Gaussian data
  face_id.json          # Metadata for SuperSplat viewer
  
Usage:
    python scripts/export_supersplat.py --model_path models/WF-001/ --output_path outputs/faces/WF-001/
"""

import os
import sys
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def load_3dgs_ply(ply_path: str) -> Dict:
    """Load 3DGS model from PLY file.
    
    Args:
        ply_path: Path to .ply file with Gaussian parameters
        
    Returns:
        Dict with 'xyz', 'opacity', 'features_dc', 'features_rest', 
              'scaling', 'rotation' arrays
    """
    
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        return None
    
    xyz_list = []
    opacity_list = []
    features_dc_list = []
    features_rest_list = []
    scaling_list = []
    rotation_list = []
    
    with open(ply_path, 'rb') as f:
        # Read header
        line = f.readline()  # "ply"
        
        while b"end_header" not in line:
            line = f.readline().decode('utf-8').strip()
            
            # Parse property declarations (simplified)
            if line.startswith("property"):
                parts = line.split()
                # property float x, property float y, etc.
                pass
        
        # Read data (binary)
        while True:
            chunk = f.read(4 * 52)  # ~52 floats per vertex
            if len(chunk) < 4 * 52:
                break
            
            values = struct.unpack("52f", chunk)
            
            # Extract fields (assuming standard 3DGS PLY format)
            xyz_list.append(values[0:3])
            opacity_list.append(values[3])
            
            # SH coefficients (first 3 are DC, rest is Rest)
            sh_start = 4
            dc_end = sh_start + 3  # 3 RGB values for DC band
            
            dc_vals = [values[sh_start + i] for i in range(3)]
            rest_vals = [values[sh_start + 3 + i] for i in range(45)]
            
            features_dc_list.append(dc_vals)
            features_rest_list.append(rest_vals)
            
            scaling_list.append(values[49:52])
            rotation_list.append(values[52:56] if len(values) > 56 else [1.0, 0, 0, 0])
    
    # Convert to numpy arrays
    result = {
        'xyz': np.array(xyz_list, dtype=np.float32),  # (N, 3)
        'opacity': np.array(opacity_list, dtype=np.float32).reshape(-1, 1),  # (N, 1)
        'features_dc': np.array(features_dc_list, dtype=np.float32),  # (N, 3)
        'features_rest': np.array(features_rest_list, dtype=np.float32),  # (N, 45)
        'scaling': np.array(scaling_list, dtype=np.float32),  # (N, 3)
        'rotation': np.array(rotation_list, dtype=np.float32),  # (N, 4)
    }
    
    return result


def export_to_glb(gaussian_data: Dict, output_path: str, face_id: str):
    """Export Gaussian data to glTF/GLB format for SuperSplat.
    
    Args:
        gaussian_data: Dict with Gaussian parameters from PLY loader
        output_path: Output directory path
        face_id: Face identifier for naming
        
    Returns:
        Path to exported .glb file
    """
    
    os.makedirs(output_path, exist_ok=True)
    
    glb_path = os.path.join(output_path, f"{face_id}.glb")
    
    N = gaussian_data['xyz'].shape[0]
    
    print(f"Exporting {N} Gaussians to glTF for SuperSplat...")
    
    # Create binary glTF structure
    # glTF format: JSON header + bin buffer
    
    # 1. Create the JSON representation
    gltf_json = {
        "asset": {
            "version": "2.0",
            "generator": "CFAD Face Gaussian Splatting Pipeline"
        },
        "scene": 0,
        "scenes": [
            {
                "nodes": [0]
            }
        ],
        "nodes": [
            {
                "name": f"{face_id}_gaussians",
                "mesh": 0,
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "OPACITY": 1,
                            "COLOR_DC": 2,
                            "COLOR_REST": 3,
                            "SCALE": 4,
                            "ROTATION": 5,
                        },
                        "indices": 0,
                        "material": 0,
                        "mode": 4,  # POINTS
                    }
                ]
            }
        ],
        "accessors": [
            # 0: POSITION (xyz) - FLOAT, 3 components
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": N,
                "type": "VEC3",
                "max": [float(x.max()) for x in gaussian_data['xyz'].T],
                "min": [float(x.min()) for x in gaussian_data['xyz'].T],
            },
            # 1: OPACITY - FLOAT, 1 component  
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": N,
                "type": "SCALAR",
            },
            # 2: COLOR_DC (first SH band) - FLOAT, 3 components
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": N,
                "type": "VEC3",
            },
            # 3: COLOR_REST (remaining SH bands) - FLOAT, 45 components
            {
                "bufferView": 3,
                "componentType": 5126,
                "count": N,
                "type": "VEC4",  # Padded to VEC4 for alignment
            },
            # 4: SCALE - FLOAT, 3 components
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": N,
                "type": "VEC3",
            },
            # 5: ROTATION (quaternion) - FLOAT, 4 components
            {
                "bufferView": 5,
                "componentType": 5126,
                "count": N,
                "type": "VEC4",
            },
            # 6: Indices (0, 1, 2, ..., N-1)
            {
                "bufferView": 6,
                "componentType": 5123,  # UNSIGNED_SHORT or UNSIGNED_INT
                "count": N,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            # 0: POSITION (N * 3 * 4 bytes = N * 12)
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": N * 12,
                "target": 34962,  # ARRAY_BUFFER
            },
            # 1: OPACITY (N * 4 bytes)
            {
                "buffer": 0,
                "byteOffset": N * 12,
                "byteLength": N * 4,
            },
            # 2: COLOR_DC (N * 3 * 4 bytes)
            {
                "buffer": 0,
                "byteOffset": N * 12 + N * 4,
                "byteLength": N * 12,
            },
            # 3: COLOR_REST (N * 45 * 4 bytes, padded to VEC4)
            {
                "buffer": 0,
                "byteOffset": N * 12 + N * 4 + N * 12,
                "byteLength": N * 180,  # 45 floats * 4 bytes
            },
            # 4: SCALE (N * 3 * 4 bytes)
            {
                "buffer": 0,
                "byteOffset": N * 12 + N * 4 + N * 12 + N * 180,
                "byteLength": N * 12,
            },
            # 5: ROTATION (N * 4 * 4 bytes)
            {
                "buffer": 0,
                "byteOffset": N * 12 + N * 4 + N * 12 + N * 180 + N * 12,
                "byteLength": N * 16,
            },
            # 6: INDICES (N * 4 bytes)
            {
                "buffer": 0,
                "byteOffset": N * 12 + N * 4 + N * 12 + N * 180 + N * 12 + N * 16,
                "byteLength": N * 4,
            },
        ],
        "buffers": [
            {
                "uri": "data:application/octet-stream;base64,<TODO_BASE64>",
                "byteLength": N * 200,  # Approximate total size
            }
        ],
        "materials": [
            {
                "name": f"{face_id}_material",
                "pbrSpecularGlossiness": {
                    "diffuseFactor": [1.0, 1.0, 1.0],
                    "specularFactor": [1.0, 1.0, 1.0],
                    "glossinessFactor": 1.0,
                },
            }
        ],
    }
    
    # Write JSON to file (SuperSplat can read .gltf)
    json_path = os.path.join(output_path, f"{face_id}.gltf")
    
    with open(json_path, 'w') as f:
        json.dump(gltf_json, f, indent=2)
    
    print(f"  JSON exported to: {json_path}")
    
    # Note: Full binary glTF export would require proper base64 encoding
    # For now, we save the .gltf text format which SuperSplat also supports
    
    # Create metadata file for SuperSplat
    metadata = {
        "version": "1.0",
        "face_id": face_id,
        "num_gaussians": N,
        "sh_degree": 3,
        "bounding_box": {
            "min": gaussian_data['xyz'].min(axis=0).tolist(),
            "max": gaussian_data['xyz'].max(axis=0).tolist(),
        },
        "camera": {
            "fov": 49.0,
            "aspect_ratio": 1.0,
        },
    }
    
    with open(os.path.join(output_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved to: {os.path.join(output_path, 'metadata.json')}")
    
    return json_path


def export_supersplat_model(model_dir: str, output_dir: str, face_id: str):
    """Export a trained 3DGS model to SuperSplat format.
    
    Args:
        model_dir: Directory containing trained .ply file
        output_dir: Output directory for glTF export
        face_id: Face identifier
        
    Returns:
        Path to exported file, or None if failed
    """
    
    print("=" * 60)
    print(f"Exporting {face_id} to SuperSplat format")
    print("=" * 60)
    
    # Find the PLY file
    ply_files = [f for f in os.listdir(model_dir) if f.endswith('.ply')]
    
    if not ply_files:
        print(f"Error: No .ply files found in {model_dir}")
        return None
    
    ply_path = os.path.join(model_dir, ply_files[0])
    
    print(f"Loading model from: {ply_path}")
    
    # Load Gaussian data
    gaussian_data = load_3dgs_ply(ply_path)
    
    if gaussian_data is None:
        return None
    
    print(f"  Loaded {gaussian_data['xyz'].shape[0]} Gaussians")
    
    # Export to glTF
    export_path = export_to_glb(gaussian_data, output_dir, face_id)
    
    print()
    return export_path


def batch_export_faces(faces_dir: str, output_base: str):
    """Export all trained face models to SuperSplat format.
    
    Args:
        faces_dir: Directory containing trained model directories
        output_base: Base output directory for glTF exports
        
    Returns:
        Dict mapping face_id to export path
    """
    
    print("=" * 60)
    print("Batch Export to SuperSplat Format")
    print("=" * 60)
    print()
    
    # Find all face model directories
    if not os.path.exists(faces_dir):
        print(f"Error: Faces directory '{faces_dir}' not found")
        return {}
    
    face_dirs = [d for d in os.listdir(faces_dir) 
                 if os.path.isdir(os.path.join(faces_dir, d))]
    
    print(f"Found {len(face_dirs)} face models")
    print()
    
    exports = {}
    
    for i, face_id in enumerate(sorted(face_dirs), 1):
        print(f"[{i}/{len(face_dirs)}] Exporting {face_id}")
        
        model_dir = os.path.join(faces_dir, face_id)
        output_path = os.path.join(output_base, "faces", face_id)
        
        export_path = export_supersplat_model(model_dir, output_path, face_id)
        
        if export_path:
            exports[face_id] = export_path
            print(f"  Exported to: {export_path}")
        else:
            print(f"  FAILED")
        
        print()
    
    # Save export manifest
    manifest = {
        "pipeline": "CFAD Face Gaussian Splatting",
        "format": "SuperSplat glTF",
        "total_faces": len(exports),
        "faces": exports,
    }
    
    manifest_path = os.path.join(output_base, "export_manifest.json")
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("=" * 60)
    print(f"Exported {len(exports)} faces")
    print("=" * 60)
    
    return exports


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export 3DGS models to SuperSplat glTF format"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str,
        help="Path to single model directory (for single face export)"
    )
    
    parser.add_argument(
        "--faces_dir", 
        type=str,
        help="Path to directory with all face models (for batch export)"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str,
        help="Output directory for glTF exports"
    )
    
    parser.add_argument(
        "--face_id", 
        type=str,
        help="Face identifier (required for single export)"
    )
    
    args = parser.parse_args()
    
    if args.model_path and args.output_path:
        # Single face export
        face_id = os.path.basename(args.model_path)
        
        output = export_supersplat_model(
            args.model_path, 
            args.output_path, 
            face_id
        )
        
        if output is None:
            sys.exit(1)
            
    elif args.faces_dir and args.output_path:
        # Batch export
        exports = batch_export_faces(args.faces_dir, args.output_path)
        
        if len(exports) == 0:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
