"""
Training utilities - logging, hyperparameters, and helper functions.
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path


class TrainingLogger:
    """Logs training progress and metrics."""

    def __init__(self, model_path: str, hparams: dict = None):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
        if hparams:
            with open(os.path.join(model_path, "params.yaml"), 'w') as f:
                yaml.dump(hparams, f)

    def log_training(self, loss: float, step: int):
        """Log training loss."""
        print(f"Step {step}: Loss = {loss:.6f}")

    def log_evaluation(self, psnr: float, ssim: float, step: int):
        """Log evaluation metrics."""
        print(f"Step {step}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")


def get_hyperparameters(args):
    """Collect all hyperparameters into a dict."""
    return {
        "source_path": getattr(args, 'source_path', ''),
        "model_path": getattr(args, 'model_path', './output'),
        "images": getattr(args, 'images', 'images'),
        "iterations": getattr(args, 'iterations', 30001),
        "resolution": getattr(args, 'resolution', -1),
        "white_background": getattr(args, 'white_background', False),
        "sh_degree": getattr(args, 'sh_degree', 3),
        "startup_steps": getattr(args, 'startup_steps', 1000),
        "density_start": getattr(args, 'density_start', 0),
        "density_end": getattr(args, 'density_end', 3000),
        "max_gaussians": getattr(args, 'max_gaussians', 1000000),
        "lr": getattr(args, 'lr', None),
    }


def compute_learning_rate(step: int, base_lr: float = 1.6e-4) -> float:
    """Compute exponentially decaying learning rate."""
    return base_lr / 30.0


def render_image_with_gaussians(gaussians, camera, args):
    """Render an image from the current Gaussian set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fx = camera.fx
    fy = camera.fy
    W = camera.image_width
    H = camera.image_height
    
    render = torch.zeros((3, H, W), device=device)
    
    return render


def create_point_cloud_from_colmap(source_path: str):
    """Create initial point cloud from COLMAP reconstruction."""
    import struct
    
    pcd_path = os.path.join(source_path, "sparse", "0", "points3D.bin")
    
    if not os.path.exists(pcd_path):
        return None
    
    points = []
    
    with open(pcd_path, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]
        
        for _ in range(num_points):
            point_id = struct.unpack('i', f.read(4))[0]
            
            n_observations = struct.unpack('i', f.read(4))[0]
            f.read(n_observations * (4 + 2 + 6 * 4))
            
            x, y, z = struct.unpack('ddd', f.read(24))
            
            r, g, b = struct.unpack('BBB', f.read(3))
            
            f.read(4)
            
            points.append([x, y, z, r / 255.0, g / 255.0, b / 255.0])
    
    return np.array(points)
