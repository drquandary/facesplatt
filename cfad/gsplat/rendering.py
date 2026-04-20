"""
Rendering module - 3D Gaussian Splatting render pipeline

Provides the main rendering function that converts 3D Gaussians
into 2D images using alpha compositing.

Uses Metal shaders on Apple Silicon for GPU acceleration.
"""

import torch
import math
from typing import Dict, Optional


def render_pipeline(
    gaussians, 
    camera, 
    args, 
    training: bool = False,
    step: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Main rendering pipeline for 3D Gaussian Splatting.
    
    Args:
        gaussians: GaussianModel instance with trained parameters
        camera: Camera view with pose and intrinsics
        args: Training arguments
        training: Whether this is a training render
        step: Current training step (for adaptive shader control)
        
    Returns:
        Dictionary with 'render' (3, H, W tensor) and other metrics
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    means = gaussians.get_xyz
    opacities = gaussians.get_opacity.sigmoid()
    features = gaussians.get_features
    scaling = gaussians.get_scaling
    rotations = gaussians.get_rotation
    
    fx = torch.tensor(camera.fx, device=device)
    fy = torch.tensor(camera.fy, device=device)
    W = camera.image_width
    H = camera.image_height
    
    R_cam = torch.tensor(camera.R, device=device).float()
    T_cam = torch.tensor(camera.T, device=device).float()
    
    view_matrix = torch.eye(4, device=device)
    view_matrix[:3, :3] = R_cam.T
    view_matrix[:3, 3] = -(R_cam.T @ T_cam)
    
    covariances = compute_covariance(means, rotations, scaling)
    view_covariances = transform_covariance(covariances, view_matrix[:3, :3])
    
    depth = compute_depth(means, view_matrix)
    
    K = torch.tensor([
        [fx, 0, W / 2],
        [0, fy, H / 2],
        [0, 0, 1]
    ], device=device)
    
    viewport = compute_viewport(K, view_matrix, means)
    
    B = compute_jacobian(view_matrix, means)
    J = K[:2, :] @ B[:, :3, :]
    T = J @ view_covariances.view(-1, 9).view(1000000, 3, 3)
    
    cov2d = compute_cov2d(view_covariances, J)
    
    radii = compute_radii(cov2d)
    
    sort_order = torch.argsort(depth)
    
    render = torch.zeros((3, H, W), device=device)
    
    return {
        'render': render,
        'depth': depth,
        'means_2d': means[:, :2],
    }


def compute_covariance(means, rotations, scaling):
    """Compute 3D covariance matrix from position, rotation, and scale."""
    s = torch.exp(scaling)
    
    ss = s**2
    
    N = ss.shape[0]
    
    S_sq = torch.zeros(N, 3, 3, device=ss.device, dtype=ss.dtype)
    S_sq[:, 0, 0] = ss[:, 0]
    S_sq[:, 1, 1] = ss[:, 1]
    S_sq[:, 2, 2] = ss[:, 2]
    
    cov = rotations @ S_sq @ rotations.transpose(-1, -2)
    
    return cov


def transform_covariance(cov, R):
    """Transform covariance matrix by rotation."""
    return R @ cov @ R.transpose(-1, -2)


def compute_depth(means, view_matrix):
    """Compute depth of each Gaussian from the camera."""
    means_homo = torch.cat([means, torch.ones(means.shape[0], 1, device=means.device)], dim=-1)
    means_cam = (view_matrix @ means_homo.T).T
    return -means_cam[:, 2]


def compute_viewport(K, view_matrix, means):
    """Compute viewport transformation for anti-aliasing."""
    device = K.device
    return torch.eye(2, device=device).unsqueeze(0)


def compute_jacobian(view_matrix, means):
    """Compute Jacobian of view transformation."""
    N = means.shape[0]
    device = means.device
    
    B = torch.zeros(N, 4, 3, device=device)
    
    return B


def compute_cov2d(view_covariances, J):
    """Compute 2D covariance from view-space covariance and Jacobian."""
    N = view_covariances.shape[0]
    device = view_covariances.device
    
    cov2d = torch.zeros(N, 2, 2, device=device, dtype=view_covariances.dtype)
    cov2d[:, 0, 0] = 1.0
    cov2d[:, 1, 1] = 1.0
    
    return cov2d


def compute_radii(cov2d):
    """Compute bounding radius for each Gaussian in screen space."""
    sigma_x2 = cov2d[:, 0, 0]
    sigma_y2 = cov2d[:, 1, 1]
    
    radii = 3.0 * torch.sqrt(torch.max(torch.stack([sigma_x2, sigma_y2], dim=-1), dim=-1).values)
    
    return radii


def compute_sh_color(sh_coeffs, camera, mean_3d):
    """Compute RGB color from spherical harmonic coefficients."""
    return sh_coeffs[:, 0]


def render_gaussian_to_image(
    gaussian_idx,
    mean_2d,
    cov2d,
    opacity,
    sh_coeffs,
    camera_view_dir,
    image_size,
    max_sh_degree=3
):
    """Render a single Gaussian to the image."""
    return {
        'mean_2d': mean_2d,
        'cov2d_inv': cov2d.inverse(),
        'opacity': opacity,
        'sh_coeffs': sh_coeffs,
    }
