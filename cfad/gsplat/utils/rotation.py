"""
Rotation utilities for 3D Gaussian Splatting.

Provides quaternion-to-rotation-matrix conversion and related operations.
"""

import torch


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    return q / q.norm(dim=-1, keepdim=True)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to rotation matrix.
    
    Args:
        q: Quaternion tensor of shape (..., 4) with w first
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    q = normalize_quaternion(q)
    
    qw = q[..., 0]
    qx = q[..., 1]
    qy = q[..., 2]
    qz = q[..., 3]
    
    batch_dims = q.shape[:-1]
    
    qw2 = qw * qw
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    
    xy = qx * qy
    xz = qx * qz
    yw = qy * qw
    zw = qz * qw
    xw = qx * qw
    yz = qy * qz
    
    R = torch.zeros(*batch_dims, 3, 3, device=q.device, dtype=q.dtype)
    
    R[..., 0, 0] = qw2 + qx2 - qy2 - qz2
    R[..., 0, 1] = 2 * (xy + zw)
    R[..., 0, 2] = 2 * (xz - yw)
    
    R[..., 1, 0] = 2 * (xy - zw)
    R[..., 1, 1] = qw2 - qx2 + qy2 - qz2
    R[..., 1, 2] = 2 * (yz + xw)
    
    R[..., 2, 0] = 2 * (xz + yw)
    R[..., 2, 1] = 2 * (yz - xw)
    R[..., 2, 2] = qw2 - qx2 - qy2 + qz2
    
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    
    Args:
        R: Rotation matrix of shape (..., 3, 3)
        
    Returns:
        Quaternion tensor of shape (..., 4) with w first
    """
    batch_dims = R.shape[:-2]
    
    m00 = R[..., 0, 0]
    m01 = R[..., 0, 1]
    m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]
    m11 = R[..., 1, 1]
    m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]
    m21 = R[..., 2, 1]
    m22 = R[..., 2, 2]
    
    tr = m00 + m11 + m22
    
    q = torch.zeros(*batch_dims, 4, device=R.device, dtype=R.dtype)
    
    positive_mask = tr > 0
    
    if positive_mask.any():
        t = tr[positive_mask]
        s = torch.sqrt(t + 1.0) * 2
        q[positive_mask, 0] = s * 0.25
        q[positive_mask, 1] = (m21 - m12)[positive_mask] / s
        q[positive_mask, 2] = (m02 - m20)[positive_mask] / s
        q[positive_mask, 3] = (m10 - m01)[positive_mask] / s
    
    max0_mask = (~positive_mask) & (m00 > m11) & (m00 > m22)
    if max0_mask.any():
        s = torch.sqrt(1.0 + m00[max0_mask] - m11[max0_mask] - m22[max0_mask]) * 2
        q[max0_mask, 0] = (m12 - m21)[max0_mask] / s
        q[max0_mask, 1] = s * 0.25
        q[max0_mask, 2] = (m01 + m10)[max0_mask] / s
        q[max0_mask, 3] = (m02 + m20)[max0_mask] / s
    
    max1_mask = (~positive_mask) & (~max0_mask) & (m11 > m22)
    if max1_mask.any():
        s = torch.sqrt(1.0 + m11[max1_mask] - m00[max1_mask] - m22[max1_mask]) * 2
        q[max1_mask, 0] = (m20 - m02)[max1_mask] / s
        q[max1_mask, 1] = (m01 + m10)[max1_mask] / s
        q[max1_mask, 2] = s * 0.25
        q[max1_mask, 3] = (m12 + m21)[max1_mask] / s
    
    max2_mask = (~positive_mask) & (~max0_mask) & (~max1_mask)
    if max2_mask.any():
        s = torch.sqrt(1.0 + m22[max2_mask] - m00[max2_mask] - m11[max2_mask]) * 2
        q[max2_mask, 0] = (m01 - m10)[max2_mask] / s
        q[max2_mask, 1] = (m02 + m20)[max2_mask] / s
        q[max2_mask, 2] = (m12 + m21)[max2_mask] / s
        q[max2_mask, 3] = s * 0.25
    
    return normalize_quaternion(q)


def scale_to_quaternion(scale: torch.Tensor, rotation: torch.Tensor = None) -> torch.Tensor:
    """Convert scale + rotation to a single quaternion representation."""
    if rotation is None:
        batch_dims = scale.shape[:-1]
        q = torch.zeros(*batch_dims, 4, device=scale.device, dtype=scale.dtype)
        q[..., 0] = 1.0
        return q
    
    return rotation_matrix_to_quaternion(rotation)
