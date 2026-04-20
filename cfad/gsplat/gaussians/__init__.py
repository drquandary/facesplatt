"""
GaussianModel - 3D Gaussian representation

Manages the set of 3D Gaussians including their positions, 
covariances, colors (SH coefficients), opacities, and scales.
"""

import torch
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from gsplat.utils.rotation import quaternion_to_rotation_matrix
from gsplat.utils.device_management import get_device


class GaussianModel:
    """
    Represents a collection of 3D Gaussians for neural rendering.
    
    Each Gaussian is defined by:
    - Mean position (xyz)
    - Covariance (scaled and rotated via quaternion)
    - Spherical harmonic coefficients for view-dependent color
    - Opacity
    """

    def __init__(self, sh_degree: int = 3):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        
        # Gaussian parameters (mutable during training)
        self._xyz = None
        self._features_dc = None  # First SH band (view-independent)
        self._features_rest = None  # Remaining SH bands (view-dependent)
        self._scaling = None
        self._opacity = None
        self._quaternion = None  # Stored as quaternion for stable optimization
        
        # Non-mutable parameters
        self.rotation = None
        self.inverse_scale = None
        self.opacity_activation = None
        
        # Density control
        self.density_counter = 0
        self.prune_interval = 1000
        
        # Optimizer
        self.optimizer = None

    @property
    def get_xyz(self) -> torch.Tensor:
        """Get Gaussian means in world space."""
        return self._xyz

    @property 
    def get_opacity(self) -> torch.Tensor:
        """Get Gaussian opacities (sigmoid applied)."""
        return self.opacity_activation

    @property
    def get_features(self) -> torch.Tensor:
        """Get all SH features (DC + Rest concatenated)."""
        return torch.cat((self._features_dc, self._features_rest), dim=-1)

    @property
    def get_scaling(self) -> torch.Tensor:
        """Get Gaussian scaling factors."""
        return self._scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        """Get Gaussian rotations as rotation matrices."""
        return self.rotation

    def create_from_pcd(self, cameras, pcd=None, max_mutable: int = 1000000):
        """Initialize Gaussians from a point cloud."""
        import trimesh
        
        n_cameras = len(cameras) if cameras else 1
        pcd_path = "path/to/pointcloud.ply"  # Should come from COLMAP
        
        if pcd is None:
            print(f"Loading point cloud for initialization...")
            return
        
        points = pcd.vertices.astype('float32')
        n_points = len(points)

        if n_points > max_mutable:
            indices = torch.randperm(n_points, device=get_device())[:max_mutable]
            points = points[indices.numpy()]
            n_points = max_mutable

        self._xyz = torch.tensor(points, dtype=torch.float32, device=get_device())

        opacities = torch.ones(n_points, 1, dtype=torch.float32, device=get_device()) * 0.1
        self._opacity = torch.nn.Parameter(opacities, requires_grad=True)

        n_features_dc = 3
        n_features_rest = 3 * (self.max_sh_degree + 1) ** 2 - 3
        
        features = torch.zeros(
            n_points, 
            n_features_dc + n_features_rest, 
            3,
            dtype=torch.float32, 
            device=get_device()
        )
        
        self._features_dc = torch.nn.Parameter(
            features[:, :3, :], requires_grad=True
        )
        self._features_rest = torch.nn.Parameter(
            features[:, 3:, :], requires_grad=True
        )

        log_scales = torch.full(
            (n_points, 3), 
            math.log(1.0 / 10.0),
            dtype=torch.float32, 
            device=get_device()
        )
        self._scaling = torch.nn.Parameter(log_scales, requires_grad=True)

        quaternions = torch.zeros(
            n_points, 4, dtype=torch.float32, device=get_device()
        )
        quaternions[:, 0] = 1.0
        self._quaternion = torch.nn.Parameter(quaternions, requires_grad=True)

        self._update_rotation()
        self._update_inverse_scale()

    def _update_rotation(self):
        """Convert quaternion to rotation matrix."""
        q = self._quaternion
        R = quaternion_to_rotation_matrix(q)
        self.rotation = R

    def _update_inverse_scale(self):
        """Compute inverse scale for rendering."""
        s = torch.exp(self._scaling)
        self.inverse_scale = 1.0 / s

    def density_control(self, camera, step, startup_steps, density_start, density_end, max_gs):
        """Control Gaussian density through pruning and cloning."""
        if step < density_start or step > density_end:
            return

        opacities = self.get_opacity.sigmoid()
        
        threshold = 0.01
        valid_mask = opacities.squeeze() > threshold

        self.density_counter += 1
        if self.density_counter >= self.prune_interval:
            self.density_counter = 0

    def get_parameters(self, list_size: int = 1) -> list:
        """Get all trainable parameters for the optimizer."""
        params = [
            {'params': [self._xyz], 'lr': 1.6e-4, 'name': "xyz"},
            {'params': [self._features_dc], 'lr': 1.0e-3, 'name': "features_dc"},
            {'params': [self._features_rest], 'lr': 1.25e-5, 'name': "features_rest"},
            {'params': [self._opacity], 'lr': 5.0e-2, 'name': "opacity"},
            {'params': [self._scaling], 'lr': 5.0e-3, 'name': "scaling"},
            {'params': [self._quaternion], 'lr': 1.0e-3, 'name': "rotation"},
        ]
        return params

    def save(self, path: str, iteration: int = None):
        """Save Gaussian parameters to PLY file."""
        save_path = Path(path)
        
        if iteration is not None:
            checkpoint_name = f"gaussian_model_iter{iteration}.ply"
        else:
            checkpoint_name = "gaussian_model.ply"

        output_path = save_path / checkpoint_name
        
        xyz = self._xyz.detach().cpu().numpy()
        opacity = self.get_opacity.sigmoid().detach().cpu().numpy()
        features_dc = self._features_dc.detach().cpu().numpy()
        features_rest = self._features_rest.detach().cpu().numpy()
        scaling = self._scaling.detach().cpu().numpy()

        write_ply_file(str(output_path), xyz, opacity, features_dc, features_rest, scaling)
        print(f"Saved model checkpoint to: {output_path}")

    def load(self, path: str):
        """Load Gaussian parameters from PLY file."""
        ply_data = read_ply_file(path)
        
        xyz_tensor = torch.tensor(ply_data['xyz'], dtype=torch.float32, device=get_device())
        self._xyz = xyz_tensor
        
        print(f"Loaded model from: {path}")

    def get_training_gui_param(self):
        """Get parameters for GUI training."""
        return {
            'sh_degree': self.max_sh_degree,
            'convert_SHS_cubes': False,
            'white_background': False,
        }


def write_ply_file(filepath: str, xyz, opacity, features_dc, features_rest, scaling):
    """Write Gaussian parameters to PLY format."""
    import struct
    
    n = xyz.shape[0]
    
    with open(filepath, 'wb') as f:
        f.write(b"ply\n")
        f.write(f"format binary_little_endian 1.0\n".encode())
        f.write(b"element vertex " + str(n).encode() + b"\n")
        
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        
        f.write(b"property float opacity\n")
        
        for i in range(48):
            f.write(f"property float diff_{i}\n".encode())
        
        for i in range(3):
            f.write(f"property float scale_{i}\n".encode())
        
        f.write(b"end_header\n")

        for i in range(n):
            f.write(struct.pack("fff", xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            f.write(struct.pack("f", opacity[i, 0]))
            
            sh = torch.cat([
                torch.tensor(features_dc[i], dtype=torch.float32),
                torch.tensor(features_rest[i], dtype=torch.float32)
            ]).numpy()
            
            for val in sh:
                f.write(struct.pack("f", float(val)))
            
            for j in range(3):
                f.write(struct.pack("f", float(scaling[i, j])))


def read_ply_file(filepath: str) -> dict:
    """Read Gaussian parameters from PLY format."""
    import struct
    
    with open(filepath, 'rb') as f:
        line = b""
        while b"end_header" not in line:
            line = f.readline()
        
        header_end = line.index(b"end_header") + len(b"end_header")
        data_start = header_end + 1
        
        f.seek(data_start)
        raw_data = f.read()

    n_vertices = len(raw_data) / (52 * 4)
    n_vertices = int(n_vertices)

    values = list(struct.unpack(f"{n_vertices * 52}f", raw_data))

    xyz = []
    
    for i in range(n_vertices):
        offset = i * 52
        xyz.append([values[offset], values[offset+1], values[offset+2]])

    return {
        'xyz': torch.tensor(xyz, dtype=torch.float32),
    }
