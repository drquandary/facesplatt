"""
Scene - Camera and dataset management

Handles loading and managing camera poses from COLMAP/NeRF++ format.
"""

import os
import glob
import torch
import numpy as np
from pathlib import Path
import cv2


class Camera:
    """Represents a single camera view with pose and intrinsics."""

    def __init__(self, colmap_id, R, T, FoVx, Fovy, image, gt_image, 
                 image_width, image_height, fx, fy, device='cpu'):
        self.id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.Fovy = Fovy
        self.image = image
        self.gt_image = gt_image
        self.image_width = image_width
        self.image_height = image_height
        self.fx = fx
        self.fy = fy
        
        self.device = device

    def original_image(self):
        return self.gt_image.to(self.device)


class Scene:
    """Manages a scene's cameras and dataset."""

    def __init__(self, args, gaussians=None):
        self.args = args
        self.gaussians = gaussians
        
        self.train_cameras = []
        self.test_cameras = []
        
        self.image_iter = None
        
        self.load_cameras()

    def load_cameras(self):
        """Load cameras from COLMAP or NeRF++ format."""
        source_path = self.args.source_path
        
        cameras_file = os.path.join(source_path, "sparse", "0", "cameras.bin")
        images_file = os.path.join(source_path, "sparse", "0", "images.bin")
        
        if os.path.exists(cameras_file) and os.path.exists(images_file):
            print("Loading COLMAP format")
            self._load_colmap(source_path)
        else:
            print("Loading NeRF++ format")
            self._load_nerfpp(source_path)

    def _load_colmap(self, source_path):
        """Load cameras from COLMAP binary format."""
        import struct
        
        cameras = {}
        with open(os.path.join(source_path, "sparse", "0", "cameras.bin"), 'rb') as f:
            num_cameras = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_cameras):
                camera_id = struct.unpack('i', f.read(4))[0]
                camera_type = struct.unpack('i', f.read(4))[0]
                
                if camera_type == 1:
                    width = struct.unpack('Q', f.read(8))[0]
                    height = struct.unpack('Q', f.read(8))[0]
                    fx = struct.unpack('d', f.read(8))[0]
                    fy = struct.unpack('d', f.read(8))[0]
                    cx = struct.unpack('d', f.read(8))[0]
                    cy = struct.unpack('d', f.read(8))[0]
                    
                    cameras[camera_id] = {
                        'width': width, 'height': height,
                        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                    }

        with open(os.path.join(source_path, "sparse", "0", "images.bin"), 'rb') as f:
            num_images = struct.unpack('i', f.read(4))[0]
            
            for _ in range(num_images):
                is_registered = struct.unpack('i', f.read(4))[0]
                image_id = struct.unpack('i', f.read(4))[0]
                
                if not is_registered:
                    continue
                
                f.read(32 + 24)
                
                camera_id = struct.unpack('i', f.read(4))[0]
                name_parts = f.readline().strip().split()
                image_name = name_parts[0] if name_parts else ""
                
                cam_info = cameras.get(camera_id)
                if not cam_info:
                    continue
                
                image_path = os.path.join(source_path, self.args.images, image_name)
                
                if not os.path.exists(image_path):
                    continue
                
                img = cv2.imread(image_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                
                fov_x = 2 * np.arctan(cam_info['width'] / (2 * cam_info['fx']))
                fov_y = 2 * np.arctan(cam_info['height'] / (2 * cam_info['fy']))
                
                cam = Camera(
                    colmap_id=image_id,
                    R=np.eye(3),
                    T=np.zeros(3),
                    FoVx=fov_x,
                    Fovy=fov_y,
                    image=None,
                    gt_image=img,
                    image_width=cam_info['width'],
                    image_height=cam_info['height'],
                    fx=cam_info['fx'],
                    fy=cam_info['fy'],
                )
                
                self.train_cameras.append(cam)

    def _load_nerfpp(self, source_path):
        """Load cameras from NeRF++ format."""
        import json
        
        transforms_file = os.path.join(source_path, "transforms_train.json")
        
        if not os.path.exists(transforms_file):
            print(f"Warning: transforms_train.json not found at {transforms_file}")
            return
        
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
        
        img_dir = os.path.join(source_path, self.args.images)
        
        for frame in transforms['frames']:
            file_name = frame['file_path'].split('/')[-1]
            image_path = os.path.join(img_dir, file_name)
            
            if not os.path.exists(image_path):
                continue
            
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            
            transform = np.array(frame['transform_matrix'])
            
            R = transform[:3, :3].T
            T = -R @ transform[:3, 3]
            
            fx = fy = 0.5 * img.shape[2] / np.tan(0.5 * transforms.get('camera_angle_x', 1.0))
            
            fov_x = 2 * np.arctan(img.shape[2] / (2 * fx))
            fov_y = 2 * np.arctan(img.shape[1] / (2 * fy))
            
            cam = Camera(
                colmap_id=0,
                R=R.astype(np.float32),
                T=T.astype(np.float32),
                FoVx=fov_x,
                Fovy=fov_y,
                image=None,
                gt_image=img,
                image_width=img.shape[2],
                image_height=img.shape[1],
                fx=fx,
                fy=fy,
            )
            
            self.train_cameras.append(cam)

    def getTrainCameras(self, scale=0):
        """Get list of training cameras."""
        return self.train_cameras

    def getTestCameras(self, scale=0):
        """Get list of test cameras."""
        return self.test_cameras

    @property
    def dataset_params(self):
        return self._dataset_params

    @dataset_params.setter
    def dataset_params(self, params):
        self._dataset_params = params
