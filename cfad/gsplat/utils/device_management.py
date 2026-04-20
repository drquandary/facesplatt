"""
Device management utilities for Apple Silicon (Metal) support.

Provides device detection and selection for optimal hardware usage.
"""

import torch


def get_device() -> str:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def get_torch_device(device_name: str = None) -> torch.device:
    """Get a PyTorch device object."""
    if device_name is None:
        device_name = get_device()
    
    return torch.device(device_name)


def is_metal_available() -> bool:
    """Check if Metal (Apple Silicon GPU) is available."""
    return torch.backends.mps.is_available()


def is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA GPU) is available."""
    return torch.cuda.is_available()


def select_device(prefer_metal: bool = True) -> str:
    """Select the best device, preferring Metal on Macs."""
    if prefer_metal and is_metal_available():
        return 'mps'
    elif is_cuda_available():
        return 'cuda'
    else:
        return 'cpu'


class DeviceManager:
    """Manages device selection and tensor placement."""

    def __init__(self, prefer_metal: bool = True):
        self.device_name = select_device(prefer_metal)
        self.device = torch.device(self.device_name)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the selected device."""
        return tensor.to(self.device)

    def create_tensor(self, *args, **kwargs) -> torch.Tensor:
        """Create a tensor on the selected device."""
        return torch.tensor(*args, device=self.device, **kwargs)

    @property
    def is_mps(self):
        return self.device_name == 'mps'

    @property
    def is_cuda(self):
        return self.device_name == 'cuda'

    @property
    def is_cpu(self):
        return self.device_name == 'cpu'
