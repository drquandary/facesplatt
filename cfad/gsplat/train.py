"""
3D Gaussian Splatting - Training Script

Main training loop for optimizing 3D Gaussians from image sets.
Optimized for Apple Silicon using Metal backends.
"""

import argparse
import os
import sys
import time
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from gsplat.scenes import Scene
from gsplat.gaussians import GaussianModel
from gsplat.utils.training_utils import TrainingLogger, get_hyperparameters
from gsplat.rendering import render_pipeline


def load_args():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting model")

    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to training dataset (COLMAP/NeRF++ format)")
    parser.add_argument("--model_path", type=str, default="./output/model",
                        help="Path to save the trained model")
    parser.add_argument("--images", type=str, default="images",
                        help="Subdirectory containing training images")
    parser.add_argument("--evaluation_mode", type=str, default="render",
                        choices=["train", "test", "render"],
                        help="Evaluation mode")

    parser.add_argument("--iterations", type=int, default=30001,
                        help="Number of training iterations")
    parser.add_argument("--resolution", type=float, default=-1,
                        help="Resolution scale (-1 for original)")
    parser.add_argument("--white_background", action="store_true",
                        help="Use white background instead of black")

    parser.add_argument("--sh_degree", type=float, default=3,
                        help="Number of SH bands")
    parser.add_argument("--convert_shs", action="store_true",
                        help="Convert existing SHS to new format")
    parser.add_argument("--startup_steps", type=int, default=1000,
                        help="Steps before density updates")
    parser.add_argument("--density_start", type=int, default=0,
                        help="Starting step for density update")
    parser.add_argument("--density_end", type=int, default=3000,
                        help="Ending step for density update")
    parser.add_argument("--max_gaussians", type=int, default=1000000,
                        help="Maximum number of Gaussians")

    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--eval", action="store_true",
                        help="Enable evaluation mode")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")

    parser.add_argument("--no_metal", action="store_true",
                        help="Disable Metal backend (use CPU)")

    return parser.parse_args()


def training(args):
    """Main training loop."""
    
    os.makedirs(args.model_path, exist_ok=True)

    hparams = {
        "source_path": args.source_path,
        "model_path": args.model_path,
        "images": args.images,
        "iterations": args.iterations,
        "resolution": args.resolution,
        "white_background": args.white_background,
        "sh_degree": args.sh_degree,
        "startup_steps": args.startup_steps,
        "density_start": args.density_start,
        "density_end": args.density_end,
        "max_gaussians": args.max_gaussians,
    }

    print(f"Loading dataset: {args.source_path}")
    scene = Scene(args, gaussians=None)

    gaussians = GaussianModel(args.sh_degree)
    scene.dataset_params = gaussians.get_training_gui_param()

    print(f"Loading training data from: {os.path.join(args.source_path, args.images)}")
    train_iter = scene.getTrainCameras()

    gaussians.create_from_pcd(scene.train_cameras, max_mutable=args.max_gaussians)

    optimizer = torch.optim.Adam(gaussians.get_parameters(), lr=0.0, eps=1e-15)

    logger = TrainingLogger(args.model_path, hparams)

    print(f"Training started. Saving to: {args.model_path}")
    print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")

    first_iter = 0
    gaussians.optimizer = optimizer
    image_idx = 0

    for iteration in range(first_iter + 1, args.iterations + 1):
        try:
            viewpoint_cam = train_iter[image_idx % len(train_iter)]
            image_idx += 1

            render_results = render_pipeline(
                gaussians, viewpoint_cam, args, 
                training=True, 
                step=iteration
            )

            loss = compute_loss(render_results, viewpoint_cam)

            loss.backward()

            with torch.no_grad():
                lr = compute_lr(iteration, args)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()

                if iteration <= args.density_end:
                    gaussians.density_control(
                        viewpoint_cam, 
                        iteration,
                        args.startup_steps,
                        args.density_start,
                        args.density_end,
                        args.max_gaussians
                    )

        except Exception as e:
            print(f"Error at iteration {iteration}: {e}")
            continue

        if iteration % 1000 == 0 or iteration == 1:
            logger.log_training(loss.item, iteration)

        if iteration % 5000 == 0 or iteration == args.iterations:
            gaussians.save(args.model_path, iteration)

    print(f"Training complete. Model saved to: {args.model_path}")


def compute_loss(render_results, viewpoint_cam):
    """Compute training loss (L1 + SSIM)."""
    render_img = render_results['render']
    gt_image = viewpoint_cam.original_image()

    l1 = torchvision.ops.smooth_l1_loss(render_img, gt_image)

    ssim_loss = 1.0 - torchvision.ops.ssim(
        render_img.unsqueeze(0), 
        gt_image.unsqueeze(0),
        window_size=11,
        size_average=True
    )

    loss = (1.0 - ssim_loss) * 0.8 + l1 * 0.2
    return loss


def compute_lr(iteration, args):
    """Compute parametric learning rate."""
    base_lr = 1.6e-4
    return base_lr / 30


def main():
    args = load_args()
    training(args)


if __name__ == "__main__":
    main()
