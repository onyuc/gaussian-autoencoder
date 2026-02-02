"""
GS_Merge: Gaussian Splatting Compression via Learned Merging

Voxel 단위로 Gaussian들을 압축하는 Transformer 기반 AutoEncoder
"""

__version__ = "2.1.0"

from gs_merge.model import GaussianMergingAE
from gs_merge.data import GaussianData, load_ply, save_ply, OctreeVoxelizer, VoxelNode, VoxelDataset
from gs_merge.loss import GMAELoss, ChamferLoss, parse_gaussian_tensor, model_output_to_dict
from gs_merge.utils import denormalize_gaussian, merge_gaussians_from_voxels, save_merged_ply
from gs_merge.config import load_config, merge_config_with_args, get_training_parser, parse_args_with_config
from gs_merge.training import Trainer, GumbelNoiseScheduler, CompressionRatioScheduler

__all__ = [
    # Model
    "GaussianMergingAE",
    # Data
    "GaussianData",
    "load_ply",
    "save_ply",
    "OctreeVoxelizer",
    "VoxelNode",
    "VoxelDataset",
    # Loss
    "GMAELoss",
    "ChamferLoss",
    "parse_gaussian_tensor",
    "model_output_to_dict",
    # Utils
    "denormalize_gaussian",
    "merge_gaussians_from_voxels",
    "save_merged_ply",
    # Config
    "load_config",
    "merge_config_with_args",
    "get_training_parser",
    "parse_args_with_config",
    # Training
    "Trainer",
    "GumbelNoiseScheduler",
    "CompressionRatioScheduler",
]


