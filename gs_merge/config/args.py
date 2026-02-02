"""
Command-line Argument Parser

Centralized argparse configuration for training scripts
"""

import argparse
from typing import Optional


def get_training_parser() -> argparse.ArgumentParser:
    """
    Get argument parser for training script
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train Gaussian Merging AutoEncoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--ply", 
        type=str, 
        required=True, 
        help="Path to input PLY file with Gaussian point cloud"
    )
    
    # Config
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to YAML config file (uses default if not specified)"
    )
    
    # Data arguments
    data_group = parser.add_argument_group('data', 'Data loading and voxelization')
    data_group.add_argument(
        "--voxel_size", 
        type=float, 
        default=None,
        help="Base voxel size for octree level 1"
    )
    data_group.add_argument(
        "--max_level", 
        type=int, 
        default=None,
        help="Maximum octree subdivision level"
    )
    data_group.add_argument(
        "--max_gaussians", 
        type=int, 
        default=None,
        help="Maximum Gaussians per voxel"
    )
    data_group.add_argument(
        "--val_split", 
        type=float, 
        default=None,
        help="Validation split ratio (0.0-1.0)"
    )
    data_group.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Directory to cache voxelized data"
    )
    data_group.add_argument(
        "--force_rebuild", 
        action="store_true",
        help="Force rebuild voxel cache"
    )
    
    # Model arguments
    model_group = parser.add_argument_group('model', 'Model architecture')
    model_group.add_argument(
        "--num_queries", 
        type=int, 
        default=None,
        help="Number of output Gaussians (compressed count)"
    )
    model_group.add_argument(
        "--latent_dim", 
        type=int, 
        default=None,
        help="Latent dimension for transformer"
    )
    model_group.add_argument(
        "--nhead", 
        type=int, 
        default=None,
        help="Number of attention heads"
    )
    model_group.add_argument(
        "--num_enc_layers", 
        type=int, 
        default=None,
        help="Number of encoder layers"
    )
    model_group.add_argument(
        "--num_dec_layers", 
        type=int, 
        default=None,
        help="Number of decoder layers"
    )
    
    # Gumbel Noise Scheduling arguments
    gumbel_group = parser.add_argument_group('gumbel', 'Gumbel noise scheduling for query selection')
    gumbel_group.add_argument(
        "--gumbel_initial_scale", 
        type=float, 
        default=None,
        help="Initial gumbel noise scale (higher = more exploration)"
    )
    gumbel_group.add_argument(
        "--gumbel_final_scale", 
        type=float, 
        default=None,
        help="Final gumbel noise scale (0 = deterministic selection)"
    )
    gumbel_group.add_argument(
        "--gumbel_warmup_epochs", 
        type=int, 
        default=None,
        help="Epochs to keep initial noise scale before annealing"
    )
    gumbel_group.add_argument(
        "--gumbel_schedule_type", 
        type=str, 
        default=None,
        choices=["linear", "cosine", "exponential", "warmup_cosine", "constant"],
        help="Type of noise annealing schedule"
    )
    
    # Compression Ratio arguments
    compression_group = parser.add_argument_group('compression', 'Compression ratio scheduling')
    compression_group.add_argument(
        "--compression_ratio_min", 
        type=float, 
        default=None,
        help="Minimum compression ratio for random sampling (0.0 ~ 1.0)"
    )
    compression_group.add_argument(
        "--compression_ratio_max", 
        type=float, 
        default=None,
        help="Maximum compression ratio for random sampling (0.0 ~ 1.0)"
    )
    
    # Training arguments
    train_group = parser.add_argument_group('training', 'Training hyperparameters')
    train_group.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Batch size for training"
    )
    train_group.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--lr", 
        type=float, 
        default=None,
        help="Learning rate"
    )
    train_group.add_argument(
        "--weight_decay", 
        type=float, 
        default=None,
        help="Weight decay (L2 regularization)"
    )
    
    # Loss arguments
    loss_group = parser.add_argument_group('loss', 'Loss function configuration')
    loss_group.add_argument(
        "--loss_type", 
        type=str, 
        default=None,
        choices=["gmae", "chamfer"],
        help="Loss function type"
    )
    loss_group.add_argument(
        "--lambda_density", 
        type=float, 
        default=None,
        help="Weight for density KL divergence loss"
    )
    loss_group.add_argument(
        "--lambda_render", 
        type=float, 
        default=None,
        help="Weight for rendering loss"
    )
    loss_group.add_argument(
        "--lambda_sparsity", 
        type=float, 
        default=None,
        help="Weight for sparsity regularization"
    )
    loss_group.add_argument(
        "--render_resolution", 
        type=int, 
        default=None,
        help="Resolution for rendering loss (NxN)"
    )
    loss_group.add_argument(
        "--warmup_iterations", 
        type=int, 
        default=None,
        help="Number of warmup iterations for rendering loss annealing"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('output', 'Checkpoints and logging')
    output_group.add_argument(
        "--save_dir", 
        type=str, 
        default=None,
        help="Directory to save checkpoints"
    )
    output_group.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # System arguments
    sys_group = parser.add_argument_group('system', 'System configuration')
    sys_group.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )
    
    return parser


def parse_args_with_config(
    parser: Optional[argparse.ArgumentParser] = None,
    args_list: Optional[list] = None
) -> argparse.Namespace:
    """
    Parse arguments with automatic config loading
    
    Args:
        parser: Custom parser (uses default if None)
        args_list: List of args to parse (uses sys.argv if None)
        
    Returns:
        Parsed and merged arguments
    """
    from .loader import load_config, merge_config_with_args, get_default_config_path
    
    if parser is None:
        parser = get_training_parser()
    
    args = parser.parse_args(args_list)
    
    # Load config if provided, otherwise use default
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
    else:
        default_config_path = get_default_config_path()
        if default_config_path.exists():
            print(f"Loading default config from: {default_config_path}")
            config = load_config(str(default_config_path))
            args = merge_config_with_args(config, args)
        else:
            print("⚠ No config file found, using command-line arguments only")
            # Set fallback defaults
            _set_fallback_defaults(args)
    
    return args


def _set_fallback_defaults(args: argparse.Namespace) -> None:
    """Set hardcoded fallback defaults if no config available"""
    defaults = {
        'voxel_size': 100.0,
        'max_level': 16,
        'min_gaussians': 8,
        'max_gaussians': 128,
        'num_queries': 32,
        'input_dim': 59,
        'latent_dim': 256,
        'nhead': 8,
        'num_enc_layers': 4,
        'num_dec_layers': 4,
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'val_split': 0.1,
        'loss_type': 'gmae',
        'lambda_density': 1.0,
        'lambda_render': 2.0,
        'lambda_sparsity': 0.01,
        'n_density_samples': 1024,
        'render_resolution': 64,
        'warmup_iterations': 200,
        'save_dir': './checkpoints',
        'save_interval': 10,
        'debug_save_interval': 100,
        # Gumbel noise scheduling defaults
        'gumbel_initial_scale': 0.5,
        'gumbel_final_scale': 0.0,
        'gumbel_warmup_epochs': 10,
        'gumbel_schedule_type': 'warmup_cosine',
        # Compression ratio defaults
        'compression_ratio_min': 0.1,
        'compression_ratio_max': 1.0,
    }
    
    for key, value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
