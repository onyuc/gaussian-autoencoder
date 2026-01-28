"""
YAML Configuration Loader

Load and merge YAML configs with command-line arguments
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML config file
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Config dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge config with command-line args (args override config)
    
    Args:
        config: Config dictionary from YAML
        args: Command-line arguments
        
    Returns:
        Merged namespace with config defaults + arg overrides
    """
    merged = argparse.Namespace()
    
    # Data config
    merged.voxel_size = config['data']['voxel_size']
    merged.max_level = config['data']['max_level']
    merged.min_gaussians = config['data'].get('min_gaussians', 8)
    merged.max_gaussians = config['data']['max_gaussians']
    merged.val_split = config['data']['val_split']
    
    # Model config
    merged.input_dim = config['model']['input_dim']
    merged.latent_dim = config['model']['latent_dim']
    merged.num_queries = config['model']['num_queries']
    merged.nhead = config['model']['nhead']
    merged.num_enc_layers = config['model']['num_enc_layers']
    merged.num_dec_layers = config['model']['num_dec_layers']
    
    # Training config
    merged.batch_size = config['training']['batch_size']
    merged.epochs = config['training']['epochs']
    merged.lr = config['training']['lr']
    merged.weight_decay = config['training']['weight_decay']
    merged.grad_clip = config['training']['grad_clip']
    
    # Loss config
    merged.loss_type = config['loss']['type']
    merged.lambda_density = config['loss']['lambda_density']
    merged.lambda_render = config['loss']['lambda_render']
    merged.lambda_sparsity = config['loss']['lambda_sparsity']
    merged.n_density_samples = config['loss']['n_density_samples']
    merged.render_resolution = config['loss']['render_resolution']
    merged.warmup_iterations = config['loss'].get('warmup_iterations', 200)
    
    # Output config
    merged.save_dir = config['output']['save_dir']
    merged.save_interval = config['output']['save_interval']
    merged.debug_save_interval = config['output']['debug_save_interval']
    
    # Override with command-line args (if provided and not None)
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            setattr(merged, key, value)
    
    # Copy args-only parameters (not in config)
    merged.ply = args.ply
    merged.cache_dir = getattr(args, 'cache_dir', None)
    merged.force_rebuild = getattr(args, 'force_rebuild', False)
    merged.device = getattr(args, 'device', 'cuda')
    merged.resume = getattr(args, 'resume', None)
    merged.config = getattr(args, 'config', None)
    
    return merged


def get_default_config_path() -> Path:
    """Get default config path relative to package"""
    return Path(__file__).parent.parent.parent / "configs" / "default.yaml"
