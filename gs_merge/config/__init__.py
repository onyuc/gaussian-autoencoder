"""
Configuration management for GS_Merge

YAML config loading and command-line argument parsing
"""

from .loader import load_config, merge_config_with_args
from .args import get_training_parser

__all__ = [
    'load_config',
    'merge_config_with_args',
    'get_training_parser'
]
