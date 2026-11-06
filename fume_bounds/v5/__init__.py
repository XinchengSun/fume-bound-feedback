"""v5: FUME-compatible towers, no fuzzy branch, multi-label evidence with unit priors."""
from .model import ModelConfig, MultiLabelV5
from .losses import LossConfig
__all__ = ['ModelConfig', 'MultiLabelV5', 'LossConfig']
