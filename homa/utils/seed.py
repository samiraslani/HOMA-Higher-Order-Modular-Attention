"""Reproducibility utilities."""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducible experiments.

    Covers Python's built-in ``random``, NumPy, and PyTorch (CPU + all GPUs).
    Also enables cuDNN deterministic mode and disables benchmark mode.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d (deterministic mode ON)", seed)
