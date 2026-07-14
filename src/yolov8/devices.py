"""Device resolution shared by every entrypoint."""

import torch
from loguru import logger


def resolve_device(name):
    """Return a torch.device, with a CPU fallback when CUDA is absent."""
    if str(name).startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device('cpu')
    return torch.device(str(name))
