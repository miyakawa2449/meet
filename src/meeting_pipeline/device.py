"""Device resolution logic."""

import logging
import sys

from .models import DeviceInfo

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> DeviceInfo:
    """
    Resolve device based on availability.
    Priority: CUDA > MPS > CPU (when requested='auto')
    """
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            print(
                "Error: CUDA device requested but CUDA is not available",
                file=sys.stderr,
            )
            sys.exit(2)
        resolved = "cuda"
    elif requested == "mps":
        if not torch.backends.mps.is_available():
            print(
                "Error: MPS device requested but MPS is not available",
                file=sys.stderr,
            )
            sys.exit(2)
        resolved = "mps"
    else:
        resolved = "cpu"

    device_info = DeviceInfo(requested=requested, resolved=resolved)
    logger.info("Device: requested=%s, resolved=%s", requested, resolved)
    return device_info
