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

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    logger.info(
        "Device availability: CUDA=%s, MPS=%s, platform=%s",
        cuda_available,
        mps_available,
        sys.platform,
    )

    if requested == "auto":
        if cuda_available:
            resolved = "cuda"
        elif mps_available:
            resolved = "mps"
        else:
            resolved = "cpu"
        logger.info("Auto device selection: %s", resolved)
    elif requested == "cuda":
        if not cuda_available:
            print(
                "Error: CUDA device requested but CUDA is not available",
                file=sys.stderr,
            )
            sys.exit(2)
        resolved = "cuda"
    elif requested == "mps":
        if not mps_available:
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
