from __future__ import annotations

import logging

import torch


def resolve_device(logger: logging.Logger | None = None) -> torch.device:
    if not torch.cuda.is_available():
        if logger is not None:
            logger.info("Using CPU because CUDA is not available.")
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)

    if major < 7:
        if logger is not None:
            logger.warning(
                "Falling back to CPU because GPU '%s' has compute capability %d.%d, "
                "which is unsupported by the current PyTorch build.",
                device_name,
                major,
                minor,
            )
        return torch.device("cpu")

    if logger is not None:
        logger.info(
            "Using CUDA on GPU '%s' with compute capability %d.%d.",
            device_name,
            major,
            minor,
        )
    return torch.device("cuda")
