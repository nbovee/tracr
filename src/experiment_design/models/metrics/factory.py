"""Factory module for creating power monitoring instances"""

import logging
from typing import Any, Literal

from .base import PowerMonitor
from .cpu import CPUPowerMonitor
from .exceptions import MonitoringInitError
from .jetson import JetsonMonitor
from .nvidia import NvidiaGPUMonitor


logger = logging.getLogger("split_computing_logger")


def create_power_monitor(
    device_type: Literal["auto", "nvidia", "jetson", "cpu"] = "auto",
    force_cpu: bool = False,
    **kwargs: Any,
) -> PowerMonitor:
    """Create the appropriate power monitor for the current hardware.

    Selects and instantiates the optimal power monitoring implementation
    based on hardware detection or explicit configuration.
    """
    if force_cpu:
        logger.info("Forcing CPU monitoring as requested")
        return CPUPowerMonitor(**kwargs)

    try:
        if device_type == "auto":
            # Auto-detect hardware
            device_type = _detect_device_type()
            logger.info(f"Auto-detected device type: {device_type}")

        # Create the appropriate monitor
        if device_type == "nvidia":
            return NvidiaGPUMonitor(**kwargs)
        elif device_type == "jetson":
            return JetsonMonitor(**kwargs)
        else:  # default to CPU
            return CPUPowerMonitor(**kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize power monitor: {e}")
        raise MonitoringInitError(f"Failed to initialize power monitor: {str(e)}")


def _detect_device_type() -> str:
    """Detect the hardware platform through progressive feature detection.

    Performs platform detection in priority order:
    1. Check for Jetson-specific sysfs entries
    2. Check for NVIDIA GPU availability via CUDA or NVML
    3. Fall back to CPU monitoring
    """
    # Check for Jetson first (edge device)
    if _is_jetson():
        return "jetson"

    # Check for NVIDIA GPU
    if _has_nvidia_gpu():
        return "nvidia"

    # Default to CPU
    return "cpu"


def _is_jetson() -> bool:
    """Check for Jetson platform by examining platform-specific sysfs entries."""
    from pathlib import Path

    jetson_paths = [
        "/sys/bus/i2c/drivers/ina3221x",
        "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device",
        "/usr/bin/tegrastats",
    ]
    return any(Path(p).exists() for p in jetson_paths)


def _has_nvidia_gpu() -> bool:
    """Check for NVIDIA GPU availability using multiple detection methods.

    Attempts detection through:
    1. PyTorch CUDA runtime detection
    2. NVIDIA Management Library (NVML)
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return device_count > 0
        except (ImportError, Exception):
            return False
