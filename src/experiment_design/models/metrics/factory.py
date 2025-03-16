"""Factory module for creating power monitoring instances."""

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

    This factory function detects the available hardware and creates
    the most appropriate monitor instance.

    Args:
        device_type: The type of device to monitor. If "auto", automatically detect.
        force_cpu: If True, force CPU monitoring even if GPU is available.
        **kwargs: Additional arguments to pass to the specific monitor constructor.

    Returns:
        A PowerMonitor instance for the detected or specified hardware.

    Raises:
        MonitoringInitError: If monitor initialization fails.
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
    """Detect the hardware platform.

    Returns:
        String identifier for the detected hardware ("nvidia", "jetson", or "cpu").
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
    """Check if running on a Jetson platform.

    Returns:
        True if running on Jetson, False otherwise.
    """
    from pathlib import Path

    jetson_paths = [
        "/sys/bus/i2c/drivers/ina3221x",
        "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device",
        "/usr/bin/tegrastats",
    ]
    return any(Path(p).exists() for p in jetson_paths)


def _has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is available.

    Returns:
        True if NVIDIA GPU is available, False otherwise.
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
