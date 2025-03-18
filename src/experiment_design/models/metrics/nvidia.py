"""NVIDIA GPU power and energy monitoring using NVML"""

import logging
from typing import Dict, Any

from .base import PowerMonitor
from .exceptions import MonitoringInitError, MeasurementError

logger = logging.getLogger("split_computing_logger")


class NvidiaGPUMonitor(PowerMonitor):
    """NVIDIA GPU power and energy monitoring using NVIDIA Management Library.

    Uses NVML through pynvml Python bindings to access GPU-specific hardware
    sensors, providing accurate power consumption and utilization metrics.
    """

    def __init__(self) -> None:
        """Initialize NVIDIA GPU monitoring with NVML."""
        super().__init__("nvidia")
        self._nvml_initialized = False
        self._nvml_handle = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_initialized = True
            # Assume the first GPU (index 0) is used
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logger.info("NVIDIA GPU monitoring initialized")
        except ImportError:
            logger.error(
                "Failed to import pynvml. Install with: pip install nvidia-ml-py"
            )
            raise MonitoringInitError("NVIDIA monitoring requires pynvml package")
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA monitoring: {e}")
            raise MonitoringInitError(
                f"Failed to initialize NVIDIA monitoring: {str(e)}"
            )

    def get_current_power(self) -> float:
        """Get current GPU power consumption in watts using hardware sensors."""
        if not self._nvml_initialized or not self._nvml_handle:
            raise MeasurementError("NVIDIA monitoring not initialized")

        try:
            import pynvml

            return (
                pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
            )  # Convert mW to W
        except Exception as e:
            logger.error(f"Error reading NVIDIA power: {e}")
            raise MeasurementError(f"Failed to read NVIDIA power: {str(e)}")

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive GPU metrics including power, compute and memory utilization."""
        try:
            if not self._nvml_initialized or not self._nvml_handle:
                raise MeasurementError("NVIDIA monitoring not initialized")

            import pynvml

            power = (
                pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
            )  # Convert mW to W
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu

            # Get memory information
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            memory_utilization = (
                (mem_info.used / mem_info.total) * 100.0 if mem_info.total > 0 else 0.0
            )

            return {
                "power_reading": power,
                "gpu_utilization": float(util),
                "memory_utilization": memory_utilization,
                "device_type": "nvidia",
            }
        except Exception as e:
            logger.error(f"Error getting NVIDIA metrics: {e}")
            return {
                "power_reading": 0.0,
                "gpu_utilization": 0.0,
                "memory_utilization": 0.0,
                "device_type": "nvidia",
                "error": str(e),
            }

    def cleanup(self) -> None:
        """Release NVML library resources."""
        if self._nvml_initialized:
            try:
                import pynvml

                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                logger.debug("NVIDIA monitoring shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down NVML: {e}")
