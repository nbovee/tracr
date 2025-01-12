# src/experiment_design/models/power_monitor.py

import time
import logging
from typing import Tuple
from pathlib import Path

logger = logging.getLogger("split_computing_logger")

try:
    import pynvml  # For NVIDIA GPU monitoring

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    logger.debug("NVIDIA monitoring not available")

try:
    from jtop import JtopException, jtop

    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    logger.debug("Jetson monitoring not available")

logger = logging.getLogger("split_computing_logger")


class GPUEnergyMonitor:
    """Monitor GPU energy consumption for both desktop NVIDIA and Jetson devices."""

    def __init__(self, device_type: str = "auto"):
        """Initialize GPU energy monitoring.

        Args:
            device_type: One of ["auto", "nvidia", "jetson"]. If "auto", will attempt to detect.
        """
        self.device_type = (
            self._detect_device_type() if device_type == "auto" else device_type
        )
        self._nvml_initialized = False
        self._jtop = None
        self._nvml_handle = None
        self._setup_monitoring()
        logger.info(f"Initialized GPU monitoring for {self.device_type}")

    def _detect_device_type(self) -> str:
        """Detect whether running on Jetson or desktop NVIDIA."""
        if JTOP_AVAILABLE and self._is_jetson():
            return "jetson"
        elif NVIDIA_AVAILABLE:
            return "nvidia"
        else:
            raise RuntimeError("No supported GPU monitoring available")

    def _is_jetson(self) -> bool:
        """Check if running on Jetson platform."""
        jetson_paths = [
            "/sys/bus/i2c/drivers/ina3221x",
            "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device",
            "/usr/bin/tegrastats",
        ]
        return any(Path(p).exists() for p in jetson_paths)

    def _setup_monitoring(self) -> None:
        """Setup appropriate monitoring based on device type."""
        if self.device_type == "jetson":
            try:
                self._jtop = jtop()
                self._jtop.start()
            except JtopException as e:
                logger.error(f"Failed to initialize Jetson monitoring: {e}")
                raise

        elif self.device_type == "nvidia":
            try:
                if not self._nvml_initialized:
                    pynvml.nvmlInit()
                    self._nvml_initialized = True
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                    0
                )  # Assuming first GPU
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIA monitoring: {e}")
                raise

    def start_measurement(self) -> float:
        """Start energy measurement and return start timestamp."""
        if not hasattr(self, "_start_power"):
            self._start_power = self.get_current_power()
            self._start_metrics = self.get_system_metrics()
        self._start_time = time.time()
        return self._start_time

    def end_measurement(self) -> Tuple[float, float]:
        """End measurement and return (energy consumed in joules, elapsed time).

        The energy calculation uses trapezoidal approximation of the power curve.
        """
        end_time = time.time()
        end_power = self.get_current_power()

        # Calculate energy using trapezoidal approximation
        avg_power = (self._start_power + end_power) / 2
        elapsed_time = end_time - self._start_time
        energy_joules = avg_power * elapsed_time

        # Reset for next measurement
        delattr(self, "_start_power")

        return energy_joules, elapsed_time

    def get_current_power(self) -> float:
        """Get current power consumption in watts."""
        try:
            if self.device_type == "jetson":
                return self._get_jetson_power()
            else:
                return self._get_nvidia_power()
        except Exception as e:
            logger.error(f"Error getting power consumption: {e}")
            return 0.0

    def get_system_metrics(self) -> dict:
        """Get essential power and GPU metrics for energy analysis."""
        if not self._jtop or not self._jtop.is_alive():
            return {}

        try:
            stats = self._jtop.stats
            if not isinstance(stats, dict):
                return {}

            metrics = {
                # Essential metrics for energy analysis
                "power_reading": float(stats.get("Power VDD_CPU_GPU_CV", 0))
                / 1000.0,  # Convert mW to W
                "gpu_utilization": float(
                    stats.get("GPU", 0)
                ),  # GPU utilization percentage
            }

            logger.debug(f"Collected power metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error collecting power metrics: {e}")
            return {}

    def _get_jetson_power(self) -> float:
        """Get Jetson GPU power consumption in watts.
        Returns the VDD_CPU_GPU_CV value which represents GPU power consumption.
        """
        if not self._jtop or not self._jtop.is_alive():
            raise RuntimeError("Jetson monitoring not initialized")

        try:
            stats = self._jtop.stats

            # Look for the exact key "Power VDD_CPU_GPU_CV"
            power_key = "Power VDD_CPU_GPU_CV"
            if power_key in stats:
                try:
                    # Value is already in mW, just convert to W
                    power = float(stats[power_key]) / 1000.0
                    logger.debug(f"Found GPU power reading: {power}W")
                    return power
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing power value: {e}")
                    return 0.0

            logger.warning("Could not find Power VDD_CPU_GPU_CV in stats")
            logger.debug(
                f"Available power keys: {[k for k in stats.keys() if k.startswith('Power')]}"
            )
            return 0.0

        except Exception as e:
            logger.error(f"Error reading Jetson power stats: {e}")
            return 0.0

    def _get_nvidia_power(self) -> float:
        """Get NVIDIA GPU power consumption in watts."""
        if not self._nvml_initialized or not self._nvml_handle:
            raise RuntimeError("NVIDIA monitoring not initialized")

        try:
            return (
                pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
            )  # Convert mW to W
        except Exception as e:
            logger.error(f"Error reading NVIDIA power: {e}")
            return 0.0

    def cleanup(self) -> None:
        """Clean up monitoring resources."""
        if self._jtop and self._jtop.is_alive():
            try:
                self._jtop.close()
            except Exception as e:
                logger.warning(f"Error closing jtop connection: {e}")

        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except Exception as e:
                logger.warning(f"Error shutting down NVML: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start_measurement()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
