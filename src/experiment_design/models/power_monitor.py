# src/experiment_design/models/power_monitor.py

import time
import logging
from typing import Tuple
from pathlib import Path

logger = logging.getLogger("split_computing_logger")

# Try to import pynvml for NVIDIA GPU monitoring.
try:
    import pynvml

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    logger.debug("NVIDIA monitoring not available")

# Try to import jtop for Jetson monitoring.
try:
    from jtop import JtopException, jtop

    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    logger.debug("Jetson monitoring not available")


class GPUEnergyMonitor:
    """Monitor GPU energy consumption for both desktop NVIDIA and Jetson devices."""

    def __init__(self, device_type: str = "auto"):
        """Initialize GPU energy monitoring.

        Args:
            device_type: One of ["auto", "nvidia", "jetson"]. If "auto", will attempt to detect the correct device type.
        """
        # Auto-detect device type if set to "auto"; otherwise, use the provided type.
        self.device_type = (
            self._detect_device_type() if device_type == "auto" else device_type
        )
        self._nvml_initialized = False  # Flag to track NVML initialization.
        self._jtop = None  # jtop instance for Jetson monitoring.
        self._nvml_handle = None  # NVML handle for NVIDIA GPU.
        self._setup_monitoring()  # Setup monitoring based on detected device type.
        logger.info(f"Initialized GPU monitoring for {self.device_type}")

    def _detect_device_type(self) -> str:
        """Detect whether running on a Jetson or a desktop NVIDIA device.

        Returns:
            "jetson" if Jetson-specific monitoring is available and detected,
            "nvidia" if NVIDIA monitoring is available.

        Raises:
            RuntimeError if no supported GPU monitoring is available.
        """
        if JTOP_AVAILABLE and self._is_jetson():
            return "jetson"
        elif NVIDIA_AVAILABLE:
            return "nvidia"
        else:
            raise RuntimeError("No supported GPU monitoring available")

    def _is_jetson(self) -> bool:
        """Check if running on a Jetson platform by verifying the existence of typical Jetson file paths."""
        jetson_paths = [
            "/sys/bus/i2c/drivers/ina3221x",
            "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device",
            "/usr/bin/tegrastats",
        ]
        return any(Path(p).exists() for p in jetson_paths)

    def _setup_monitoring(self) -> None:
        """Setup the appropriate monitoring system based on the device type."""
        if self.device_type == "jetson":
            try:
                self._jtop = jtop()  # Create a jtop instance.
                self._jtop.start()  # Start monitoring via jtop.
            except JtopException as e:
                logger.error(f"Failed to initialize Jetson monitoring: {e}")
                raise
        elif self.device_type == "nvidia":
            try:
                if not self._nvml_initialized:
                    pynvml.nvmlInit()  # Initialize NVML.
                    self._nvml_initialized = True
                # Assume the first GPU (index 0) is used.
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIA monitoring: {e}")
                raise

    def start_measurement(self) -> float:
        """Start energy measurement and return the start timestamp.
        This function stores the current power and system metrics, then records the start time.
        """
        if not hasattr(self, "_start_power"):
            self._start_power = self.get_current_power()
            self._start_metrics = self.get_system_metrics()
        self._start_time = time.time()
        return self._start_time

    def end_measurement(self) -> Tuple[float, float]:
        """End measurement and return a tuple (energy consumed in joules, elapsed time).
        Uses a trapezoidal approximation to calculate energy consumption over the measurement interval.
        """
        end_time = time.time()
        end_power = self.get_current_power()

        # Compute average power over the interval.
        avg_power = (self._start_power + end_power) / 2
        elapsed_time = end_time - self._start_time
        energy_joules = avg_power * elapsed_time

        # Clean up for next measurement.
        delattr(self, "_start_power")

        return energy_joules, elapsed_time

    def get_current_power(self) -> float:
        """Return the current power consumption in watts.
        Chooses the appropriate method based on the device type."""
        try:
            if self.device_type == "jetson":
                return self._get_jetson_power()
            else:
                return self._get_nvidia_power()
        except Exception as e:
            logger.error(f"Error getting power consumption: {e}")
            return 0.0

    def get_system_metrics(self) -> dict:
        """Retrieve essential system metrics (e.g., power reading and GPU utilization) for energy analysis.
        Returns an empty dictionary if jtop is not initialized or not alive."""
        if not self._jtop or not self._jtop.is_alive():
            return {}

        try:
            stats = self._jtop.stats
            if not isinstance(stats, dict):
                return {}

            metrics = {
                # Extract and convert power reading from mW to W.
                "power_reading": float(stats.get("Power VDD_CPU_GPU_CV", 0)) / 1000.0,
                # GPU utilization percentage.
                "gpu_utilization": float(stats.get("GPU", 0)),
            }

            logger.debug(f"Collected power metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error collecting power metrics: {e}")
            return {}

    def _get_jetson_power(self) -> float:
        """Retrieve Jetson GPU power consumption in watts.
        This function extracts the "Power VDD_CPU_GPU_CV" value from jtop stats and converts it from mW to W.
        """
        if not self._jtop or not self._jtop.is_alive():
            raise RuntimeError("Jetson monitoring not initialized")

        try:
            stats = self._jtop.stats

            # Use the specific key "Power VDD_CPU_GPU_CV" to retrieve GPU power.
            power_key = "Power VDD_CPU_GPU_CV"
            if power_key in stats:
                try:
                    power = float(stats[power_key]) / 1000.0  # Convert mW to W.
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
        """Retrieve NVIDIA GPU power consumption in watts using NVML."""
        if not self._nvml_initialized or not self._nvml_handle:
            raise RuntimeError("NVIDIA monitoring not initialized")

        try:
            # NVML returns power in mW; convert to watts.
            return pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
        except Exception as e:
            logger.error(f"Error reading NVIDIA power: {e}")
            return 0.0

    def cleanup(self) -> None:
        """Clean up and release monitoring resources."""
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
        """Enter the context manager; start energy measurement."""
        self.start_measurement()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager; clean up monitoring resources."""
        self.cleanup()

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()
