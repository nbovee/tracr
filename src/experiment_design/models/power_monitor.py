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

# Keep the psutil import for battery check only
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil monitoring not available")


class GPUEnergyMonitor:
    """Monitor GPU energy consumption for both desktop NVIDIA and Jetson devices."""

    def __init__(self, device_type: str = "auto"):
        """Initialize GPU/CPU energy monitoring.

        Args:
            device_type: One of ["auto", "nvidia", "jetson", "cpu"]. If "auto", will attempt to detect the correct device type.
        """
        try:
            self.device_type = (
                self._detect_device_type() if device_type == "auto" else device_type
            )
        except RuntimeError:
            logger.info("No GPU detected, falling back to CPU monitoring")
            self.device_type = "cpu"

        self._nvml_initialized = False
        self._jtop = None
        self._nvml_handle = None
        self._battery_initialized = PSUTIL_AVAILABLE and self._has_battery()
        self._initial_battery = None
        self._current_split = None
        self._measurements = {}

        self.initialize_battery_monitoring()
        if self.device_type != "cpu":
            self._setup_monitoring()
        logger.info(f"Initialized monitoring for {self.device_type}")

    def _detect_device_type(self) -> str:
        """Detect whether running on a Jetson, desktop NVIDIA, or CPU device."""
        if JTOP_AVAILABLE and self._is_jetson():
            return "jetson"
        elif NVIDIA_AVAILABLE:
            return "nvidia"
        else:
            return "cpu"

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
        """Start energy measurement and return the start timestamp."""
        if not hasattr(self, "_start_power"):
            self._start_power = self.get_current_power()
            self._start_metrics = self.get_system_metrics()
        self._start_time = time.time()
        return self._start_time

    def end_measurement(self) -> Tuple[float, float]:
        """End measurement and return a tuple (energy consumed in joules, elapsed time)."""
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
        """Return the current power consumption in watts."""
        try:
            if self.device_type == "jetson":
                return self._get_jetson_power()
            elif self.device_type == "nvidia":
                return self._get_nvidia_power()
            else:  # CPU mode
                if self._battery_initialized:
                    battery = psutil.sensors_battery()
                    if battery and not battery.power_plugged:
                        # Estimate power from battery percentage change
                        return self._estimate_cpu_power()
                return 0.0
        except Exception as e:
            logger.error(f"Error getting power consumption: {e}")
            return 0.0

    def _estimate_cpu_power(self) -> float:
        """Estimate CPU power consumption from battery changes."""
        try:
            if not hasattr(self, "_last_power_reading"):
                self._last_power_reading = (
                    time.time(),
                    psutil.sensors_battery().percent,
                )
                return 0.0

            current_time = time.time()
            current_battery = psutil.sensors_battery()

            if current_battery and not current_battery.power_plugged:
                last_time, last_percent = self._last_power_reading
                time_diff = current_time - last_time
                percent_diff = last_percent - current_battery.percent

                if time_diff > 0 and percent_diff > 0:
                    # Typical laptop battery capacity (50Wh = 50000mWh)
                    BATTERY_CAPACITY_WH = 50.0
                    # Convert percent/hour to watts
                    power = (
                        (percent_diff / 100.0)
                        * BATTERY_CAPACITY_WH
                        * (3600 / time_diff)
                    )

                    # Update last reading
                    self._last_power_reading = (current_time, current_battery.percent)
                    return power

            return 0.0
        except Exception as e:
            logger.error(f"Error estimating CPU power: {e}")
            return 0.0

    def get_system_metrics(self) -> dict:
        """Get current system metrics."""
        metrics = {}

        if self.device_type in ["nvidia", "jetson"]:
            # Existing GPU metrics collection
            if self.device_type == "jetson":
                if not self._jtop or not self._jtop.is_alive():
                    return metrics
                try:
                    stats = self._jtop.stats
                    if isinstance(stats, dict):
                        metrics.update(
                            {
                                "power_reading": float(
                                    stats.get("Power VDD_CPU_GPU_CV", 0)
                                )
                                / 1000.0,
                                "gpu_utilization": float(stats.get("GPU", 0)),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error collecting Jetson metrics: {e}")
        else:  # CPU mode
            try:
                power = self._estimate_cpu_power()
                metrics.update(
                    {
                        "power_reading": power,
                        "gpu_utilization": 0.0,  # No GPU utilization in CPU mode
                    }
                )
            except Exception as e:
                logger.error(f"Error collecting CPU metrics: {e}")

        logger.debug(f"Collected system metrics: {metrics}")
        return metrics

    def _get_jetson_power(self) -> float:
        """Retrieve Jetson GPU power consumption in watts."""
        if not self._jtop or not self._jtop.is_alive():
            raise RuntimeError("Jetson monitoring not initialized")

        try:
            stats = self._jtop.stats
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

    def _calculate_battery_draw(self, battery) -> float:
        """Calculate battery energy consumption in mWh."""
        try:
            if not hasattr(self, "_last_battery_reading"):
                self._last_battery_reading = (
                    battery.percent,
                    time.time(),
                    battery.power_plugged,
                )
                logger.debug(
                    f"Initial battery state: {battery.percent}%, plugged={battery.power_plugged}"
                )
                return 0.0

            last_percent, last_time, was_plugged = self._last_battery_reading
            current_time = time.time()

            # Update reading for next time
            self._last_battery_reading = (
                battery.percent,
                current_time,
                battery.power_plugged,
            )

            # Only measure when on battery power
            if battery.power_plugged or was_plugged:
                logger.debug("Battery is/was plugged in, skipping measurement")
                return 0.0

            # Calculate energy used
            percent_diff = last_percent - battery.percent
            if percent_diff > 0:
                # Typical laptop battery capacity in mWh
                TYPICAL_BATTERY_CAPACITY = 50000  # 50Wh = 50000mWh
                energy_used = (percent_diff / 100.0) * TYPICAL_BATTERY_CAPACITY
                logger.debug(
                    f"Battery dropped {percent_diff}%, estimated {energy_used:.2f}mWh used"
                )
                return energy_used

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating battery energy: {e}")
            return 0.0

    def _has_battery(self) -> bool:
        """Check if the system has a battery."""
        try:
            battery = psutil.sensors_battery()
            return battery is not None
        except Exception as e:
            logger.debug(f"Error checking battery: {e}")
            return False

    def initialize_battery_monitoring(self):
        battery = psutil.sensors_battery()
        if battery:
            self._battery_initialized = True
            logger.info(
                f"Initial battery state: plugged={battery.power_plugged}, percent={battery.percent}%"
            )

    def start_split_measurement(self, split_layer: int):
        """Start a fresh battery measurement for a new split layer"""
        if self._battery_initialized:
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged:
                # Get fresh battery reading for this split layer
                current_percent = battery.percent

                # Start fresh measurement state for this split
                self._current_split = split_layer
                self._measurements[split_layer] = {
                    "start_percent": current_percent,
                    "start_time": time.time(),
                }
                logger.info(
                    f"Starting split layer {split_layer} with battery at {current_percent:.3f}%"
                )

    def get_battery_energy(self) -> float:
        """Calculate battery energy used for the current split"""
        if not self._battery_initialized or self._current_split is None:
            return 0.0

        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged:
            current_percent = battery.percent
            split_data = self._measurements.get(self._current_split)

            if split_data:
                start_percent = split_data["start_percent"]
                total_percent_diff = start_percent - current_percent

                if total_percent_diff > 0:
                    # HP Laptop battery with Full Charge Capacity of 57,356 mWh
                    BATTERY_CAPACITY = 57356  # mWh
                    total_energy_used = (total_percent_diff / 100.0) * BATTERY_CAPACITY

                    # Log the final measurement for this split
                    elapsed_time = time.time() - split_data["start_time"]
                    logger.info(
                        f"Split {self._current_split} completed: "
                        f"Battery dropped {total_percent_diff:.3f}% "
                        f"({start_percent:.3f}% -> {current_percent:.3f}%), "
                        f"used {total_energy_used:.2f}mWh in {elapsed_time:.1f}s"
                    )

                    # Store final results
                    split_data.update(
                        {
                            "end_percent": current_percent,
                            "energy_used": total_energy_used,
                            "elapsed_time": elapsed_time,
                        }
                    )

                    return total_energy_used

        return 0.0

    def get_split_measurements(self):
        """Return all split measurements for logging to Excel"""
        return self._measurements
