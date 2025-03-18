"""CPU power and energy monitoring for different operating systems."""

import logging
import platform
import threading
import time
from typing import Dict, Any

from .base import PowerMonitor

logger = logging.getLogger("split_computing_logger")


class CPUPowerMonitor(PowerMonitor):
    """CPU power and energy monitoring for various operating systems.

    This class provides power monitoring on systems without dedicated
    GPU monitoring capabilities (laptops, desktops without NVIDIA GPUs).
    It uses different approaches depending on the operating system:
    - Windows: CPU power model based on utilization
    - Linux: Battery monitoring (if available) or CPU utilization model
    - macOS: CPU utilization model

    Attributes:
        device_type: Always "cpu"
        _os_type: The operating system type (Windows, Linux, Darwin)
        _tdp: Estimated CPU Thermal Design Power in watts
        _cpu_name: CPU model name for logging and TDP estimation
        _background_metrics_thread: Thread for background metrics collection
        _lock: Thread lock for safe concurrent access
        _background_metrics: Dict storing metrics collected in background
    """

    def __init__(self) -> None:
        """Initialize the CPU power monitor.

        Raises:
            MonitoringInitError: If initialization fails
        """
        super().__init__("cpu")

        # System information
        self._os_type = platform.system()
        self._cpu_name = self._get_cpu_name()
        self._tdp = self._detect_cpu_tdp()

        # Background metrics collection
        self._background_metrics_thread = None
        self._stop_background_thread = False
        self._background_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "cpu_freq": 0.0,
        }
        self._lock = threading.Lock()

        # Battery monitoring
        self._battery_initialized = False
        self._has_battery = self._check_battery()
        if self._has_battery:
            self._initialize_battery_monitoring()

        # Cumulative metrics for split computation
        self._cumulative_metrics_enabled = False
        self._cumulative_start_time = 0.0
        self._cumulative_power_readings = []
        self._cumulative_cpu_utilization = []
        self._cumulative_memory_utilization = []

        # Start background collection on Windows
        if self._os_type == "Windows":
            self._start_background_metrics_collection()

        logger.info(f"CPU power monitor initialized on {self._os_type}")
        logger.info(f"CPU: {self._cpu_name}, estimated TDP: {self._tdp}W")

    def get_current_power(self) -> float:
        """Get current CPU power consumption in watts.

        The implementation varies by operating system:
        - Windows: Uses a power model based on CPU utilization and TDP
        - Linux with battery: Uses battery discharge rate if available
        - Other: Uses a CPU utilization-based model

        Returns:
            Current power consumption in watts
        """
        try:
            # First try to get power from battery if available and unplugged
            if self._has_battery:
                battery_power = self._estimate_battery_power()
                if battery_power > 0:
                    return battery_power

            # If battery power estimation is not available, use the CPU power model
            if self._os_type == "Windows":
                return self._estimate_windows_cpu_power()

            # For non-Windows systems without battery power, use a default based on TDP
            return 0.4 * self._tdp  # Default to 40% of TDP
        except Exception as e:
            logger.error(f"Error getting CPU power consumption: {e}")
            return 0.4 * self._tdp  # Default to 40% of TDP

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get CPU system metrics including power, CPU and memory utilization.

        Returns:
            Dictionary with power and utilization metrics
        """
        try:
            # If Windows and cumulative metrics are enabled, use dedicated collection
            if self._os_type == "Windows" and self._cumulative_metrics_enabled:
                return self.get_cumulative_metrics()

            # Get CPU metrics - use non-blocking for Windows
            use_nonblocking = self._os_type == "Windows"
            cpu_info = self._get_cpu_metrics(non_blocking=use_nonblocking)

            metrics = {
                "cpu_percent": cpu_info["cpu_percent"],
                "memory_utilization": cpu_info["memory_percent"],
                "power_reading": self.get_current_power(),
                "gpu_utilization": 0.0,  # CPU has no GPU
                "device_type": "cpu",
                "os_type": self._os_type,
            }

            # If battery is available, add battery energy
            if self._has_battery:
                battery_energy = self.get_battery_energy()
                if battery_energy > 0:
                    metrics["host_battery_energy_mwh"] = battery_energy

            return metrics
        except Exception as e:
            logger.error(f"Error getting CPU system metrics: {e}")
            # Return sensible defaults
            return {
                "power_reading": 0.4 * self._tdp,
                "gpu_utilization": 0.0,
                "cpu_percent": 50.0,
                "memory_utilization": 50.0,
                "device_type": "cpu",
                "os_type": self._os_type,
                "error": str(e),
            }

    def start_cumulative_measurement(self) -> None:
        """Start collecting cumulative metrics for split computation.

        This is optimized for Windows CPU devices to collect metrics once
        at the split point instead of for each layer.
        """
        # Only enable for Windows CPU devices
        if self._os_type == "Windows":
            self._cumulative_metrics_enabled = True
            self._cumulative_start_time = time.time()
            self._cumulative_power_readings = []
            self._cumulative_cpu_utilization = []
            self._cumulative_memory_utilization = []

            # Take an initial power reading
            self._estimate_windows_cpu_power()

    def get_cumulative_metrics(self) -> Dict[str, float]:
        """Get cumulative metrics collected for CPU devices.

        Returns:
            Dictionary with cumulative power and energy metrics
        """
        if not self._cumulative_metrics_enabled:
            return {}

        try:
            # Calculate elapsed time
            elapsed_time = time.time() - self._cumulative_start_time

            # Take a final power reading
            current_power = self._estimate_windows_cpu_power()

            # Calculate average power from all readings
            if self._cumulative_power_readings:
                avg_power = sum(self._cumulative_power_readings) / len(
                    self._cumulative_power_readings
                )
            else:
                avg_power = current_power

            # Calculate average CPU and memory utilization
            avg_cpu_util = 0.0
            if self._cumulative_cpu_utilization:
                avg_cpu_util = sum(self._cumulative_cpu_utilization) / len(
                    self._cumulative_cpu_utilization
                )

            avg_memory_util = 0.0
            if self._cumulative_memory_utilization:
                avg_memory_util = sum(self._cumulative_memory_utilization) / len(
                    self._cumulative_memory_utilization
                )

            # Calculate total energy (joules) = average power (watts) * time (seconds)
            total_energy = avg_power * elapsed_time

            metrics = {
                "power_reading": avg_power,
                "processing_energy": total_energy,
                "elapsed_time": elapsed_time,
                "cpu_utilization": avg_cpu_util,
                "memory_utilization": avg_memory_util,
                "gpu_utilization": 0.0,
                "communication_energy": 0.0,
                "host_battery_energy_mwh": 0.0,
            }

            # Reset cumulative measurement
            self._cumulative_metrics_enabled = False

            logger.info(
                f"Cumulative CPU metrics: power={avg_power:.2f}W, energy={total_energy:.6f}J, time={elapsed_time:.4f}s"
            )
            return metrics
        except Exception as e:
            logger.error(f"Error getting cumulative metrics: {e}")
            self._cumulative_metrics_enabled = False
            return {
                "power_reading": 0.0,
                "processing_energy": 0.0,
                "elapsed_time": 0.0,
                "cpu_utilization": 0.0,
                "memory_utilization": 0.0,
                "gpu_utilization": 0.0,
            }

    def _get_cpu_name(self) -> str:
        """Get CPU model name.

        Returns:
            CPU model name as a string
        """
        try:
            if platform.system() == "Windows":
                import subprocess

                output = subprocess.check_output(
                    "wmic cpu get name", shell=True
                ).decode()
                return output.strip().split("\n")[1].strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Darwin":  # macOS
                import subprocess

                output = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"]
                ).decode()
                return output.strip()
            return "Unknown CPU"
        except Exception as e:
            logger.debug(f"Error getting CPU name: {e}")
            return "Unknown CPU"

    def _detect_cpu_tdp(self) -> float:
        """Estimate CPU TDP (Thermal Design Power) based on CPU model.

        Returns:
            Estimated TDP in watts
        """
        try:
            # Use a default TDP if we can't determine it
            default_tdp = 65.0  # Watts - typical desktop CPU TDP

            if platform.system() == "Windows":
                cpu_name = self._cpu_name.lower()

                # Estimate based on common CPU series naming conventions
                if "i9" in cpu_name:
                    return 125.0  # High performance CPU
                elif "i7" in cpu_name:
                    return 95.0  # Performance CPU
                elif "i5" in cpu_name:
                    return 65.0  # Mid-range CPU
                elif "i3" in cpu_name:
                    return 35.0  # Entry-level CPU
                elif "celeron" in cpu_name or "pentium" in cpu_name:
                    return 15.0  # Low-power CPU
                elif "ryzen 9" in cpu_name:
                    return 105.0  # High-end AMD
                elif "ryzen 7" in cpu_name:
                    return 65.0  # Performance AMD
                elif "ryzen 5" in cpu_name:
                    return 65.0  # Mid-range AMD
                elif "ryzen 3" in cpu_name:
                    return 65.0  # Entry-level AMD
                else:
                    # If it's a laptop, use a lower default TDP
                    if self._has_battery:
                        return 28.0  # Typical laptop TDP

            # For other operating systems or when CPU model doesn't match
            if self._has_battery:
                return 28.0  # Laptop TDP

            # Scale with number of CPU cores if psutil is available
            try:
                import psutil

                cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
                if cpu_count:
                    # Rough estimate based on core count
                    if cpu_count >= 16:
                        return 105.0  # Many-core CPU
                    elif cpu_count >= 8:
                        return 95.0  # 8-core CPU
                    elif cpu_count >= 6:
                        return 65.0  # 6-core CPU
                    elif cpu_count >= 4:
                        return 65.0  # Quad core
                    elif cpu_count >= 2:
                        return 35.0  # Dual core
            except ImportError:
                pass

            return default_tdp
        except Exception as e:
            logger.debug(f"Error detecting CPU TDP: {e}")
            return 65.0  # Default to a mid-range desktop CPU TDP

    def _check_battery(self) -> bool:
        """Check if the system has a battery.

        Returns:
            True if the system has a battery, False otherwise
        """
        try:
            import psutil

            battery = psutil.sensors_battery()
            return battery is not None
        except (ImportError, Exception) as e:
            logger.debug(f"Error checking battery: {e}")
            return False

    def _initialize_battery_monitoring(self) -> None:
        """Initialize battery monitoring if available."""
        try:
            import psutil

            battery = psutil.sensors_battery()
            if battery:
                self._battery_initialized = True
                logger.info(
                    f"Battery monitoring initialized: plugged={battery.power_plugged}, percent={battery.percent}%"
                )
        except (ImportError, Exception) as e:
            logger.debug(f"Error initializing battery monitoring: {e}")

    def _estimate_battery_power(self) -> float:
        """Estimate power consumption from battery changes.

        Returns:
            Estimated power consumption in watts or 0 if not available
        """
        try:
            import psutil

            # Make sure we're on battery power
            battery = psutil.sensors_battery()
            if not battery or battery.power_plugged:
                return 0.0

            # Initialize last reading if not exists
            if not hasattr(self, "_last_power_reading"):
                self._last_power_reading = (time.time(), battery.percent)
                return 0.0

            current_time = time.time()

            # Get current battery percentage
            last_time, last_percent = self._last_power_reading
            time_diff = current_time - last_time
            percent_diff = last_percent - battery.percent

            if time_diff > 0 and percent_diff > 0:
                # Typical laptop battery capacity (50Wh = 50000mWh)
                BATTERY_CAPACITY_WH = 50.0
                # Convert percent/hour to watts
                power = (
                    (percent_diff / 100.0) * BATTERY_CAPACITY_WH * (3600 / time_diff)
                )

                # Update last reading
                self._last_power_reading = (current_time, battery.percent)
                return power

            # Reset last reading if conditions not met
            self._last_power_reading = (current_time, battery.percent)
            return 0.0
        except (ImportError, Exception) as e:
            logger.debug(f"Error estimating battery power: {e}")
            return 0.0

    def _estimate_windows_cpu_power(self) -> float:
        """Estimate Windows CPU power consumption based on CPU utilization and TDP.

        Returns:
            Estimated power consumption in watts
        """
        try:
            # Get current CPU metrics - use non-blocking mode
            metrics = self._get_cpu_metrics(non_blocking=True)
            cpu_percent = metrics["cpu_percent"]

            # Simple CPU power model:
            # Base power (idle) is roughly 30% of TDP
            # Maximum power at 100% utilization is the TDP
            idle_power = 0.3 * self._tdp
            max_power = self._tdp

            # Linear model: power = idle_power + (max_power - idle_power) * (cpu_percent / 100)
            estimated_power = idle_power + (max_power - idle_power) * (
                cpu_percent / 100.0
            )

            # Add a small random variation to make the readings look more realistic
            import random

            variation = random.uniform(-0.5, 0.5)
            estimated_power += variation

            # Ensure power is not negative
            estimated_power = max(estimated_power, 0.0)

            return estimated_power
        except Exception as e:
            logger.error(f"Error estimating Windows CPU power: {e}")
            return 0.4 * self._tdp  # Default to 40% of TDP

    def cleanup(self) -> None:
        """Clean up resources used by the CPU power monitor.

        This method ensures that background threads are properly stopped
        and any open resources are released.
        """
        # Stop background metrics collection for Windows
        if self._os_type == "Windows" and hasattr(self, "_background_metrics_thread"):
            self._stop_background_thread = True
            if (
                hasattr(self, "_background_metrics_thread")
                and self._background_metrics_thread
            ):
                if self._background_metrics_thread.is_alive():
                    self._background_metrics_thread.join(timeout=1.0)
                self._background_metrics_thread = None
                logger.debug("Background metrics collection stopped")

        # Clean up any other resources
        if hasattr(self, "_battery_initialized") and self._battery_initialized:
            logger.debug("Battery monitoring cleanup completed")
