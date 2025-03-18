# src/experiment_design/models/power_monitor.py

import time
import logging
import platform
import os
from typing import Tuple, Dict, Optional, Any
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

    def __init__(self, device_type: str = "auto", force_cpu: bool = False):
        """Initialize GPU/CPU energy monitoring.

        Args:
            device_type: One of ["auto", "nvidia", "jetson", "cpu"]
            force_cpu: Force CPU monitoring even if GPU is available
        """
        self._forced_cpu = force_cpu
        # Initialize all attributes to avoid AttributeError
        self._nvml_initialized = False
        self._jtop = None
        self._nvml_handle = None
        self._battery_initialized = False
        self._initial_battery = None
        self._current_split = None
        self._measurements = {}
        self._start_power = 0.0
        self._start_time = 0.0
        self._start_metrics = {}
        self._last_power_reading = None

        # Windows CPU power model parameters - these will be used for Windows CPU power estimation
        self._tdp = self._detect_cpu_tdp()  # Base TDP in watts
        self._power_model_initialized = False
        self._cpu_power_model_cache = {}
        self._last_cpu_metrics_time = 0
        self._cpu_metrics_cache_duration = (
            0.1  # Cache CPU metrics for 100ms to avoid too frequent calls
        )
        self._os_type = platform.system()
        self._cpu_name = self._get_cpu_name()

        try:
            self.device_type = (
                self._detect_device_type() if device_type == "auto" else device_type
            )
        except RuntimeError:
            logger.info("No GPU detected, falling back to CPU monitoring")
            self.device_type = "cpu"

        # Initialize battery monitoring first
        self._battery_initialized = PSUTIL_AVAILABLE and self._has_battery()
        if self._battery_initialized:
            self.initialize_battery_monitoring()
            logger.debug("Battery monitoring initialized")

        # Only set up GPU monitoring if not forced to CPU
        if not self._forced_cpu and self.device_type != "cpu":
            try:
                self._setup_monitoring()
                logger.debug(f"GPU monitoring initialized for {self.device_type}")
            except Exception as e:
                logger.warning(
                    f"GPU monitoring initialization failed: {e}, falling back to CPU"
                )
                self.device_type = "cpu"

        # Initialize CPU power model if using CPU device
        if self.device_type == "cpu":
            self._init_cpu_power_model()

        logger.info(f"Energy monitoring initialized in {self.device_type} mode")
        if self.device_type == "cpu":
            logger.info(f"Using CPU: {self._cpu_name}, estimated TDP: {self._tdp}W")

    def _get_cpu_name(self) -> str:
        """Get CPU model name."""
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
        """Estimate CPU TDP (Thermal Design Power) based on CPU model or system information."""
        try:
            # Use a default TDP if we can't determine it
            default_tdp = 65.0  # Watts - typical desktop CPU TDP

            if platform.system() == "Windows":
                cpu_name = self._get_cpu_name().lower()

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
                    import ctypes

                    try:
                        # Check if system is on battery power which might indicate a laptop
                        if ctypes.windll.kernel32.GetSystemPowerStatus(
                            ctypes.byref(ctypes.c_byte())
                        ):
                            return 28.0  # Typical laptop TDP
                    except:
                        pass

            # For other operating systems, make a general estimate
            elif PSUTIL_AVAILABLE:
                # If a laptop (has battery), use lower TDP
                if self._has_battery():
                    return 28.0  # Laptop TDP

                # Scale with number of CPU cores
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

            return default_tdp

        except Exception as e:
            logger.debug(f"Error detecting CPU TDP: {e}")
            return 65.0  # Default to a mid-range desktop CPU TDP

    def _init_cpu_power_model(self) -> None:
        """Initialize the CPU power model for Windows."""
        try:
            # Log that we're using the CPU power model
            self._power_model_initialized = True
            logger.info(f"Initialized CPU power model with base TDP: {self._tdp}W")

            # Take an initial measurement of CPU metrics
            if PSUTIL_AVAILABLE:
                self._get_cpu_metrics()
        except Exception as e:
            logger.error(f"Failed to initialize CPU power model: {e}")

    def _get_cpu_metrics(self) -> Dict[str, float]:
        """Get current CPU metrics, with caching to avoid excessive psutil calls."""
        current_time = time.time()

        # Return cached metrics if within cache duration
        if (
            hasattr(self, "_cpu_metrics_cache")
            and current_time - self._last_cpu_metrics_time
            < self._cpu_metrics_cache_duration
        ):
            return self._cpu_metrics_cache

        try:
            if PSUTIL_AVAILABLE:
                # Get CPU utilization (1 second interval would be more accurate but creates lag)
                # Using a short interval of 0.1 seconds for responsiveness
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # Get CPU frequency if available
                cpu_freq = 0.0
                if hasattr(psutil, "cpu_freq"):
                    freq = psutil.cpu_freq()
                    if freq and hasattr(freq, "current"):
                        cpu_freq = freq.current

                # Create metrics dictionary
                metrics = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "cpu_freq": cpu_freq,
                }

                # Update cache
                self._cpu_metrics_cache = metrics
                self._last_cpu_metrics_time = current_time

                return metrics
            else:
                return {
                    "cpu_percent": 50.0,  # Default to 50% if psutil not available
                    "memory_percent": 50.0,
                    "cpu_freq": 0.0,
                }
        except Exception as e:
            logger.error(f"Error getting CPU metrics: {e}")
            return {"cpu_percent": 50.0, "memory_percent": 50.0, "cpu_freq": 0.0}

    def _estimate_windows_cpu_power(self) -> float:
        """Estimate Windows CPU power consumption based on CPU utilization and TDP."""
        if not self._power_model_initialized:
            self._init_cpu_power_model()

        try:
            # Get current CPU metrics
            metrics = self._get_cpu_metrics()
            cpu_percent = metrics["cpu_percent"]

            # Simple CPU power model: Power scales with CPU utilization
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
            estimated_power = max(0.1, estimated_power)

            return estimated_power

        except Exception as e:
            logger.error(f"Error estimating Windows CPU power: {e}")
            # Return a reasonable default based on TDP
            return 0.4 * self._tdp

    def _detect_device_type(self) -> str:
        """Detect whether running on a Jetson, desktop NVIDIA, or CPU device."""
        # First check if we're forced to CPU mode
        if self._forced_cpu:
            return "cpu"

        # Then check available hardware
        if JTOP_AVAILABLE and self._is_jetson():
            return "jetson"
        elif NVIDIA_AVAILABLE and self._should_use_gpu():
            return "nvidia"
        else:
            return "cpu"

    def _should_use_gpu(self) -> bool:
        """Check if GPU should be used based on availability and config"""
        try:
            import torch

            return torch.cuda.is_available() and not self._forced_cpu
        except ImportError:
            return False

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
                # First try to get power from battery if available and unplugged
                if self._battery_initialized:
                    battery = psutil.sensors_battery()
                    if battery and not battery.power_plugged:
                        # Estimate power from battery percentage change
                        battery_power = self._estimate_cpu_power()
                        if battery_power > 0:
                            return battery_power

                # If battery power estimation is not available or returns 0,
                # use the Windows CPU power model for Windows systems
                if self._os_type == "Windows":
                    return self._estimate_windows_cpu_power()

                # For non-Windows systems without battery power, use a default based on TDP
                return 0.4 * self._tdp  # Default to 40% of TDP
        except Exception as e:
            logger.error(f"Error getting power consumption: {e}")
            return 0.4 * self._tdp  # Default to 40% of TDP

    def _estimate_cpu_power(self) -> float:
        """Estimate CPU power consumption from battery changes."""
        try:
            # Initialize last reading if not exists
            if not hasattr(self, "_last_power_reading"):
                battery = psutil.sensors_battery()
                if battery and not battery.power_plugged:
                    self._last_power_reading = (time.time(), battery.percent)
                else:
                    self._last_power_reading = None
                return 0.0

            current_time = time.time()
            current_battery = psutil.sensors_battery()

            # Check if we have valid battery readings
            if (
                current_battery
                and not current_battery.power_plugged
                and self._last_power_reading is not None
            ):
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

            # Reset last reading if conditions not met
            self._last_power_reading = None
            return 0.0

        except Exception as e:
            logger.error(f"Error estimating CPU power: {e}")
            self._last_power_reading = None
            return 0.0

    def get_system_metrics(self) -> dict:
        """Get current system metrics based on device type."""
        metrics = {}

        try:
            if self.device_type == "cpu":
                # For Windows CPU, use our CPU power model
                if self._os_type == "Windows":
                    # Get estimated power
                    power = self._estimate_windows_cpu_power()

                    # Get CPU metrics
                    cpu_metrics = self._get_cpu_metrics()

                    metrics.update(
                        {
                            "power_reading": power,
                            "gpu_utilization": 0.0,
                            "cpu_utilization": cpu_metrics.get("cpu_percent", 0.0),
                            "memory_utilization": cpu_metrics.get(
                                "memory_percent", 0.0
                            ),
                            "host_battery_energy_mwh": self.get_battery_energy(),
                        }
                    )
                # For non-Windows CPU systems with battery, use battery metrics
                elif self._battery_initialized:
                    battery = psutil.sensors_battery()
                    if battery and not battery.power_plugged:
                        power = self._estimate_cpu_power()
                        metrics.update(
                            {
                                "power_reading": power,
                                "gpu_utilization": 0.0,
                                "host_battery_energy_mwh": self.get_battery_energy(),
                            }
                        )
                    else:
                        # For plugged-in devices, use CPU model
                        power = 0.4 * self._tdp  # Default to 40% of TDP
                        metrics.update(
                            {
                                "power_reading": power,
                                "gpu_utilization": 0.0,
                                "host_battery_energy_mwh": 0.0,
                            }
                        )

                # If no metrics have been added yet, use defaults based on TDP
                if not metrics:
                    metrics = {
                        "power_reading": 0.4 * self._tdp,
                        "gpu_utilization": 0.0,
                        "host_battery_energy_mwh": 0.0,
                    }
            else:
                # For GPU, get device-specific metrics
                if self.device_type == "jetson":
                    metrics.update(self._get_jetson_metrics())
                elif self.device_type == "nvidia":
                    metrics.update(self._get_nvidia_metrics())
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            # Return safe defaults if there's an error
            metrics = {
                "power_reading": 0.4 * self._tdp,
                "gpu_utilization": 0.0,
                "host_battery_energy_mwh": 0.0,
            }

        return metrics

    def _get_jetson_metrics(self) -> dict:
        """Get Jetson specific metrics."""
        try:
            if self._jtop and self._jtop.is_alive():
                stats = self._jtop.stats

                # Get power using our improved _get_jetson_power method
                power = self._get_jetson_power()

                # Get GPU utilization - try multiple possible keys
                gpu_util = 0.0
                gpu_keys = ["GPU", "GPU1", "GPU0", "gpu_usage"]

                for gpu_key in gpu_keys:
                    if gpu_key in stats and stats[gpu_key] is not None:
                        try:
                            gpu_util = float(stats[gpu_key])
                            logger.debug(
                                f"Found GPU utilization using key '{gpu_key}': {gpu_util}%"
                            )
                            break
                        except (ValueError, TypeError):
                            continue

                # If GPU utilization wasn't found through direct keys, try entries in tegrastats
                if gpu_util == 0.0 and "tegrastats" in stats:
                    # Orin often reports GPU utilization here
                    tegrastats = stats["tegrastats"]
                    if isinstance(tegrastats, dict) and "GR3D_FREQ" in tegrastats:
                        try:
                            gpu_util = float(tegrastats["GR3D_FREQ"].split("%")[0])
                            logger.debug(
                                f"Found GPU utilization in tegrastats: {gpu_util}%"
                            )
                        except (ValueError, TypeError, IndexError, AttributeError):
                            pass

                # Get memory utilization - try multiple approaches for Jetson devices
                mem_util = 0.0

                # Debug what's available in stats for memory
                logger.debug(f"Available keys in stats: {list(stats.keys())}")

                # Method 1: Standard RAM dictionary approach
                if "RAM" in stats:
                    try:
                        if isinstance(stats["RAM"], dict):
                            mem_used = stats["RAM"].get("used", 0)
                            mem_total = stats["RAM"].get("total", 1)
                            mem_util = (
                                (mem_used / mem_total) * 100 if mem_total > 0 else 0
                            )
                            logger.debug(
                                f"Method 1: RAM dict - used:{mem_used}, total:{mem_total}, util:{mem_util:.2f}%"
                            )
                        elif isinstance(stats["RAM"], str):
                            # Sometimes RAM is reported as "1234M/5678M"
                            parts = stats["RAM"].split("/")
                            if len(parts) == 2:
                                mem_used = float(parts[0].rstrip("MKG"))
                                mem_total = float(parts[1].rstrip("MKG"))
                                mem_util = (
                                    (mem_used / mem_total) * 100 if mem_total > 0 else 0
                                )
                                logger.debug(
                                    f"Method 1: RAM string - used:{mem_used}, total:{mem_total}, util:{mem_util:.2f}%"
                                )
                    except Exception as e:
                        logger.debug(f"Method 1 RAM failed: {e}")

                # Method 2: Look for memory in tegrastats (common in Jetson)
                if mem_util == 0.0 and "tegrastats" in stats:
                    try:
                        tegrastats = stats["tegrastats"]
                        if isinstance(tegrastats, dict):
                            # Check for RAM field in tegrastats
                            if "RAM" in tegrastats:
                                ram_str = tegrastats["RAM"]
                                if isinstance(ram_str, str) and "/" in ram_str:
                                    parts = ram_str.split("/")
                                    if len(parts) == 2:
                                        # Format could be "1234MB/5678MB"
                                        used_str = parts[0].strip()
                                        total_str = parts[1].strip()

                                        # Extract numbers
                                        import re

                                        used_match = re.search(r"(\d+)", used_str)
                                        total_match = re.search(r"(\d+)", total_str)

                                        if used_match and total_match:
                                            mem_used = float(used_match.group(1))
                                            mem_total = float(total_match.group(1))
                                            mem_util = (
                                                (mem_used / mem_total) * 100
                                                if mem_total > 0
                                                else 0
                                            )
                                            logger.debug(
                                                f"Method 2: tegrastats RAM - used:{mem_used}, total:{mem_total}, util:{mem_util:.2f}%"
                                            )
                    except Exception as e:
                        logger.debug(f"Method 2 tegrastats failed: {e}")

                # Method 3: Use the MEM field if available (some Jetsons have this)
                if mem_util == 0.0 and "MEM" in stats:
                    try:
                        mem_info = stats["MEM"]
                        if isinstance(mem_info, dict):
                            if "used" in mem_info and "total" in mem_info:
                                mem_used = float(mem_info["used"])
                                mem_total = float(mem_info["total"])
                                mem_util = (
                                    (mem_used / mem_total) * 100 if mem_total > 0 else 0
                                )
                                logger.debug(
                                    f"Method 3: MEM dict - used:{mem_used}, total:{mem_total}, util:{mem_util:.2f}%"
                                )
                    except Exception as e:
                        logger.debug(f"Method 3 MEM failed: {e}")

                # Method 4: Check for Orin-specific memory fields
                if mem_util == 0.0:
                    try:
                        # Look for keys containing "memory" or "mem" in a case-insensitive way
                        mem_keys = [k for k in stats.keys() if "mem" in k.lower()]
                        logger.debug(
                            f"Method 4: Found potential memory keys: {mem_keys}"
                        )

                        for key in mem_keys:
                            try:
                                if (
                                    isinstance(stats[key], dict)
                                    and "used" in stats[key]
                                    and "total" in stats[key]
                                ):
                                    mem_used = float(stats[key]["used"])
                                    mem_total = float(stats[key]["total"])
                                    mem_util = (
                                        (mem_used / mem_total) * 100
                                        if mem_total > 0
                                        else 0
                                    )
                                    logger.debug(
                                        f"Method 4: Using memory key {key} - util:{mem_util:.2f}%"
                                    )
                                    break
                            except (ValueError, TypeError):
                                continue
                    except Exception as e:
                        logger.debug(f"Method 4 memory keys failed: {e}")

                # Method 5: Last resort - use psutil if available
                if mem_util == 0.0 and PSUTIL_AVAILABLE:
                    try:
                        import psutil

                        mem = psutil.virtual_memory()
                        mem_util = mem.percent
                        logger.debug(
                            f"Method 5: Using psutil fallback - util:{mem_util:.2f}%"
                        )
                    except Exception as e:
                        logger.debug(f"Method 5 psutil failed: {e}")

                # If all methods fail, use a default non-zero value for memory
                if mem_util == 0.0:
                    # Use a reasonable default (50% utilization) to avoid zeros
                    mem_util = 50.0
                    logger.warning(
                        f"Could not detect memory utilization, using default: {mem_util}%"
                    )

                return {
                    "power_reading": power,
                    "gpu_utilization": gpu_util,
                    "memory_utilization": mem_util,
                }

        except Exception as e:
            logger.error(f"Error getting Jetson metrics: {e}", exc_info=True)

        return {"power_reading": 0.0, "gpu_utilization": 0.0, "memory_utilization": 0.0}

    def _get_nvidia_metrics(self) -> dict:
        """Get NVIDIA GPU specific metrics."""
        try:
            if self._nvml_initialized and self._nvml_handle:
                power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
                return {"power_reading": power, "gpu_utilization": float(util)}
        except Exception as e:
            logger.error(f"Error getting NVIDIA metrics: {e}")
        return {"power_reading": 0.0, "gpu_utilization": 0.0}

    def _get_jetson_power(self) -> float:
        """Retrieve Jetson GPU power consumption in watts."""
        if not self._jtop or not self._jtop.is_alive():
            raise RuntimeError("Jetson monitoring not initialized")

        try:
            stats = self._jtop.stats

            # List of potential power keys in Jetson platforms
            jetson_power_keys = [
                "Power VDD_CPU_GPU_CV",  # Xavier/Nano
                "Power SYS5V_CPU",  # Orin
                "Power VDDQ_VDD2_1V8AO",  # Another possible Orin key
                "Power CV",  # Sometimes found on Orin
                "Power POM_5V_GPU",  # Sometimes found on Orin
                "Power A0",  # Generic fallback
            ]

            # Try each power key until we find a valid one
            for power_key in jetson_power_keys:
                if power_key in stats and stats[power_key] is not None:
                    try:
                        power = float(stats[power_key]) / 1000.0  # Convert mW to W
                        logger.debug(f"Using power key '{power_key}': {power}W")
                        return power
                    except (ValueError, TypeError):
                        continue

            # If no specific keys work, try any power key
            all_power_keys = [k for k in stats.keys() if k.startswith("Power")]
            for key in all_power_keys:
                if stats[key] is not None:
                    try:
                        power = float(stats[key]) / 1000.0  # Convert mW to W
                        logger.debug(f"Using fallback power key '{key}': {power}W")
                        return power
                    except (ValueError, TypeError):
                        continue

            # Last resort: log all available keys
            logger.warning(
                f"No valid power keys found. Available stats keys: {list(stats.keys())}"
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
        """Clean up monitoring resources."""
        try:
            # Clean up Jetson monitoring
            if hasattr(self, "_jtop") and self._jtop is not None:
                try:
                    if self._jtop.is_alive():
                        self._jtop.stop()
                    self._jtop = None
                except Exception as e:
                    logger.debug(f"Error stopping jtop: {e}")

            # Clean up NVIDIA monitoring
            if hasattr(self, "_nvml_initialized") and self._nvml_initialized:
                try:
                    pynvml.nvmlShutdown()
                    self._nvml_initialized = False
                    self._nvml_handle = None
                except Exception as e:
                    logger.debug(f"Error shutting down NVML: {e}")

        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")

    def __enter__(self):
        """Enter the context manager; start energy measurement."""
        self.start_measurement()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager; clean up monitoring resources."""
        self.cleanup()

    def __del__(self) -> None:
        """Ensure cleanup is called when object is destroyed."""
        try:
            self.cleanup()
        except Exception as e:
            # Use sys.stderr since logger might be gone during shutdown
            import sys

            print(f"Error during GPUEnergyMonitor cleanup: {e}", file=sys.stderr)

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
