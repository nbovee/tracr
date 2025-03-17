"""Jetson power and energy monitoring using jtop"""

import logging
from typing import Dict, Any

from .base import PowerMonitor
from .exceptions import MonitoringInitError, MeasurementError

logger = logging.getLogger("split_computing_logger")


class JetsonMonitor(PowerMonitor):
    """NVIDIA Jetson power and energy monitoring using the jtop library.

    Implements specialized monitoring for Jetson edge computing devices
    by accessing hardware-specific power sensors through the jetson-stats package.
    """

    def __init__(self) -> None:
        """Initialize Jetson monitoring through jtop library."""
        super().__init__("jetson")
        self._jtop = None

        try:
            from jtop import jtop

            self._jtop = jtop()
            self._jtop.start()
            logger.info("Jetson monitoring initialized")
        except ImportError:
            logger.error(
                "Failed to import jtop. Install with: pip install jetson-stats"
            )
            raise MonitoringInitError("Jetson monitoring requires jtop package")
        except Exception as e:
            logger.error(f"Failed to initialize Jetson monitoring: {e}")
            raise MonitoringInitError(
                f"Failed to initialize Jetson monitoring: {str(e)}"
            )

    def get_current_power(self) -> float:
        """Get power consumption from Jetson-specific power sensors.

        Handles different sensor naming schemes across Jetson platforms
        (Xavier, Nano, Orin) by trying multiple known power rail keys.
        """
        if not self._jtop or not self._jtop.is_alive():
            raise MeasurementError("Jetson monitoring not initialized")

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

            # Last resort: log all available keys and return default
            logger.warning(
                f"No valid power keys found. Available stats keys: {list(stats.keys())}"
            )
            return 0.0

        except Exception as e:
            logger.error(f"Error reading Jetson power stats: {e}")
            return 0.0

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics from Jetson hardware."""
        try:
            if not self._jtop or not self._jtop.is_alive():
                raise MeasurementError("Jetson monitoring not initialized")

            stats = self._jtop.stats

            # Get power using our dedicated method
            power = self.get_current_power()

            # Get GPU utilization
            gpu_util = 0.0
            gpu_keys = ["GPU", "GPU1", "GPU0", "gpu_usage"]

            for gpu_key in gpu_keys:
                if gpu_key in stats and stats[gpu_key] is not None:
                    try:
                        gpu_util = float(stats[gpu_key])
                        break
                    except (ValueError, TypeError):
                        continue

            # If GPU utilization wasn't found through direct keys, try tegrastats
            if gpu_util == 0.0 and "tegrastats" in stats:
                tegrastats = stats["tegrastats"]
                if isinstance(tegrastats, dict) and "GR3D_FREQ" in tegrastats:
                    try:
                        gpu_util = float(tegrastats["GR3D_FREQ"].split("%")[0])
                    except (ValueError, TypeError, IndexError, AttributeError):
                        pass

            # Get memory utilization using our best effort across multiple approaches
            mem_util = self._get_memory_utilization(stats)

            return {
                "power_reading": power,
                "gpu_utilization": gpu_util,
                "memory_utilization": mem_util,
                "device_type": "jetson",
            }
        except Exception as e:
            logger.error(f"Error getting Jetson metrics: {e}")
            return {
                "power_reading": 0.0,
                "gpu_utilization": 0.0,
                "memory_utilization": 50.0,  # Default to avoid zeros
                "device_type": "jetson",
                "error": str(e),
            }

    def _get_memory_utilization(self, stats: Dict[str, Any]) -> float:
        """Extract memory utilization using multiple fallback approaches.

        Implements a multi-layered approach to ensure memory data is available:
        1. Standard RAM dictionary with used/total fields
        2. RAM string parsing for formats like "1234M/5678M"
        3. Search for alternative memory keys in stats
        4. Use psutil as fallback for system memory
        """
        mem_util = 0.0

        # Method 1: Standard RAM dictionary approach
        if "RAM" in stats:
            try:
                if isinstance(stats["RAM"], dict):
                    mem_used = stats["RAM"].get("used", 0)
                    mem_total = stats["RAM"].get("total", 1)
                    mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                elif isinstance(stats["RAM"], str):
                    # Sometimes RAM is reported as "1234M/5678M"
                    parts = stats["RAM"].split("/")
                    if len(parts) == 2:
                        mem_used = float(parts[0].rstrip("MKG"))
                        mem_total = float(parts[1].rstrip("MKG"))
                        mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0
            except Exception:
                pass

        # Method 2: Check for other memory keys if method 1 failed
        if mem_util == 0.0:
            mem_keys = [k for k in stats.keys() if "mem" in k.lower()]
            for key in mem_keys:
                try:
                    if (
                        isinstance(stats[key], dict)
                        and "used" in stats[key]
                        and "total" in stats[key]
                    ):
                        mem_used = float(stats[key]["used"])
                        mem_total = float(stats[key]["total"])
                        mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                        break
                except (ValueError, TypeError):
                    continue

        # Method 3: Use psutil if available
        if mem_util == 0.0:
            try:
                import psutil

                mem = psutil.virtual_memory()
                mem_util = mem.percent
            except ImportError:
                pass

        # If all methods fail, use a default non-zero value
        if mem_util == 0.0:
            mem_util = 50.0

        return mem_util

    def cleanup(self) -> None:
        """Stop jtop monitoring and release resources."""
        if hasattr(self, "_jtop") and self._jtop and self._jtop.is_alive():
            try:
                self._jtop.close()
                self._jtop = None
                logger.debug("Jetson monitoring shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down jtop: {e}")
