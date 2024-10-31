# src/utils/power_meter.py

import time
import logging
from typing import Union, Optional

import psutil  # type: ignore
import torch

logger = logging.getLogger("split_computing_logger")

try:
    from jtop import JtopException, jtop  # type: ignore

    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    logger.warning("jtop not available - falling back to basic power monitoring")


class PowerMeter:
    """Measures CPU and GPU energy usage over time."""

    def __init__(self, device: Union[str, torch.device]):
        """Initialize with the specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.reset()
        self.is_jetson = JTOP_AVAILABLE
        self._jtop: Optional[jtop] = None
        if self.is_jetson:
            try:
                self._jtop = jtop()
                self._jtop.start()
                logger.info("Successfully initialized Jetson power monitoring")
            except JtopException as e:
                logger.error(f"Failed to initialize jtop: {e}")
                self.is_jetson = False

    def reset(self):
        """Reset all measurements."""
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=None)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_gpu_memory = torch.cuda.max_memory_allocated(self.device)

    def __enter__(self):
        """Context manager entry."""
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._jtop and self._jtop.is_alive():
            try:
                self._jtop.close()
            except Exception as e:
                logger.warning(f"Error closing jtop connection: {e}")

    def get_energy(self) -> float:
        """Calculate the energy used since initialization."""
        if self.is_jetson and self._jtop and self._jtop.is_alive():
            try:
                # Get stats from jtop
                stats = self._jtop.stats
                if not stats:
                    logger.warning("No stats available from Jetson")
                    return self._get_fallback_energy()

                # Get power readings from all components
                power_readings = []

                # Get power readings from stats
                if "power" in stats:
                    power_dict = stats["power"]
                    # Iterate through all power rails
                    for rail_name, rail_data in power_dict.items():
                        try:
                            # Different Jetson models might report power differently
                            if isinstance(rail_data, dict):
                                if "power" in rail_data:
                                    power_readings.append(float(rail_data["power"]))
                                elif "instant" in rail_data:
                                    power_readings.append(float(rail_data["instant"]))
                            elif isinstance(rail_data, (int, float)):
                                power_readings.append(float(rail_data))
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Skipping rail {rail_name}: {e}")
                            continue

                if power_readings:
                    # Return the sum of all available power readings
                    total_power = sum(power_readings)
                    logger.debug(f"Jetson power readings: {total_power:.2f}W")
                    return total_power
                else:
                    logger.warning("No valid power readings available from Jetson")
                    return self._get_fallback_energy()

            except Exception as e:
                logger.warning(f"Failed to get Jetson power metrics: {e}")
                return self._get_fallback_energy()
        return self._get_fallback_energy()

    def _get_fallback_energy(self) -> float:
        """Fallback method for energy calculation."""
        current_cpu_percent = psutil.cpu_percent(interval=None)
        energy = current_cpu_percent - self.start_cpu_percent

        if self.device.type == "cuda":
            current_gpu_memory = torch.cuda.max_memory_allocated(self.device)
            energy += (current_gpu_memory - self.start_gpu_memory) / (1024 * 1024)  # MB

        return energy

    def get_total_power(self) -> float:
        """Compute the average power usage since initialization."""
        elapsed_time = time.time() - self.start_time
        energy_used = self.get_energy()
        return energy_used / elapsed_time if elapsed_time > 0 else 0.0

    def __del__(self):
        """Cleanup jtop connection on deletion."""
        if self._jtop and self._jtop.is_alive():
            try:
                self._jtop.close()
            except Exception as e:
                logger.warning(f"Error closing jtop connection during cleanup: {e}")
