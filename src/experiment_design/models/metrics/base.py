"""Base class for power and energy monitoring"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

logger = logging.getLogger("split_computing_logger")


class PowerMonitor(ABC):
    """Abstract base class for power and energy monitoring.

    Defines the interface for hardware-specific power monitoring implementations
    and provides the common trapezoidal integration method for energy calculation.
    """

    def __init__(self, device_type: str) -> None:
        """Initialize the power monitor."""
        self.device_type = device_type
        self._start_time = 0.0
        self._start_power = 0.0
        self._start_metrics = {}
        logger.info(f"Initialized {self.device_type} power monitor")

    def start_measurement(self) -> float:
        """Start energy measurement and return the start timestamp."""
        self._start_power = self.get_current_power()
        self._start_metrics = self.get_system_metrics()
        self._start_time = time.time()
        return self._start_time

    def end_measurement(self) -> Tuple[float, float]:
        """End measurement and calculate energy consumed using trapezoidal integration.

        Uses two-point trapezoidal approximation of the power curve between
        start and end points to calculate energy consumption.
        """
        end_time = time.time()
        end_power = self.get_current_power()

        # Compute average power over the interval using trapezoidal rule
        avg_power = (self._start_power + end_power) / 2
        elapsed_time = end_time - self._start_time
        energy_joules = avg_power * elapsed_time

        return energy_joules, elapsed_time

    @abstractmethod
    def get_current_power(self) -> float:
        """Get instantaneous power consumption in watts."""
        pass

    @abstractmethod
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics (CPU, memory, temperature, etc.)."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release hardware monitors and clean up resources."""
        pass
