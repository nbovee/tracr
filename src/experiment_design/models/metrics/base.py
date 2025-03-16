"""Base class for power and energy monitoring."""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

logger = logging.getLogger("split_computing_logger")


class PowerMonitor(ABC):
    """Abstract base class for power and energy monitoring.

    This class defines the interface for all power monitoring implementations
    and provides common functionality for measurement timing and energy calculation.

    Attributes:
        device_type: Type of device being monitored
        _start_time: Start time of current measurement
        _start_power: Power level at start of measurement
        _start_metrics: System metrics at start of measurement
    """

    def __init__(self, device_type: str) -> None:
        """Initialize the power monitor.

        Args:
            device_type: Type of device being monitored
        """
        self.device_type = device_type
        self._start_time = 0.0
        self._start_power = 0.0
        self._start_metrics = {}
        logger.info(f"Initialized {self.device_type} power monitor")

    def start_measurement(self) -> float:
        """Start energy measurement and return the start timestamp.

        Returns:
            Start timestamp in seconds since epoch
        """
        self._start_power = self.get_current_power()
        self._start_metrics = self.get_system_metrics()
        self._start_time = time.time()
        return self._start_time

    def end_measurement(self) -> Tuple[float, float]:
        """End measurement and return energy consumed and elapsed time.

        Returns:
            Tuple of (energy consumed in joules, elapsed time in seconds)
        """
        end_time = time.time()
        end_power = self.get_current_power()

        # Compute average power over the interval
        avg_power = (self._start_power + end_power) / 2
        elapsed_time = end_time - self._start_time
        energy_joules = avg_power * elapsed_time

        return energy_joules, elapsed_time

    @abstractmethod
    def get_current_power(self) -> float:
        """Get current power consumption in watts.

        Returns:
            Current power consumption in watts
        """
        pass

    @abstractmethod
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.

        Returns:
            Dictionary of system metrics
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the monitor."""
        pass
