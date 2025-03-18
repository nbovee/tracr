"""Exceptions for power monitoring"""


class PowerMonitorError(Exception):
    """Base exception for power monitoring errors."""

    pass


class MonitoringInitError(PowerMonitorError):
    """Exception raised when monitor initialization fails."""

    pass


class MeasurementError(PowerMonitorError):
    """Exception raised when a measurement fails."""

    pass


class HardwareNotSupportedError(PowerMonitorError):
    """Exception raised when the hardware is not supported."""

    pass
