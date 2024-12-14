# src/api/power_monitor.py

import time
import logging
from typing import Union, Optional, Dict, Any, List, Tuple
from pathlib import Path

import psutil  # type: ignore
import torch
import pandas as pd

logger = logging.getLogger("split_computing_logger")

# Optional imports for different platforms
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


class PowerMeter:
    """Generic power meter for different computing devices."""

    def __init__(self, device: Union[str, torch.device]):
        """Initialize power monitoring for the specified device.

        Args:
            device: Device to monitor ('cpu', 'cuda', or torch.device)
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.platform = self._detect_platform()
        self._nvml_initialized = False

        self.reset()
        self._setup_monitors()
        logger.info(f"Initialized power monitoring for {self.platform} platform")

    def _detect_platform(self) -> str:
        """Detect the computing platform."""
        if JTOP_AVAILABLE and self._is_jetson():
            return "jetson"
        elif NVIDIA_AVAILABLE and self.device.type == "cuda":
            return "nvidia"
        elif self.device.type == "cpu":
            return "cpu"
        return "unknown"

    def _is_jetson(self) -> bool:
        """Check if running on Jetson platform."""
        jetson_paths = [
            "/sys/bus/i2c/drivers/ina3221x",
            "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device",
            "/usr/bin/tegrastats",
        ]
        return any(Path(p).exists() for p in jetson_paths)

    def _initialize_nvml(self) -> bool:
        """Initialize NVML if not already initialized."""
        if not self._nvml_initialized and NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                logger.debug("NVML initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
        return False

    def _setup_monitors(self) -> None:
        """Setup appropriate monitoring tools based on platform."""
        self._jtop = None
        self._nvml_handle = None

        if self.platform == "jetson" and JTOP_AVAILABLE:
            try:
                self._jtop = jtop()
                self._jtop.start()
            except JtopException as e:
                logger.error(f"Failed to initialize Jetson monitoring: {e}")

        elif self.platform == "nvidia" and NVIDIA_AVAILABLE:
            try:
                self._initialize_nvml()
                device_idx = self.device.index if self.device.index is not None else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIA monitoring: {e}")

    def reset(self) -> None:
        """Reset all measurements."""
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=None)
        self.start_memory = psutil.Process().memory_info().rss

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_gpu_memory = torch.cuda.max_memory_allocated(self.device)

    def get_power_metrics(self) -> Dict[str, Any]:
        """Get comprehensive power and resource usage metrics."""
        current_time = time.time()
        metrics = {
            "timestamp": current_time - self.start_time,
            "datetime": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(current_time)
            ),
            "platform": self.platform,
            "cpu": self._get_cpu_metrics(),
            "memory": self._get_memory_metrics(),
        }

        if self.device.type == "cuda":
            metrics["gpu"] = self._get_gpu_metrics()

        return metrics

    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU-related metrics."""
        return {
            "percent": psutil.cpu_percent(interval=None),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "temperature": self._get_cpu_temperature(),
        }

    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory usage metrics."""
        process = psutil.Process()
        return {
            "rss": process.memory_info().rss,
            "vms": process.memory_info().vms,
            "percent": process.memory_percent(),
        }

    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU-related metrics."""
        metrics = {}

        if self.platform == "nvidia" and NVIDIA_AVAILABLE:
            try:
                if not self._nvml_initialized:
                    self._initialize_nvml()
                    device_idx = (
                        self.device.index if self.device.index is not None else 0
                    )
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

                if self._nvml_handle:
                    # Get basic metrics
                    power_watts = (
                        pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                    )
                    temp = pynvml.nvmlDeviceGetTemperature(
                        self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)

                    # Get utilization rates
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)

                    # Get clock speeds
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(
                        self._nvml_handle, pynvml.NVML_CLOCK_GRAPHICS
                    )
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(
                        self._nvml_handle, pynvml.NVML_CLOCK_MEM
                    )

                    # Calculate memory utilization percentage
                    mem_used_gb = mem_info.used / (1024**3)  # Convert to GB
                    mem_total_gb = mem_info.total / (1024**3)
                    mem_util_percent = (mem_info.used / mem_info.total) * 100

                    metrics = {
                        "power": {"current": round(power_watts, 2), "unit": "W"},
                        "temperature": {"current": temp, "unit": "°C"},
                        "memory": {
                            "used": round(mem_used_gb, 2),
                            "total": round(mem_total_gb, 2),
                            "utilization": round(mem_util_percent, 1),
                            "unit": "GB",
                        },
                        "utilization": {
                            "gpu": util_rates.gpu,  # GPU utilization percentage
                            "memory": util_rates.memory,  # Memory controller utilization
                            "unit": "%",
                        },
                        "clocks": {
                            "graphics": graphics_clock,
                            "memory": memory_clock,
                            "unit": "MHz",
                        },
                    }

                    # Add power efficiency metric (GFLOPS/Watt if available)
                    if util_rates.gpu > 0 and power_watts > 0:
                        efficiency = util_rates.gpu / power_watts
                        metrics["efficiency"] = {
                            "ops_per_watt": round(efficiency, 2),
                            "unit": "util%/W",
                        }

            except Exception as e:
                logger.error(f"Error getting NVIDIA GPU metrics: {e}")
                self._nvml_initialized = False
                try:
                    self._initialize_nvml()
                    device_idx = (
                        self.device.index if self.device.index is not None else 0
                    )
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                except Exception as reinit_error:
                    logger.error(f"Failed to reinitialize NVML: {reinit_error}")

        return metrics

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available."""
        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:
                return sum(t.current for t in temps["coretemp"]) / len(
                    temps["coretemp"]
                )
            return None
        except Exception:
            return None

    def __enter__(self):
        """Context manager entry."""
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def cleanup(self):
        """Clean up monitoring resources."""
        if self._jtop and self._jtop.is_alive():
            try:
                self._jtop.close()
            except Exception as e:
                logger.warning(f"Error closing jtop connection: {e}")

        # Only shutdown NVML if we initialized it
        if self._nvml_initialized and NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                logger.debug("NVML shutdown successful")
            except Exception as e:
                logger.warning(f"Error shutting down NVML: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


class PowerAnalyzer:
    """Analyzes and formats power monitoring data."""

    @staticmethod
    def analyze_metrics(power_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze power metrics and return summary statistics."""
        if not power_metrics:
            return {}

        try:
            # Calculate total power consumption
            total_power = sum(m["gpu"]["power"]["current"] for m in power_metrics)

            power_analysis = {
                # Store original metrics for timestamp reference
                "metrics": power_metrics,
                # Power metrics
                "total_power": total_power,  # Add total power
                "avg_power_watts": total_power / len(power_metrics),
                # Efficiency metrics
                "avg_efficiency": sum(
                    m["gpu"].get("efficiency", {}).get("ops_per_watt", 0)
                    for m in power_metrics
                )
                / len(power_metrics),
                # Memory utilization
                "memory_utils": [
                    m["gpu"]["memory"]["utilization"] for m in power_metrics
                ],
                "max_memory_util": max(
                    m["gpu"]["memory"]["utilization"] for m in power_metrics
                ),
                "avg_memory_util": sum(
                    m["gpu"]["memory"]["utilization"] for m in power_metrics
                )
                / len(power_metrics),
                # GPU utilization pattern (keep relative time)
                "gpu_util_pattern": [
                    (m["timestamp"], m["gpu"]["utilization"]["gpu"])
                    for m in power_metrics
                ],
                # Temperature pattern (keep relative time)
                "temp_pattern": [
                    (m["timestamp"], m["gpu"]["temperature"]["current"])
                    for m in power_metrics
                ],
                # Additional derived metrics
                "peak_power": max(m["gpu"]["power"]["current"] for m in power_metrics),
                "peak_temperature": max(
                    m["gpu"]["temperature"]["current"] for m in power_metrics
                ),
                "avg_gpu_util": sum(
                    m["gpu"]["utilization"]["gpu"] for m in power_metrics
                )
                / len(power_metrics),
            }
            return power_analysis
        except Exception as e:
            logger.error(f"Error analyzing power metrics: {e}")
            return {}

    @staticmethod
    def create_analysis_dataframes(
        power_analysis: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create DataFrames for power analysis results."""
        # Create analysis DataFrame
        analysis_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Power (W)",  # Add Total Power metric
                    "Average Power (W)",
                    "Peak Power (W)",
                    "Power Efficiency (util%/W)",
                    "Maximum Memory Utilization (%)",
                    "Average Memory Utilization (%)",
                    "Average GPU Utilization (%)",
                    "Peak Temperature (°C)",
                ],
                "Value": [
                    round(power_analysis["total_power"], 2),  # Add total power value
                    round(power_analysis["avg_power_watts"], 2),
                    round(power_analysis["peak_power"], 2),
                    round(power_analysis["avg_efficiency"], 2),
                    round(power_analysis["max_memory_util"], 1),
                    round(power_analysis["avg_memory_util"], 1),
                    round(power_analysis["avg_gpu_util"], 1),
                    power_analysis["peak_temperature"],
                ],
            }
        )

        # Create time series DataFrames with both relative and absolute time
        gpu_util_df = pd.DataFrame(
            [
                {
                    "Relative Time (s)": timestamp,
                    "Datetime": m["datetime"],
                    "GPU Utilization (%)": util,
                }
                for (timestamp, util), m in zip(
                    power_analysis["gpu_util_pattern"], power_analysis["metrics"]
                )
            ]
        )

        temp_df = pd.DataFrame(
            [
                {
                    "Relative Time (s)": timestamp,
                    "Datetime": m["datetime"],
                    "Temperature (°C)": temp,
                }
                for (timestamp, temp), m in zip(
                    power_analysis["temp_pattern"], power_analysis["metrics"]
                )
            ]
        )

        return analysis_df, gpu_util_df, temp_df
