# src/api/power_monitor.py

import time
import logging
from typing import Union, Optional, Dict, Any, List, Tuple
from pathlib import Path

import psutil  # type: ignore
import torch
import pandas as pd
import platform
import subprocess

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

# Add these constants
WINDOWS_BATTERY_CMD = (
    "WMIC PATH Win32_Battery Get EstimatedChargeRemaining,EstimatedRunTime /Format:CSV"
)
LINUX_BATTERY_PATH = "/sys/class/power_supply"
MACOS_BATTERY_CMD = "pmset -g batt"


class PowerMeter:
    """Generic power meter for different computing devices."""

    def __init__(self, device: Union[str, torch.device]):
        """Initialize power monitoring for the specified device.

        Args:
            device: Device to monitor ('cpu', 'cuda', or torch.device)
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.os_type = platform.system().lower()
        self.platform = self._detect_platform()
        self._nvml_initialized = False
        self.battery_metrics = {}
        self.initial_battery_info = None  # Store initial battery info

        self.reset()
        self._setup_monitors()
        logger.info(
            f"Initialized power monitoring for {self.platform} platform on {self.os_type}"
        )

    def _detect_platform(self) -> str:
        """Detect the computing platform and OS."""
        if JTOP_AVAILABLE and self._is_jetson():
            return "jetson"
        elif NVIDIA_AVAILABLE and self.device.type == "cuda":
            return "nvidia"
        elif self.os_type == "windows":
            return "windows"
        elif self.os_type == "linux":
            return "linux"
        elif self.os_type == "darwin":
            return "macos"
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

        # Store initial battery info
        self.initial_battery_info = self._get_battery_info()
        logger.debug(f"Initial battery info: {self.initial_battery_info}")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_gpu_memory = torch.cuda.max_memory_allocated(self.device)

    def get_power_metrics(self) -> Dict[str, Any]:
        """Get comprehensive power and resource usage metrics."""
        current_time = time.time()
        current_battery_info = self._get_battery_info()

        # Calculate time-based power consumption
        if self.initial_battery_info and current_battery_info:
            initial_charge = self.initial_battery_info.get("charge_remaining", 0)
            current_charge = current_battery_info.get("charge_remaining", 0)
            time_diff = current_battery_info.get(
                "timestamp", current_time
            ) - self.initial_battery_info.get("timestamp", self.start_time)

            if time_diff > 0:
                charge_rate = (
                    initial_charge - current_charge
                ) / time_diff  # % per second
                power_draw = (
                    charge_rate * 3600 / 100
                )  # Convert to watts (assuming 100% = 1 hour of battery life)
                current_battery_info["power_draw"] = abs(power_draw)

        metrics = {
            "timestamp": current_time - self.start_time,
            "datetime": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(current_time)
            ),
            "platform": self.platform,
            "os_type": self.os_type,
            "cpu": self._get_cpu_metrics(),
            "memory": self._get_memory_metrics(),
            "battery": {
                "current": current_battery_info,
                "initial": self.initial_battery_info,
                "elapsed_time": current_time - self.start_time,
            },
        }

        # Add GPU metrics if available
        if self.device.type == "cuda":
            gpu_metrics = self._get_gpu_metrics()
            if gpu_metrics:
                metrics["gpu"] = gpu_metrics

        return metrics

    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU-related metrics."""
        metrics = {
            "percent": psutil.cpu_percent(interval=None),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "temperature": self._get_cpu_temperature(),
            "power": self._get_cpu_power(),
        }

        # Add per-core metrics
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        metrics["per_core_percent"] = per_core

        return metrics

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

    def _get_cpu_power(self) -> Optional[float]:
        """Get CPU power consumption if available."""
        try:
            if self.platform == "linux":
                # Try reading from RAPL (Running Average Power Limit) interface
                rapl_path = Path("/sys/class/powercap/intel-rapl")
                if rapl_path.exists():
                    for domain in rapl_path.glob("intel-rapl:*"):
                        energy_uj = float((domain / "energy_uj").read_text())
                        return energy_uj / 1_000_000  # Convert to watts

            elif self.platform == "windows":
                # Use Windows Management Instrumentation (WMI)
                import wmi  # type: ignore

                w = wmi.WMI()
                for processor in w.Win32_Processor():
                    return processor.CurrentVoltage * processor.CurrentClockSpeed / 1000

        except Exception as e:
            logger.debug(f"Error getting CPU power: {e}")
        return None

    def _get_battery_info(self) -> Dict[str, Any]:
        """Get battery information based on OS."""
        try:
            if self.os_type == "windows":
                return self._get_windows_battery_info()
            elif self.os_type == "linux":
                return self._get_linux_battery_info()
            elif self.os_type == "darwin":
                return self._get_macos_battery_info()
        except Exception as e:
            logger.debug(f"Error getting battery info: {e}")
        return {}

    def _get_windows_battery_info(self) -> Dict[str, Any]:
        """Get battery information on Windows."""
        try:
            result = subprocess.check_output(WINDOWS_BATTERY_CMD, shell=True).decode()
            lines = result.strip().split("\n")
            if len(lines) >= 2:  # Header + data
                data = lines[1].split(",")
                # Add timestamp to track when measurement was taken
                return {
                    "charge_remaining": float(data[1]),
                    "time_remaining": float(data[2]) if data[2] != "65535" else None,
                    "timestamp": time.time(),
                }
        except Exception as e:
            logger.debug(f"Error getting Windows battery info: {e}")
        return {}

    def _get_linux_battery_info(self) -> Dict[str, Any]:
        """Get battery information on Linux."""
        try:
            battery_path = Path(LINUX_BATTERY_PATH)
            for bat_dir in battery_path.glob("BAT*"):
                energy_now = float((bat_dir / "energy_now").read_text())
                energy_full = float((bat_dir / "energy_full").read_text())
                power_now = float((bat_dir / "power_now").read_text())

                return {
                    "charge_remaining": (energy_now / energy_full) * 100,
                    "power_draw": power_now / 1000000,  # Convert to watts
                    "energy_remaining": energy_now / 1000000,  # Convert to watt-hours
                    "timestamp": time.time(),
                }
        except Exception as e:
            logger.debug(f"Error getting Linux battery info: {e}")
        return {}

    def _get_macos_battery_info(self) -> Dict[str, Any]:
        """Get battery information on macOS."""
        try:
            result = subprocess.check_output(MACOS_BATTERY_CMD, shell=True).decode()
            if "InternalBattery" in result:
                percentage = float(result.split("\t")[1].split(";")[0].rstrip("%"))
                return {
                    "charge_remaining": percentage,
                }
        except Exception as e:
            logger.debug(f"Error getting macOS battery info: {e}")
        return {}

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
            analysis = {
                "metrics": power_metrics,
                "avg_cpu_percent": sum(m["cpu"]["percent"] for m in power_metrics)
                / len(power_metrics),
                "peak_cpu_percent": max(m["cpu"]["percent"] for m in power_metrics),
                "avg_memory_percent": sum(m["memory"]["percent"] for m in power_metrics)
                / len(power_metrics),
                "peak_memory_percent": max(
                    m["memory"]["percent"] for m in power_metrics
                ),
            }

            # Add battery metrics if available
            battery_metrics = [
                m.get("battery", {}) for m in power_metrics if m.get("battery")
            ]
            if battery_metrics:
                # Get initial and final measurements
                initial_info = battery_metrics[0].get("initial", {})
                final_info = battery_metrics[-1].get("current", {})

                initial_charge = initial_info.get("charge_remaining", 0)
                final_charge = final_info.get("charge_remaining", 0)

                # Calculate total elapsed time
                total_time = battery_metrics[-1].get("elapsed_time", 0)

                # Calculate consumption
                charge_consumed = max(
                    0, initial_charge - final_charge
                )  # Ensure non-negative

                # Calculate average power consumption
                power_draws = [
                    b.get("current", {}).get("power_draw", 0) for b in battery_metrics
                ]
                avg_power = sum(power_draws) / len(power_draws) if power_draws else None

                analysis["battery"] = {
                    "initial_charge": initial_charge,
                    "final_charge": final_charge,
                    "charge_consumed": charge_consumed,
                    "power_consumption": avg_power,
                    "total_time": total_time,
                }

            # Add GPU metrics if available
            if any("gpu" in m for m in power_metrics):
                analysis.update(
                    {
                        "total_power": sum(
                            m["gpu"]["power"]["current"] for m in power_metrics
                        ),
                        "avg_power_watts": sum(
                            m["gpu"]["power"]["current"] for m in power_metrics
                        )
                        / len(power_metrics),
                        "peak_power": max(
                            m["gpu"]["power"]["current"] for m in power_metrics
                        ),
                        "avg_efficiency": sum(
                            m["gpu"].get("efficiency", {}).get("ops_per_watt", 0)
                            for m in power_metrics
                        )
                        / len(power_metrics),
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
                        "gpu_util_pattern": [
                            (m["timestamp"], m["gpu"]["utilization"]["gpu"])
                            for m in power_metrics
                        ],
                        "temp_pattern": [
                            (m["timestamp"], m["gpu"]["temperature"]["current"])
                            for m in power_metrics
                        ],
                        "peak_temperature": max(
                            m["gpu"]["temperature"]["current"] for m in power_metrics
                        ),
                        "avg_gpu_util": sum(
                            m["gpu"]["utilization"]["gpu"] for m in power_metrics
                        )
                        / len(power_metrics),
                    }
                )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing power metrics: {e}")
            logger.exception(e)  # This will log the full traceback
            return {}

    @staticmethod
    def create_analysis_dataframes(
        power_analysis: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create DataFrames for power analysis results."""
        # Basic metrics that are always available
        metrics = [
            ("Average CPU Usage (%)", power_analysis.get("avg_cpu_percent", 0)),
            ("Peak CPU Usage (%)", power_analysis.get("peak_cpu_percent", 0)),
            ("Average Memory Usage (%)", power_analysis.get("avg_memory_percent", 0)),
            ("Peak Memory Usage (%)", power_analysis.get("peak_memory_percent", 0)),
        ]

        # Add battery metrics if available
        if "battery" in power_analysis:
            battery_metrics = [
                (
                    "Initial Battery Charge (%)",
                    power_analysis["battery"].get("initial_charge", 0),
                ),
                (
                    "Final Battery Charge (%)",
                    power_analysis["battery"].get("final_charge", 0),
                ),
                (
                    "Battery Charge Consumed (%)",
                    power_analysis["battery"].get("charge_consumed", 0),
                ),
            ]
            if power_analysis["battery"].get("power_consumption") is not None:
                battery_metrics.append(
                    (
                        "Average Battery Power Consumption (W)",
                        power_analysis["battery"]["power_consumption"],
                    )
                )
            metrics.extend(battery_metrics)

        # Add GPU metrics if available
        if "total_power" in power_analysis:
            gpu_metrics = [
                ("Total GPU Power (W)", power_analysis["total_power"]),
                ("Average GPU Power (W)", power_analysis["avg_power_watts"]),
                ("Peak GPU Power (W)", power_analysis["peak_power"]),
                ("Power Efficiency (util%/W)", power_analysis["avg_efficiency"]),
                ("Maximum Memory Utilization (%)", power_analysis["max_memory_util"]),
                ("Average Memory Utilization (%)", power_analysis["avg_memory_util"]),
                ("Average GPU Utilization (%)", power_analysis["avg_gpu_util"]),
                ("Peak Temperature (°C)", power_analysis["peak_temperature"]),
            ]
            metrics.extend(gpu_metrics)

        # Create analysis DataFrame
        analysis_df = pd.DataFrame(metrics, columns=["Metric", "Value"])
        analysis_df["Value"] = analysis_df["Value"].apply(
            lambda x: round(x, 2) if isinstance(x, float) else x
        )

        # Create empty DataFrames for GPU utilization and temperature if not available
        gpu_util_df = pd.DataFrame(
            columns=["Relative Time (s)", "Datetime", "GPU Utilization (%)"]
        )
        temp_df = pd.DataFrame(
            columns=["Relative Time (s)", "Datetime", "Temperature (°C)"]
        )

        # Fill GPU utilization and temperature DataFrames if data is available
        if "gpu_util_pattern" in power_analysis:
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

        if "temp_pattern" in power_analysis:
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
