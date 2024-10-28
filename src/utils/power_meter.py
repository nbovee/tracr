# src/utils/power_meter.py

import time
from typing import Union

import psutil  # type: ignore
import torch


class PowerMeter:
    """Measures CPU and GPU energy usage over time."""

    def __init__(self, device: Union[str, torch.device]):
        """Initialize with the specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=None)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_gpu_memory = torch.cuda.max_memory_allocated(self.device)

    def get_energy(self) -> float:
        """Calculate the energy used since initialization."""
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
