import torch
import time
import psutil  # type: ignore


class PowerMeter:
    def __init__(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.start_time = time.time()
        self.start_cpu_percent = psutil.cpu_percent(interval=None)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_gpu_memory = torch.cuda.max_memory_allocated(self.device)

    def get_energy(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        energy = cpu_percent - self.start_cpu_percent

        if self.device.type == "cuda":
            current_gpu_memory = torch.cuda.max_memory_allocated(self.device)
            energy += (current_gpu_memory - self.start_gpu_memory) / (
                1024 * 1024
            )  # Convert to MB

        return energy

    def get_total_power(self):
        end_time = time.time()
        energy_used = self.get_energy()
        time_elapsed = end_time - self.start_time
        return energy_used / time_elapsed if time_elapsed > 0 else 0
