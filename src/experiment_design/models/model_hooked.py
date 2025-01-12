# src/experiment_design/models/model_hooked.py

import atexit
import copy
import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Union, ClassVar

import numpy as np
import torch
from PIL import Image
from torchinfo import summary  # type: ignore

from .base import BaseModel
from .hooks import (
    create_forward_prehook,
    create_forward_posthook,
    EarlyOutput,
    HookExitException,
)
from .templates import LAYER_TEMPLATE
from src.api import MasterDict
from src.interface import ModelInterface
from .power_monitor import GPUEnergyMonitor

atexit.register(torch.cuda.empty_cache)
logger = logging.getLogger("split_computing_logger")


class WrappedModel(BaseModel, ModelInterface):
    """Model wrapper implementing hook-based functionality for edge computing experiments."""

    DEFAULT_DEPTH: ClassVar[int] = 2
    DEFAULT_BUFFER_SIZE: ClassVar[int] = 100
    DEFAULT_WARMUP_ITERS: ClassVar[int] = 2

    def __init__(
        self,
        config: Dict[str, Any],
        master_dict: Optional[MasterDict] = None,
        **kwargs,
    ) -> None:
        """Initialize wrapped model with configuration and optional master dictionary."""
        BaseModel.__init__(self, config)
        ModelInterface.__init__(self, config)
        logger.debug(f"Initializing WrappedModel with config: {config}")

        # Basic model attributes
        self.timer = time.perf_counter_ns
        self.master_dict = master_dict
        self.io_buffer = {}
        self.inference_info = {}  # Stores per-inference metrics
        self.forward_info = {}  # Stores per-layer metrics
        self.forward_hooks = []
        self.forward_post_hooks = []
        self.save_layers = getattr(self.model, "save", {})
        self.layer_times = {}  # Temporary storage for layer timing
        self.layer_timing_data = {}  # Historical timing data
        self.layer_energy_data = {}  # Historical energy data

        # Initialize energy monitoring
        try:
            self.energy_monitor = GPUEnergyMonitor()
            logger.info("GPU energy monitoring initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GPU energy monitoring: {e}")
            self.energy_monitor = None

        # Hook state tracking
        self.start_i: Optional[int] = None  # First layer to process
        self.stop_i: Optional[int] = None  # Last layer to process
        self.banked_output: Optional[Any] = None  # Store intermediate outputs
        self.log = False  # Enable/disable metric collection
        self.current_energy_start = None  # Track energy measurement timing

        # Setup model layers and hooks
        self._setup_model()
        logger.info("WrappedModel initialization complete")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "energy_monitor") and self.energy_monitor:
            self.energy_monitor.cleanup()

    def _setup_model(self) -> None:
        """Set up model layers, hooks, and initialize state."""
        self.torchinfo_summary = summary(
            self.model, (1, *self.input_size), device=self.device, verbose=0
        )
        self.layer_count = self._walk_modules(self.model.children(), depth=1, walk_i=0)
        del self.torchinfo_summary

        # Store empty forward info for resets
        self.forward_info_empty = copy.deepcopy(self.forward_info)
        logger.debug(f"Model initialized with {self.layer_count} layers")

        # Perform warmup iterations
        self.warmup(iterations=self.warmup_iterations)

    def _walk_modules(self, modules: Any, depth: int, walk_i: int) -> int:
        """Register hooks on model layers recursively."""
        for child in modules:
            child_name = child.__class__.__name__
            children = list(child.children())

            if children and depth < self.depth:
                logger.debug(f"{'-' * depth}Module {child_name} has children")
                walk_i = self._walk_modules(children, depth + 1, walk_i)
            elif isinstance(child, torch.nn.Module):
                walk_i = self._register_layer(child, child_name, depth, walk_i)

        return walk_i

    def _register_layer(
        self, layer: torch.nn.Module, layer_name: str, depth: int, walk_i: int
    ) -> int:
        """Register hooks and info for a single layer."""
        layer_info = next(
            (
                info
                for info in self.torchinfo_summary.summary_list
                if info.layer_id == id(layer)
            ),
            None,
        )

        if layer_info:
            # Initialize metrics storage for layer
            self.forward_info[walk_i] = copy.deepcopy(LAYER_TEMPLATE)
            self.forward_info[walk_i].update(
                {
                    "layer_id": walk_i,
                    "layer_type": layer_info.class_name,
                    "output_bytes": layer_info.output_bytes,  # Initial output size estimate
                    "inference_time": None,  # Placeholder for timing
                }
            )

            # Attach pre and post hooks for metric collection
            self.forward_hooks.append(
                layer.register_forward_pre_hook(
                    create_forward_prehook(
                        self, walk_i, layer_name, (0, 0), self.device
                    )
                )
            )
            self.forward_post_hooks.append(
                layer.register_forward_hook(
                    create_forward_posthook(
                        self, walk_i, layer_name, (0, 0), self.device
                    )
                )
            )
            logger.debug(f"Registered layer {walk_i}: {layer_info.class_name}")
            walk_i += 1

        return walk_i

    def forward(
        self,
        x: Union[torch.Tensor, Image.Image],
        inference_id: Optional[str] = None,
        start: int = 0,
        end: Union[int, float] = np.inf,
        log: bool = True,
    ) -> Any:
        """Execute forward pass with optional slicing and logging."""
        start_time = self.timer()
        end = self.layer_count if end == np.inf else end
        logger.info(
            f"Starting forward pass: id={inference_id}, start={start}, end={end}, log={log}"
        )

        # Configure forward pass
        self.log = log  # Make sure logging is enabled
        logger.debug(f"Logging is {'enabled' if self.log else 'disabled'}")
        self.start_i = start
        self.stop_i = end
        self._setup_inference_id(inference_id)

        # Execute forward pass
        try:
            output = self._execute_forward(x)
        except HookExitException as e:
            output = self._handle_early_exit(e)

        # Handle results
        self._handle_results(start_time)
        return output

    def _setup_inference_id(self, inference_id: Optional[str]) -> None:
        """Set up inference ID and logging state."""
        if inference_id:
            base_id, *suffix = inference_id.rsplit(".", maxsplit=1)
            suffix = int(suffix[0]) + 1 if suffix else 0
            self.inference_info["inference_id"] = f"{base_id}.{suffix}"
        else:
            # During warmup or when no inference_id is provided, we still want to collect timing data
            # but we don't need to store it in the master_dict
            self.inference_info["inference_id"] = "warmup"

    def _execute_forward(self, x: Union[torch.Tensor, Image.Image]) -> Any:
        """Execute model forward pass with appropriate context."""
        context = torch.no_grad() if self.get_mode() == "eval" else nullcontext()
        with context:
            logger.debug("Starting model forward pass")
            output = self.model(x)
            logger.debug("Completed model forward pass")
            return output

    def _handle_early_exit(self, exception: HookExitException) -> EarlyOutput:
        """Handle early exit from forward pass."""
        output = EarlyOutput(exception.result)

        # Keep timing data for completed layers
        completed_layers = {
            k: v for k, v in self.forward_info.items() if k <= self.stop_i
        }

        # Ensure timing data is preserved for completed layers
        for layer_idx in completed_layers:
            if layer_idx in self.layer_times:
                end_time = time.perf_counter()
                start_time = self.layer_times[layer_idx]
                elapsed_time = end_time - start_time
                completed_layers[layer_idx]["inference_time"] = elapsed_time
                logger.debug(
                    f"Preserved timing for layer {layer_idx}: {elapsed_time:.6f} seconds"
                )

        # Store the completed layers back
        self.forward_info = completed_layers
        logger.debug(
            f"Preserved timing data for {len(completed_layers)} layers during early exit"
        )
        return output

    def _handle_results(self, start_time: int) -> None:
        """Handle forward pass results and logging."""
        # Calculate total inference time
        total_time = self.timer() - start_time
        self.inference_info["total_time"] = total_time
        logger.debug(f"Total forward pass time: {total_time/1e9:.6f} seconds")

        # Store layer metrics for current inference
        current_forward_info = copy.deepcopy(self.forward_info)
        self.inference_info["layer_information"] = current_forward_info

        # Update historical timing data
        for layer_id, info in current_forward_info.items():
            if info.get("inference_time") is not None:
                logger.debug(
                    f"Layer {layer_id} time: {info['inference_time']:.6f} seconds"
                )
                # Add to historical timing data
                if not hasattr(self, "layer_timing_data"):
                    self.layer_timing_data = {}
                if layer_id not in self.layer_timing_data:
                    self.layer_timing_data[layer_id] = []
                self.layer_timing_data[layer_id].append(info["inference_time"])
            else:
                logger.debug(f"No timing data for layer {layer_id}")

        # Buffer results if logging enabled
        if self.log and self.master_dict:
            base_id = self.inference_info["inference_id"].split(".", maxsplit=1)[0]
            self.io_buffer[base_id] = copy.deepcopy(self.inference_info)

            # Flush buffer if full
            if len(self.io_buffer) >= self.flush_buffer_size:
                self.update_master_dict()

        # Reset state for next inference
        self.inference_info.clear()
        self.forward_info = copy.deepcopy(self.forward_info_empty)
        self.layer_times.clear()
        self.banked_output = None

    def update_master_dict(self) -> None:
        """Update master dictionary with buffered data."""
        if self.master_dict and self.io_buffer:
            self.master_dict.update(self.io_buffer)
            self.io_buffer.clear()

    def get_state_dict(self) -> Dict[str, Any]:
        """Return model state dictionary."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into model."""
        self.model.load_state_dict(state_dict)

    def get_layer_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Get collected metrics for all layers."""
        metrics = {}
        for layer_idx, info in self.forward_info.items():
            # Combine all metrics per layer
            metrics[layer_idx] = {
                "layer_id": info.get("layer_id"),
                "layer_type": info.get("layer_type"),
                "inference_time": info.get("inference_time"),
                "output_bytes": info.get("output_bytes"),
                "processing_energy": info.get("processing_energy", 0.0),
                "communication_energy": info.get("communication_energy", 0.0),
                "power_reading": info.get("power_reading", 0.0),
                "gpu_utilization": info.get("gpu_utilization", 0.0),
                "total_energy": info.get("total_energy", 0.0),
            }
        return metrics
