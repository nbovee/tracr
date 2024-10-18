# src/experiment_design/models/model_hooked.py

import atexit
import copy
import logging
import os
import sys
import time
from contextlib import nullcontext
from typing import Any, Optional, Union, Dict

import numpy as np
import torch
from PIL import Image
from torchinfo import summary  # type: ignore

# Add parent module (src) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.api.master_dict import MasterDict
from .base import BaseModel
from .hooks import (
    create_forward_prehook,
    create_forward_posthook,
    NotDict,
    HookExitException,
)
from .templates import LAYER_TEMPLATE
from src.utils.power_meter import PowerMeter

# Register atexit handler to clear CUDA cache
atexit.register(torch.cuda.empty_cache)

# Configure logger
logger = logging.getLogger("tracr_logger")


class WrappedModel(BaseModel):
    """Wraps a pretrained model with features necessary for edge computing tests."""

    def __init__(
        self,
        config: Dict[str, Any],
        master_dict: Optional[MasterDict] = None,
        **kwargs,
    ):
        super().__init__(config)
        logger.info(f"Initializing WrappedModel with config: {config}")

        self.timer = time.perf_counter_ns
        self.master_dict = master_dict
        self.io_buffer = {}
        self.inference_info = {}
        self.forward_info = {}
        self.forward_hooks = []
        self.forward_post_hooks = []
        self.flush_buffer_size = self.flush_buffer_size

        # Load model using the BaseModel's load_model method
        self.model = self.load_model()
        self.drop_save_dict = getattr(self.model, "save", {})
        logger.debug(f"Model loaded with drop_save_dict: {self.drop_save_dict}")

        # Load model summary and layer count
        self.torchinfo_summary = summary(self.model, (1, *self.input_size), verbose=0)
        self.layer_count = self._walk_modules(self.model.children(), depth=1, walk_i=0)
        del self.torchinfo_summary
        self.forward_info_empty = copy.deepcopy(self.forward_info)
        logger.debug(f"Model initialized with {self.layer_count} layers")

        # Hook-related attributes
        self.model_start_i: Optional[int] = None
        self.model_stop_i: Optional[int] = None
        self.banked_input: Optional[Any] = None
        self.log = False

        self.node_name = config.get("default", {}).get("device_type", "UNKNOWN")
        self.power_meter = PowerMeter(
            self.device
        )  # Implement a PowerMeter class to measure power usage

        self.to_device(self.device)
        self.warmup(iterations=2)
        logger.info("WrappedModel initialization complete")

    def _walk_modules(self, modules, depth: int, walk_i: int) -> int:
        """Recursively registers hooks on model layers."""
        for child in modules:
            child_name = child.__class__.__name__
            children = list(child.children())
            if children and depth < self.depth:
                logger.debug(
                    f"{'-' * depth}Module {child_name} has children; hooking children instead."
                )
                walk_i = self._walk_modules(children, depth + 1, walk_i)
                logger.debug(f"{'-' * depth}Finished hooking children of {child_name}.")
            else:
                if isinstance(child, torch.nn.Module):
                    layer_info = next(
                        (
                            layer
                            for layer in self.torchinfo_summary.summary_list
                            if layer.layer_id == id(child)
                        ),
                        None,
                    )
                    if layer_info:
                        self.forward_info[walk_i] = copy.deepcopy(LAYER_TEMPLATE)
                        self.forward_info[walk_i].update(
                            {
                                "depth": depth,
                                "layer_id": walk_i,
                                "class": layer_info.class_name,
                                "parameters": layer_info.num_params,
                                "parameter_bytes": layer_info.param_bytes,
                                "input_size": layer_info.input_size,
                                "output_size": layer_info.output_size,
                                "output_bytes": layer_info.output_bytes,
                            }
                        )
                        logger.debug(
                            f"Registered layer {walk_i}: {layer_info.class_name}"
                        )

                    pre_hook = child.register_forward_pre_hook(
                        create_forward_prehook(
                            self, walk_i, child_name, (0, 0), self.device
                        )
                    )
                    post_hook = child.register_forward_hook(
                        create_forward_posthook(
                            self, walk_i, child_name, (0, 0), self.device
                        )
                    )
                    self.forward_hooks.append(pre_hook)
                    self.forward_post_hooks.append(post_hook)
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
        """Performs a forward pass with optional slicing and logging."""
        start_time = time.perf_counter_ns()
        start_energy = self.power_meter.get_energy()
        end = self.layer_count if end == np.inf else end
        logger.info(
            f"Starting forward pass: inference_id={inference_id}, start={start}, end={end}, log={log}"
        )

        # Configure hooks
        self.log = log
        self.model_start_i = start
        self.model_stop_i = end

        # Prepare inference ID
        if inference_id:
            base_id, *suffix = inference_id.rsplit(".", maxsplit=1)
            suffix = int(suffix[0]) + 1 if suffix else 0
            current_id = f"{base_id}.{suffix}"
        else:
            current_id = "unlogged"
            self.log = False

        self.inference_info["inference_id"] = current_id
        logger.info(f"Inference {current_id} started.")

        # Run forward pass
        try:
            context = torch.no_grad() if self.get_mode() == "eval" else nullcontext()
            with context:
                output = self.model(x)
        except HookExitException as e:
            logger.debug("Early exit from forward pass due to stop index.")
            output = NotDict(e.result)
            for i in range(self.model_stop_i, self.layer_count):
                self.forward_info.pop(i, None)
        end_time = time.perf_counter_ns()
        end_energy = self.power_meter.get_energy()

        total_time = end_time - start_time
        total_energy = end_energy - start_energy

        # Update inference info with timing and power usage
        self.inference_info["total_time"] = total_time
        self.inference_info["total_energy"] = total_energy

        # Handle inference info
        self.inference_info["layer_information"] = self.forward_info
        if self.log and self.master_dict:
            base_id = current_id.split(".", maxsplit=1)[0]
            self.io_buffer[base_id] = copy.deepcopy(self.inference_info)
            if len(self.io_buffer) >= self.flush_buffer_size:
                self.update_master_dict()

        # Reset for next inference
        self.inference_info.clear()
        self.forward_info = copy.deepcopy(self.forward_info_empty)
        self.banked_input = None
        logger.info(f"Inference {current_id} ended.")
        return output

    def update_master_dict(self) -> None:
        """Flushes the IO buffer to the MasterDict."""
        logger.debug("Updating MasterDict with IO buffer.")
        if self.master_dict and self.io_buffer:
            logger.info("Flushing IO buffer to MasterDict.")
            self.master_dict.update(self.io_buffer)
            self.io_buffer.clear()
        else:
            logger.info("MasterDict not updated: buffer empty or MasterDict is None.")
