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

        # Initialize basic attributes
        self.timer = time.perf_counter_ns
        self.master_dict = master_dict
        self.io_buffer = {}
        self.inference_info = {}
        self.forward_info = {}
        self.forward_hooks = []
        self.forward_post_hooks = []
        self.save_layers = getattr(self.model, "save", {})

        # Initialize hook-related attributes
        self.start_i: Optional[int] = None
        self.stop_i: Optional[int] = None
        self.banked_output: Optional[Any] = None
        self.log = False

        # Setup model layers and hooks
        self._setup_model()
        logger.info("WrappedModel initialization complete")

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
            # Store layer information
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

            # Register hooks
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
            f"Starting forward pass: id={inference_id}, start={start}, end={end}"
        )

        # Configure forward pass
        self.log = log
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
            self.inference_info["inference_id"] = "unlogged"
            self.log = False

    def _execute_forward(self, x: Union[torch.Tensor, Image.Image]) -> Any:
        """Execute model forward pass with appropriate context."""
        context = torch.no_grad() if self.get_mode() == "eval" else nullcontext()
        with context:
            return self.model(x)

    def _handle_early_exit(self, exception: HookExitException) -> EarlyOutput:
        """Handle early exit from forward pass."""
        output = EarlyOutput(exception.result)
        for i in range(self.stop_i, self.layer_count):
            self.forward_info.pop(i, None)
        return output

    def _handle_results(self, start_time: int) -> None:
        """Handle forward pass results and logging."""
        self.inference_info["total_time"] = self.timer() - start_time
        self.inference_info["layer_information"] = self.forward_info

        if self.log and self.master_dict:
            base_id = self.inference_info["inference_id"].split(".", maxsplit=1)[0]
            self.io_buffer[base_id] = copy.deepcopy(self.inference_info)

            if len(self.io_buffer) >= self.flush_buffer_size:
                self.update_master_dict()

        # Reset state
        self.inference_info.clear()
        self.forward_info = copy.deepcopy(self.forward_info_empty)
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
