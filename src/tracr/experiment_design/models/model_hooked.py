"""Module for wrapping and hooking models for edge computing experiments."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union
import copy
import logging
import time
import numpy as np

from src.tracr.app_api.model_interface import ModelInterface, ModelFactoryInterface
from .model_utils import HookExitException, NotDict, read_model_config

logger = logging.getLogger("tracr_logger")


class WrappedModel(ModelInterface):
    """
    Wraps a pretrained model with features necessary for edge computing tests.

    Uses PyTorch hooks to perform benchmarking, grab intermediate layers, and slice the
    Sequential to provide input to intermediate layers or exit early.
    """

    layer_template_dict = {
        "layer_id": None,
        "completed_by_node": None,
        "class": None,
        "inference_time": 0,
        "parameters": None,
        "parameter_bytes": None,
        "cpu_cycles_used": None,
        "watts_used": None,
    }

    def __init__(
        self,
        *args,
        config_path: Optional[str] = None,
        master_dict: Any = None,
        splittable_layer_count: Any = None,
        flush_buffer_size: int = 100,
        **kwargs,
    ):
        super().__init__(*args)
        self.timer = time.perf_counter_ns
        self.master_dict = master_dict
        self.splittable_layer_count = splittable_layer_count
        self.io_buf_dict: Dict[str, Any] = {}
        self.inference_dict: Dict[str, Any] = {}
        self.forward_dict: Dict[int, Dict[str, Any]] = {}

        self.__dict__.update(read_model_config(config_path))
        self.training = self.mode in ["train", "training"]
        print("model_name", self.model_name)
        self.model = self._model_selector(self.model_name)
        print("model is", self.model)
        self.drop_save_dict = self._find_save_layers()
        self.flush_buffer_size = flush_buffer_size

        self.f_hooks = []
        self.f_pre_hooks = []

        self._setup_model()

        self.model_start_i: Optional[int] = None
        self.model_stop_i: Optional[int] = None
        self.banked_input: Optional[Dict[int, Any]] = None
        self.log = False

        self._finalize_setup()
        self._setup_cuda()

    def _model_selector(self, model_name: str) -> Any:
        """Select and return a model based on the given name."""
        from .model_utils import model_selector

        return model_selector(model_name)

    def _setup_model(self):
        """Set up the model, including torchinfo summary and module walking."""
        from torchinfo import summary

        self.torchinfo = summary(self.model, (1, *self.input_size), verbose=0)
        self.layer_count = self._walk_modules(self.model.children(), 1, 0)
        del self.torchinfo
        self.forward_dict_empty = copy.deepcopy(self.forward_dict)

    def _finalize_setup(self):
        """Finalize model setup, including device placement and warmup."""
        if self.mode == "eval":
            self.model.eval()
        if self.device == "cuda":
            if self._cuda_is_available():
                logger.info("Loading Model to CUDA.")
            else:
                logger.info("Loading Model to CPU. CUDA not available.")
                self.device = "cpu"
        self._move_model_to_device()
        self.warmup(iterations=2)

    def _setup_cuda(self):
        """Set up CUDA if available."""
        if self._cuda_is_available():
            import torch
            import atexit

            atexit.register(torch.cuda.empty_cache)
            torch.cuda.empty_cache()

    def _cuda_is_available(self) -> bool:
        """Check if CUDA is available."""
        import torch

        return torch.cuda.is_available()

    def _move_model_to_device(self):
        """Move the model to the specified device."""
        self.model.to(self.device)

    def _find_save_layers(self) -> Dict[Any, Any]:
        """Find layers with skip connections."""
        return self.model.save if hasattr(self.model, "save") else {}

    def _walk_modules(self, module_generator, depth: int, walk_i: int) -> int:
        """Recursively walk and mark Modules for hooks in a DFS manner."""
        import torch

        for child in module_generator:
            childname = str(child).split("(", maxsplit=1)[0]
            if len(list(child.children())) > 0 and depth < self.depth:
                logger.debug(
                    f"{'-'*depth}Module {childname} with children found, hooking children instead of module."
                )
                walk_i = self._walk_modules(child.children(), depth + 1, walk_i)
                logger.debug(f"{'-'*depth}End of Module {childname}'s children.")
            elif isinstance(child, torch.nn.Module):
                self._setup_layer_hooks(child, depth, walk_i, childname)
                walk_i += 1
        return walk_i

    def _setup_layer_hooks(self, child: Any, depth: int, walk_i: int, childname: str):
        """Set up hooks for a specific layer."""
        for layer in self.torchinfo.summary_list:
            if layer.layer_id == id(child):
                self._update_forward_dict(walk_i, depth, layer)
                break

        self.f_hooks.append(
            child.register_forward_pre_hook(
                self.forward_prehook(walk_i, childname, (0, 0)),
                with_kwargs=False,
            )
        )
        self.f_pre_hooks.append(
            child.register_forward_hook(
                self.forward_posthook(walk_i, childname, (0, 0)),
                with_kwargs=False,
            )
        )
        logger.debug(f"{'-'*depth}Layer {walk_i}: {childname} had hooks applied.")

    def _update_forward_dict(self, walk_i: int, depth: int, layer: Any):
        """Update forward_dict with layer information."""
        self.forward_dict[walk_i] = copy.deepcopy(self.layer_template_dict)
        self.forward_dict[walk_i].update(
            {
                "depth": depth,
                "layer_id": walk_i,
                "class": layer.class_name,
                "parameters": layer.num_params,
                "parameter_bytes": layer.param_bytes,
                "input_size": layer.input_size,
                "output_size": layer.output_size,
                "output_bytes": layer.output_bytes,
            }
        )

    def forward_prehook(self, fixed_layer_i: int, layer_name: str, input_shape: tuple):
        """Structure a forward prehook for a given layer index."""

        def pre_hook(module, layer_input):
            logger.debug(f"{'Prehook': >8} {fixed_layer_i:>3} started.")
            new_input = None
            if fixed_layer_i == 0:
                new_input = self._handle_first_layer(layer_input)
            if self.log and (fixed_layer_i >= self.model_start_i):
                self._update_forward_dict_for_logging(fixed_layer_i)
            logger.debug(f"{'Prehook': >8} {fixed_layer_i:>3} ended.")
            return new_input

        return pre_hook

    def _handle_first_layer(self, layer_input):
        """Handle logic for the first layer in prehook."""
        import torch

        if self.model_start_i == 0:
            logger.debug("\tresetting input bank")
            self.banked_input = {}
        else:
            logger.debug("\timporting input bank from initiating network")
            self.banked_input = layer_input[0]()
            return torch.randn(1, *self.input_size)
        return None

    def _update_forward_dict_for_logging(self, fixed_layer_i: int):
        """Update forward_dict with logging information."""
        self.forward_dict[fixed_layer_i]["completed_by_node"] = self.node_name
        self.forward_dict[fixed_layer_i]["inference_time"] = -self.timer()

    def forward_posthook(
        self, fixed_layer_i: int, layer_name: str, input_shape: tuple, **kwargs
    ):
        """Structure a forward posthook for a given layer index."""

        def hook(module, layer_input, output):
            logger.debug(f"Posthook {fixed_layer_i:>3} started.")
            if self.log and fixed_layer_i >= self.model_start_i:
                self.forward_dict[fixed_layer_i]["inference_time"] += self.timer()

            output = self._handle_layer_output(fixed_layer_i, output)

            if self.model_stop_i <= fixed_layer_i < self.layer_count:
                logger.info(f"\texit signal: during posthook {fixed_layer_i}")
                self.banked_input[fixed_layer_i] = output
                raise HookExitException(self.banked_input)

            logger.debug(f"Posthook {fixed_layer_i:>3} ended.")
            return output

        return hook

    def _handle_layer_output(self, fixed_layer_i: int, output: Any) -> Any:
        """Handle layer output in posthook."""
        if fixed_layer_i in self.drop_save_dict or (
            0 < self.model_start_i == fixed_layer_i
        ):
            if self.model_start_i == 0:
                logger.debug(f"\tstoring layer {fixed_layer_i} into input bank")
                self.banked_input[fixed_layer_i] = output
            elif self.model_start_i >= fixed_layer_i:
                logger.debug(
                    f"\toverwriting layer {fixed_layer_i} with input from bank"
                )
                output = self.banked_input[fixed_layer_i]
        return output

    def forward(
        self,
        x: Any,
        inference_id: Optional[str] = None,
        start: int = 0,
        end: Union[int, float] = np.inf,
        log: bool = True,
    ) -> Any:
        """Wrap the model forward pass to utilize slicing."""
        end = self.layer_count if end == np.inf else end
        self.log, self.model_stop_i, self.model_start_i = log, end, start

        _inference_id = self._prepare_inference_id(inference_id)

        try:
            out = self._run_forward_pass(x)
        except HookExitException as e:
            logger.debug("Exited early from forward pass due to stop index.")
            out = NotDict(e.result)
            for i in range(self.model_stop_i, self.layer_count):
                del self.forward_dict[i]

        self._process_inference_results(_inference_id)
        logger.info(f"{_inference_id} end.")
        return out

    def _prepare_inference_id(self, inference_id: Optional[str]) -> str:
        """Prepare the inference ID for logging."""
        if inference_id is None:
            _inference_id = "unlogged"
            self.log = False
        else:
            _inference_id = inference_id
        if len(str(_inference_id).split(".")) > 1:
            suffix = int(str(_inference_id).rsplit(".", maxsplit=1)[-1]) + 1
        else:
            suffix = 0
        _inference_id = f"{str(_inference_id).split('.', maxsplit=1)[0]}.{suffix}"
        self.inference_dict["inference_id"] = _inference_id
        logger.info(f"{_inference_id} id beginning.")
        return _inference_id

    def _run_forward_pass(self, x: Any) -> Any:
        """Run the actual forward pass."""
        import torch

        if self.mode != "train":
            with torch.no_grad():
                return self.model(x)
        else:
            return self.model(x)

    def _process_inference_results(self, _inference_id: str):
        """Process and clean dicts after forward pass."""
        self.inference_dict["layer_information"] = self.forward_dict
        if self.log and self.master_dict:
            self.io_buf_dict[str(_inference_id).split(".", maxsplit=1)[0]] = (
                copy.deepcopy(self.inference_dict)
            )
            if len(self.io_buf_dict) >= self.flush_buffer_size:
                self.update_master_dict()
        self.inference_dict = {}
        self.forward_dict = copy.deepcopy(self.forward_dict_empty)
        self.banked_input = None

    def update_master_dict(self):
        """Update the linked MasterDict object with recent data and clear buffer."""
        logger.debug("WrappedModel.update_master_dict called")
        if self.master_dict is not None and self.io_buf_dict:
            logger.info("flushing IO buffer dict to MasterDict")
            self.master_dict.update(self.io_buf_dict)
            self.io_buf_dict = {}
            return
        logger.info(
            "MasterDict not updated; either buffer is empty or MasterDict is None"
        )

    def parse_input(self, _input: Any) -> Any:
        """Check if the input is appropriate at the given stage of the network."""
        import torch
        from PIL import Image

        if isinstance(_input, Image.Image):
            return self._process_image_input(_input)
        elif isinstance(_input, torch.Tensor):
            return self._process_tensor_input(_input)
        else:
            raise ValueError(f"Bad input given to WrappedModel: type {type(_input)}")

    def _process_image_input(self, image: Any) -> Any:
        """Process PIL Image input."""
        from torchvision.transforms import ToTensor

        if image.size != self.base_input_size:
            image = image.resize(self.base_input_size)
        input_tensor = ToTensor()(image).unsqueeze(0)
        return self._move_to_device(input_tensor)

    def _process_tensor_input(self, tensor: Any) -> Any:
        """Process torch.Tensor input."""
        return self._move_to_device(tensor)

    def _move_to_device(self, tensor: Any) -> Any:
        """Move tensor to the correct device."""
        import torch

        if (
            torch.cuda.is_available()
            and self.device == "cuda"
            and tensor.device != self.device
        ):
            return tensor.to(self.device)
        return tensor

    def warmup(self, iterations: int = 50, force: bool = False):
        """Run specified passes on the NN to warm up GPU if enabled."""
        self._perform_warmup(iterations, force)

    def _perform_warmup(self, iterations: int, force: bool):
        """Perform the actual warmup process."""
        import torch

        if self.device != "cuda" and not force:
            logger.info("Warmup not required.")
        else:
            logger.info("Starting warmup.")
            with torch.no_grad():
                for _ in range(iterations):
                    self(torch.randn(1, *self.input_size), log=False)
            logger.info("Warmup complete.")

    def prune_layers(self, newlow: int, newhigh: int):
        """NYE: Trim network layers."""
        raise NotImplementedError()

    @property
    def base_input_size(self):
        """Get the base input size for the model."""
        return self.input_size[1:]

    def __str__(self):
        """String representation of the WrappedModel."""
        return (
            f"WrappedModel({self.model_name}, device={self.device}, mode={self.mode})"
        )

    def __repr__(self):
        """Detailed string representation of the WrappedModel."""
        return (
            f"WrappedModel(model_name={self.model_name}, "
            f"device={self.device}, mode={self.mode}, "
            f"depth={self.depth}, layer_count={self.layer_count})"
        )


class WrappedModelFactory(ModelFactoryInterface):
    def create_model(
        self,
        config_path: Optional[str] = None,
        master_dict: Any = None,
        flush_buffer_size: int = 100,
    ) -> ModelInterface:
        """Create and return a WrappedModel instance."""
        return WrappedModel(
            config_path=config_path,
            master_dict=master_dict,
            flush_buffer_size=flush_buffer_size,
        )
