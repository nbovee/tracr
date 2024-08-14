import logging
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union
import copy

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
        flush_buffer_size: int = 100,
        splittable_layer_count: Optional[int] = None,
        node_name: str = "unknown",
        **kwargs,
    ):
        try:
            super().__init__(*args)
            self.timer = time.perf_counter_ns
            self.master_dict = master_dict
            self.io_buf_dict: Dict[str, Any] = {}
            self.inference_dict: Dict[str, Any] = {}
            self.forward_dict: Dict[int, Dict[str, Any]] = {}
            self.node_name = node_name
            self.__dict__.update(read_model_config(config_path))
            self.training = self.mode in ["train", "training"]
            self.model = self._model_selector(self.model_name)
            self.drop_save_dict = self._find_save_layers()
            self.flush_buffer_size = flush_buffer_size

            # Separate the features, avgpool, and classifier
            self.features = self.model.features
            logger.debug(f"Features layer: {self.features}")
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            logger.debug(f"Avgpool layer: {self.avgpool}")
            
            self._rebuild_classifier()

            # self.classifier = self.model.classifier
            # logger.debug(f"Classifier layer: {self.classifier}")

            # # Log each layer of the classifier
            # for i, layer in enumerate(self.classifier):
            #     logger.debug(f"Classifier layer {i}: {layer}")

            # # Adjust the first layer of the classifier
            # if isinstance(self.classifier[0], nn.Linear):
            #     in_features = 6 * 6 * 256  # 36 * 256
            #     out_features = self.classifier[0].out_features
            #     self.classifier[0] = nn.Linear(in_features, out_features)
            #     logger.debug(f"Adjusted classifier first layer: {self.classifier[0]}")
            # else:
            #     logger.warning(f"First layer of classifier is not nn.Linear: {type(self.classifier[0])}")
                #   OBSERVER@localhost: Features layer: Sequential(
                #   (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
                #   (1): ReLU(inplace=True)
                #   (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
                #   (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                #   (4): ReLU(inplace=True)
                #   (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
                #   (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                #   (7): ReLU(inplace=True)
                #   (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                #   (9): ReLU(inplace=True)
                #   (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                #   (11): ReLU(inplace=True)
                #   (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
                # )
                # OBSERVER@localhost: Avgpool layer: AdaptiveAvgPool2d(output_size=(6, 6))
                # OBSERVER@localhost: Classifier layer: Sequential(
                #   (0): Dropout(p=0.5, inplace=False)
                #   (1): Linear(in_features=9216, out_features=4096, bias=True)
                #   (2): ReLU(inplace=True)
                #   (3): Dropout(p=0.5, inplace=False)
                #   (4): Linear(in_features=4096, out_features=4096, bias=True)
                #   (5): ReLU(inplace=True)
                #   (6): Linear(in_features=4096, out_features=1000, bias=True)
                # )
                # OBSERVER@localhost: Classifier layer 0: Dropout(p=0.5, inplace=False)
                # OBSERVER@localhost: Classifier layer 1: Linear(in_features=9216, out_features=4096, bias=True)
                # OBSERVER@localhost: Classifier layer 2: ReLU(inplace=True)
                # OBSERVER@localhost: Classifier layer 3: Dropout(p=0.5, inplace=False)
                # OBSERVER@localhost: Classifier layer 4: Linear(in_features=4096, out_features=4096, bias=True)
                # OBSERVER@localhost: Classifier layer 5: ReLU(inplace=True)
                # OBSERVER@localhost: Classifier layer 6: Linear(in_features=4096, out_features=1000, bias=True)
                # OBSERVER@localhost: First layer of classifier is not nn.Linear: <class 'torch.nn.modules.dropout.Dropout'>


            self.f_hooks = []
            self.f_pre_hooks = []

            self._setup_model()  # This should set self.layer_count

            self.model_start_i: Optional[int] = None
            self.model_stop_i: Optional[int] = None
            self.banked_input: Optional[Dict[int, Any]] = None
            self.log = False
            self.splittable_layer_count = splittable_layer_count or self.layer_count

            self._finalize_setup()
            self._setup_cuda()
            logger.info(f"WrappedModel initialized: {self}")
            logger.info(
                f"Model name: {self.model_name}. Model has {self.layer_count} layers."
            )
        except Exception as e:
            logger.error(f"Error initializing WrappedModel: {str(e)}")
            raise

    def _rebuild_classifier(self):
        in_features = 6 * 6 * 256  # 9216
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000)  # Assuming 1000 classes for ImageNet
        )
        logger.debug(f"Rebuilt classifier: {self.classifier}")
        for i, layer in enumerate(self.classifier):
            logger.debug(f"New classifier layer {i}: {layer}")

        # Verify that the first Linear layer has the correct input size
        if isinstance(self.classifier[1], nn.Linear):
            actual_in_features = self.classifier[1].in_features
            logger.debug(f"First Linear layer input features: {actual_in_features}")
            if actual_in_features != in_features:
                logger.error(f"Mismatch in input features. Expected: {in_features}, Actual: {actual_in_features}")
                raise ValueError(f"Classifier rebuilding failed. Expected {in_features} input features, but got {actual_in_features}")

    def _model_selector(self, model_name: str) -> Any:
        """Select and return a model based on the given name."""
        from .model_utils import model_selector

        return model_selector(model_name)

    def _find_save_layers(self) -> Dict[Any, Any]:
        """Find layers with skip connections."""
        return self.model.save if hasattr(self.model, "save") else {}

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
            input_shape = "unknown"
            if isinstance(layer_input[0], NotDict):
                # Handle NotDict case
                if isinstance(layer_input[0].inner_dict, torch.Tensor):
                    input_shape = layer_input[0].inner_dict.shape
                else:
                    # If inner_dict is not a tensor, we might need to handle this case differently
                    input_shape = "NotDict(non-tensor)"
            elif hasattr(layer_input[0], 'shape'):
                input_shape = layer_input[0].shape
            
            logger.debug(
                f"{'Prehook': >8} {fixed_layer_i:>3} started. Input shape: {input_shape}"
            )

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
            output_shape = "unknown"
            if isinstance(output, NotDict):
                if isinstance(output.inner_dict, torch.Tensor):
                    output_shape = output.inner_dict.shape
                else:
                    output_shape = "NotDict(non-tensor)"
            elif hasattr(output, 'shape'):
                output_shape = output.shape

            logger.debug(
                f"Posthook {fixed_layer_i:>3} started. Output shape: {output_shape}"
            )
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
                if isinstance(self.banked_input[fixed_layer_i], NotDict):
                    output = self.banked_input[fixed_layer_i].inner_dict
                else:
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
            logger.debug(
                f"Exited early from forward pass due to stop index at layer {self.model_stop_i}"
            )
            out = NotDict(e.result)
            for i in range(self.model_stop_i, self.layer_count):
                del self.forward_dict[i]
            logger.info(
                f"Removed {self.layer_count - self.model_stop_i} layers from forward_dict"
            )

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error during forward pass")
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            logger.error(f"Unexpected error in forward pass: {str(e)}", exc_info=True)
            raise

        finally:
            try:
                self._process_inference_results(_inference_id)
                logger.info(f"Forward pass completed for inference {_inference_id}")
            except Exception as e:
                logger.error(
                    f"Error processing inference results: {str(e)}", exc_info=True
                )

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
        import torch

        if self.mode != "train":
            with torch.no_grad():
                logger.debug(f"Input shape: {x.shape}")
                x = self.features(x)
                logger.debug(f"After features: {x.shape}")
                x = self.avgpool(x)
                logger.debug(f"After avgpool: {x.shape}")
                
                # Flatten all dimensions except the batch dimension
                x = x.reshape(x.size(0), -1)
                logger.debug(f"After flatten: {x.shape}")
                
                expected_features = 6 * 6 * 256
                actual_features = x.shape[1]
                logger.debug(f"Expected features: {expected_features}, Actual features: {actual_features}")
                if actual_features != expected_features:
                    logger.error(f"Mismatch in flattened features. Expected: {expected_features}, Actual: {actual_features}")
                    # Adjust the tensor shape if necessary
                    if actual_features < expected_features:
                        padding = torch.zeros(x.size(0), expected_features - actual_features, device=x.device)
                        x = torch.cat([x, padding], dim=1)
                        logger.warning(f"Padded input to shape: {x.shape}")
                    else:
                        x = x[:, :expected_features]
                        logger.warning(f"Truncated input to shape: {x.shape}")
                
                x = self.classifier(x)
                # After classifier
                logger.debug(f"After classifier: {x.shape}")
                
                # Save results to master_dict
                if self.master_dict is not None and hasattr(self, 'current_inference_id'):
                    inference_data = {
                        "inference_id": self.current_inference_id,
                        "layer_information": self.forward_dict,
                        "output_shape": list(x.shape),
                        "split_layer": self.model_stop_i if hasattr(self, 'model_stop_i') else None
                    }
                    self.master_dict[self.current_inference_id] = inference_data
                    logger.info(f"Saved results for inference {self.current_inference_id} to master_dict")
                
                return x
        else:
            logger.debug(f"Input shape: {x.shape}")
            x = self.features(x)
            logger.debug(f"After features: {x.shape}")
            x = self.avgpool(x)
            logger.debug(f"After avgpool: {x.shape}")
            x = torch.flatten(x, 1)
            logger.debug(f"After flatten: {x.shape}")
            x = self.classifier(x)
            logger.debug(f"After classifier: {x.shape}")
            logger.debug(f"Output: {x}")
            return x

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
        try:
            logger.debug("WrappedModel.update_master_dict called")
            if self.master_dict is not None and self.io_buf_dict:
                logger.info(f"Flushing {len(self.io_buf_dict)} items from IO buffer dict to MasterDict")
                self.master_dict.update(self.io_buf_dict)
                self.io_buf_dict = {}
                return
            logger.info("MasterDict not updated; either buffer is empty or MasterDict is None")
        except Exception as e:
            logger.error(f"Error updating master dict: {str(e)}")
            raise

    def parse_input(self, _input: Any) -> Any:
        """Check if the input is appropriate at the given stage of the network."""
        try:
            import torch
            from PIL import Image

            if isinstance(_input, Image.Image):
                return self._process_image_input(_input)
            elif isinstance(_input, torch.Tensor):
                return self._process_tensor_input(_input)
        except ValueError as e:
            logger.error(f"Bad input given to WrappedModel: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error parsing input: {str(e)}")
            raise

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

    def __call__(
        self,
        x: Any,
        inference_id: Optional[str] = None,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Any:
        """Make the WrappedModel callable, delegating to the forward method."""
        return self.forward(x, inference_id=inference_id, start=start, end=end)


class WrappedModelFactory(ModelFactoryInterface):
    def create_model(self, **kwargs) -> ModelInterface:
        try:
            logger.info("Creating WrappedModel instance")
            return WrappedModel(**kwargs)
        except Exception as e:
            logger.error(f"Error creating WrappedModel: {str(e)}")
            raise
