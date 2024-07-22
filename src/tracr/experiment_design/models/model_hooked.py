"""model_hooked module"""

import atexit
import copy
import logging
import time
from typing import Any, Union

import numpy as np
import torch
from PIL import Image
from torchinfo import summary
from torchvision.transforms import ToTensor

from src.tracr.experiment_design.records.master_dict import MasterDict
from .model_config import read_model_config
from .model_selector import model_selector

atexit.register(torch.cuda.empty_cache)
logger = logging.getLogger("tracr_logger")


class NotDict:
    """Wrapper for a dict to circumenvent some of Ultralytics forward pass handling. Uses a class instead of tuple in case additional handling is added later."""

    def __init__(self, passed_dict) -> None:
        self.inner_dict = passed_dict

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.inner_dict


class HookExitException(Exception):
    """Exception to early exit from inference in naive running."""

    def __init__(self, out, *args: object) -> None:
        super().__init__(*args)
        self.result = out


class WrappedModel(torch.nn.Module):
    """Wraps a pretrained model with the features necesarry to perform edge computing tests.
    Uses pytorch hooks to perform benchmarkings, grab intermediate layers, and slice the
    Sequential to provide input to intermediate layers or exit early.
    """

    layer_template_dict = {
        "layer_id": None,  # this one may prove unneeded
        "completed_by_node": None,
        "class": None,
        "inference_time": 0,
        "parameters": None,
        "parameter_bytes": None,
        # "precision": None, precision is not technically per layer, disabled for now
        "cpu_cycles_used": None,
        "watts_used": None,
    }

    def __init__(
        self,
        *args,
        config_path=None,
        master_dict: Union[MasterDict, None] = None,
        flush_buffer_size: int = 100,
        **kwargs,
    ):
        logger.debug(f"{args=}")
        super().__init__(*args)
        self.timer = time.perf_counter_ns
        self.master_dict = master_dict  # this should be the externally accessible dict
        self.io_buf_dict = {}
        self.inference_dict = (
            {}
        )  # collation dict for the current partition of a given inference
        self.forward_dict = {}  # dict for the results from the current forward pass
        # assigns config vars to the wrapper
        self.__dict__.update(read_model_config(config_path))
        self.training = True if self.mode in ["train", "training"] else False
        self.model = model_selector(self.model_name)
        self.drop_save_dict = self._find_save_layers()
        self.flush_buffer_size = flush_buffer_size
        # self.selected_out = OrderedDict()  # could be useful for skips
        self.f_hooks = []
        self.f_pre_hooks = []
        # run torchinfo here to get parameters/flops/mac for entry into dict
        """ INFO: YOLO() model wrapper appears to map .eval() that torchinfo calls to .train()
        I don't have a fix tonight outside of popping the model out of the wrapper after setup."""
        self.torchinfo = summary(self.model, (1, *self.input_size), verbose=0)
        self.layer_count = self._walk_modules(
            self.model.children(), 1, 0
        )  # depth starts at 1 to match torchinfo depths
        del self.torchinfo
        self.forward_dict_empty = copy.deepcopy(self.forward_dict)
        # ---- class scope values that the hooks and forward pass use ----
        self.model_start_i = None
        self.model_stop_i = None
        self.banked_input = None
        self.log = False

        if self.mode == "eval":
            self.model.eval()
        if self.device == "cuda":
            if torch.cuda.is_available():
                logger.info("Loading Model to CUDA.")
            else:
                logger.info("Loading Model to CPU. CUDA not available.")
                self.device = "cpu"
        self.model.to(self.device)
        self.warmup(iterations=2)

    def _find_save_layers(self):
        """Interrogate the model to find skip connections.
        Requires the model to have knowledge of its structure (for now)."""
        drop_save_dict = {}
        drop_save_dict = self.model.save if self.model.save else {}
        return drop_save_dict

    def _walk_modules(self, module_generator, depth, walk_i):
        """Recursively walks and marks Modules for hooks in a DFS. Most NN have an
        intended or intuitive depth to split at, but it is not obvious to the naive program.
        """
        for child in module_generator:
            childname = str(child).split("(", maxsplit=1)[0]
            if len(list(child.children())) > 0 and depth < self.depth:
                # either has children we want to look at, or is max depth
                logger.debug(
                    f"{'-'*depth}Module {childname} "
                    "with children found, hooking children instead of module."
                )
                walk_i = self._walk_modules(child.children(), depth + 1, walk_i)
                logger.debug(f"{'-'*depth}End of Module {childname}'s children.")
            elif isinstance(child, torch.nn.Module):
                # if not iterable/too deep, we have found a layer to hook
                for layer in self.torchinfo.summary_list:
                    if layer.layer_id == id(child):
                        self.forward_dict[walk_i] = copy.deepcopy(
                            WrappedModel.layer_template_dict
                        )
                        self.forward_dict[walk_i].update(
                            {
                                "depth": depth,
                                "layer_id": walk_i,
                                "class": layer.class_name,
                                # "precision": None,
                                "parameters": layer.num_params,
                                "parameter_bytes": layer.param_bytes,
                                "input_size": layer.input_size,
                                "output_size": layer.output_size,
                                "output_bytes": layer.output_bytes,
                            }
                        )

                self.f_hooks.append(
                    child.register_forward_pre_hook(
                        self.forward_prehook(
                            walk_i,
                            childname,
                            (0, 0),
                        ),
                        with_kwargs=False,
                    )
                )
                self.f_pre_hooks.append(
                    child.register_forward_hook(
                        self.forward_posthook(
                            walk_i,
                            childname,
                            (0, 0),
                        ),
                        with_kwargs=False,
                    )
                )
                logger.debug(
                    f"{'-'*depth}Layer {walk_i}: {childname} had hooks applied."
                )
                # back hooks left out for now
                walk_i += 1
        return walk_i

    def forward_prehook(self, fixed_layer_i, layer_name, input_shape):
        """Prehook a layer for benchmarking."""

        def pre_hook(module, layer_input):  # hook signature format is required
            logger.debug(f"start prehook {fixed_layer_i}")
            hook_output = layer_input
            # previous layer exit
            if (
                self.model_stop_i <= fixed_layer_i < self.layer_count
                and self.hook_style == "pre"
            ):
                logger.info(f"exit signal: during prehook {fixed_layer_i}")
                # wait to allow non torch.nn.Modules to modify input as needed (ex flatten)
                self.banked_input[fixed_layer_i - 1] = layer_input[0]
                raise HookExitException(self.banked_input)
            if fixed_layer_i == 0:
                # if at first layer, prepare self.banked_input
                if self.model_start_i == 0:
                    logger.debug("reseting input bank")
                    # initiating pass: reset bank
                    self.banked_input = {}
                else:
                    logger.debug("importing input bank from initiating network")
                    # completing pass: store input dict until the correct layer arrives
                    self.banked_input = layer_input[
                        0
                    ]()  # wrapped dict expected, deepcopy may help
                    hook_output = torch.randn(1, *self.input_size)
            elif (
                fixed_layer_i in self.drop_save_dict
                or self.model_start_i == fixed_layer_i
            ):
                # if not at first layer, not exiting, at a marked layer
                if self.model_start_i == 0 and self.hook_style == "pre":
                    logger.debug(f"storing layer {fixed_layer_i} into input bank")
                    # initiating pass case: store inputs into dict
                    self.banked_input[fixed_layer_i] = layer_input
                if 0 < self.model_start_i > fixed_layer_i and self.hook_style == "pre":
                    logger.debug(
                        f"overwriting layer {fixed_layer_i} with input from bank"
                    )
                    # completing pass: overwrite dummy pass with stored input
                    hook_output = self.banked_input[
                        fixed_layer_i - (1 if self.hook_style == "pre" else 0)
                    ]
            # lastly, prepare timestamps for current layer
            if self.log and (fixed_layer_i >= self.model_start_i):
                self.forward_dict[fixed_layer_i]["completed_by_node"] = self.node_name
                self.forward_dict[fixed_layer_i]["inference_time"] = -self.timer()
            logger.debug(f"end prehook {fixed_layer_i}")
            return hook_output

        return pre_hook

    def forward_posthook(self, fixed_layer_i, layer_name, input_shape, **kwargs):
        """Posthook a layer for output capture and benchmarking."""

        def hook(module, layer_input, output):
            logger.debug(f"start posthook {fixed_layer_i}")
            if self.log and fixed_layer_i >= self.model_start_i:
                self.forward_dict[fixed_layer_i]["inference_time"] += self.timer()
            if (
                fixed_layer_i in self.drop_save_dict
                or (0 < self.model_start_i == fixed_layer_i)
                and self.hook_style == "post"
            ):
                # if not at first layer, not exiting, at a marked layer
                if self.model_start_i == 0:
                    logger.debug(f"storing layer {fixed_layer_i} into input bank")
                    # initiating pass case: store inputs into dict
                    self.banked_input[fixed_layer_i] = output
                elif self.hook_style == "post" and self.model_start_i >= fixed_layer_i:
                    logger.debug(
                        f"overwriting layer {fixed_layer_i} with input from bank"
                    )
                    # completing pass: overwrite dummy pass with stored input
                    output = self.banked_input[fixed_layer_i]
            if (
                self.model_stop_i <= fixed_layer_i < self.layer_count
                and self.hook_style == "post"
            ):
                logger.info(f"exit signal: during posthook {fixed_layer_i}")
                self.banked_input[fixed_layer_i] = output
                raise HookExitException(self.banked_input)
            logger.debug(f"end posthook {fixed_layer_i}")
            return output

        return hook

    def forward(
        self,
        x,
        inference_id: Union[str, None] = None,
        start: int = 0,
        end: Union[int, float] = np.inf,
        log: bool = True,
    ):
        """Wraps the model forward pass to utilize our slicing."""
        end = self.layer_count if end == np.inf else end

        # set values for the hooks to see
        self.log = log
        self.model_stop_i = end
        self.model_start_i = start

        # prepare inference_id for storing results
        if inference_id is None:
            _inference_id = "unlogged"
            self.log = False
        else:
            _inference_id = inference_id
        if len(str(_inference_id).split(".")) > 1:
            suffix = int(str(_inference_id).rsplit(".", maxsplit=1)[-1]) + 1
        else:
            suffix = 0
        _inference_id = str(str(_inference_id).split(".", maxsplit=1)[0]) + f".{suffix}"
        self.inference_dict["inference_id"] = _inference_id
        logger.info(f"{_inference_id} id beginning.")
        # actually run the forward pass
        try:
            if self.mode != "train":
                with torch.no_grad():
                    out = self.model(x)
            else:
                out = self.model(x)
        except HookExitException as e:
            logger.debug("Exited early from forward pass due to stop index.")
            out = NotDict(e.result)
            for i in range(self.model_stop_i, self.layer_count):
                del self.forward_dict[i]

        # process and clean dicts before leaving forward
        self.inference_dict["layer_information"] = self.forward_dict
        if log and self.master_dict:
            self.io_buf_dict[str(_inference_id).split(".", maxsplit=1)[0]] = (
                copy.deepcopy(self.inference_dict)
            )  # only one deepcopy needed
            if len(self.io_buf_dict) >= self.flush_buffer_size:
                self.update_master_dict()
        self.inference_dict = {}
        self.forward_dict = copy.deepcopy(self.forward_dict_empty)
        self.banked_input = None
        logger.info(f"{_inference_id} end.")
        return out

    def update_master_dict(self):
        """Updates the linked MasterDict object with recent data, and clears buffer"""
        logger.debug("WrappedModel.update_master_dict called")
        if self.master_dict is not None and self.io_buf_dict:
            logger.info("flushing IO buffer dict to MasterDict")
            self.master_dict.update(self.io_buf_dict)
            self.io_buf_dict = {}
            return
        logger.info(
            "MasterDict not updated; either buffer is empty or MasterDict is None"
        )

    def parse_input(self, _input):
        """Checks if the input is appropriate at the given stage of the network.
        Does not yet check Tensor shapes for intermediate layers."""
        if isinstance(_input, Image.Image):
            if _input.size != self.base_input_size:
                _input = _input.resize(self.base_input_size)
            transform = ToTensor()
            input_tensor = transform(_input)
            input_tensor = input_tensor.unsqueeze(0)
        elif isinstance(_input, torch.Tensor):
            input_tensor = _input
        else:
            raise ValueError(f"Bad input given to WrappedModel: type {type(_input)}")
        if (
            torch.cuda.is_available()
            and self.mode == "cuda"
            and input_tensor.device != self.mode
        ):
            input_tensor = input_tensor.to(self.mode)
        return input_tensor

    def warmup(self, iterations=50, force=False):
        """runs specified passes on the nn to warm up gpu if enabled"""
        if self.device != "cuda" and force is not False:
            logger.info("Warmup not required.")
        else:
            logger.info("Starting warmup.")
            with torch.no_grad():
                for _ in range(iterations):
                    self(torch.randn(1, *self.input_size), log=False)
            logger.info("Warmup complete.")

    def prune_layers(self, newlow, newhigh):
        """NYE: Trim network layers. inputs specify the lower and upper layers to REMAIN.
        Used to attempt usage on low compute power devices, such as early Raspberry Pi models.
        """
        raise NotImplementedError()
