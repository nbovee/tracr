from typing import Union
import numpy as np
import logging
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import ToTensor
import time
import atexit
from collections import OrderedDict
import copy
from torchinfo import summary

from src.experiment_design.records.master_dict import MasterDict


logger = logging.getLogger("tracr_logger")


class HookExitException(Exception):
    """Exception to early exit from inference in naive running."""

    def __init__(self, out, *args: object) -> None:
        super().__init__(*args)
        self.result = out


class WrappedModel(nn.Module):
    """Wraps a pretrained model with the features necesarry to perform edge computing tests. Uses pytorch
    hooks to perform benchmarkings, grab intermediate layers, and slice the Sequential to provide input to intermediate layers or exit early.
    """

    layer_template_dict = {
        "layer_id": None, # this one may prove unneeded
        "completed_by_node": None,
        "class": None,
        "inference_time": None,
        "parameters": None,
        "parameter_bytes": None,
        # "precision": None, precision is not technically per layer, disabled for now
        "cpu_cycles_used": None,
        "watts_used": None,
    }

    def __init__(self, *args, master_dict: Union[MasterDict, None] = None, flush_buffer_size: int = 100, **kwargs):
        print(*args)
        super().__init__(*args)
        self.timer = time.perf_counter_ns
        self.master_dict = master_dict # this should be the externally accessible dict
        self.io_buf_dict = {}
        self.inference_dict = {} # collation dict for the current partition of a given inference
        self.forward_dict = {} # dict for the results from the current forward pass
        self.device = kwargs.get("device", "cpu")
        self.mode = kwargs.get("mode","eval")
        self.hook_depth = kwargs.get("depth", np.inf)
        self.base_input_size = kwargs.get("image_size", (3, 224, 224))
        self.node_name = kwargs.get("node_name", "unknown")
        self.flush_buffer_size = flush_buffer_size
        atexit.register(self.safeClose)
        self.pretrained = kwargs.pop("pretrained", models.alexnet(pretrained=True))
        self.splittable_layer_count = 0
        self.selected_out = OrderedDict()  # could be useful for skips
        self.f_hooks = []
        self.f_pre_hooks = []
       
        # run torchinfo here to get parameters/flops/mac for entry into dict      
        self.torchinfo = summary(self.pretrained, (1, *self.base_input_size), verbose=0)
        self.walk_modules(self.pretrained.children(), 1) # depth starts at 1 to match torchinfo depths
        del self.torchinfo
        self.empty_buffer_dict = copy.deepcopy(self.forward_dict)
       
        # ---- class scope values that the hooks and forward pass use ----
        self.current_module_start_index = None
        self.current_module_stop_index = None
        self.current_module_index = None
        self.banked_input = None
        
        # layers before this index are not added to tracking dictionary, as they are not based upon the given input tensor
        
        # will not perform inference at this layer or above, watch if pruned.
        self.max_ignore_layer_index = self.splittable_layer_count - 1
        

        if self.mode == "eval":
            self.pretrained.eval()
        if self.device == "cuda":
            if torch.cuda.is_available():
                print("Loading Model to CUDA.")
            else:
                print("Loading Model to CPU. CUDA not available.")
                self.device = "cpu"
        self.pretrained.to(self.device)
        self.warmup(iterations=2)

    def walk_modules(self, module_generator, depth):
        """Recursively walks and marks Modules for hooks in a DFS. Most NN have an intended or intuitive
        depth to split at, but it is not obvious to the naive program."""
        for child in module_generator:
            if len(list(child.children())) > 0 and depth < self.hook_depth:
                # either has children we want to look at, or is max depth
                print(
                    f"{'-'*depth}Module {str(child).split('(')[0]} with children found, hooking children instead of module."
                )
                self.walk_modules(child.children(), depth + 1)
                print(
                    f"{'-'*depth}End of Module {str(child).split('(')[0]}'s children."
                )
            elif isinstance(child, nn.Module):
                # if not iterable/too deep, we have found a layer to hook
                this_layer = None
                for layer in self.torchinfo.summary_list:
                    if layer.layer_id == id(child):
                        this_layer = layer
                        break
                if this_layer is None:
                    raise Exception("module id not find while adding hooks.")

                this_layer_id = self.splittable_layer_count
                self.forward_dict[this_layer_id] = copy.deepcopy(WrappedModel.layer_template_dict)
                self.forward_dict[this_layer_id]["depth"] = depth
                # block of data from torchinfo
                self.forward_dict[this_layer_id]["layer_id"] = this_layer_id
                self.forward_dict[this_layer_id]["class"] = this_layer.class_name
                self.forward_dict[this_layer_id]["precision"] = None
                self.forward_dict[this_layer_id]["parameters"] = this_layer.num_params
                self.forward_dict[this_layer_id]["parameter_bytes"] = this_layer.param_bytes
                self.forward_dict[this_layer_id]["input_size"] = this_layer.input_size
                self.forward_dict[this_layer_id]["output_size"] = this_layer.output_size
                self.forward_dict[this_layer_id]["output_bytes"] = this_layer.output_bytes

                self.f_hooks.append(
                    child.register_forward_pre_hook(
                        self.forward_prehook(
                            this_layer_id,
                            str(child).split("(")[0],
                            (0, 0),
                        )
                    )
                )
                self.f_pre_hooks.append(
                    child.register_forward_hook(
                        self.forward_posthook(
                            this_layer_id,
                            str(child).split("(")[0],
                            (0, 0),
                        )
                    )
                )
                print(
                    f"{'-'*depth}Layer {this_layer_id}: {str(child).split('(')[0]} hooks applied."
                )
                # back hooks left out for now
                self.splittable_layer_count += 1

    def forward_prehook(self, layer_index, layer_name, input_shape):
        """Prehook a layer for benchmarking."""

        def pre_hook(module, input):
            assert self.current_module_index is not None and self.current_module_start_index is not None
            if (self.current_module_stop_index is not None and
                layer_index >= self.current_module_stop_index and 
                layer_index < self.max_ignore_layer_index
            ): # exit at the start of the layer to complete, in order to catch modifications to the input from non-Modules
                raise HookExitException(input)
            if self.log and (self.current_module_index >= self.current_module_start_index):
                self.forward_dict[layer_index]["completed_by_node"] = self.node_name
                self.forward_dict[layer_index]['inference_time'] = -self.timer()
            # store input until the correct layer arrives
            if self.current_module_index == 0 and self.current_module_start_index > 0:
                self.banked_input = copy.deepcopy(*input)
                return torch.randn(1, *self.base_input_size)
            # swap correct input back in now that we are at the right layer
            elif self.banked_input is not None and self.current_module_index == self.current_module_start_index:
                input = self.banked_input
                self.banked_input = None
                return input

        return pre_hook

    def forward_posthook(self, layer_index, layer_name, input_shape, **kwargs):
        """Posthook a layer for output capture and benchmarking."""

        def hook(module, input, output):
            assert self.current_module_index is not None and self.current_module_start_index is not None
            if self.log and self.current_module_index >= self.current_module_start_index:
                self.forward_dict[layer_index]['inference_time'] += self.timer()
            self.current_module_index += 1
            # print(f"stop at layer: {self.current_module_stop_index -1}")
            # print(f"L{layer_index}-{layer_name} returned.")

        return hook

    def forward(self, x,
                inference_id: Union[str, None] = None,
                start: int = 0,
                end: Union[int, float] = np.inf,
                log: bool = True
                ):
        """Wraps the pretrained forward pass to utilize our slicing."""
        end = self.splittable_layer_count if end == np.inf else end

        # set values for the hooks to see
        self.log = log
        self.current_module_stop_index = end
        self.current_module_index = 0
        self.current_module_start_index = start

        # prepare inference_id for storing results
        _inference_id = "unlogged" if inference_id is None else inference_id
        self.inference_dict['inference_id'] = _inference_id
        print(f"{_inference_id} id beginning.")
        # actually run the forward pass
        try:
            if self.mode != "train":
                with torch.no_grad():
                    out = self.pretrained(x)
            else:
                out = self.pretrained(x)
        except HookExitException as e:
            print("Exit early from hook.")
            out = e.result
            assert isinstance(self.current_module_stop_index, int)
            for i in range(self.current_module_stop_index, self.splittable_layer_count):
                del self.forward_dict[i]

        # process and clean dicts before leaving forward
        self.inference_dict['layer_information'] = self.forward_dict
        if log and self.master_dict:
            self.io_buf_dict[str(_inference_id).split(".")[0]] = copy.deepcopy(self.inference_dict) # only one deepcopy needed
            if len(self.io_buf_dict) >= self.flush_buffer_size:
                self.update_master_dict()
        self.inference_dict = {}
        self.forward_dict = copy.deepcopy(self.empty_buffer_dict)

        # reset hook variables
        self.current_module_stop_index = None
        self.current_module_index = None
        print(f"{_inference_id} end.")
        return out

    def update_master_dict(self):
        logger.debug("WrappedModel.update_master_dict called")
        if self.master_dict is not None and self.io_buf_dict:
            logger.info("flushing IO buffer dict to MasterDict")
            self.master_dict.update(self.io_buf_dict)
            self.io_buf_dict = {}
            return
        logger.info("MasterDict not updated; either buffer is empty or MasterDict is None")

    def parse_input(self, input):
        """Checks if the input is appropriate at the given stage of the network. Does not yet check Tensor shapes for intermediate layers."""
        if isinstance(input, Image.Image):
            if input.size != self.base_input_size:
                input = input.resize(self.base_input_size)
            transform = ToTensor()
            input_tensor = transform(input)
            input_tensor = input_tensor.unsqueeze(0)
        elif isinstance(input, torch.Tensor):
            input_tensor = input
        else:
            raise ValueError(f"Bad input given to WrappedModel: type {type(input)}")
        if (
            torch.cuda.is_available()
            and self.mode == "cuda"
            and input_tensor.device != self.mode
        ):
            input_tensor = input_tensor.to(self.mode)
        return input_tensor

    def warmup(self, iterations=50, force=False):
        if self.device != "cuda" and force is not False:
            print("Warmup not required.")
        else:
            print("Starting warmup.")
            with torch.no_grad():
                for i in range(iterations):
                    self(torch.randn(1, *self.base_input_size), log = False)
            print("Warmup complete.")

    def prune_layers(self, newlow, newhigh):
        """NYE: Trim network layers. inputs specify the lower and upper layers to REMAIN. Used to attempt usage on low compute power devices, such as early Raspberry Pi models."""
        raise NotImplementedError

    def safeClose(self):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # running as main will test baselines on the running platform
    m = WrappedModel()

