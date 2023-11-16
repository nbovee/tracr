from .partitioner import Partitioner
from typing import Any
import torch
import numpy
import os
import csv
import copy
import pickle
# import matplotlib.pyplot as plt

class RegressionPartitioner(Partitioner):
    _TYPE = "regression"

    class linreg():

        def __init__(self, *args) -> None:
            self.model = torch.nn.Linear(1,1)
            self.criterion = torch.nn.MSELoss()
            self.optim = torch.optim.SGD(self.model.parameters(), lr=1e-2)
            self.training_iter = 20
            self.loss_list = []
        
        def forward(self, x):
            return self.model(torch.unsqueeze(x, 0))
        
        def train_pass(self, y, pred):
            loss = self.criterion(pred, torch.as_tensor([y]))
            # storing the calculated loss in a list
            self.loss_list.append(loss.data)
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        def manually_set_weights(self, w, b):
            self.model.weight.requires_grad = False
            self.model.bias.requires_grad = False
            self.model.weight.fill_(w)
            self.model.bias.fill_(b)

        def manually_scale_bias(self, scale_factor):
            self.model.weight.requires_grad = False
            self.model.bias.requires_grad = False
            temp = scale_factor* self.model.bias
            self.model.bias.data = temp
        

    def __init__(self, num_breakpoints, clip_min_max=True) -> None:
        super().__init__()
        self.start = 0 # needed if we start dropping modules from the Model class
        self.breakpoints = num_breakpoints
        self.clip = clip_min_max
        self.regression = {}
        self.module_sequence = []
        self.num_modules = None
        self._dir = 'TestCases/AlexnetSplit/partitioner_datapoints/local/'
        self.server_regression = None

    def pass_regression_copy(self):
        return pickle.dumps(self.regression)
    
    def add_server_module(self, server_modules):
        self.server_regression = server_modules

    def estimate_split_point(self, starting_layer):
        '''returns the index of the active model to split before. To mandate layer 0 is run on edge, provide starting_layer = 1'''
        for module, param_bytes, output_bytes in self.module_sequence:
            local_time_est_s = int(self.regression[module].forward(torch.as_tensor(float(param_bytes))))*1e-9
            if self.server_regression is None:
                server_time_est_s = 0
            else:
                server_time_est_s = int(self.server_regression[module].forward(torch.as_tensor(float(param_bytes))))*1e-9 # get server_regression from grpc
            output_transfer_time = output_bytes/self._get_network_speed_bytes()

            if local_time_est_s < output_transfer_time + server_time_est_s:
                starting_layer += 1
            else:
                return starting_layer

    def create_data(self, model, iterations = 10):
        for f in os.listdir(self._dir):
            os.remove(os.path.join(self._dir, f))
        temp_data = []
        for i in range(iterations):
            model(torch.randn(1, *model.base_input_size), inference_id = f"profile")
            from_model = model.master_dict.pop("profile")["layer_information"].values()
            temp_data.extend(from_model)
            # build a simple sequence from the first row of data
        for i in range(self.breakpoints):
            self.module_sequence.append((temp_data[i]["class"], temp_data[i]["parameter_bytes"], temp_data[i]["output_bytes"]))
        output_bytes = None # store output size bytes for usage in following layers that may not have true parameters
        for datapoint in temp_data:
            with open(os.path.join(self._dir, f'{datapoint["class"]}.csv') , 'a') as f:
                selected_value = datapoint['parameter_bytes'] if datapoint['parameter_bytes'] != 0 else output_bytes
                f.write(f"{selected_value}, {datapoint['inference_time']}\n")
            output_bytes = datapoint['output_bytes']

    def update_regression(self):
        for layer_type in os.listdir(self._dir):
            x, y = [], []
            self.regression[layer_type.split(".")[0]] = RegressionPartitioner.linreg()
            current_linreg = self.regression[layer_type.split(".")[0]]
            with open(os.path.join(self._dir, layer_type) , 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    x.append(float(line[0]))
                    y.append(float(line[1]))
                # normalize to avoid explosion
                x = torch.as_tensor(x)
                y = torch.as_tensor(y)
                if max(x) == min(x):
                    print(f"Insufficient data for linreg, setting w=0 b=middle quantile {layer_type.split('.')[0]}")
                    current_linreg.manually_set_weights(torch.as_tensor(0), torch.quantile(y, q=0.5))
                else:
                    # normalize to avoid explosion
                    mmax = max(max(x),max(y))    # pyright: ignore
                    x = x/mmax
                    y = y/mmax                             
                    for i in range(current_linreg.training_iter):
                        # variable dataset so batchsize of 1
                        for v, z in zip(x, y):
                            pred = current_linreg.forward(v)
                            current_linreg.train_pass(z, pred)
                    # then scale b by mmax, our data is not normalized to the same range
                    current_linreg.manually_scale_bias(mmax)

    def _get_network_speed_bytes(self, artificial_value = 4 * 1024**2):
        # needs work, ideal methodology to have a thread checking this continuously.
        return artificial_value if artificial_value else None # change none to the thread value 
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.estimate_split_point(starting_layer=0)
