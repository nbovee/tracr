from partitioner.linreg_partitioner import RegressionPartitioner
from partitioner.iter_partitioner import CyclePartitioner
from model.model_hooked import WrappedModel
from participant_rpc_node import EdgeService
import uuid
import atexit
import torch
import rpyc
from rpyc import ThreadedServer
import threading
import blosc2
import time
import sys
import json
import pickle

timer = time.perf_counter_ns

# this module reads configs and sets up the modules and local rpc
keepalive = True  

def parse_input(self, input):
    """Checks if the input is appropriate at the given stage of the network. Does not yet check Tensor shapes for intermediate layers."""
    if isinstance(input, Image.Image):
        if input.size != self.base_input_size:
            input = input.resize(self.base_input_size)
        input_tensor = self.preprocess(input)
        input_tensor = input_tensor.unsqueeze(0)
    elif isinstance(input, torch.Tensor):
        input_tensor = input
    if (
        torch.cuda.is_available()
        and self.mode == "cuda"
        and input_tensor.device != self.mode
    ):
        input_tensor = input_tensor.to(self.mode)
    return input_tensor

def parse_output(self, predictions):
    """Take the final output tensor of the wrapped model and map it to its appropriate human readable results."""
    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    # Show top categories per image
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    prediction = self.categories[top1_catid]
    return prediction

def safeClose():
    keepalive = False

def start_server(server_stub):
    
    def thread_loop():
        server_stub.start()
        while keepalive:
            time.sleep(5)
        server_stub.close()

    thread = threading.Thread(target=thread_loop)
    thread.daemon = True
    thread.start()

def callback_with_result(parent_uuid, result):
    global master_dictionary
    master_dictionary[parent_uuid]["result"] = result
    print(f"callback for {parent_uuid}")
    # signal observer here
    obs_conn.root.inference_completed_signal(parent_uuid)

atexit.register(safeClose)
if __name__ == "__main__":
    global master_dictionary
    master_dictionary = dict()

    m = WrappedModel(dict = master_dictionary, depth = 2)
    Scheduler = RegressionPartitioner(m.splittable_layer_count)
    # Scheduler = CyclePartitioner(m.splittable_layer_count, clip_min_max=False)
    Scheduler.create_data(m)
    Scheduler.update_regression()
    this_server = ThreadedServer(EdgeService(), auto_register=True)
    this_server.service.link_dict(master_dictionary)

    # this block could easily be cleaned up
    cloud_conn = None
    obs_conn = None
    while cloud_conn is None:
        try:
            cloud_conn = rpyc.connect_by_service("cloud", service=EdgeService())
        except:
            print("waiting for cloud server.")
            time.sleep(1)
    while obs_conn is None:
        try:
            obs_conn = rpyc.connect_by_service("observer")
        except:
            print("waiting for observer server.")
            time.sleep(1)


    # start our server after finding cloud, so we don't grab it! Would be easier to avoid if we parsed our own IP

    start_server(this_server)
    Scheduler.add_server_module(pickle.loads(cloud_conn.root.get_scheduler_dict()))

    while keepalive:
        s = Scheduler()
        input = torch.randn(1, *m.base_input_size)
        x_uuid = str(uuid.uuid4())
        print(f"Split point: {s} Start inference {x_uuid}")
        x = m(input, inference_id = x_uuid, end=s)
        x = blosc2.pack_tensor(x)
        master_dictionary[x_uuid]["compressed_size"] = sys.getsizeof(x)
        timestamp = timer()
        cloud_conn.root.complete_inference(x, x_uuid, s)
        master_dictionary[x_uuid]["transfer_time"] = timer() - timestamp
        time.sleep(.1) 
        obs_conn.root.inference_completed_signal(x_uuid)
        time.sleep(1) # server callback and queue need some work so this is our ratelimit for now
        
    keepalive = False