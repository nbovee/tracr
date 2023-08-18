import rpyc
import uuid
from rpyc.utils.server import ThreadedServer
from rpyc.utils.helpers import classpartial
import blosc2
import time
import atexit
import threading
import pickle

timer = time.perf_counter_ns

class ParticipantService(rpyc.Service):
    """Base class for all particpant nodes"""

    def on_connect(self, conn):
        # code that runs when a connection is created
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass
    
    def link_dict(self, _dict):
        self.master_dict = _dict

    def link_model(self, m):
        self.model = m    

    def link_scheduler(self, s):
        self.s = s

    def exposed_get_inference_dict(self, uuid): # this is an exposed method
        """pickled to decouple from server"""
        return pickle.dumps(self.master_dict.pop(str(uuid), []))
    
class EdgeService(ParticipantService):

    ALIASES = ["Edge","Participant"]

    def exposed_get_result(self, _uuid, result):
        self.master_dict[_uuid.split(".")]["result"] = result

class CloudService(ParticipantService):

    ALIASES = ["Cloud","Participant"]

    def exposed_get_scheduler_dict(self):
        # pass linear estimators back to client for usage
        return self.s.pass_regression_copy()
    
    def exposed_complete_inference(self, serialized_tensor, parent_uuid, start_layer):
        '''sends a tensor TO the service being called. UUID has a suffix.'''
        timestamp = timer()
        print(f"called for {parent_uuid}")
        x = blosc2.unpack_tensor(serialized_tensor)
        
        def complete_inference(x, inference_id = parent_uuid+".0", start = start_layer):
            x = self.model(x, inference_id = inference_id, start = start)
            self.master_dict[inference_id.split(".")[0]]["result"] = x
        # complete the inference outside of this timed method
        threading.Thread(target=complete_inference, args = [x], kwargs={"start":start_layer,"inference_id":parent_uuid+".0"}).start()
        return timer() - timestamp

# rpc_service = classpartial(ParticipantService) # decoupled early just in case
# node1 = ThreadedServer(rpc_service, port=18861) # rpyc 4.0+ only, single server for all incoming requests to Node to reduce overhead

if __name__ == "__main__":
    from model.model_hooked import WrappedModel
    from partitioner.linreg_partitioner import RegressionPartitioner
    # start in server mode
    global master_dictionary
    master_dictionary = dict()
    print("Init Model.")
    m = WrappedModel(dict = master_dictionary, depth = 2)
    print("Init Scheduler.")
    Scheduler = RegressionPartitioner(m.splittable_layer_count)
    print("Running Scheduler regression.")
    Scheduler.create_data(m)
    Scheduler.update_regression()
    this_server = ThreadedServer(CloudService(), auto_register=True, protocol_config={'allow_public_attrs': True})
    this_server.service.link_dict(master_dictionary)
    this_server.service.link_model(m)
    this_server.service.link_scheduler(Scheduler)
    print("Starting server.")
    atexit.register(this_server.close)
    this_server.start()