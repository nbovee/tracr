import sys
import logging
import os
import io
from concurrent import futures
import grpc
# from timeit import default_timer as timer
import time
# from time import perf_counter_ns as timer, process_time_ns as cpu_timer
from time import time as timer
import uuid
import pickle
import blosc2 as blosc
import numpy as np
from PIL import Image

sys.path.append(".")
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from alexnet_pytorch_split import Model
from test_data import test_data_loader as data_loader

from . import colab_vision
from . import colab_vision_pb2
from . import colab_vision_pb2_grpc

class FileServer(colab_vision_pb2_grpc.colab_visionServicer):
    def __init__(self):

        class Servicer(colab_vision_pb2_grpc.colab_visionServicer):
            def __init__(self):
                self.tmp_folder = './temp/'
                # self.model = Model()

            def constantInference(self, request_iterator, context):
                #unpack msg contents
                current_chunks = []
                last_id = None
                for i, msg in enumerate(request_iterator):
                    print(f"Message received with id {msg.id}. Responding with Dummy.")
                    yield colab_vision_pb2.Response_Dict(
                            id = f"test response for {msg.id} is {i}",
                            keypairs = None,
                            results = None,
                            actions = None
                        )

            def constantInference_1(self, request_iterator, context):
                #unpack msg contents
                current_chunks = []
                last_id = None
                print("inside bidirectional")
                for msg in request_iterator:
                #     print("Received message from client with contents: ")
                #     for thingy in msg:
                #         print(thingy)
                    # if 4 in msg.action:
                    #     break #exit
                    # if 1 in msg.action:
                    #     #reset operation regardless of current progress
                    #     current_chunks = []
                    #     last_id = msg.layer
                    # if msg.id == last_id:
                    #     current_chunks.append(msg.chunk)
                    #     #continue the same inference
                    # else:
                    #     current_chunks = [].append(msg.chunk)
                    # #continue the same inference
                    # if 2 in msg.action: 
                    #     #convert chunks into object and save at appropriate layer
                    #     current_chunks = save_chunks_to_object(current_chunks)
                    #     if 5 in msg.action: # decompress
                    #         current_chunks = blosc.decompress(current_chunks)
                    #     pickle.loads(current_chunks)
                    #     pass #not yet implemented
                    # if 3 in msg.action:
                    #     #convert chunks into object and perform inference
                    #     if 5 in msg.action: # decompress
                    #         current_chunks = blosc.decompress(current_chunks)
                    #     pickle.loads(current_chunks)
                    print(f"Message received with id {msg.id}. Responding.")
                    yield colab_vision_pb2.Response_Dict(
                            id = "test",
                            keypairs = None,
                            results = None,
                            actions = None
                        )
      
                    
                #deal with chunks

                #do flag actions


        logging.basicConfig()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        colab_vision_pb2_grpc.add_colab_visionServicer_to_server(Servicer(), self.server)

    def start(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        print("Server started.")
        self.server.wait_for_termination()