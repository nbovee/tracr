from concurrent import futures
from email import message
import enum
from fileinput import filename
from logging.handlers import WatchedFileHandler
from multiprocessing.connection import wait
import sys
import logging
import os
import io
import grpc
# from timeit import default_timer as timer
import time
# from time import perf_counter_ns as timer, process_time_ns as cpu_timer
from time import time as timer
import uuid
import pickle
import blosc
import numpy as np
from PIL import Image

sys.path.append(".")
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from alexnet_pytorch_split import Model
from test_data import test_data_loader as data_loader

import colab_vision
import colab_vision_pb2
import colab_vision_pb2_grpc

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
                for msg in request_iterator:
                    print("Received message from client with contents: ")
                    for thingy in msg:
                        print(thingy)
                    if 4 in msg.action:
                        break #exit
                    if 1 in msg.action:
                        #reset operation regardless of current progress
                        current_chunks = []
                        last_id = msg.layer
                    if msg.id == last_id:
                        current_chunks.append(msg.chunk)
                        #continue the same inference
                    else:
                        current_chunks = [].append(msg.chunk)
                    #continue the same inference
                    if 2 in msg.action: 
                        #convert chunks into object and save at appropriate layer
                        current_chunks = colab_vision.save_chunks_to_object(current_chunks)
                        if 5 in msg.action: # decompress
                            current_chunks = blosc.decompress(current_chunks)
                        pickle.loads(current_chunks)
                        pass #not yet implemented
                    if 3 in msg.action:
                        #convert chunks into object and perform inference
                        if 5 in msg.action: # decompress
                            current_chunks = blosc.decompress(current_chunks)
                        pickle.loads(current_chunks)
                        pass
                        
                    
                #deal with chunks

                #do flag actions
                pass

        logging.basicConfig()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        colab_vision_pb2_grpc.add_colab_visionServicer_to_server(Servicer(), self.server)

    def start(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        self.server.wait_for_termination()

