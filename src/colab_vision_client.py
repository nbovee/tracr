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

from src.colab_vision import USE_COMPRESSION
import alexnet_pytorch_split.Model as model

sys.path.append(".")
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from alexnet_pytorch_split import Model
from test_data import test_data_loader as data_loader

from . import colab_vision
from . import colab_vision_pb2
from . import colab_vision_pb2_grpc

class FileClient:
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address)
        self.stub = colab_vision_pb2_grpc.colab_visionStub(self.channel)
        self.results_dict = {}
        logging.basicConfig()
        self.model = model()

    def safeClose(self):
        self.channel.close()
        
    def initiateInference(self, target):
        #stuff
        messages = self.stub.constantInference(self.inference_generator(target))
        for received_msg in messages:
            print("Received message from server with contents: ")
            print(received_msg)
            # results_dict[received_msg.pop(id)] = received_msg

    def inference_generator_test(self, data_loader):
        for i in range(5):
            yield colab_vision_pb2.Info_Chunk(id = "test")

    def inference_generator(self, data_loader):
        print("image available.")
        tmp = data_loader.next()
        while(tmp):
            try:
                [ current_obj, exit_layer, filename ] = next(tmp)
            except StopIteration:
                return
            current_obj = model.predict(current_obj, end_layer=exit_layer)
            message = colab_vision_pb2.Info_Chunk()
            message.ClearField('action')#colab_vision_pb2.Action()
            message.id = uuid.uuid4().hex # uuid4().bytes is utf8 not unicode like grpc wants
            self.results_dict[message.id] = {} 
            self.results_dict[message.id]["filename"] = filename
            message.layer = exit_layer + 1 # the server begins inference 1 layer above where the edge exited
            if colab_vision.USE_COMPRESSION:
                message.action.append(5)
                current_obj = blosc.compress(current_obj)
            for i, piece in enumerate(colab_vision.get_object_chunks(current_obj)):
                message.chunk.CopyFrom(piece)
                message.ClearField('action')#colab_vision_pb2.Action()
                if i == 0:
                    message.action.append(1)
                if piece is None: #current behavior will send the entirety of the current_obj, then when generator ends, follow up with action flags. small efficiency boost possible if has_next is altered
                    message.action.append(3)
                print(message)
                yield message

    # def start(self, port):
    #     self.server.add_insecure_port(f'[::]:{port}')
    #     self.server.start()
    #     self.server.wait_for_termination()