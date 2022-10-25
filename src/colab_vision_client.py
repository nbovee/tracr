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

from src.colab_vision import USE_COMPRESSION

sys.path.append(".")
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from alexnet_pytorch_split import Model
from test_data import test_data_loader as data_loader

import colab_vision
import colab_vision_pb2
import colab_vision_pb2_grpc

class FileClient:
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address)
        self.stub = colab_vision_pb2_grpc.colab_visionStub(self.channel)
        logging.basicConfig()

    def safeClose(self):
        self.channel.close()
        
    def initiateConstantInference(self, target):
        #stuff
        messages = self.stub.constantInference(inference_generator(target))
        for received_msg in messages:
            print("Received message from server with contents: ")
            print(received_msg)
            # results_dict[received_msg.pop(id)] = received_msg



    def inference_generator(data_loader):
        for i in range(5):
            yield colab_vision_pb2.Info_Chunk(id = "test")

    def inference_generator_1(data_loader):
        print("inference generator")
        print(data_loader.has_next())
        while data_loader.has_next():
            print("inside while")
            tmp = data_loader.next()
            yield colab_vision_pb2.Info_Chunk()
            if tmp is not None:
                [ current_obj, exit_layer, filename ] = tmp
            print(f"{current_obj}  {exit_layer}  {filename}")
            print("msg1")
            message = colab_vision_pb2.Info_Chunk()
            message.action = colab_vision_pb2.Action()
            message.id = uuid.UUID()
            results_dict[message.id] = {}
            results_dict[message.id]["filename"] = filename
            message.layer = exit_layer + 1 # the server begins inference 1 layer above where the edge exited
            print("midway")
            if compress:
                message.action.append(5)
                current_obj = blosc.compress(current_obj)
            for i, piece in enumerate(get_object_chunks(current_obj)):
                message.chunk = colab_vision_pb2.Chunk(piece = piece)
                if i == 0:
                    message.action.append(1)
                if piece is None: #current behavior will send the entirety of the current_obj, then when generator ends, follow up with action flags. small efficiency boost possible if has_next is altered
                    message.action.append(3)
                for i in message:
                    print(i)
                yield message

    def start(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        self.server.wait_for_termination()
