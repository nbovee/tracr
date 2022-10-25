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
        channel = grpc.insecure_channel(address)
        self.stub = colab_vision_pb2_grpc.colab_visionStub(channel)

    def initiateConstantInference(self, target):
        #stuff
        for received_msg in self.stub.constantInference(colab_vision.inference_generator(target)):
            print("Received message from server with contents: ")
            for i in received_msg:
                print(i)
            results_dict[received_msg.pop(id)] = received_msg
        return None

        logging.basicConfig()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        colab_vision_pb2_grpc.add_colab_visionServicer_to_server(Servicer(), self.server)

    def inference_generator(data_loader):
    while data_loader.has_next():
        [ current_obj, exit_layer, filename ] = data_loader.next()
        message = colab_vision_pb2.Info_Chunk()
        message.action = colab_vision_pb2.Action()
        message.id = uuid.UUID()
        results_dict[message.id] = {}
        results_dict[message.id]["filename"] = filename
        # getting split layer should be broken out and methodized
        # for current_split_layer in range(1, Model.max_layers + 1): # we will be iterating over split layers to generate test results. 0 = server handles full inference (tbi). Max_layers + 1 = client handles full inference (tbi)
        message.layer = exit_layer + 1 # the server begins inference 1 layer above where the edge exited
        #split into chunks, set values, add message to messages list
        if USE_COMPRESSION:
            message.action.append(5)
            current_obj = blosc.compress(current_obj)
        for i, piece in enumerate(get_object_chunks(current_obj)):
            message.chunk = piece
            if i == 0:
                message.action.append(1)
            if piece is None: #current behavior will send the entirety of the current_obj, then when generator ends, follow up with action flags. small efficiency boost possible if has_next is altered
                message.action.append(3)
            yield message

    def start(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        self.server.wait_for_termination()
