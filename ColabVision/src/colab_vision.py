from concurrent import futures
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
import json
import pickle
import blosc
import numpy as np
from PIL import Image

sys.path.append(".")
# from model_wrapper_torch import Model
from . import colab_vision_pb2
from . import colab_vision_pb2_grpc


bitrate = 0.1 * 2 ** 20# byte/s

CHUNK_SIZE = 1024#reduce size for testing * 1024  # 1MB
# this should probably be an independant database that client and server can both interact with async


def get_object_chunks(filename):
    # first yield is always the filename. Later we will make this explicit
    # yield colab_vision_pb2.Chunk(chunk = filename)
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE);
            if len(piece) == 0:
                return
            yield colab_vision_pb2.Chunk(chunk=piece)

def save_chunks_to_object(chunks):
    chunk_bytes = []
    for c in chunks:
        chunk_bytes.append(c.chunk)
    img_bytes = b''.join(chunk_bytes)
    print(len(chunk_bytes))
    image = Image.open(io.BytesIO(img_bytes))
    return image

def inference_generator(model_wrapper):
    while model_wrapper.has_next():
        current_obj = model_wrapper.next()
        message = colab_vision_pb2.Info_Chunk()
        message.id = uuid.UUID()
        #split into chunks, set values, add message to messages list
        for i, piece in enumerate(get_object_chunks(current_obj)):
            message.action = []
            message.chunk = piece
            if i == 0:
                message.action.append(1)
            if piece is None: #current behavior will send the entirety of the current_obj, then when generator ends, follow up with action flags. small efficiency boost possible if has_next is altered
                message.action.append(3)
            yield message

class FileClient:
    def __init__(self, address):
        channel = grpc.insecure_channel(address)
        self.stub = colab_vision_pb2_grpc.colab_visionStub(channel)

    def initiateConstantInference(self, target):
        #stuff
        for received_msg in self.stub.constantInference(inference_generator(target)):
            print("Received message from server with contents: ")
            for i in received_msg:
                print(i)

        return None

class FileServer(colab_vision_pb2_grpc.colab_visionServicer):
    def __init__(self):

        class Servicer(colab_vision_pb2_grpc.colab_visionServicer):
            def __init__(self):
                self.tmp_folder = './tmp/server_tmp/'
                # self.model = Model()
            
            def constantInference(self, request_iterator, context):
                #unpack msg contents
                current_chunks = []
                for msg in request_iterator:
                    print("Received message from client with contents: ")
                    for thingy in msg:
                        print(thingy)
                    if 4 in msg.action:
                        break #exit
                    if 1 in msg.action:
                        #reset operation regardless of current progress
                        current_chunks = []
                    if msg.id == last_id:
                        current_chunks.append(msg.chunk)
                        #continue the same inference
                    else:
                        current_chunks = [].append(msg.chunk)
                    #continue the same inference
                    if 2 in msg.action: 
                        #convert chunks into object and save at appropriate layer
                        pass #not yet implemented
                    if 3 in msg.action:
                        #convert chunks into object and perform inference
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
