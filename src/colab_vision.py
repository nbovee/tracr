from bz2 import compress
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

import colab_vision_pb2
import colab_vision_pb2_grpc


BITRATE = 0.1 * 2 ** 20# byte/s
USE_COMPRESSION = False
CHUNK_SIZE = 1024 #reduce size for testing * 1024  # 1MB
# this should probably be an independant database that client and server can both interact with async
results_dict = {}

def get_object_chunks(object):
    object = pickle.dumps(object)
    for pos in range(0, len(object), CHUNK_SIZE):
        piece = object[pos:pos + CHUNK_SIZE]
        if len(piece) == 0:
            return
        yield colab_vision_pb2.Chunk(chunk=piece)

def save_chunks_to_object(chunks):
    chunk_byte_list = []
    for c in chunks:
        chunk_byte_list.append(c.chunk)
    obj_bytes = b''.join(chunk_byte_list)
    return obj_bytes
