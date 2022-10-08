from concurrent import futures
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
import numpy as np
from PIL import Image

sys.path.append(".")
from model_wrapper_torch import Model
from . import colab_vision_pb2
from . import colab_vision_pb2_grpc


bitrate = 0.1 * 2 ** 20# byte/s

CHUNK_SIZE = 1024 * 1024  # 1MB
# this should probably be an independant database that client and server can both interact with async
dict_pattern = {
    'uuid' : None,
    'results' : None,
    'overall' : None,
    'upload' : None,
    'upload_delay' : None,
    'inference' : None,
    'download' : None,
    'download_delay' : None
}    

def get_file_chunks(filename):
    # first yield is always the filename. Later we will make this explicit
    # yield colab_vision_pb2.Chunk(chunk = filename)
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE);
            if len(piece) == 0:
                return
            yield colab_vision_pb2.Chunk(chunk=piece)


def save_chunks_to_file(chunks, filename):
    # old_filename = next(chunks).chunk
    # print(old_filename)
    # file_ext = os.path.splitext(old_filename)[1]
    full_filename = filename
    with open(full_filename, 'wb') as f:
        for c in chunks:
            f.write(c.chunk)

def save_chunks_to_object(chunks):
    chunk_bytes = []
    for c in chunks:
        chunk_bytes.append(c.chunk)
    img_bytes = b''.join(chunk_bytes)
    print(len(chunk_bytes))
    image = Image.open(io.BytesIO(img_bytes))
    return image

class FileClient:
    def __init__(self, address):
        channel = grpc.insecure_channel(address)
        self.stub = colab_vision_pb2_grpc.colab_visionStub(channel)

    def upload(self, in_file_name):
        start = timer()
        chunks_generator = get_file_chunks(in_file_name)
        response = self.stub.uploadImage(chunks_generator)
        result_dict = pickle.loads(response.chunk)
        result_dict['download'] = (timer() - result_dict['download'])
        time.sleep(result_dict['download_delay'])
        end = timer()
        result_dict['overall'] = ( end - start)
        # assert response.code == os.path.getsize(in_file_name)
        return result_dict

    def download(self, target_name, out_file_name):
        # have this time download time and report to server
        start = timer()
        response = self.stub.downloadFile(colab_vision_pb2.uuid(id=target_name))
        wait_time = (timer() - start)/1e9
        save_chunks_to_file(response, out_file_name)
        transfer_time = os.path.getsize(out_file_name) / (bitrate)
        print(f"Wait time: {wait_time} Trans time: {transfer_time}")
        time.sleep(transfer_time - wait_time) if wait_time < transfer_time else None   

    def processingTime(self, target_name):
        # print("target:", end='')
        # print(target_name)
        response = self.stub.resultTimeDownload(colab_vision_pb2.uuid(id=target_name))
        # response better be a dict serialized in json
        new_dict = json.loads(response.dict)
        return new_dict

class FileServer(colab_vision_pb2_grpc.colab_visionServicer):
    def __init__(self):

        class Servicer(colab_vision_pb2_grpc.colab_visionServicer):
            def __init__(self):
                self.tmp_folder = './tmp/server_tmp/'
                self.transaction_dict = {}
                self.filetype = ".jpg" # parametrize this later based on the upload name
                self.model = Model()
            
            def uploadImage(self, request_iterator, context):
                start = timer()
                data_dict = dict_pattern.copy()
                new_id = uuid.uuid4()
                data_dict['uuid'] = str(new_id)
                object = True
                filepath = self.tmp_folder + str(new_id) + self.filetype
                if object:
                    #img is an object
                    image = save_chunks_to_object(request_iterator)
                else:
                    #img is a filepath
                    save_chunks_to_file(request_iterator, filepath)
                    image = filepath
                transfer_timer = timer()
                data_dict['upload'] = (transfer_timer - start)
                result = self.model.predict(image)
                data_dict['results'] = result
                prediction_timer = timer()
                data_dict['inference'] = (prediction_timer - transfer_timer)
                # get filename and size from first pop of request iterator
                artificial_upload_speed = sys.getsizeof(image)/(bitrate) - data_dict['upload']
                if artificial_upload_speed < 0:
                    artificial_upload_speed = 0
                data_dict['upload_delay'] = artificial_upload_speed
                artificial_download_speed = sys.getsizeof(data_dict) / (bitrate)
                data_dict['download_delay'] = artificial_download_speed
                # download sleep must be client side to be accurate
                data_dict['download'] = timer()
                sleep_time = data_dict['upload_delay']
                time.sleep(sleep_time) if sleep_time > 0 else None   
                # print(sys.getsizeof(data_dict))
                # for i in data_dict:
                #     print(f"{i}: {data_dict[i]}")
                # return colab_vision_pb2.Ack(code=os.path.getsize(filepath), id=str(new_id))
                return colab_vision_pb2.Chunk(chunk=pickle.dumps(data_dict))

            def downloadFile(self, uuid, context):
                # print(uuid)
                if uuid.id:
                    filepath = self.tmp_folder + str(uuid.id) + self.filetype
                    transfer_time = os.path.getsize(filepath) / (bitrate)
                    # this will be fractionally long size the transfer still must happen
                    time.sleep(transfer_time)
                    return get_file_chunks(filepath)

            def resultTimeDownload(self, request, context):
                result = self.transaction_dict[uuid.UUID(request.id)]
                # print(result)
                return colab_vision_pb2.result_Time_Dict(dict = json.dumps(result))

        logging.basicConfig()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        colab_vision_pb2_grpc.add_colab_visionServicer_to_server(Servicer(), self.server)

    def start(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        self.server.wait_for_termination()
        # I think wait_for_termination() is more ideal?
        # try:
        #     while True:
        #         time.sleep(60*60*24)
        # except KeyboardInterrupt:
        #     self.server.stop(0)