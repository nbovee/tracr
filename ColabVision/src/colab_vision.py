from concurrent import futures
from fileinput import filename
from logging.handlers import WatchedFileHandler
from multiprocessing.connection import wait
import sys
import logging
import os
import grpc
# from timeit import default_timer as timer
import time
from time import perf_counter_ns as timer, process_time_ns as cpu_timer
import uuid
import json

sys.path.append(".")
from . import colab_vision_pb2
from . import colab_vision_pb2_grpc


bitrate = 0.100 # MB/s

CHUNK_SIZE = 1024 * 1024  # 1MB
# this should probably be an independant database that client and server can both interact with async
dict_pattern = {
    'overall' : None,
    'upload' : None,
    'inference' : None,
    'download' : None
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



class FileClient:
    def __init__(self, address):
        channel = grpc.insecure_channel(address)
        self.stub = colab_vision_pb2_grpc.colab_visionStub(channel)

    def upload(self, in_file_name):
        chunks_generator = get_file_chunks(in_file_name)
        response = self.stub.uploadFile(chunks_generator)
        # assert response.code == os.path.getsize(in_file_name)
        return response

    def download(self, target_name, out_file_name):
        # have this time download time and report to server
        start = timer()
        response = self.stub.downloadFile(colab_vision_pb2.uuid(id=target_name))
        wait_time = (timer() - start)/1e9
        save_chunks_to_file(response, out_file_name)
        transfer_time = os.path.getsize(out_file_name) / (bitrate * 2**20)
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

            def uploadFile(self, request_iterator, context):
                start = timer()
                new_id = uuid.uuid4()
                # get filename and size from first pop of request iterator
                filepath = self.tmp_folder + str(new_id) + self.filetype
                save_chunks_to_file(request_iterator, filepath)
                wait_time = (timer() - start)/1e9
                transfer_time = os.path.getsize(filepath) / (bitrate * 2**20)
                print(f"Wait time: {wait_time} Trans time: {transfer_time}")
                self.transaction_dict[new_id] = dict_pattern.copy()
                time.sleep(transfer_time - wait_time) if wait_time < transfer_time else None   
                print(f"File Saved at {filepath}")
                self.transaction_dict[new_id]['upload'] = wait_time*1e9
                self.transaction_dict[new_id]['upload_slowed'] = transfer_time*1e9 - wait_time*1e9
                # print(self.transaction_dict[new_id])
                # trigger inference
                 
                return colab_vision_pb2.Ack(code=os.path.getsize(filepath), id=str(new_id))

            def downloadFile(self, uuid, context):
                # print(uuid)
                if uuid.id:
                    filepath = self.tmp_folder + str(uuid.id) + self.filetype
                    transfer_time = os.path.getsize(filepath) / (bitrate * 2**20)
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
        # self.server.wait_for_termination()
        # I think wait_for_termination() is more ideal?
        try:
            while True:
                time.sleep(60*60*24)
        except KeyboardInterrupt:
            self.server.stop(0)