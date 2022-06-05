from concurrent import futures
import sys
import logging
import os
import grpc
from timeit import default_timer as timer
import uuid
import json


import colab_vision_pb2
import colab_vision_pb2_grpc

CHUNK_SIZE = 1024 * 1024  # 1MB
dict_pattern = {
    'upload' : None,
    'handling' : None,
    'inference' : None,
    'download' : None
}


def get_file_chunks(filename):
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE);
            if len(piece) == 0:
                return
            yield colab_vision_pb2.Chunk(chunk=piece)


def save_chunks_to_file(chunks, filename):
    start = timer()
    with open(filename, 'wb') as f:
        for c in chunks:
            f.write(c.chunk)
    return timer() - start


class FileClient:
    def __init__(self, address):
        channel = grpc.insecure_channel(address)
        self.stub = colab_vision_pb2_grpc.colab_visionStub(channel)

    def upload(self, in_file_name):
        chunks_generator = get_file_chunks(in_file_name)
        response = self.stub.uploadFile(chunks_generator)
        assert response.code == os.path.getsize(in_file_name)

    def download(self, target_name, out_file_name):
        response = self.stub.downloadFile(colab_vision_pb2.Request(target=target_name))
        save_chunks_to_file(response, out_file_name)

    def processingTime(self, uuid):
        response = self.stub.resultTimeDownload(colab_vision_pb2.result_Time_Dict(id=str(uuid)))
        # response better be a dict serialized in json
        return json.loads(response)

class FileServer(colab_vision_pb2_grpc.colab_visionServicer):
    def __init__(self):

        class Servicer(colab_vision_pb2_grpc.colab_visionServicer):
            def __init__(self):
                self.tmp_file_name = '/tmp/server_tmp'
                self.transaction_dict = {}

            def uploadFile(self, request_iterator, context):
                upload_time = save_chunks_to_file(request_iterator, self.tmp_file_name)
                id = uuid.uuid4()
                self.transaction_dict[id] = dict_pattern.copy()
                self.transaction_dict[id]['upload'] = upload_time 
                print(f"File Saved as {id}")
                return colab_vision_pb2.Ack(code=os.path.getsize(self.tmp_file_name). id=str(id))

            def downloadFile(self, request, context):
                if request.target:
                    return get_file_chunks(self.tmp_file_name)

            def resultTimeDownload(self, request, context):
                return super().resultTimeDownload(request, context)

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