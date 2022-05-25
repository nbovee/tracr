from concurrent import futures
import logging

import grpc
import colab_vision_pb2
import colab_vision_pb2_grpc


class colab_vision_bridge(colab_vision_pb2_grpc.colab_vision_bridgeServicer):

    def UploadImage(self, request, context):
        # process bytes into image
        # alert ML module to new target
        # acknowledge receipt of file (processing occurs, then remote will contact client with results)
        return colab_vision_pb2.Ack(code=1)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    colab_vision_pb2_grpc.add_colab_vision_bridgeServicer_to_server(colab_vision_bridge(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

def start_server():
    logging.basicConfig()
    serve()


if __name__ == '__main__':
    start_server()
