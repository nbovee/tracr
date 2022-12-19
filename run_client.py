import os
import src.colab_vision_client as cv
import time
import atexit
from test_data import test_data_loader as data_loader
if __name__ == '__main__':
    # client = cv.FileClient('grpc_server:8893')
    # client = cv.FileClient('DESKTOP-HF7K570:8893')
    client = cv.FileClient('192.168.1.254:8893')
    atexit.register(client.safeClose)

    # Model = blah
    # client.setModel(Model)
    # data_loader = blah
    # client.setDataLoader(data_loader)

    # for i in range(num_tests):
    # print(os.path.exists(in_file_name))
    test_data = data_loader()
    while test_data.has_next():
        client.initiateInference(test_data)
        # overall += results["inference"]
        # for i in results:
        #     val = results[i]
        #     if i not in ["uuid", "results"]:
        #         val = float(val)
        #         print(f"{i} : {results[i]:.04f}")
        #     else:
        #         print(f"{i} : {results[i]}")
    # print(f"Average over {num_tests} runs: {overall/num_tests:0.04f}")
    # for i in results:
    #     val = results[i]
    #     if i not in ["uuid", "results"]:
    #         val = float(val)
    #         print(f"{i} : {results[i]:.04f}")
    #     else:
    #         print(f"{i} : {results[i]}")
 # docker run --gpus all --name "grpc_server" -t --mount type=bind,source="F:\Nick\Documents\Code\Work\Summer Research\collab-vision",target=/app nvcr.io/nvidia/pytorch:21.08-py3 nvidia-smi
# docker run --gpus all --name "grpc_server" --memory=16g --oom-kill-disable -t --mount type=bind,source="F:\Nick\Documents\Code\Work\Summer Research\collab-vision",target=/app nvcr.io/nvidia/pytorch:21.08-py3
# docker run --name "grpc_client" --memory=4g --oom-kill-disable -t --mount type=bind,source="F:\Nick\Documents\Code\Work\Summer Research\collab-vision",target=/app nvcr.io/nvidia/pytorch:21.08-py3
# arm64 pytorch l4t-pytorch:r32.7.1-pth1.10-py3

# wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
# pip install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl