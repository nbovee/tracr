from grpc_tools import protoc
import os

print(os.getcwd())

protoc.main((
    '',
    '--proto_path=/ColabVision/protos',
    '--python_out=../src',
    '--grpc_python_out=../src',
    '/ColabVision/protos/*.proto',
))

# (.venv) nick@DESKTOP-HF7K570:~/collaborative-vision-research/ColabVision/protos$ 
# python -m grpc_tools.protoc -I. --python_out=../src --grpc_python_out=../src ./colab_vision.proto
# python3 -m grpc_tools.protoc -I./ColabVision/protos --python_out=./ColabVision/src/ --grpc_python_out=./ColabVision/src ./ColabVision/protos/colab_vision.proto