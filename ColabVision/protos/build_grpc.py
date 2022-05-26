from grpc_tools import protoc

protoc.main((
    '',
    '--proto_path=.',
    '--python_out=../src',
    '--grpc_python_out=../src',
    './colab_vision.proto',
))

# (.venv) nick@DESKTOP-HF7K570:~/collaborative-vision-research/ColabVision/protos$ 
# python -m grpc_tools.protoc -I. --python_out=../src --grpc_python_out=../src ./colab_vision.proto