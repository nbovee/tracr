from grpc_tools import protoc

protoc.main((
    '',
    '--proto_path=.',
    '--python_out=../src',
    '--grpc_python_out=../src',
    './colab_vision.proto',
))

# python -m grpc_tools.protoc -I. --python_out=../src --grpc_python_out=../src ./chunk.proto