import logging
import pickle
import socket
import sys
import time
from pathlib import Path

import torch

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.experiment_design.models.model_hooked import WrappedModel, NotDict
from src.experiment_design.utils import DetectionUtils, DataUtils

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")


# ------------------ FIX THIS PART (START) ------------------
weight_path = 'data/runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
yaml_file_path = 'config/model_config.yaml'
font_path = 'fonts/DejaVuSans-Bold.ttf'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WrappedModel(config=yaml_file_path)

# ------------------ FIX THIS PART (END) ------------------




server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '0.0.0.0'
port = 12345
server_socket.bind((host, port))
server_socket.listen(1)
print("Server is listening...")

conn, addr = server_socket.accept()
print(f"Connected by {addr}")
print(device)

 # Process data
model.eval()

# Server Code Snippet for Processing Each Image
try:
    while True:
        # Receiving split layer index
        split_layer_index_bytes = conn.recv(4)
        if not split_layer_index_bytes:
            break
        split_layer_index = int.from_bytes(split_layer_index_bytes, 'big')

        # Receiving data length
        length_data = conn.recv(4)
        expected_length = int.from_bytes(length_data, 'big')

        # Receiving compressed data
        compressed_data = DataUtils.receive_full_message(conn, expected_length)
 
        # Assuming decompress_data function returns the deserialized object
        received_data = DataUtils.decompress_data(compressed_data)

        # Unpack the received data
        out, original_img_size = received_data

        # Start timing server processing
        server_start_time = time.time()

        with torch.no_grad():
            if isinstance(out, NotDict):
                inner_dict = out.inner_dict  # Access the inner dictionary
                # Move all tensors in the dictionary to the correct device
                for key in inner_dict:
                    if isinstance(inner_dict[key], torch.Tensor):
                        inner_dict[key] = inner_dict[key].to(model.device)  # Move tensors to the correct device
                        print(f"Intermediate tensors of {key} moved to the correct device.")
            else:
                print("out is not an instance of NotDict")
            res, layer_outputs = model(out, start=split_layer_index)
            detection_utils = DetectionUtils(class_names=class_names, font_path=font_path)
            detections = detection_utils.postprocess(res, original_img_size)

        # End timing server processing
        server_processing_time = time.time() - server_start_time

        # Send back prediction and server processing time
        response_data = pickle.dumps((detections, server_processing_time))
        conn.sendall(response_data)

except Exception as e:
    print(f"Encountered exception: {e}")
finally:
    conn.close()
    server_socket.close()
    print("Server socket closed.")



