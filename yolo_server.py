import torch
import torchvision
import os
import zlib
import pickle
import time
import sys
import socket
from pathlib import Path
import os
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import blosc2
from src.tracr.experiment_design.models.model_hooked import WrappedModel, HookExitException, NotDict

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

weight_path = 'runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
model = WrappedModel(config_path=yaml_file_path, weights_path=weight_path, participant_key = 'server')

# def decompress_data(compressed_data):
#     decompressed_data = zlib.decompress(compressed_data)
#     data = pickle.loads(decompressed_data)
#     return data

# Blosc2 Decompression
def decompress_data(compressed_data):
    decompressed_data = blosc2.decompress(compressed_data)
    data = pickle.loads(decompressed_data)
    return data

def receive_full_message(conn, expected_length):
    data_chunks = []
    bytes_recd = 0
    while bytes_recd < expected_length:
        chunk = conn.recv(min(expected_length - bytes_recd, 4096))
        if chunk == b'':
            raise RuntimeError("Socket connection broken")
        data_chunks.append(chunk)
        bytes_recd += len(chunk)
    return b''.join(data_chunks)

def postprocess(outputs, original_img_size, conf_threshold=0.25, iou_threshold=0.45):
    """
    Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.
    """
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # Adjust based on the structure of outputs

    outputs = outputs.detach().cpu().numpy()
    outputs = np.transpose(np.squeeze(outputs))
    rows = outputs.shape[0]

    boxes = []
    scores = []
    class_ids = []

    img_w, img_h = original_img_size
    input_height, input_width = 640, 640

    x_factor = img_w / input_width
    y_factor = img_h / input_height

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= conf_threshold:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    detections = []
    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            print(f"Class: {class_names[class_id]}, Score: {score:.2f}, Box: {box}")
            detections.append((box, score, class_id))

    return detections

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
        compressed_data = receive_full_message(conn, expected_length)
 
        # Assuming decompress_data function returns the deserialized object
        received_data = decompress_data(compressed_data)

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
            detections = postprocess(res, original_img_size)

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


