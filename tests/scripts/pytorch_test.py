pytorch_test_script = """#!/usr/bin/env python3
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    # Load pre-trained AlexNet model
    model = models.alexnet(pretrained=True)
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Using device:", device)

    # Prepare input
    input_size = [3, 224, 224]
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(dummy_input)
    
    print("Output shape:", output.shape)
    print("Top 5 predicted classes:", output.topk(5).indices.cpu().numpy())

if __name__ == "__main__":
    main()
"""