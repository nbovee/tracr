from operator import mod
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

image_size = (224, 224)
# model = torch.hub.load('pytorch/vision:v0.10.0', selected, pretrained=True)
model = None
mode = 'cuda'
max_layers = 21

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SplitAlex(models.AlexNet):

    # almost exactly like pytorch AlexNet, but we cannot split out of a Sequential so ModuleList is used instead
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__() # have to do this to get some stuff out of the way.
        # _log_api_usage_once(self) #idk what this is

        self.features = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.ModuleList([
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ])

    def forward(self, x: torch.Tensor, start_layer = 0, end_layer = np.inf) -> torch.Tensor:
        i = 0
        for i in range(start_layer, min(len(self.features), end_layer)):
            x = self.features[i].forward(x)
        for i in range(start_layer + len(self.features), min(len(self.features) + 1, end_layer)):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        for i in range(start_layer + len(self.features) + 1, min(len(self.features) + len(self.classifier) + 1, end_layer)):
            x = self.classifier[i - 14].forward(x) #fix magic offset later
        return x


class Model:
    def __init__(self,) -> None:
        global model 
        # values = models.alexnet(pretrained=True).state_dict()
        model = SplitAlex()
        model.load_state_dict(models.alexnet(pretrained=True).state_dict())
        # print(model)
        model.eval()
        self.max_layers =  max_layers
        if torch.cuda.is_available() and mode == 'cuda':
            print("Loading Model to CUDA.")
            model.to(mode)
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        print("Imagenet categories loaded.")
        self.warmup()


    def predict(self, payload, start_layer = 0, end_layer = np.inf):
        if isinstance(payload, Image.Image):
            if payload.size != image_size:
                payload = payload.resize(image_size)
                # img = Image.load_img(img, target_size=image_size) #?
            input_tensor = preprocess(payload)
            input_tensor = input_tensor.unsqueeze(0)
        elif isinstance(payload, torch.Tensor):
            input_tensor = payload 
        if torch.cuda.is_available() and mode == 'cuda':
            input_tensor = input_tensor.to(mode)
        with torch.no_grad():
            predictions = model(input_tensor, start_layer = start_layer, end_layer = end_layer)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        # Show top categories per image
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        prediction = self.categories[top1_catid]
        return prediction

    
    def warmup(self, iterations = 50):
        if mode != 'cuda':
            print("Warmup not required.")
        else:
            print("Starting warmup.")
            imarray = np.random.rand(*image_size, 3) * 255
            for i in range(iterations):
                warmup_image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
                _ = self.predict(warmup_image)
            print("Warmup complete.")

if __name__ == "__main__":
    m = Model()