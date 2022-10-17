import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

image_size = (224, 224)
# model = torch.hub.load('pytorch/vision:v0.10.0', selected, pretrained=True)
model = models.alexnet(pretrained=True)
mode = 'cuda'
max_layers = 8

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SplitAlex(models.AlexNet):

    # almost exactly like pytorch AlexNet, but we cannot split out of a Sequential
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__() # have to do this to get some stuff out of the way.
        # _log_api_usage_once(self) #idk what this is

        # we still need weights but that actually doesnt affect our test cases at the moment
        self.features = [
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
        ]
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = [
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ]

    def forward(self, x: torch.Tensor, start_layer = 0, end_layer = 9999) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Model:
    def __init__(self,) -> None:
        global model 
        model.eval()
        self.max_layers =  max_layers
        if torch.cuda.is_available():
            model.to(mode)
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        # self.model = model(weights='imagenet')
        self.warmup()


    def predict(self, img):
        if isinstance(img, Image.Image):
            if img.size != image_size:
                img = img.resize(image_size)
        else:
            img = Image.load_img(img, target_size=image_size)
        input_tensor = preprocess(img)
        x = input_tensor.unsqueeze(0)
        # check if tf is lazily loading by running second image
        if torch.cuda.is_available():
            input_batch = x.to(mode)
        with torch.no_grad():
            predictions = model(input_batch)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        # Show top categories per image
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        prediction = self.categories[top1_catid]
        return prediction

    
    def warmup(self, iterations = 100):
        imarray = np.random.rand(*image_size, 3) * 255
        for i in range(iterations):
            warmup_image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
            _ = self.predict(warmup_image)
        print("Warmup Complete.")
