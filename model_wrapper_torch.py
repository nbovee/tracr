import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models

selected = "MobileNet"
if selected == "AlexNet":
    selected ="alexnet"
elif selected == "SqueezeNet":
    selected= "squeezenet1_1"
elif selected == "MobileNet":
    selected ="mobilenet_v3s"
image_size = (224, 224)
# model = torch.hub.load('pytorch/vision:v0.10.0', selected, pretrained=True)
model = models.vgg16(pretrained=True)
mode = 'cuda'

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Model:
    def __init__(self,) -> None:
        global model 
        model.eval()
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
