
class Model:
    def __init__(self, selected = "MobileNet") -> None:
        import numpy as np
        from PIL import Image
        self.image_size = (224, 224)
        from tensorflow.keras.preprocessing import image
        if selected == "VGG":
            from tensorflow.keras.applications.vgg16 import VGG16 as model
            from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
            # image_size = (224, 224)
        if selected == "EfficientNet":
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S as model
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
            # image_size = (384, 384)
        if selected == "MobileNet":
            from tensorflow.keras.applications.mobilenet import MobileNet as model
            from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
            # image_size = (224, 224)
        else:
            from tensorflow.keras.applications.resnet50 import ResNet50 as model
            from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
            # image_size = (224, 224)
        model = model(weights='imagenet')
        self.warmup()

    def predict(self, img):
        if isinstance(img, self.Image):
            if img.size != self.image_size:
                img = img.resize(self.image_size)
        else:
            img = self.image.load_img(img, target_size=self.image_size)
        x = self.image.img_to_array(img)
        x = self.np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        # check if tf is lazily loading by running second image
        predictions = self.model.predict(x)
        prediction = self.decode_predictions(predictions, top=3)
        return prediction

    
    def warmup(self, iterations = 5):
        imarray = self.np.random.rand(*self.image_size, 3) * 255
        warmup_image = self.Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        for i in range(iterations):
            _ = self.predict(warmup_image)
