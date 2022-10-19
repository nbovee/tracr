from .data_wrapper import data_wrapper
from PIL import Image
import glob

class test_data_loader(data_wrapper):

    def __init__(self) -> None:
        super().__init__()
        self.image_list = []
        self.load_data("test/*");

    def has_next(self):
        return bool(len(self.image_list))

    def next(self):
        val = self.image_list.pop(-1)
        for i in range(1,22): # hardcoded split layers for AlexNet - no full processing yet
            yield [ val, i ]

    def load_data(self, path):
        max_images = 2
        for image in glob.iglob(path):
            print(image)
            self.image_list.append(Image.Image.open(image))
            if len(self.image_list) >= max_images:
                break