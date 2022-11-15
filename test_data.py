from data_wrapper import data_wrapper
from PIL import Image
import glob

class test_data_loader(data_wrapper):

    def __init__(self):
        self.image_list = []
        self.load_data("test/*.JPEG")
    

    def has_next(self):
        return bool(len(self.image_list))

    def next(self):
        if self.has_next() is None:
            return None
        else:
            [val, filename] = self.image_list.pop()
            print(f"Testing \"{filename}\": # left={len(self.image_list)}")
            for i in range(1,19): # hardcoded split layers for AlexNet - no full processing yet -1 would be full server, 20 would be full client            
                yield [ val, i, filename ]

    def load_data(self, path):
        max_images = 1000
        self.image_list.clear()
        # print(f"len iglob {len(list(glob.iglob(path)))}")
        for image in glob.iglob(path):
            print(image)
            self.image_list.append([ Image.open(image), image ])
            if len(self.image_list) >= max_images:
                break