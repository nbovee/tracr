import pathlib
import logging
from PIL import Image

from src.tracr.experiment_design.datasets.dataset import BaseDataset
import torchvision.transforms as transforms


logger = logging.getLogger("tracr_logger")


class ImagenetDataset(BaseDataset):
    """
    Here's an example of a 'user-defined' dataset class extending the included BaseDataset
    class. Below the class definitions are some instances of this class that can later be
    referenced by participating nodes for experiments.
    """

    CLASS_TEXTFILE: pathlib.Path
    IMG_DIRECTORY: pathlib.Path

    def __init__(self, max_iter: int = -1, transform=None, target_transform=None):
        self.CLASS_TEXTFILE = (
            self.DATA_SOURCE_DIRECTORY / "imagenet" / "imagenet_classes.txt"
        )
        self.IMG_DIRECTORY = self.DATA_SOURCE_DIRECTORY / "imagenet" / "sample_images"

        with open(self.CLASS_TEXTFILE) as file:
            img_labels = file.read().split("\n")
        if len(img_labels) > max_iter:
            img_labels = img_labels[:max_iter]
        img_labels = [label.replace(" ", "_") for label in img_labels]

        self.img_labels = img_labels
        self.img_dir = self.IMG_DIRECTORY
        self.transform = transform
        self.target_transform = target_transform
        self.img_map = {}

        for i in range(len(self.img_labels), 0, -1):
            i -= 1
            img_name = self.img_labels[i]
            try:
                self.img_map[img_name] = next(self.img_dir.glob(f"*{img_name}*"))
            except StopIteration:
                logger.warning(
                    f"Couldn't find image with name {img_name} in directory. Skipping."
                )
                self.img_labels.pop(i)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]
        img_fp = self.img_map[label]
        image = Image.open(img_fp).convert("RGB")
        image = image.resize((224, 224))

        if self.transform:
            image = self.transform(image)
            image = image.unsqueeze(0)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


# Here are the dataset instances the observer can offer to the participants

# All 999 images as PIL objects converted to RGB
imagenet999_rgb = ImagenetDataset()
# Same, but just the first 10
imagenet10_rgb = ImagenetDataset(max_iter=10)

# This gives all 999 images as torch Tensors
imagenet999_tr = ImagenetDataset(transform=transforms.Compose([transforms.ToTensor()]))
# And this gives the same, but only the first 10
imagenet10_tr = ImagenetDataset(
    transform=transforms.Compose([transforms.ToTensor()]), max_iter=10
)

# And here's the sad little dataset I've been using for tests
imagenet2_tr = ImagenetDataset(
    transform=transforms.Compose([transforms.ToTensor()]), max_iter=2
)


if __name__ == "__main__":
    print(f"Output size: {imagenet2_tr[0][0].element_size() * imagenet2_tr[0][0].nelement()}")  # type: ignore
