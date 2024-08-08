import pathlib
import logging
from typing import Optional, Callable
import torchvision.transforms as transforms
from PIL import Image
from .dataset import BaseDataset

logger = logging.getLogger("tracr_logger")


class ImagenetDataset(BaseDataset):
    """
    A dataset class for loading and processing ImageNet data.

    This class extends the BaseDataset class and provides functionality to load
    ImageNet images and labels, apply transformations, and retrieve items.

    Attributes:
        CLASS_TEXTFILE (pathlib.Path): Path to the file containing ImageNet class labels.
        IMG_DIRECTORY (pathlib.Path): Path to the directory containing ImageNet images.
        img_labels (list): List of image labels.
        img_dir (pathlib.Path): Directory containing the images.
        transform (Optional[Callable]): Optional transform to be applied on images.
        target_transform (Optional[Callable]): Optional transform to be applied on labels.
        img_map (dict): Mapping of image labels to file paths.

    Sample Data Source:
        https://github.com/EliSchwartz/imagenet-sample-images
    """

    CLASS_TEXTFILE: pathlib.Path
    IMG_DIRECTORY: pathlib.Path

    def __init__(
        self,
        max_iter: int = -1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the ImagenetDataset.

        Args:
            max_iter (int): Maximum number of images to load. -1 for all images.
            transform (Optional[Callable]): Optional transform to be applied on images.
            target_transform (Optional[Callable]): Optional transform to be applied on labels.
        """
        self.CLASS_TEXTFILE = (
            self.DATA_SOURCE_DIRECTORY / "imagenet" / "imagenet_classes.txt"
        )
        self.IMG_DIRECTORY = self.DATA_SOURCE_DIRECTORY / "imagenet" / "sample_images"

        self.img_labels = self._load_labels(max_iter)
        self.img_dir = self.IMG_DIRECTORY
        self.transform = transform
        self.target_transform = target_transform
        self.img_map = self._create_image_map()

    def _load_labels(self, max_iter: int) -> list:
        """
        Load and process image labels from the class text file.

        Args:
            max_iter (int): Maximum number of labels to load.

        Returns:
            list: Processed list of image labels.
        """
        with open(self.CLASS_TEXTFILE) as file:
            img_labels = file.read().split("\n")
        if max_iter > 0:
            img_labels = img_labels[:max_iter]
        return [label.replace(" ", "_") for label in img_labels]

    def _create_image_map(self) -> dict:
        """
        Create a mapping of image labels to their file paths.

        Returns:
            dict: Mapping of image labels to file paths.
        """
        img_map = {}
        for i in range(len(self.img_labels) - 1, -1, -1):
            img_name = self.img_labels[i]
            try:
                img_map[img_name] = next(self.img_dir.glob(f"*{img_name}*"))
            except StopIteration:
                logger.warning(
                    f"Couldn't find image with name {img_name} in directory. Skipping."
                )
                self.img_labels.pop(i)
        return img_map

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get an item (image and label) from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and its label.
        """
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


# Dataset instances for different configurations
imagenet999_rgb = ImagenetDataset()
imagenet10_rgb = ImagenetDataset(max_iter=10)
imagenet999_tr = ImagenetDataset(transform=transforms.Compose([transforms.ToTensor()]))
imagenet10_tr = ImagenetDataset(
    transform=transforms.Compose([transforms.ToTensor()]), max_iter=10
)
imagenet2_tr = ImagenetDataset(
    transform=transforms.Compose([transforms.ToTensor()]), max_iter=2
)

if __name__ == "__main__":
    print(
        f"Output size: {imagenet2_tr[0][0].element_size() * imagenet2_tr[0][0].nelement()}"
    )
