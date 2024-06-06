import os
import shutil
import xml.etree.ElementTree as ET


class CustomYOLODataLoader:
    def __init__(self, file_path_info, dataset_type):
        self.path_info = file_path_info
        self.file_info = file_path_info["Data_splitting_file_information"]
        self.dataset_type = dataset_type

    def prepare_dataset(self):
        # Process XML annotations
        self._process_xml_annotations(
            self.path_info["File_path"][self.dataset_type][
                "source_annotation_folder_location"
            ],
            self.path_info["File_path"][self.dataset_type][
                "modified_annotation_folder_location"
            ],
        )

        # Arrange files
        self._arrange_files(
            self.path_info["File_path"][self.dataset_type][
                "images_split_files_location"
            ],
            self.path_info["File_path"][self.dataset_type][
                "actual_images_files_location"
            ],
            self.path_info["File_path"][self.dataset_type][
                "modified_annotation_folder_location"
            ],
            self.path_info["File_path"][self.dataset_type][
                "actual_images_files_split_location"
            ],
        )

    def _process_xml_annotations(self, source_path, destination_path):
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        files = os.listdir(source_path)
        classification_label = {"with_weeds": 0, "without_weeds": 1}

        for file in files:
            file_location = os.path.join(source_path, file)
            tree = ET.parse(file_location)
            root = tree.getroot()
            image_size = root.find("size")
            image_width = int(image_size.find("width").text)
            image_height = int(image_size.find("height").text)

            output_file = os.path.join(destination_path, file.replace(".xml", ".txt"))
            with open(output_file, "w") as f:
                for obj in root.findall("object"):
                    cls_name = "_".join(
                        obj.find("name").text.strip().lower().split(" ")
                    )
                    if cls_name not in classification_label:
                        raise ValueError(f"Unknown class {cls_name}")

                    bbox = obj.find("bndbox")
                    xmin = int(bbox.find("xmin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymin = int(bbox.find("ymin").text)
                    ymax = int(bbox.find("ymax").text)

                    x_center = (xmin + xmax) / 2 / image_width
                    width = (xmax - xmin) / image_width
                    y_center = (ymin + ymax) / 2 / image_height
                    height = (ymax - ymin) / image_height

                    f.write(
                        f"{classification_label[cls_name]} {x_center} {y_center} {width} {height}\n"
                    )

    def _arrange_files(
        self, files_location, images_location, annotations_location, target_location
    ):
        for file_key in self.file_info:
            target_file = os.path.join(files_location, self.file_info[file_key])
            destination = os.path.join(target_location, file_key)

            if os.path.exists(destination):
                shutil.rmtree(destination)
            os.makedirs(destination)

            with open(target_file) as f:
                file_names = [name.strip() for name in f.readlines()]

            for name in file_names:
                image_path = os.path.join(images_location, name + ".jpg")
                annotation_path = os.path.join(annotations_location, name + ".txt")
                shutil.copy2(image_path, destination)
                shutil.copy2(annotation_path, destination)
