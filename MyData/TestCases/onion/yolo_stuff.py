def prepare_yolo_dataset(self):
    data_loader = CustomYOLODataLoader(self.config_details, self.dataset_type)
    data_loader.prepare_dataset()
    print("Dataset prepared for YOLO training.")


def train_model(self, model):
    self.prepare_yolo_dataset()
    if self.dataset_type.strip().lower() == "balanced":
        yaml_data_path = self.config_details["File_path"]["balanced"][
            "yaml_label_information_path"
        ]
    elif self.dataset_type.strip().lower() == "unbalanced":
        yaml_data_path = self.config_details["File_path"]["unbalanced"][
            "yaml_label_information_path"
        ]
    model.train(data=yaml_data_path, imgsz=512, batch=24, epochs=150)
