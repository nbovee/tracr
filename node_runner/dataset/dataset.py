class Dataset:
    def has_next(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def normalize(self):
        self.preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
        
    def parse_output(self, predictions):
        """Take the final output tensor of the wrapped model and map it to its appropriate human readable results."""
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        # Show top categories per image
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        prediction = self.categories[top1_catid]
        return prediction