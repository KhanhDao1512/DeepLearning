from torch import nn


class ResNet50HF(nn.Module):
    def __init__(self, num_classes, pretrained_model="microsoft/resnet-50", freeze_backbone=False):
        super().__init__()
        try:
            from transformers import ResNetForImageClassification
        except ImportError as error:
            raise ImportError("Install transformers to use the HuggingFace ResNet50 assignment") from error

        self.model = ResNetForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        if freeze_backbone:
            for parameter in self.model.resnet.parameters():
                parameter.requires_grad = False
            for parameter in self.model.classifier.parameters():
                parameter.requires_grad = True

    def forward(self, images):
        return self.model(pixel_values=images).logits
