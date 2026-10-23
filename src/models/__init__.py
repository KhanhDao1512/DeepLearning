from .lenet import LeNet
from .googlenet import GoogLeNet
from .resnet18 import ResNet18
from .resnet50_hf import ResNet50HF


def create_model(config, num_classes):
    name = config["model"]["name"].lower()
    if name == "lenet":
        return LeNet(num_classes=num_classes)
    if name == "googlenet":
        model_config = config["model"]
        return GoogLeNet(
            num_classes=num_classes,
            first_conv_padding=int(model_config.get("first_conv_padding", 3)),
            maxpool_ceil_mode=bool(model_config.get("maxpool_ceil_mode", True)),
        )
    if name == "resnet18":
        pool_config = config["model"].get("between_stages_pool", {})
        return ResNet18(
            num_classes=num_classes,
            pool_kernel_size=int(pool_config.get("kernel_size", 3)),
            pool_stride=int(pool_config.get("stride", 2)),
            pool_padding=int(pool_config.get("padding", 0)),
        )
    if name == "resnet50_hf":
        model_config = config["model"]
        return ResNet50HF(
            num_classes=num_classes,
            pretrained_model=model_config["pretrained_model"],
            freeze_backbone=bool(model_config.get("freeze_backbone", False)),
        )
    raise ValueError(f"Unknown model: {name}")


__all__ = ["LeNet", "GoogLeNet", "ResNet18", "ResNet50HF", "create_model"]
