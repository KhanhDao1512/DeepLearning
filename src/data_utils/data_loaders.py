from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from .mnist_dataset import MNISTDataset
from .vinafood_dataset import make_vinafood_datasets


def build_loaders(config, model_name=None):
    dataset_config = config["dataset"]
    training_config = config["training"]
    batch_size = int(training_config["batch_size"])
    workers = int(training_config.get("num_workers", 0))
    seed = int(training_config.get("seed", 42))

    if dataset_config["name"].lower() == "mnist":
        train_dataset = MNISTDataset(
            dataset_config["train_images"],
            dataset_config["train_labels"],
        )
        eval_dataset = MNISTDataset(
            dataset_config["test_images"],
            dataset_config["test_labels"],
        )
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers),
            DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=workers),
            int(dataset_config.get("num_classes", 10)),
            [str(index) for index in range(int(dataset_config.get("num_classes", 10)))],
        )

    image_size = int(dataset_config.get("image_size", 224))
    if model_name == "resnet50_hf":
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(config["model"]["pretrained_model"])

        class ProcessorTransform:
            def __call__(self, image):
                return processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]

        train_transform = ProcessorTransform()
        eval_transform = ProcessorTransform()
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        eval_transform = train_transform

    train_dataset, eval_dataset, classes = make_vinafood_datasets(
        dataset_config["root"], train_transform, eval_transform, seed=seed
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers),
        DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=workers),
        len(classes),
        classes,
    )
