from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder


class ImageFolderWithSharedLabels(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.dataset = ImageFolder(str(root), transform=transform)
        missing = set(self.dataset.classes) - set(class_to_idx)
        if missing:
            raise ValueError(f"Evaluation split contains unknown classes: {sorted(missing)}")
        self.class_to_idx = class_to_idx
        self.samples = [
            (path, class_to_idx[self.dataset.classes[old_label]])
            for path, old_label in self.dataset.samples
        ]
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.dataset.loader(path)
        if self.dataset.transform is not None:
            image = self.dataset.transform(image)
        return image, label


def make_vinafood_datasets(root, train_transform, eval_transform, seed=42):
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not val_dir.exists():
        val_dir = root / "validation"
    test_dir = root / "test"

    if train_dir.is_dir():
        train_dataset = ImageFolder(str(train_dir), transform=train_transform)
        class_names = train_dataset.classes
        eval_dir = val_dir if val_dir.is_dir() else test_dir
        if eval_dir.is_dir():
            eval_dataset = ImageFolderWithSharedLabels(
                eval_dir,
                train_dataset.class_to_idx,
                transform=eval_transform,
            )
        else:
            full_eval = ImageFolder(str(train_dir), transform=eval_transform)
            train_size = int(0.8 * len(train_dataset))
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(train_dataset), generator=generator).tolist()
            train_dataset = Subset(train_dataset, indices[:train_size])
            eval_dataset = Subset(full_eval, indices[train_size:])
        return train_dataset, eval_dataset, class_names

    if not root.is_dir():
        raise FileNotFoundError(f"VinaFood21 directory was not found: {root}")

    full_train = ImageFolder(str(root), transform=train_transform)
    full_eval = ImageFolder(str(root), transform=eval_transform)
    train_size = int(0.8 * len(full_train))
    eval_size = len(full_train) - train_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_subset = Subset(full_train, indices[:train_size])
    eval_subset = Subset(full_eval, indices[train_size:])
    return train_subset, eval_subset, full_train.classes
