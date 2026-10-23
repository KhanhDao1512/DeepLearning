import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        image_path = Path(image_path)
        label_path = Path(label_path)

        with image_path.open("rb") as file:
            magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid MNIST image file: {image_path}")
            images = np.frombuffer(file.read(), dtype=np.uint8)
            images = images.reshape(count, rows, cols).copy()

        with label_path.open("rb") as file:
            magic, count_labels = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid MNIST label file: {label_path}")
            labels = np.frombuffer(file.read(), dtype=np.uint8).copy()

        if len(images) != count_labels:
            raise ValueError("MNIST image count does not match label count")

        self.images = torch.from_numpy(images).float().div_(255.0).unsqueeze(1)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
