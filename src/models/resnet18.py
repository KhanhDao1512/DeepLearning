import torch
from torch import nn

from .residual_block import ResidualBlock


class ResNet18(nn.Module):
    def __init__(self, num_classes=21, pool_kernel_size=3, pool_stride=2, pool_padding=0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._stage(64, 64)
        self.stage2 = self._stage(64, 128)
        self.stage3 = self._stage(128, 256)
        self.stage4 = self._stage(256, 512)
        self.between_stages_pool = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=pool_padding,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.stage1(x)
        x = self.between_stages_pool(x)
        x = self.stage2(x)
        x = self.between_stages_pool(x)
        x = self.stage3(x)
        x = self.between_stages_pool(x)
        x = self.stage4(x)
        x = self.avgpool(x).flatten(start_dim=1)
        return self.fc(x)
