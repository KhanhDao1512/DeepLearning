import torch
from torch import nn

from .inception_block import ConvRelu, InceptionBlock


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=21, first_conv_padding=3, maxpool_ceil_mode=True):
        super().__init__()
        self.conv1 = ConvRelu(3, 64, kernel_size=7, stride=2, padding=first_conv_padding)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=maxpool_ceil_mode)
        self.conv2 = ConvRelu(64, 64, kernel_size=1)
        self.conv3 = ConvRelu(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=maxpool_ceil_mode)

        self.inception3a = InceptionBlock(192, (64, 96, 128, 16, 32, 32))
        self.inception3b = InceptionBlock(256, (128, 128, 192, 32, 96, 64))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=maxpool_ceil_mode)

        self.inception4a = InceptionBlock(480, (192, 96, 208, 16, 48, 64))
        self.inception4b = InceptionBlock(512, (160, 112, 224, 24, 64, 64))
        self.inception4c = InceptionBlock(512, (128, 128, 256, 24, 64, 64))
        self.inception4d = InceptionBlock(512, (112, 144, 288, 32, 64, 64))
        self.inception4e = InceptionBlock(528, (256, 160, 320, 32, 128, 128))
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=maxpool_ceil_mode)

        self.inception5a = InceptionBlock(832, (256, 160, 320, 32, 128, 128))
        self.inception5b = InceptionBlock(832, (384, 192, 384, 48, 128, 128))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x).flatten(start_dim=1)
        return self.classifier(self.dropout(x))
