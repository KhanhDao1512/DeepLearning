import torch
from torch import nn


class ConvRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self[0].bias)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        one, reduce_three, three, reduce_five, five, pool_proj = channels
        self.branch1 = ConvRelu(in_channels, one, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvRelu(in_channels, reduce_three, kernel_size=1),
            ConvRelu(reduce_three, three, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvRelu(in_channels, reduce_five, kernel_size=1),
            ConvRelu(reduce_five, five, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ConvRelu(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x))
        return torch.cat(branches, dim=1)
