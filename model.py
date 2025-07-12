import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class EDSR_baseline(nn.Module):
    def __init__(self, num_blocks=4, channels=32, scale=4):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = self.upsample(x)
        return x
