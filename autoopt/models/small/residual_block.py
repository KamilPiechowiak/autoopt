from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, c, downsample: bool = False) -> None:
        super(ResidualBlock, self).__init__()
        layers = [
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU()
        ]
        if downsample:
            layers += [
                nn.Conv2d(c, c * 2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(c * 2),
            ]
            self.downsampling_conv = nn.Conv2d(c, c * 2, kernel_size=3, padding=1, stride=2)
        else:
            layers += [
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
            ]
        self.block = nn.Sequential(*layers)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsample:
            return self.relu(self.block(x) + self.downsampling_conv(x))
        return self.relu(self.block(x) + x)
