import torch
from torch import nn
from torch.nn.modules.dropout import Dropout

# the default momentum in pytorch is 0.1, but in keras it's 0.99, therefore chosing this number
momentum_BN = 0.99

# this is the implementation of a single VGG block with dropout and BN
class _VGG_Block_BN_Dropout(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, momentum=momentum_BN),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, momentum=momentum_BN),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.block(x)


class VGG3_BN_Dropput(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_block_1 = _VGG_Block_BN_Dropout(3, 32, 0.2)
        self.vgg_block_2 = _VGG_Block_BN_Dropout(32, 64, 0.3)
        self.vgg_block_3 = _VGG_Block_BN_Dropout(64, 128, 0.4)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, momentum=momentum_BN),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.vgg_block_1(x)
        x = self.vgg_block_2(x)
        x = self.vgg_block_3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
