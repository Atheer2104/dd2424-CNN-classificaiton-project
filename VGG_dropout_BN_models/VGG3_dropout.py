from VGG1_dropout import _VGG_Block_dropout
import torch
from torch import nn

class VGG3_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_block_1 = _VGG_Block_dropout(3, 32, 0.2)
        self.vgg_block_2 = _VGG_Block_dropout(32, 64, 0.2)
        self.vgg_block_3 = _VGG_Block_dropout(64, 128, 0.2)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.vgg_block_1(x)
        x = self.vgg_block_2(x)
        x = self.vgg_block_3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
