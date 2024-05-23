import torch
from torch import nn

momentum_BN = 0.01

class _VGG_Block_BN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=momentum_BN),
            nn.ReLU(inplace=True),	
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=momentum_BN),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

    def forward(self, x):
        return self.block(x)

class VGG1_BN(nn.Module):
	def __init__(self):
		super().__init__()
		self.vgg_block_1 = _VGG_Block_BN(3,32)
		self.classifier = nn.Sequential(
			nn.Linear(32 * 16 * 16, 128),
			nn.BatchNorm1d(128, momentum=momentum_BN),
			nn.ReLU(inplace=True),
			nn.Linear(128, 10),
		)

	def forward(self, x):
		x = self.vgg_block_1(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x