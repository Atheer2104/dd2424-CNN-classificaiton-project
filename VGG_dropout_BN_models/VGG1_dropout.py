import torch
from torch import nn

class _VGG_Block_dropout(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.block(x)

class VGG1_dropout(nn.Module):
	def __init__(self):
		super().__init__()
		self.vgg_block_1 = _VGG_Block_dropout(3,32, 0.2)
		self.classifier = nn.Sequential(
			nn.Linear(32 * 16 * 16, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			nn.Linear(128, 10),
		)

	def forward(self, x):
		x = self.vgg_block_1(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x