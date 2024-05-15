from VGG1_BN import _VGG_Block_BN
import torch
from torch import nn

momentum_BN = 0.01

class VGG3_BN(nn.Module):
	def __init__(self):
		super().__init__()
		self.vgg_block_1 = _VGG_Block_BN(3,32)
		self.vgg_block_2 = _VGG_Block_BN(32, 64)
		self.vgg_block_3 = _VGG_Block_BN(64, 128)
		self.classifier = nn.Sequential(
			nn.Linear(128 * 4 * 4, 128),
			nn.BatchNorm1d(128, momentum=momentum_BN),
			nn.ReLU(inplace=True),
			nn.Linear(128, 10),
		)

	def forward(self, x):
		x = self.vgg_block_1(x)
		x = self.vgg_block_2(x)
		x = self.vgg_block_3(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x