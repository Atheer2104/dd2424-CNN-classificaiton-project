import torch
from torch import nn
from blocks.ResNetBlock import ResNetBlock
from blocks.SE_ResNetBlock import SE_ResNetBlock
from blocks.Patchify_Embed_Block import Patchify_EmbedBlock


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes, use_ViT):
		super().__init__()
		self.use_ViT = use_ViT
		# taking performing patchify and embed with patch size 2, this will reduce the image size to 16x16
		self.patchify_embed_block = Patchify_EmbedBlock(3, 3, 2)
		
		self.current_filter_size = 16

		# inital operation to be applied
		self.initial = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
		)

		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		# using AdaptiveAvgPool2d instead of AvgPool2d since here we just have to define the target dimension we want in this case we want the
		# to have width = 1, height = 1 so when we later flatten we get 64 as the size since we have so many channels/depth
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.classifier = nn.Linear(64, num_classes)
		# explicity setting the HE initalization for the classifier here instad of previous where did this the below but for all linear
		# layers, now we don't want this since we using this class for SE ResNet and there the linear layers don't have any bias
		torch.nn.init.kaiming_normal_(
			self.classifier.weight, mode="fan_in", nonlinearity="relu"
		)
		torch.nn.init.zeros_(self.classifier.bias)

	def _make_layer(self, block, out_channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.current_filter_size, out_channels, stride))
			self.current_filter_size = out_channels * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		if self.use_ViT:
			x = self.patchify_embed_block(x)
		
		x = self.initial(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)

		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x


def resnet20(num_classes):
	return ResNet(ResNetBlock, [3, 3, 3], num_classes, False)


def resnet32(num_classes):
	return ResNet(ResNetBlock, [5, 5, 5], num_classes, False)


def resnet44(num_classes):
	return ResNet(ResNetBlock, [7, 7, 7], num_classes, False)


def resnet56(num_classes):
	return ResNet(ResNetBlock, [9, 9, 9], num_classes, False)


def resnet110(num_classes):
	return ResNet(ResNetBlock, [18, 18, 18], num_classes, False)


def SE_resnet20(num_classes):
	return ResNet(SE_ResNetBlock, [3, 3, 3], num_classes, False)


def SE_resnet32(num_classes):
	return ResNet(SE_ResNetBlock, [5, 5, 5], num_classes, False)


def SE_resnet44(num_classes):
	return ResNet(SE_ResNetBlock, [7, 7, 7], num_classes, False)


def SE_resnet56(num_classes):
	return ResNet(SE_ResNetBlock, [9, 9, 9], num_classes, False)


def SE_resnet110(num_classes):
	return ResNet(ResNetBlock, [18, 18, 18], num_classes, False)

def ViT_resnet20(num_classes):
	return ResNet(ResNetBlock, [3, 3, 3], num_classes, True)


def ViT_resnet32(num_classes):
	return ResNet(ResNetBlock, [5, 5, 5], num_classes, True)


def ViT_resnet44(num_classes):
	return ResNet(ResNetBlock, [7, 7, 7], num_classes, True)


def ViT_resnet56(num_classes):
	return ResNet(ResNetBlock, [9, 9, 9], num_classes, True)

def ViT_resnet110(num_classes):
	return ResNet(ResNetBlock, [18, 18, 18], num_classes, True)