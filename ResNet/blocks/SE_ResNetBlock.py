from torch import nn

class SE_block(nn.Module):
	def __init__(self, num_channels, reduction_ratio=16):
		super().__init__()

		self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
		self.se_operations = nn.Sequential(
			# setting bias to False as we have down conv in ResNet
			nn.Linear(
				num_channels, num_channels // reduction_ratio, bias=False
			),
		 	nn.ReLU(inplace=True),
			nn.Linear(
				num_channels // reduction_ratio, num_channels, bias=False
			),
			nn.Sigmoid()
		)
		
	
	def forward(self, x):
		batch_size, channels, _, _ =  x.size()

		y = self.avg_pool(x).view(batch_size, channels)
		y = self.se_operations(y).view(batch_size, channels, 1, 1)

		# performing final scaling dot product of what we have computed so far and the original feature_map values
		# here we are expanding the output to match size of x which is out feature maps U in the papper, this is also important
  		# the size will differ when going from one layer to the next and here we don't have to deal with it
		return x * y.expand_as(x)


class SE_ResNetBlock(nn.Module):
	expansion = 1

	def __init__(
		self, in_channels, out_channels, stride=1, option="B", reducation_ratio=16
	):
		super().__init__()
		self.conv_1 = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=3,
			stride=stride,
			padding=1,
			bias=False,
		)
		self.bn_1 = nn.BatchNorm2d(out_channels)
		self.conv_2 = nn.Conv2d(
			out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
		)
		self.bn_2 = nn.BatchNorm2d(out_channels)

		self.se_layer = SE_block(out_channels, reducation_ratio)

		self.relu = nn.ReLU(inplace=True)

		# this is how define the identity just using the sequl
		self.residual = nn.Sequential()

		# we also have to think about when we pass the residual to another layer that has a smaller input size
		# there are two options in how this is done in the ResNet paper

		# this happens in two cases when we define that stride is not 1 i.e start a new layer or when the in_channel is not the same
		# as the out_channel indicating that we are not in the same layer
		if stride != 1 or in_channels != out_channels:
			if option == "A":
				# self.residual =
				raise ValueError("Not implemented")
			elif option == "B":
				# here we use a 1x1 conv to match the dimension but we also have to add a batch normalization since this is a conv layer
				# note how we reduce the output depth here
				self.residual = nn.Sequential(
					nn.Conv2d(
						in_channels,
						out_channels * self.expansion,
						kernel_size=1,
						stride=stride,
						bias=False,
					),
					nn.BatchNorm2d(self.expansion * out_channels),
				)
			else:
				raise ValueError("uncorrect option choosen")

	def forward(self, x):
		identity = x

		x = self.conv_1(x)
		x = self.bn_1(x)
		x = self.relu(x)

		x = self.conv_2(x)
		x = self.bn_2(x)
		
		x = self.se_layer(x)

		# add residual connection this happen before second activation as described in the paper
		x = x + self.residual(identity)

		x = self.relu(x)
		return x
