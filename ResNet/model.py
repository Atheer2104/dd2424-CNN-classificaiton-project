import torch
from torch import nn


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, option="B"):
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

        # add residual connection this happen before second activation as described in the paper
        x = x + self.residual(identity)

        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
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

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.current_filter_size, out_channels, stride))
            self.current_filter_size = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        print(x.shape)
        x = self.avgpool(x)

        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.classifier(x)
        return x


def resnet20():
    return ResNet(ResNetBlock, [3, 3, 3])


def resnet32():
    return ResNet(ResNetBlock, [5, 5, 5])


def resnet44():
    return ResNet(ResNetBlock, [7, 7, 7])


def resnet56():
    return ResNet(ResNetBlock, [9, 9, 9])
