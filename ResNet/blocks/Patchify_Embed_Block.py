from torch import nn


class Patchify_EmbedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()

        self.p = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        nn.init.zeros_(self.p.bias)

    def forward(self, x):
        return self.p(x)

