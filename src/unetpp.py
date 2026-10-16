"""U-Net++ (Zhou et al., 2018).

Plain U-Net connects encoder level i straight to decoder level i. Those two features
are many layers apart in the network, so they are semantically dissimilar and the
decoder has to reconcile them.

U-Net++ fills the gap with a grid of intermediate nodes. Node X[i][j] takes every
earlier node on its own level plus an upsampled node from the level below:

    X[i][j] = conv( [ X[i][0], ..., X[i][j-1], up(X[i+1][j-1]) ] )

so the skip path becomes a short dense chain instead of one long jump. The final
prediction reads X[0][4].
"""

import torch
import torch.nn as nn

from .attention_unet import DoubleConv


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base=32, deep_supervision=False):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]
        self.depth = len(c)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)

        self.conv = nn.ModuleDict()
        self.up = nn.ModuleDict()
        for i in range(self.depth):
            for j in range(self.depth - i):
                if j == 0:
                    in_ch = in_channels if i == 0 else c[i - 1]
                    self.conv[f"{i}_{j}"] = DoubleConv(in_ch, c[i])
                else:
                    self.up[f"{i}_{j}"] = nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2)
                    self.conv[f"{i}_{j}"] = DoubleConv((j + 1) * c[i], c[i])

        heads = self.depth - 1 if deep_supervision else 1
        self.heads = nn.ModuleList([nn.Conv2d(c[0], num_classes, 1) for _ in range(heads)])

    def forward(self, x):
        nodes = {}
        for j in range(self.depth):
            for i in range(self.depth - j):
                if j == 0:
                    source = x if i == 0 else self.pool(nodes[(i - 1, 0)])
                    nodes[(i, j)] = self.conv["0_0" if i == 0 else f"{i}_0"](source)
                else:
                    skips = [nodes[(i, k)] for k in range(j)]
                    skips.append(self.up[f"{i}_{j}"](nodes[(i + 1, j - 1)]))
                    nodes[(i, j)] = self.conv[f"{i}_{j}"](torch.cat(skips, dim=1))

        if self.deep_supervision:
            outputs = [head(nodes[(0, j)]) for j, head in enumerate(self.heads, start=1)]
            return torch.stack(outputs).mean(0)
        return self.heads[0](nodes[(0, self.depth - 1)])
