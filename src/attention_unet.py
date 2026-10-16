"""U-Net and Attention U-Net.

Both use padded convolutions, so output size == input size. The lecturer's `unet.py`
uses the original unpadded design (572 in -> 388 out), which forces you to crop every
mask. Padding removes that bookkeeping and makes the two models directly comparable.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Oktay et al. 2018.

    The decoder feature `g` says *what* the network is currently looking for; the skip
    connection `x` carries high-resolution detail but also a lot of irrelevant
    background. The gate scores every spatial position from both and multiplies the
    skip by that score, so only the relevant regions get passed to the decoder.
    """

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.psi(self.relu(self.w_gate(gate) + self.w_skip(skip)))
        return skip * attention, attention


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.gate = (AttentionGate(out_channels, skip_channels, out_channels // 2)
                     if attention else None)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        attention = None
        if self.gate is not None:
            skip, attention = self.gate(x, skip)
        return self.conv(torch.cat([skip, x], dim=1)), attention


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base=64, attention=False):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]

        self.enc1 = DoubleConv(in_channels, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.bottleneck = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = UpBlock(c[4], c[3], c[3], attention)
        self.up3 = UpBlock(c[3], c[2], c[2], attention)
        self.up2 = UpBlock(c[2], c[1], c[1], attention)
        self.up1 = UpBlock(c[1], c[0], c[0], attention)
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x, return_attention=False):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        x = self.bottleneck(self.pool(s4))

        x, a4 = self.up4(x, s4)
        x, a3 = self.up3(x, s3)
        x, a2 = self.up2(x, s2)
        x, a1 = self.up1(x, s1)

        logits = self.head(x)
        return (logits, [a1, a2, a3, a4]) if return_attention else logits


def AttentionUNet(in_channels=1, num_classes=1, base=64):
    return UNet(in_channels, num_classes, base, attention=True)
