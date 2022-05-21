import torch
from torch import nn


class Conv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
    ):
        super(Conv, self).__init__()
        self._conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._act1 = nn.ReLU()

    def forward(self, x):
        x = self._conv1(x)
        x = self._act1(x)
        return x


class ConvSig(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
    ):
        super(ConvSig, self).__init__()
        self._conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._act1 = nn.Sigmoid()

    def forward(self, x):
        x = self._conv1(x)
        x = self._act1(x)
        return x


class ConvBn(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
    ):
        super(ConvBn, self).__init__()
        self._conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self._batch_norm1 = nn.BatchNorm2d(out_channels)
        self._act1 = nn.ReLU()

    def forward(self, x):
        x = self._conv1(x)
        x = self._batch_norm1(x)
        x = self._act1(x)
        return x


class DownConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    ):
        super(DownConv, self).__init__()
        self._conv1 = ConvBn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._maxpool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self._conv1(x)
        x = self._maxpool2(x)
        return x


class UpConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_upsample=True
    ):
        super(UpConv, self).__init__()
        self.is_upsample = is_upsample
        self._conv1 = ConvBn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        if self.is_upsample:
            self._up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self._conv1(x)
        if self.is_upsample:
            x = self._up2(x)
        return x


class OutConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    ):
        super(OutConv, self).__init__()
        self._conv1 = ConvSig(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )

    def forward(self, x):
        x = self._conv1(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = DownConv(in_channels=3, out_channels=16)
        self.conv2 = DownConv(in_channels=16, out_channels=32)
        self.conv3 = DownConv(in_channels=32, out_channels=64)
        self.conv4 = DownConv(in_channels=64, out_channels=128)
        self.conv5 = DownConv(in_channels=128, out_channels=192)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        return out2, out3, out4, out5


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.conv6 = UpConv(in_channels=192, out_channels=32)
        self.conv7 = UpConv(in_channels=160, out_channels=32)
        self.conv8 = UpConv(in_channels=96, out_channels=32)
        self.conv9 = UpConv(in_channels=64, out_channels=32, is_upsample=False)
        self.conv10 = OutConv(in_channels=32, out_channels=out_channels)

    def forward(self, out2, out3, out4, out5):
        out6 = self.conv6(out5)
        out7 = self.conv7(torch.cat([out6, out4], dim=1))
        out8 = self.conv8(torch.cat([out7, out3], dim=1))
        out9 = self.conv9(torch.cat([out8, out2], dim=1))
        heatmaps = self.conv10(out9)
        return heatmaps


class Pose2dModel(nn.Module):
    def __init__(self, config):
        super(Pose2dModel, self).__init__()
        self.out_channels = config["model"]["n_keypoints"]
        self.encoder = Encoder()
        self.decoder = Decoder(self.out_channels)

    def forward(self, x):
        out2, out3, out4, out5 = self.encoder(x)
        heatmaps = self.decoder(out2, out3, out4, out5)
        return heatmaps, out2, out3, out4, out5
