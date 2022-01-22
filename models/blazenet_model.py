import torch
from torch import nn


class Conv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
    ):
        super(Conv, self).__init__()
        self._conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self._act1 = nn.ReLU()

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
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=1,
    ):
        super(DownConv, self).__init__()
        self._conv1 = ConvBn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._conv2 = ConvBn(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
        )

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        return x


class UpConvCat(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    ):
        super(UpConvCat, self).__init__()
        self._up = nn.Upsample(scale_factor=2, mode="nearest")
        self._conv1 = ConvBn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._conv2 = ConvBn(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )

    def forward(self, x1, x2):
        x1 = self._up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self._conv1(x)
        x = self._conv2(x)
        return x


class BridgeConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    ):
        super(BridgeConv, self).__init__()
        self._conv1 = ConvBn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._conv2 = ConvBn(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        return x


class UpConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    ):
        super(UpConv, self).__init__()
        self._up = nn.Upsample(scale_factor=2, mode="nearest")
        self._conv1 = ConvBn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._conv2 = ConvBn(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
        )

    def forward(self, x):
        x = self._up(x)
        x = self._conv1(x)
        x = self._conv2(x)
        return x


class OutConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
    ):
        super(OutConv, self).__init__()
        self._conv1 = Conv(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._conv2 = Conv(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
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
        self.conv6 = BridgeConv(in_channels=192, out_channels=32)
        self.conv7 = UpConvCat(in_channels=160, out_channels=32)
        self.conv8 = UpConvCat(in_channels=96, out_channels=32)
        self.conv9 = UpConvCat(in_channels=64, out_channels=32)
        self.conv10 = UpConv(in_channels=32, out_channels=32)
        self.conv11 = OutConv(in_channels=32, out_channels=out_channels)

    def forward(self, out2, out3, out4, out5):
        out6 = self.conv6(out5)
        out7 = self.conv7(out6, out4)
        out8 = self.conv8(out7, out3)
        out9 = self.conv9(out8, out2)
        out10 = self.conv10(out9)
        heatmaps = self.conv11(out10)
        return heatmaps


# class PoseRegressor(nn.Module):
#     def __init__(self, out_channels):
#         super(PoseRegressor, self).__init__()
#         self.out_channels = out_channels
#         self.conv12 = DownConv(in_channels=64, out_channels=32)
#         self.conv13 = DownConv(in_channels=96, out_channels=64)
#         self.conv14 = DownConv(in_channels=192, out_channels=128)
#         self.conv15 = DownConv(in_channels=320, out_channels=192)
#         self.conv16 = DownConv(in_channels=192, out_channels=192)
#         self.conv17 = OutConv(in_channels=192, out_channels=out_channels, kernel_size=2)
#         self.sigmoid1 = nn.Sigmoid()

#     def forward(self, out9, out2, out3, out4, out5):
#         B, _, _, _ = out9.shape
#         out11 = self.conv12(torch.cat([out9, out2], dim=1))
#         out12 = self.conv13(torch.cat([out11, out3], dim=1))
#         out13 = self.conv14(torch.cat([out12, out4], dim=1))
#         out14 = self.conv15(torch.cat([out13, out5], dim=1))
#         out15 = self.conv16(out14)
#         out16 = self.conv17(out15)
#         kpts3d = out16.reshape(B, self.out_channels, -1)
#         kpts3d = self.sigmoid1(kpts3d)
#         return kpts3d


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
