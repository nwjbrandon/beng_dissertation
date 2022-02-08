import torch
import torch.nn as nn

from models.resnet import resnet18
from models.semgcn import HAND_ADJ, _GraphConv, _ResGraphConv


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


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.conv6 = UpConv(in_channels=512, out_channels=32)
        self.conv7 = UpConv(in_channels=288, out_channels=32)
        self.conv8 = UpConv(in_channels=160, out_channels=32)
        self.conv9 = UpConv(in_channels=96, out_channels=32, is_upsample=False)
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
        self.resnet = resnet18()
        self.decoder = Decoder(self.out_channels)

    def forward(self, x):
        out2, out3, out4, out5 = self.resnet(x)
        heatmaps = self.decoder(out2, out3, out4, out5)
        return heatmaps, out2, out3, out4, out5


class Regressor3d(nn.Module):
    def __init__(self, config):
        super(Regressor3d, self).__init__()
        self.out_channels = config["model"]["n_keypoints"]

        # out4
        self.flat3 = nn.Flatten()
        self.gconv3 = _GraphConv(HAND_ADJ, 65536, 256, p_dropout=0.0)

        self.resgconv3 = _ResGraphConv(HAND_ADJ, 256, 256, 128, p_dropout=0.0)

        # out5
        self.flat4 = nn.Flatten()
        self.gconv4 = _GraphConv(HAND_ADJ, 32768, 256, p_dropout=0.0)

        self.resgconv4 = _ResGraphConv(HAND_ADJ, 256, 256, 128, p_dropout=0.0)

        # backbone
        self.gconv5 = _GraphConv(HAND_ADJ, 512, 256, p_dropout=0.0)

        self.resgconv5 = _ResGraphConv(HAND_ADJ, 256, 256, 128, p_dropout=0.0)

        self.resgconv6 = _ResGraphConv(HAND_ADJ, 256, 256, 128, p_dropout=0.0)

        self.gconvout = _GraphConv(HAND_ADJ, 256, 3, p_dropout=0.0)

    def forward(self, heatmaps, out2, out3, out4, out5):
        x3 = self.flat3(out4).unsqueeze(1).repeat(1, 21, 1)
        x3 = self.gconv3(x3)
        x3 = self.resgconv3(x3)

        x4 = self.flat4(out5).unsqueeze(1).repeat(1, 21, 1)
        x4 = self.gconv4(x4)
        x4 = self.resgconv4(x4)

        # feat = torch.cat([x0, x1, x2, x3, x4], dim=2)
        feat = torch.cat([x3, x4], dim=2)

        feat = self.gconv5(feat)

        feat = self.resgconv5(feat)
        feat = self.resgconv6(feat)

        out = self.gconvout(feat)

        return out


class Pose3dModel(nn.Module):
    def __init__(self, config):
        super(Pose3dModel, self).__init__()
        self.pose_2d = Pose2dModel(config)
        self.pose_3d = Regressor3d(config)

    def forward(self, x):
        heatmaps, out2, out3, out4, out5 = self.pose_2d(x)
        kpt_3d = self.pose_3d(heatmaps, out2, out3, out4, out5)
        return heatmaps, kpt_3d
