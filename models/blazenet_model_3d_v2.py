import torch
from torch import nn

from models.resnet import resnet18
from models.semgcn import _GraphConv, adj_mx_from_edges

N_JOINTS = 21

HAND_JOINTS_PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    0,
    13,
    14,
    15,
    0,
    17,
    18,
    19,
]
HAND_EDGES = list(filter(lambda x: x[1] >= 0, zip(list(range(0, N_JOINTS)), HAND_JOINTS_PARENTS)))
ADDITIONAL_EDGES = [
    (1, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (1, 17),
    (2, 6),
    (6, 10),
    (10, 14),
    (14, 18),
    (2, 18),
    (3, 7),
    (7, 11),
    (11, 15),
    (15, 19),
    (3, 19),
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (4, 20),
]
HAND_EDGES.extend(ADDITIONAL_EDGES)
HAND_ADJ = adj_mx_from_edges(N_JOINTS, HAND_EDGES, sparse=False)


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

        # cnn
        self.conv11 = DownConv(85, 32)
        self.conv12 = DownConv(160, 64)
        self.conv13 = DownConv(320, 192)
        self.conv14 = DownConv(704, 21)
        self.flat = nn.Flatten(start_dim=2)

        # gcn
        self.gconvout = _GraphConv(HAND_ADJ, 16, 3, p_dropout=0.0)

    def forward(self, heatmaps, out2, out3, out4, out5):
        out11 = self.conv11(torch.cat([heatmaps, out2], dim=1))
        out12 = self.conv12(torch.cat([out11, out3], dim=1))
        out13 = self.conv13(torch.cat([out12, out4], dim=1))
        out14 = self.conv14(torch.cat([out13, out5], dim=1))
        feat = self.flat(out14)
        kpt_3d = self.gconvout(feat)
        return kpt_3d


class Pose3dModel(nn.Module):
    def __init__(self, config):
        super(Pose3dModel, self).__init__()
        self.pose_2d = Pose2dModel(config)
        self.pose_3d = Regressor3d(config)

    def forward(self, x):
        heatmaps, out2, out3, out4, out5 = self.pose_2d(x)
        kpt_3d = self.pose_3d(heatmaps, out2, out3, out4, out5)
        return heatmaps, kpt_3d
