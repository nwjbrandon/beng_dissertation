import torch
from torch import nn

from models.blazenet_model_v5 import ConvBn, Pose2dModel
from models.resnet import BasicBlock


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


class Regressor3d(nn.Module):
    def __init__(self, config):
        super(Regressor3d, self).__init__()
        self.out_channels = config["model"]["n_keypoints"]
        self.conv13 = DownConv(64, 32)
        self.conv14 = BasicBlock(160, 160)
        self.conv15 = DownConv(160, 64)
        self.conv16 = BasicBlock(320, 320)
        self.conv17 = DownConv(320, 192)
        self.conv18 = BasicBlock(704, 704)
        self.conv19 = DownConv(704, 192)

        self.flat = nn.Flatten()
        self.fc = nn.Linear(3072, self.out_channels * 3)

    def forward(self, out2, out3, out4, out5):
        out13 = self.conv13(out2)
        out14 = self.conv14(torch.cat([out13, out3], dim=1))
        out15 = self.conv15(out14)
        out16 = self.conv16(torch.cat([out15, out4], dim=1))
        out17 = self.conv17(out16)
        out18 = self.conv18(torch.cat([out17, out5], dim=1))
        out19 = self.conv19(out18)

        x = self.flat(out19)
        kpt_3d = self.fc(x)

        # reshape
        B, _ = kpt_3d.shape
        kpt_3d = kpt_3d.view((B, self.out_channels, -1))
        return kpt_3d


class Pose3dModel(nn.Module):
    def __init__(self, config):
        super(Pose3dModel, self).__init__()
        self.pose_2d = Pose2dModel(config)
        self.pose_3d = Regressor3d(config)

    def forward(self, x):
        heatmaps, out2, out3, out4, out5 = self.pose_2d(x)
        kpt_3d = self.pose_3d(out2, out3, out4, out5)
        return heatmaps, kpt_3d
