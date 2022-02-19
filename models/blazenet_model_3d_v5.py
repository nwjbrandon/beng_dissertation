import torch
from torch import nn

from models.blazenet_model_v5 import ConvBn, Pose2dModel
from models.resnet import BasicBlock
from models.non_local import NLBlockND


class DownConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    ):
        super(DownConv, self).__init__()
        self._conv1 = ConvBn(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self._maxpool2 = nn.MaxPool2d(2, 2)
        self._conv2 = BasicBlock(out_channels, out_channels)
        self._conv3 = NLBlockND(in_channels=out_channels, mode="concatenate", dimension=2, bn_layer=True)

    def forward(self, x):
        x = self._conv1(x)
        x = self._maxpool2(x)
        x = self._conv2(x)
        x = self._conv3(x)
        return x


class Regressor3d(nn.Module):
    def __init__(self, config):
        super(Regressor3d, self).__init__()
        self.out_channels = config["model"]["n_keypoints"]
        self.conv11 = DownConv(21, 21)
        self.conv12 = DownConv(85, 32)
        self.conv13 = DownConv(160, 64)
        self.conv14 = DownConv(320, 192)
        self.conv15 = DownConv(704, 192)

        self.flat = nn.Flatten()
        self.fc = nn.Linear(3072, self.out_channels * 3)

    def forward(self, heatmaps, out2, out3, out4, out5):
        out11 = self.conv11(heatmaps)
        out12 = self.conv12(torch.cat([out11, out2], dim=1))
        out13 = self.conv13(torch.cat([out13, out3], dim=1))
        out14 = self.conv14(torch.cat([out15, out4], dim=1))
        out15 = self.conv15(torch.cat([out17, out5], dim=1))

        x = self.flat(out15)
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
        kpt_3d = self.pose_3d(heatmaps, out2, out3, out4, out5)
        return heatmaps, kpt_3d
