import torch
from torch import nn

from models.blazenet_model_v5 import ConvBn, Pose2dModel
from models.non_local import NLBlockND
from models.resnet import BasicBlock
from models.semgcn import SemGraphConv, _GraphConv, _ResGraphConv, adj_mx_from_edges

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
HAND_ADJ = adj_mx_from_edges(N_JOINTS, HAND_EDGES, sparse=False)


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
        self.conv19 = DownConv(704, 210)

        # gcn
        self.gconv21 = _GraphConv(HAND_ADJ, 160, 128, p_dropout=0.0)
        self.gconv22 = _ResGraphConv(HAND_ADJ, 128, 128, 64, p_dropout=0.0)
        self.gconv23 = NLBlockND(
            in_channels=N_JOINTS, mode="concatenate", dimension=1, bn_layer=True
        )
        self.gconv24 = _ResGraphConv(HAND_ADJ, 128, 128, 64, p_dropout=0.0)
        self.gconv25 = NLBlockND(
            in_channels=N_JOINTS, mode="concatenate", dimension=1, bn_layer=True
        )
        self.gconv26 = _ResGraphConv(HAND_ADJ, 128, 128, 64, p_dropout=0.0)
        self.gconv27 = NLBlockND(
            in_channels=N_JOINTS, mode="concatenate", dimension=1, bn_layer=True
        )
        self.gconv28 = _ResGraphConv(HAND_ADJ, 128, 128, 64, p_dropout=0.0)
        self.gconv29 = NLBlockND(
            in_channels=N_JOINTS, mode="concatenate", dimension=1, bn_layer=True
        )
        self.gconvout = SemGraphConv(128, 3, HAND_ADJ)

        self.flat = nn.Flatten()
        self.fc = nn.Linear(3072, self.out_channels * 3)

    def forward(self, out2, out3, out4, out5):
        B = out2.shape[0]

        out13 = self.conv13(out2)
        out14 = self.conv14(torch.cat([out13, out3], dim=1))
        out15 = self.conv15(out14)
        out16 = self.conv16(torch.cat([out15, out4], dim=1))
        out17 = self.conv17(out16)
        out18 = self.conv18(torch.cat([out17, out5], dim=1))
        out19 = self.conv19(out18)

        out20 = out19.view(B, self.out_channels, -1)

        out21 = self.gconv21(out20)
        out22 = self.gconv22(out21)
        out23 = self.gconv23(out22)
        out24 = self.gconv24(out23)
        out25 = self.gconv25(out24)
        out26 = self.gconv26(out25)
        out27 = self.gconv27(out26)
        out28 = self.gconv28(out27)
        out29 = self.gconv29(out28)

        kpt_3d = self.gconvout(out29)
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
