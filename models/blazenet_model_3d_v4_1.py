import torch
from torch import nn

from models.blazenet_model_v5 import ConvBn, Pose2dModel
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
        self.conv12 = DownConv(64, 32)
        self.conv13 = DownConv(160, 64)
        self.conv14 = DownConv(320, 192)
        self.conv15 = DownConv(704, 210)

        # gcn
        self.gconv17 = _GraphConv(HAND_ADJ, 160, 128, p_dropout=0.0)
        self.gconv18 = _ResGraphConv(HAND_ADJ, 128, 128, 64, p_dropout=0.0)
        self.gconv19 = _ResGraphConv(HAND_ADJ, 128, 128, 64, p_dropout=0.0)
        self.gconv20 = _ResGraphConv(HAND_ADJ, 128, 128, 64, p_dropout=0.0)
        self.gconv21 = _ResGraphConv(HAND_ADJ, 128, 128, 64, p_dropout=0.0)
        self.gconvout = SemGraphConv(128, 3, HAND_ADJ)

    def forward(self, out2, out3, out4, out5):
        B = out2.shape[0]

        out12 = self.conv12(out2)
        out13 = self.conv13(torch.cat([out12, out3], dim=1))
        out14 = self.conv14(torch.cat([out13, out4], dim=1))
        out15 = self.conv15(torch.cat([out14, out5], dim=1))

        out16 = out15.view(B, self.out_channels, -1)

        out17 = self.gconv17(out16)
        out18 = self.gconv18(out17)
        out19 = self.gconv19(out18)
        out20 = self.gconv20(out19)
        out21 = self.gconv21(out20)
        kpt_3d = self.gconvout(out21)
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
