import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.blazenet_model_v5 import ConvBn, Pose2dModel
from models.resnet import BasicBlock

N_JOINTS = 21


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


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.act = nn.Tanh()

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        X = self.act(X)
        return X


class Regressor3d(nn.Module):
    def __init__(self, config):
        super(Regressor3d, self).__init__()
        self.out_channels = config["model"]["n_keypoints"]
        self.conv11 = DownConv(21, 21)
        self.conv12 = BasicBlock(85, 85)
        self.conv13 = DownConv(85, 32)
        self.conv14 = BasicBlock(160, 160)
        self.conv15 = DownConv(160, 64)
        self.conv16 = BasicBlock(320, 320)
        self.conv17 = DownConv(320, 192)
        self.conv18 = BasicBlock(704, 704)
        self.conv19 = DownConv(704, 21)

        self.flat = nn.Flatten(start_dim=2)

        # gcn
        self.A_0 = Parameter(torch.eye(N_JOINTS, dtype=torch.float), requires_grad=True)
        self.gconv0 = GraphConv(16, 3)

    def forward(self, heatmaps, out2, out3, out4, out5):
        out11 = self.conv11(heatmaps)
        out12 = self.conv12(torch.cat([out11, out2], dim=1))
        out13 = self.conv13(out12)
        out14 = self.conv14(torch.cat([out13, out3], dim=1))
        out15 = self.conv15(out14)
        out16 = self.conv16(torch.cat([out15, out4], dim=1))
        out17 = self.conv17(out16)
        out18 = self.conv18(torch.cat([out17, out5], dim=1))
        out19 = self.conv19(out18)

        feat = self.flat(out19)

        kpt_3d = self.gconv0(feat, self.A_0)
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
