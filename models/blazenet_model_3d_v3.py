import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from models.resnet import resnet18


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


class GraphPool(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(GraphPool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)

    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class GraphUnpool(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(GraphUnpool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)

    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class Regressor3d(nn.Module):
    def __init__(self, config):
        super(Regressor3d, self).__init__()
        self.out_channels = config["model"]["n_keypoints"]
        self.conv11 = DownConv(85, 32)
        self.conv12 = DownConv(160, 64)
        self.conv13 = DownConv(320, 192)
        self.conv14 = DownConv(704, 21)
        self.flat = nn.Flatten(start_dim=2)

        self.A_0 = Parameter(torch.eye(21, dtype=torch.float), requires_grad=True)
        self.gconv0 = GraphConv(16, 3)

    def forward(self, heatmaps, out2, out3, out4, out5):
        out11 = self.conv11(torch.cat([heatmaps, out2], dim=1))
        out12 = self.conv12(torch.cat([out11, out3], dim=1))
        out13 = self.conv13(torch.cat([out12, out4], dim=1))
        out14 = self.conv14(torch.cat([out13, out5], dim=1))
        feat = self.flat(out14)
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