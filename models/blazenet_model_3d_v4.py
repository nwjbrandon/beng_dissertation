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

        self.A_0 = Parameter(torch.eye(21).float(), requires_grad=True)
        self.flat0 = nn.Flatten()
        self.gconv0 = GraphConv(86016, 2)
        self.gpool0 = GraphPool(21, 15)

        self.A_1 = Parameter(torch.eye(21).float(), requires_grad=True)
        self.flat1 = nn.Flatten()
        self.gconv1 = GraphConv(32768, 1024)
        self.gpool1 = GraphPool(21, 15)

        self.A_2 = Parameter(torch.eye(15).float(), requires_grad=True)
        self.gconv2 = GraphConv(1026, 512)
        self.gpool2 = GraphPool(15, 7)

        self.A_3 = Parameter(torch.eye(7).float(), requires_grad=True)
        self.gconv3 = GraphConv(512, 256)
        self.gpool3 = GraphPool(7, 4)

        self.A_4 = Parameter(torch.eye(4).float(), requires_grad=True)
        self.gconv4 = GraphConv(256, 128)
        self.gpool4 = GraphPool(4, 2)

        self.A_5 = Parameter(torch.eye(2).float(), requires_grad=True)
        self.gconv5 = GraphConv(128, 64)
        self.gpool5 = GraphPool(2, 1)

        self.fc1 = nn.Linear(64, 32)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(32, 64)
        self.act2 = nn.Tanh()

        self.A_6 = Parameter(torch.eye(2).float(), requires_grad=True)
        self.gpool6 = GraphUnpool(1, 2)
        self.gconv6 = GraphConv(192, 32)

        self.A_7 = Parameter(torch.eye(4).float(), requires_grad=True)
        self.gpool7 = GraphUnpool(2, 4)
        self.gconv7 = GraphConv(288, 16)

        self.A_8 = Parameter(torch.eye(7).float(), requires_grad=True)
        self.gpool8 = GraphUnpool(4, 7)
        self.gconv8 = GraphConv(528, 8)

        self.A_9 = Parameter(torch.eye(15).float(), requires_grad=True)
        self.gpool9 = GraphUnpool(7, 15)
        self.gconv9 = GraphConv(1032, 4)

        self.A_10 = Parameter(torch.eye(21).float(), requires_grad=True)
        self.gpool10 = GraphUnpool(15, 21)
        self.gconv10 = GraphConv(4, 3)

    def forward(self, heatmaps, out2, out3, out4, out5):
        x0 = self.flat0(heatmaps).unsqueeze(1).repeat(1, 21, 1)
        x0 = self.gconv0(x0, self.A_0)
        x0 = self.gpool0(x0)

        x1 = self.flat1(out5).unsqueeze(1).repeat(1, 21, 1)
        x1 = self.gconv1(x1, self.A_1)
        x1 = self.gpool1(x1)

        x01 = torch.cat([x0, x1], dim=2)
        x2 = self.gconv2(x01, self.A_2)
        x2 = self.gpool2(x2)

        x3 = self.gconv3(x2, self.A_3)
        x3 = self.gpool3(x3)

        x4 = self.gconv4(x3, self.A_4)
        x4 = self.gpool4(x4)

        x5 = self.gconv5(x4, self.A_5)
        x5 = self.gpool5(x5)

        feat = self.act1(self.fc1(x5))
        feat = self.act2(self.fc2(feat))

        x6 = self.gpool6(feat)
        x46 = torch.cat([x4, x6], dim=2)
        x6 = self.gconv6(x46, self.A_6)

        x7 = self.gpool7(x6)
        x37 = torch.cat([x3, x7], dim=2)
        x7 = self.gconv7(x37, self.A_7)

        x8 = self.gpool8(x7)
        x28 = torch.cat([x2, x8], dim=2)
        x8 = self.gconv8(x28, self.A_8)

        x9 = self.gpool9(x8)
        x19 = torch.cat([x1, x9], dim=2)
        x9 = self.gconv9(x19, self.A_9)

        x10 = self.gpool10(x9)
        x10 = self.gconv10(x10, self.A_10)
        return x10


class Pose3dModel(nn.Module):
    def __init__(self, config):
        super(Pose3dModel, self).__init__()
        self.pose_2d = Pose2dModel(config)
        self.pose_3d = Regressor3d(config)

    def forward(self, x):
        heatmaps, out2, out3, out4, out5 = self.pose_2d(x)
        kpt_3d = self.pose_3d(heatmaps, out2, out3, out4, out5)
        return heatmaps, kpt_3d
