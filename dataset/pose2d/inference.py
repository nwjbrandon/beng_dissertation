import math

import cv2
import numpy as np
import torch

from pose2d.modules.keypoints import extract_keypoints, group_keypoints
from pose2d.modules.load_state import load_state
from pose2d.modules.pose import Pose, track_poses
from pose2d.modules.with_mobilenet import PoseEstimationWithMobileNet


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(
        img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=pad_value
    )
    return padded_img, pad


def infer_fast(
    net,
    img,
    net_input_height_size,
    stride,
    upsample_ratio,
    cpu,
    pad_value=(0, 0, 0),
    img_mean=np.array([128, 128, 128], np.float32),
    img_scale=np.float32(1 / 256),
):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(
        heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC,
    )

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(
        pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC,
    )

    return heatmaps, pafs, scale, pad


class Pose2D:
    def __init__(self, path_to_model, cpu):
        self.path_to_model = path_to_model
        self.cpu = cpu
        self.net = self.load_model()

        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.height_size = 256
        self.track = 0
        self.smooth = 1
        self.previous_poses = []

    def load_model(self):
        net = PoseEstimationWithMobileNet()
        net = net.eval()
        if not self.cpu:
            net = net.cuda()
        checkpoint = torch.load(self.path_to_model, map_location="cpu")
        load_state(net, checkpoint)
        return net

    def inference(self, img):
        heatmaps, pafs, scale, pad = infer_fast(
            self.net, img, self.height_size, self.stride, self.upsample_ratio, self.cpu
        )

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]
            ) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]
            ) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if self.track:
            track_poses(self.previous_poses, current_poses, smooth=self.smooth)
            self.previous_poses = current_poses

        if current_poses == []:
            return None
        else:
            keypoints = current_poses[0].keypoints  # TODO: change to person largest in the image
        return self.remap_keypoints(keypoints)

    def remap_full_body_keypoints(self, keypoints):
        hip = (keypoints[8] + keypoints[11]) / 2
        spine = (hip + keypoints[1]) / 2
        keypoints1 = keypoints[(8, 9, 10, 11, 12, 13), :]
        keypoints2 = keypoints[(1, 0, 5, 6, 7, 2, 3, 4), :]
        return np.concatenate([[hip], keypoints1, [spine], keypoints2])

    def remap_half_body_keypoints(self, keypoints):
        hip = (keypoints[8] + keypoints[11]) / 2
        spine = (hip + keypoints[1]) / 2
        keypoints = keypoints[(1, 5, 6, 7, 2, 3, 4), :]
        return np.concatenate([[spine], keypoints])

    def remap_keypoints(self, keypoints):
        return self.remap_half_body_keypoints(keypoints)
