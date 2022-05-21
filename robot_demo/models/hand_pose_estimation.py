import sys

sys.path.insert(0, "/home/nwjbrandon/fyp/mycobot_teleop/models/hand3d")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.blazenet_model_3d import Pose3dModel
from utils.data_utils import draw_2d_skeleton, draw_3d_skeleton_on_ax
from utils.dataset_utils import heatmaps_to_coordinates

from camera_stream import LoadStreams


# https://itectec.com/matlab/matlab-how-to-calculate-roll-pitch-and-yaw-from-xyz-coordinates-of-3-planar-points/
def compute_hand_orientation(p1, p2, p3):
    p1, p2, p3 = p1[:3], p2[:3], p3[:3]
    x = (p1 + p2) / 2 - p3
    v1, v2 = p2 - p1, p3 - p1
    z = np.cross(v1, v2)
    z = z / np.linalg.norm(z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


class HandPoseEstimation:
    def __init__(
        self, model_file, n_keypoints=21, device="cpu", raw_image_size=256, model_img_size=128
    ):
        self.model_file = model_file
        self.n_keypoints = n_keypoints
        self.device = device
        self.raw_image_size = raw_image_size
        self.model_img_size = model_img_size

        self.config = {
            "model": {
                "n_keypoints": n_keypoints,
                "model_file": model_file,
                "device": device,
                "raw_image_size": raw_image_size,
                "model_img_size": model_img_size,
            }
        }
        self.model = Pose3dModel(self.config)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device(device),))

        self.image_transform = transforms.Compose(
            [transforms.Resize(raw_image_size), transforms.ToTensor(),]
        )
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = plt.axes(projection="3d")

    def compute_orientation(self, kpt_3d_pred):
        p1 = kpt_3d_pred[5]
        p2 = kpt_3d_pred[17]
        p3 = kpt_3d_pred[0]
        x, y, z = compute_hand_orientation(p1, p2, p3)
        return x, y, z

    def compute_grasp(self, kpt_3d_pred):
        p1 = kpt_3d_pred[4]
        p2 = kpt_3d_pred[8]
        dist = np.linalg.norm(p1-p2)
        return dist

    def visualize(self, img, heatmaps_pred, kpt_3d_pred, orientation_vector):
        im_width, im_height = img.size
        kpt_2d_pred = heatmaps_to_coordinates(heatmaps_pred, self.model_img_size)
        kpt_2d_pred[:, 0] = kpt_2d_pred[:, 0] * im_width
        kpt_2d_pred[:, 1] = kpt_2d_pred[:, 1] * im_height

        x, y, z = orientation_vector
        p4 = kpt_3d_pred[9]
        p4x = p4 + x * 0.1
        p4y = p4 + y * 0.1
        p4z = p4 + z * 0.1

        # draw 2d
        skeleton_overlay = draw_2d_skeleton(img, kpt_2d_pred)
        skeleton_overlay = cv2.cvtColor(skeleton_overlay, cv2.COLOR_RGB2BGR)

        # draw 3d
        self.ax.clear()
        draw_3d_skeleton_on_ax(kpt_3d_pred, self.ax)

        # draw hand frame
        self.ax.plot(
            [p4[0], p4x[0]], [p4[1], p4x[1]], [p4[2], p4x[2]], zdir="z", c="red",
        )
        self.ax.plot(
            [p4[0], p4y[0]], [p4[1], p4y[1]], [p4[2], p4y[2]], zdir="z", c="green",
        )
        self.ax.plot(
            [p4[0], p4z[0]], [p4[1], p4z[1]], [p4[2], p4z[2]], zdir="z", c="blue",
        )

        cv2.imshow("frame", skeleton_overlay)
        plt.draw()
        plt.pause(0.001)


    def detect(self, img, is_vis=True):
        img = Image.fromarray(img).convert("RGB")

        image_inp = self.image_transform(img)
        image_inp = image_inp.unsqueeze(0)
        image_inp = image_inp.float().to(self.device)
        pred = self.model(image_inp)
        heatmaps_pred = pred[0].cpu().numpy()[0]
        kpt_3d_pred = pred[1].cpu().numpy()[0]

        orientation_vector = self.compute_orientation(kpt_3d_pred)
        grasp_dist = self.compute_grasp(kpt_3d_pred)
        if is_vis:
            self.visualize(img, heatmaps_pred, kpt_3d_pred, orientation_vector)

        return orientation_vector, grasp_dist, kpt_3d_pred

    def __call__(self, img, is_vis=True):
        return self.detect(img, is_vis)


if __name__ == "__main__":
    source = "0"
    imgsz = 640
    stride = 32
    auto = True
    model_file = "model_v3_1_6.pth"
    hand_estimator = HandPoseEstimation(model_file=model_file)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=auto)

    with torch.no_grad():
        for path, im, im0s, vid_cap, s in dataset:
            inp_img = cv2.cvtColor(im0s[0], cv2.COLOR_BGR2RGB)
            inp_img = inp_img[:480, :480]
            outputs = hand_estimator(inp_img)
            if cv2.waitKey(1) == ord("q"):
                break

    vid_cap.release()
    cv2.destroyAllWindows()
