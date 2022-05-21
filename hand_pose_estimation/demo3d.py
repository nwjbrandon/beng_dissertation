import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from datasets.freihand.data_utils import draw_2d_skeleton, draw_3d_skeleton_on_ax
from datasets.freihand.dataset_utils import heatmaps_to_coordinates
from models.blazenet_model_3d_v3_1_6 import Pose3dModel

config = {
    "model": {
        "n_keypoints": 21,
        "model_file": "model_v3_1_6.pth",
        "device": "cpu",
        "raw_image_size": 256,
        "model_img_size": 128,
    }
}
model = Pose3dModel(config)
model = model.to(config["model"]["device"])
model.load_state_dict(
    torch.load(config["model"]["model_file"], map_location=torch.device(config["model"]["device"]),)
)

image_transform = transforms.Compose(
    [transforms.Resize(config["model"]["raw_image_size"]), transforms.ToTensor(),]
)

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


# define a video capture object
vid = cv2.VideoCapture(0)
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection="3d")

with torch.no_grad():
    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:480, :480]
        frame = Image.fromarray(frame).convert("RGB")
        im_width, im_height = frame.size

        image_inp = image_transform(frame)
        image_inp = image_inp.unsqueeze(0)
        image_inp = image_inp.float().to(config["model"]["device"])
        pred = model(image_inp)
        heatmaps_pred = pred[0].cpu().numpy()[0]
        kpt_3d_pred = pred[1].cpu().numpy()[0]

        p1 = kpt_3d_pred[5]
        p2 = kpt_3d_pred[17]
        p3 = kpt_3d_pred[0]
        x, y, z = compute_hand_orientation(p1, p2, p3)
        p4 = kpt_3d_pred[9]
        p4x = p4 + x * 0.1
        p4y = p4 + y * 0.1
        p4z = p4 + z * 0.1

        kpt_2d_pred = heatmaps_to_coordinates(heatmaps_pred, config["model"]["model_img_size"])
        kpt_2d_pred[:, 0] = kpt_2d_pred[:, 0] * im_width
        kpt_2d_pred[:, 1] = kpt_2d_pred[:, 1] * im_height
        skeleton_overlay = draw_2d_skeleton(frame, kpt_2d_pred)
        ax.clear()
        draw_3d_skeleton_on_ax(kpt_3d_pred, ax)
        ax.plot(
            [p4[0], p4x[0]], [p4[1], p4x[1]], [p4[2], p4x[2]], zdir="z", c="red",
        )
        ax.plot(
            [p4[0], p4y[0]], [p4[1], p4y[1]], [p4[2], p4y[2]], zdir="z", c="green",
        )
        ax.plot(
            [p4[0], p4z[0]], [p4[1], p4z[1]], [p4[2], p4z[2]], zdir="z", c="blue",
        )

        skeleton_overlay = cv2.cvtColor(skeleton_overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", skeleton_overlay)

        plt.draw()
        plt.pause(0.001)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

vid.release()
cv2.destroyAllWindows()
