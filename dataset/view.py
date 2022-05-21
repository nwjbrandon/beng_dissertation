import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

JOINTS_RIGHT = [5, 6, 7]
JOINTS_LEFT = [2, 3, 4]
PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    1,
    5,
    6,
]
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
RADIUS = 2


def draw_2d_skeleton(img, keypoints):
    for idx, keypoint in enumerate(keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        if idx in JOINTS_LEFT:
            cv2.circle(img, (x, y), RADIUS, RED, -1)
            cv2.putText(
                img, str(idx), (x - 40, y - 30), FONT, FONT_SCALE, RED, THICKNESS,
            )
        elif idx in JOINTS_RIGHT:
            cv2.circle(img, (x, y), RADIUS, BLUE, -1)
            cv2.putText(
                img, str(idx), (x - 40, y - 30), FONT, FONT_SCALE, BLUE, THICKNESS,
            )
        else:
            cv2.circle(img, (x, y), RADIUS, GREEN, -1)
            cv2.putText(
                img, str(idx), (x - 40, y - 30), FONT, FONT_SCALE, GREEN, THICKNESS,
            )

    for j, parent in enumerate(PARENTS):
        if parent == -1:
            continue
        if j in JOINTS_RIGHT:
            col = BLUE
        elif j in JOINTS_LEFT:
            col = RED
        else:
            col = GREEN
        pt1 = int(keypoints[j][0]), int(keypoints[j][1])
        pt2 = int(keypoints[parent][0]), int(keypoints[parent][1])
        img = cv2.line(img, pt1, pt2, col, thickness=2)

    return img


def read_data(img_f, pose2d_f, pose3d_f):
    img = Image.open(img_f).convert("RGB")

    width, height = img.size  # Get dimensions
    print(width, height)

    new_height, new_width = 480, 480
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))

    pose2d = np.load(pose2d_f)
    pose3d = np.load(pose3d_f)
    return img, pose2d, pose3d


def visualize_data(img_f, pose2d_f, pose3d_f):
    img, pose2d, pose3d = read_data(img_f, pose2d_f, pose3d_f)

    pose3d = pose3d - pose3d[0]

    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = draw_2d_skeleton(img, pose2d)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    plt.imshow(img)
    plt.title("Image With GT 2D joints")

    plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.axes(projection="3d")
    draw_3d_skeleton_on_ax(pose3d, ax)
    ax.set_title("GT 3D joints")

    plt.show()


def draw_3d_skeleton_on_ax(keypoints, ax):
    radius = 1.7
    azim = 90
    elev = 90
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([-radius / 2, radius / 2])

    ax.set_xlabel("x-axis", labelpad=10)
    ax.set_ylabel("y-axis", labelpad=10)
    ax.set_zlabel("z-axis", labelpad=10)

    for i, kpt in enumerate(keypoints):
        x, y, z = kpt
        ax.text(x, y, z, f"{i}")
        if i in JOINTS_LEFT:  # left
            ax.scatter(x, y, z, c="red")
        elif i in JOINTS_RIGHT:  # right:
            ax.scatter(x, y, z, c="blue")
        else:
            ax.scatter(x, y, z, c="green")

    for j, parent in enumerate(PARENTS):
        if parent == -1:
            continue
        if j in JOINTS_RIGHT:
            col = "blue"
        elif j in JOINTS_LEFT:
            col = "red"
        else:
            col = "green"
        ax.plot(
            [keypoints[j, 0], keypoints[parent, 0]],
            [keypoints[j, 1], keypoints[parent, 1]],
            [keypoints[j, 2], keypoints[parent, 2]],
            zdir="z",
            c=col,
        )


if __name__ == "__main__":
    data_index = os.getenv("data_index")

    img_path = f"data/frame_{data_index}.jpg"
    pose2d_path = f"data/pose2d_{data_index}.npy"
    pose3d_path = f"data/pose3d_{data_index}.npy"
    visualize_data(img_path, pose2d_path, pose3d_path)
