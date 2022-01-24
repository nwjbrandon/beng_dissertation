import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_2d_skeleton(
    image,
    coords_hw,
    vis=None,
    color_fixed=None,
    linewidth=3,
    order="hw",
    img_order="rgb",
    draw_kp=True,
    kp_style=None,
):
    """ Inpaints a hand stick figure into a matplotlib figure. """
    if kp_style is None:
        kp_style = (5, 3)

    image = np.squeeze(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, 2)
    s = image.shape
    assert len(s) == 3, "This only works for single images."

    convert_to_uint8 = False
    if s[2] == 1:
        # grayscale case
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
        image = np.tile(image, [1, 1, 3])
        pass
    elif s[2] == 3:
        # RGB case
        if image.dtype == np.uint8:
            convert_to_uint8 = True
            image = image.astype("float32") / 255.0
        elif image.dtype == np.float32:
            # convert to gray image
            image = np.mean(image, axis=2)
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
            image = np.expand_dims(image, 2)
            image = np.tile(image, [1, 1, 3])
    else:
        assert 0, "Unknown image dimensions."

    if order == "uv":
        coords_hw = coords_hw[:, ::-1]

    colors = np.array(
        [
            [0.4, 0.4, 0.4],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.4, 0.4, 0.0],
            [0.6, 0.6, 0.0],
            [0.8, 0.8, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.4, 0.2],
            [0.0, 0.6, 0.3],
            [0.0, 0.8, 0.4],
            [0.0, 1.0, 0.5],
            [0.0, 0.2, 0.4],
            [0.0, 0.3, 0.6],
            [0.0, 0.4, 0.8],
            [0.0, 0.5, 1.0],
            [0.4, 0.0, 0.4],
            [0.6, 0.0, 0.6],
            [0.7, 0.0, 0.8],
            [1.0, 0.0, 1.0],
        ]
    )

    if img_order == "rgb":
        colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [
        ((0, 1), colors[1, :]),
        ((1, 2), colors[2, :]),
        ((2, 3), colors[3, :]),
        ((3, 4), colors[4, :]),
        ((0, 5), colors[5, :]),
        ((5, 6), colors[6, :]),
        ((6, 7), colors[7, :]),
        ((7, 8), colors[8, :]),
        ((0, 9), colors[9, :]),
        ((9, 10), colors[10, :]),
        ((10, 11), colors[11, :]),
        ((11, 12), colors[12, :]),
        ((0, 13), colors[13, :]),
        ((13, 14), colors[14, :]),
        ((14, 15), colors[15, :]),
        ((15, 16), colors[16, :]),
        ((0, 17), colors[17, :]),
        ((17, 18), colors[18, :]),
        ((18, 19), colors[19, :]),
        ((19, 20), colors[20, :]),
    ]

    color_map = {
        "k": np.array([0.0, 0.0, 0.0]),
        "w": np.array([1.0, 1.0, 1.0]),
        "b": np.array([0.0, 0.0, 1.0]),
        "g": np.array([0.0, 1.0, 0.0]),
        "r": np.array([1.0, 0.0, 0.0]),
        "m": np.array([1.0, 1.0, 0.0]),
        "c": np.array([0.0, 1.0, 1.0]),
    }

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :].astype(np.int32)
        coord2 = coords_hw[connection[1], :].astype(np.int32)

        if (coord1[0] < 1) or (coord1[0] >= s[0]) or (coord1[1] < 1) or (coord1[1] >= s[1]):
            continue
        if (coord2[0] < 1) or (coord2[0] >= s[0]) or (coord2[1] < 1) or (coord2[1] >= s[1]):
            continue

        if color_fixed is None:
            cv2.line(
                image,
                (coord1[1], coord1[0]),
                (coord2[1], coord2[0]),
                color,
                thickness=linewidth,
                lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
            )
        else:
            c = color_map.get(color_fixed, np.array([1.0, 1.0, 1.0]))
            cv2.line(image, (coord1[1], coord1[0]), (coord2[1], coord2[0]), c, thickness=linewidth)

    if draw_kp:
        coords_hw = coords_hw.astype(np.int32)
        for i in range(21):
            if vis[i]:
                image = cv2.circle(
                    image,
                    (coords_hw[i, 1], coords_hw[i, 0]),
                    radius=3,
                    color=colors[i, :],
                    thickness=-1,
                    lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
                )

    if convert_to_uint8:
        image = (image * 255).astype("uint8")

    return image


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def draw_3d_skeleton_on_ax(pose_cam_xyz, ax):
    """
    :param pose_cam_xyz: 21 x 3
    :param ax:
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    color_hand_joints = [
        [1.0, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 1.0, 0.0],  # thumb
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 1.0],
        [0.4, 0.4, 1.0],  # index
        [0.0, 0.4, 0.4],
        [0.0, 0.6, 0.6],
        [0.0, 0.8, 0.8],
        [0.0, 1.0, 1.0],  # middle
        [0.4, 0.4, 0.0],
        [0.6, 0.6, 0.0],
        [0.8, 0.8, 0.0],
        [1.0, 1.0, 0.0],  # ring
        [0.4, 0.0, 0.4],
        [0.6, 0.0, 0.6],
        [0.8, 0.0, 0.8],
        [1.0, 0.0, 1.0],
    ]  # little

    marker_sz = 15
    x_radius = [-0.2, 0.2]
    y_radius = [-0.2, 0.2]
    z_radius = [-0.2, 0.2]
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=-85, azim=-90)
    ax.set_xlim3d(x_radius)
    ax.set_zlim3d(z_radius)
    ax.set_ylim3d(y_radius)

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(
            pose_cam_xyz[joint_ind : joint_ind + 1, 0],
            pose_cam_xyz[joint_ind : joint_ind + 1, 1],
            pose_cam_xyz[joint_ind : joint_ind + 1, 2],
            ".",
            c=color_hand_joints[joint_ind],
            markersize=marker_sz,
        )
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(
                pose_cam_xyz[[0, joint_ind], 0],
                pose_cam_xyz[[0, joint_ind], 1],
                pose_cam_xyz[[0, joint_ind], 2],
                color=color_hand_joints[joint_ind],
            )
        else:
            ax.plot(
                pose_cam_xyz[[joint_ind - 1, joint_ind], 0],
                pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                pose_cam_xyz[[joint_ind - 1, joint_ind], 2],
                color=color_hand_joints[joint_ind],
            )


def read_data(data_dir, sid, fid, cid):
    # load keypoints
    kp_data_file = os.path.join(data_dir, f"xyz/{sid:04d}/{fid:08d}.json")
    with open(kp_data_file, "r") as f:
        global_pose3d_gt = np.array(json.load(f))

    # load calibration
    calib_file = os.path.join(data_dir, f"calib/{sid:04d}/{fid:08d}.json")
    with open(calib_file, "r") as f:
        cam_proj_mat = json.load(f)

    # convert world to camera coordinates
    M = np.array(cam_proj_mat["M"])[cid]
    local_pose3d_gt = np.matmul(global_pose3d_gt, M[:3, :3].T) + M[:3, 3][None]

    # Camera intrinsics
    K = np.array(cam_proj_mat["K"])[cid]

    return local_pose3d_gt, K


def cam_projection(local_pose3d_gt, K):
    local_pose3d_gt = local_pose3d_gt / local_pose3d_gt[:, -1:]
    pose_2d = np.matmul(local_pose3d_gt, K.T)
    pose_2d = pose_2d[:, :2] / pose_2d[:, -1:]
    return pose_2d


def visualize_data(data_dir, sid, fid, cid):
    local_pose3d_gt, K = read_data(data_dir, sid, fid, cid)
    pose_2d = cam_projection(local_pose3d_gt, K)

    # load image
    image_file = os.path.join(data_dir, f"rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg")
    img = cv2.imread(image_file)

    img = draw_2d_skeleton(img, pose_2d, order="uv", img_order="rgb")
    cv2.imwrite("./hand_pose_2d.jpg", img)

    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.axes(projection="3d")
    draw_3d_skeleton_on_ax(local_pose3d_gt, ax)
    ax.set_title("GT 3D joints")

    ret = fig2data(fig)
    plt.close(fig)

    cv2.imwrite("./hand_pose_3d.jpg", ret)


if __name__ == "__main__":
    data_dir = "data/hanco/"
    visualize_data(data_dir, sid=110, fid=24, cid=3)
