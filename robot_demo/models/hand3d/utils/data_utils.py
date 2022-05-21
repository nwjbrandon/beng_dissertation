import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

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

""" General util functions. """


def _assert_exist(p):
    msg = "File does not exists: %s" % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, "r") as fi:
        d = json.load(fi)
    return d


def cam_projection(local_pose3d, cam_proj_mat):
    """ Project 3D coordinates into image space. """
    local_pose3d = np.array(local_pose3d)
    cam_proj_mat = np.array(cam_proj_mat)
    pose_2d = np.matmul(cam_proj_mat, local_pose3d.T).T
    return pose_2d[:, :2] / pose_2d[:, -1:]


def init_pose3d_labels(data_dir, is_training):
    if is_training:
        cam_param_path = os.path.join(data_dir, "train", "training_K.json")
        pose3d_gt_path = os.path.join(data_dir, "train", "training_xyz.json")
    else:
        cam_param_path = os.path.join(data_dir, "val", "evaluation_K.json")
        pose3d_gt_path = os.path.join(data_dir, "val", "evaluation_xyz.json")
    all_camera_params = json_load(cam_param_path)
    all_global_pose3d_gt = json_load(pose3d_gt_path)
    assert len(all_camera_params) == len(all_global_pose3d_gt), "Size mismatch."
    return all_camera_params, all_global_pose3d_gt


def read_data(
    image_idx,
    is_training,
    all_camera_params_train,
    all_global_pose3d_gt_train,
    all_camera_params_val,
    all_global_pose3d_gt_val,
):
    if is_training:
        return (
            np.array(all_camera_params_train[image_idx % 32560]),
            np.array(all_global_pose3d_gt_train[image_idx % 32560]),
        )
    else:
        return (
            np.array(all_camera_params_val[image_idx]),
            np.array(all_global_pose3d_gt_val[image_idx]),
        )


def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = np.copy(image)

    marker_sz = 2
    line_wd = 1
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype("int32"), pose_uv[joint_ind, 1].astype("int32")
        cv2.circle(
            skeleton_overlay,
            joint,
            radius=marker_sz,
            color=color_hand_joints[joint_ind] * np.array(255),
            thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
        )

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype("int32"), pose_uv[root_ind, 1].astype("int32")
            cv2.line(
                skeleton_overlay,
                root_joint,
                joint,
                color=color_hand_joints[joint_ind] * np.array(255),
                thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
            )
        else:
            joint_2 = (
                pose_uv[joint_ind - 1, 0].astype("int32"),
                pose_uv[joint_ind - 1, 1].astype("int32"),
            )
            cv2.line(
                skeleton_overlay,
                joint_2,
                joint,
                color=color_hand_joints[joint_ind] * np.array(255),
                thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith("2") else cv2.LINE_AA,
            )

    return skeleton_overlay


def draw_3d_skeleton_on_ax(pose_cam_xyz, ax):
    """
    :param pose_cam_xyz: 21 x 3
    :param ax:
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    marker_sz = 15
    x_radius = [-0.08, 0.08]
    y_radius = [-0.08, 0.08]
    z_radius = [-0.08, 0.08]
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
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


def visualize_data(image_name, data_dir, is_sample_training, sample_image_idx):
    all_camera_params_train, all_global_pose3d_gt_train = init_pose3d_labels(data_dir, True)
    all_camera_params_val, all_global_pose3d_gt_val = init_pose3d_labels(data_dir, False)

    cam_param, local_pose3d_gt = read_data(
        sample_image_idx,
        is_sample_training,
        all_camera_params_train,
        all_global_pose3d_gt_train,
        all_camera_params_val,
        all_global_pose3d_gt_val,
    )

    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    img_rgb = img[:, :, :3]

    im_height, im_width = img.shape[:2]
    fig = plt.figure()
    fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

    pose_2d = cam_projection(local_pose3d_gt, cam_param)

    # plt = plt.figure()
    skeleton_overlay = draw_2d_skeleton(img_rgb, pose_2d)
    plt.imshow(skeleton_overlay)
    plt.title("Image With GT 2D joints")

    ret = fig2data(fig)
    plt.close(fig)

    cv2.imwrite("./hand_pose_2d.jpg", ret)

    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.axes(projection="3d")
    local_pose3d_gt = local_pose3d_gt - local_pose3d_gt[0]
    draw_3d_skeleton_on_ax(local_pose3d_gt * 10, ax)
    ax.set_title("GT 3D joints")

    ret = fig2data(fig)
    plt.close(fig)

    cv2.imwrite("./hand_pose_3d.jpg", ret)


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
