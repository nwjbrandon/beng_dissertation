from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import math
import os.path as osp

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


def draw_mesh(mesh_renderer, image, cam_param, box, mesh_xyz):
    """
    :param mesh_renderer:
    :param image: H x W x 3
    :param cam_param: fx, fy, u0, v0
    :param box: x, y, w, h
    :param mesh_xyz: M x 3
    :return:
    """
    resize_ratio = float(image.shape[0]) / box[2]
    cam_for_render = (
        np.array([cam_param[0], cam_param[2] - box[0], cam_param[3] - box[1]]) * resize_ratio
    )

    rend_img_overlay = mesh_renderer(mesh_xyz, cam=cam_for_render, img=image, do_alpha=True)
    vps = [60.0, -60.0]
    rend_img_vps = [
        mesh_renderer.rotated(mesh_xyz, vp, cam=cam_for_render, img_size=image.shape[:2])
        for vp in vps
    ]

    return rend_img_overlay, rend_img_vps[0], rend_img_vps[1]


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

    marker_sz = 6
    line_wd = 3
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


def draw_3d_skeleton(pose_cam_xyz, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(
        float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True
    )
    ax = plt.subplot(111, projection="3d")
    draw_3d_skeleton_on_ax(pose_cam_xyz, ax)

    ret = fig2data(fig)  # H x W x 4
    plt.close(fig)
    return ret


def save_batch_image_with_mesh_joints(
    mesh_renderer,
    batch_images,
    cam_params,
    bboxes,
    est_mesh_cam_xyz,
    est_pose_uv,
    est_pose_cam_xyz,
    file_name,
    padding=2,
):
    """
    :param mesh_renderer:
    :param batch_images: B x H x W x 3 (torch.Tensor)
    :param cam_params: B x 4 (torch.Tensor)
    :param bboxes: B x 4 (torch.Tensor)
    :param est_mesh_cam_xyz: B x 1280 x 3 (torch.Tensor)
    :param est_pose_uv: B x 21 x 2 (torch.Tensor)
    :param est_pose_cam_xyz: B x 21 x 3 (torch.Tensor)
    :param file_name:
    :param padding:
    :return:
    """
    num_images = batch_images.shape[0]
    image_height = batch_images.shape[1]
    image_width = batch_images.shape[2]
    num_column = 6

    grid_image = np.zeros(
        (num_images * (image_height + padding), num_column * (image_width + padding), 3),
        dtype=np.uint8,
    )

    for id_image in range(num_images):
        image = batch_images[id_image].numpy()
        cam_param = cam_params[id_image].numpy()
        box = bboxes[id_image].numpy()
        mesh_xyz = est_mesh_cam_xyz[id_image].numpy()
        pose_uv = est_pose_uv[id_image].numpy()
        pose_xyz = est_pose_cam_xyz[id_image].numpy()

        rend_img_overlay, rend_img_vp1, rend_img_vp2 = draw_mesh(
            mesh_renderer, image, cam_param, box, mesh_xyz
        )
        skeleton_overlay = draw_2d_skeleton(image, pose_uv)
        skeleton_3d = draw_3d_skeleton(pose_xyz, image.shape[:2])

        img_list = [
            image,
            rend_img_overlay,
            rend_img_vp1,
            rend_img_vp2,
            skeleton_overlay,
            skeleton_3d,
        ]

        height_begin = (image_height + padding) * id_image
        height_end = height_begin + image_height
        width_begin = 0
        width_end = image_width
        for show_img in img_list:
            grid_image[height_begin:height_end, width_begin:width_end, :] = show_img[..., :3]
            width_begin += image_width + padding
            width_end = width_begin + image_width

    cv2.imwrite(file_name, grid_image)


def get_train_val_im_paths(image_dir, val_set_path, train_val_flag):
    """
    get training or validation image paths
    :param image_dir:
    :param val_set_path:
    :param train_val_flag:
    :return:
    """
    val_cameras = []
    with open(val_set_path) as reader:
        for line in reader:
            val_cameras.append(line.strip())
    val_cameras = set(val_cameras)

    lighting_folders = glob.glob(osp.join(image_dir, "l*"))

    image_paths = []
    for lighting_folder in lighting_folders:
        cam_folders = glob.glob(osp.join(lighting_folder, "cam*"))
        for cam_folder in cam_folders:
            cam_name = osp.basename(cam_folder)
            is_val = cam_name in val_cameras
            if (train_val_flag == "val" and is_val) or (train_val_flag == "train" and not is_val):
                image_paths += glob.glob(osp.join(cam_folder, "*.png"))

    return image_paths


def extract_pose_camera_id(im_filename):
    """
    extract pose id and camera id from image file name
    :param im_filename: e.g., 'handV2_rgt01_specTest5_gPoses_ren_25cRrRs_l21_cam01_.0001.png'
    :return: pose id (int, start from 0) and camera id (int, start from 0)
    """
    name = osp.splitext(im_filename)[0]
    fields = name.split("_")
    pose_id = int(fields[-1].replace(".", "0")) - 1
    camera_id = int(fields[-2][3:]) - 1
    return pose_id, camera_id


def load_camera_param(camera_param_path):
    """
    load camera parameters
    :param camera_param_path:
    :return: (N_pose, N_cam, 7) (focal_length, 3 translation val; 3 euler angles)
    """
    all_camera_names = np.loadtxt(camera_param_path, usecols=(0,), dtype=str)
    num_cameras = len(np.unique(all_camera_names))
    all_camera_params = np.loadtxt(camera_param_path, usecols=(1, 2, 3, 4, 5, 6, 7))
    all_camera_params = all_camera_params.reshape((-1, num_cameras, 7))
    return all_camera_params


def load_global_pose3d_gt(pose3d_gt_path):
    """
    load global 3D hand pose ground truth
    :param pose3d_gt_path:
    :return: (N_pose, 21, 3)
    """
    all_joint_names = np.loadtxt(pose3d_gt_path, usecols=(0,), dtype=str)
    num_joints = len(np.unique(all_joint_names))
    all_global_pose3d_gt = np.loadtxt(pose3d_gt_path, usecols=(1, 2, 3)).reshape(
        (-1, num_joints, 3)
    )
    return all_global_pose3d_gt


def euler_xyz_to_rot_mx(euler_angle):
    """
    convert xyz euler angles to rotation matrix
    :param euler_angle: euler angles for x, y, z axis, (degree)
    :return: rotation matrix, (3, 3)
    """
    rad = euler_angle * math.pi / 180.0
    sins = np.sin(rad)
    coss = np.cos(rad)
    rot_x = np.array([[1, 0, 0], [0, coss[0], -sins[0]], [0, sins[0], coss[0]]])
    rot_y = np.array([[coss[1], 0, sins[1]], [0, 1, 0], [-sins[1], 0, coss[1]]])
    rot_z = np.array([[coss[2], -sins[2], 0], [sins[2], coss[2], 0], [0, 0, 1]])
    rot_mx = rot_z.dot(rot_y).dot(rot_x)
    return rot_mx


def transform_global_to_cam(global_3d, camera_param, use_translation=True):
    """
    transform 3D pose in global coordinate system to camera coordinate system
    :param global_3d: (N, 3)
    :param camera_param: (7, ) focal_length, 3 translation val; 3 euler angles (degree)
    :param use_translation: bool
    :return: camera_3d: (N, 3)
    """
    if use_translation:
        translation = camera_param[1:4]  # (3, )
        pose3d = global_3d - translation
    else:
        pose3d = global_3d

    theta = camera_param[4:]  # (3, )
    rot_mx = euler_xyz_to_rot_mx(theta)
    aux_mx = np.eye(3, dtype=float)
    aux_mx[1, 1] = -1.0
    aux_mx[2, 2] = -1.0
    rot_mx = rot_mx.dot(aux_mx)

    camera_3d = pose3d.dot(rot_mx)
    return camera_3d


def cam_projection(local_pose3d, cam_proj_mat):
    """
    get 2D projection points
    :param local_pose3d: (N, 3)
    :param cam_proj_mat: (3, 3)
    :return:
    """
    xyz = local_pose3d.dot(cam_proj_mat.transpose())  # (N, 3)
    z_inv = 1.0 / xyz[:, 2]  # (N, ), 1/z
    z_inv = np.expand_dims(z_inv, axis=1)  # (N, 1), 1/z
    xyz = xyz * z_inv
    pose_2d = xyz[:, :2]  # (N, 2)
    return pose_2d


def load_mesh_from_obj(mesh_file, arm_index_range=[473, 529]):
    """
    Load mesh vertices, normals, triangle indices and vertices from obj file
    :param mesh_file: path to the hand mesh obj file
    :param arm_index_range: range of indices which belong to arm
    :return: mesh vertices, normals, triangle indices and vertices
    """
    mesh_pts = []
    mesh_tri_idx = []
    mesh_vn = []
    id_vn = 0
    state = "V"
    with open(mesh_file) as reader:
        for line in reader:
            fields = line.strip().split()
            try:
                if fields[0] == "v":
                    if state != "V":
                        break
                    mesh_pts.append([float(f) for f in fields[1:]])
                if fields[0] == "f":
                    state = "F"
                    mesh_tri_idx.append([int(f.split("/")[0]) - 1 for f in fields[1:]])
                if fields[0] == "vn":
                    state = "N"
                    if id_vn % 3 == 0:
                        mesh_vn.append([float(f) for f in fields[1:]])
                    id_vn = id_vn + 1
            except:
                pass

    mesh_pts = np.array(mesh_pts)  # (N_vertex, 3)
    mesh_vn = np.array(mesh_vn)  # (N_tris, 3)
    mesh_tri_idx = np.array(mesh_tri_idx)  # (N_tris, 3)

    if len(arm_index_range) > 1 and arm_index_range[1] > arm_index_range[0]:
        mesh_pts, mesh_vn, mesh_tri_idx = remove_arm_vertices(
            mesh_pts, mesh_vn, mesh_tri_idx, arm_index_range
        )

    return mesh_pts, mesh_vn, mesh_tri_idx


def get_mesh_tri_vertices(mesh_vertices, mesh_tri_idx):
    """
    get the 3D coordinates of three vertices in mesh triangles
    :param mesh_vertices: (N_vertex, 3)
    :param mesh_tri_idx: (N_tris, 3)
    :return: (N_tris, 3, 3)
    """
    mesh_tri_pts = np.zeros((len(mesh_tri_idx), 3, 3))  # (N_tris, 3, 3)
    for idx, tri in enumerate(mesh_tri_idx):
        mesh_tri_pts[idx, 0, :] = mesh_vertices[tri[0]]
        mesh_tri_pts[idx, 1, :] = mesh_vertices[tri[1]]
        mesh_tri_pts[idx, 2, :] = mesh_vertices[tri[2]]

    return mesh_tri_pts


def remove_arm_vertices(mesh_pts, mesh_vn, mesh_tri_idx, arm_index_range):
    """
    remove vertices belong to arm in the hand mesh
    :param mesh_pts: (N_vertex, 3)
    :param mesh_vn: (N_tris, 3)
    :param mesh_tri_idx: (N_tris, 3)
    :param arm_index_range: range of indices which belong to arm
    :return:
    """
    arm_mesh_idx = range(arm_index_range[0], arm_index_range[1])
    arm_index_set = set(arm_mesh_idx)
    hand_indices = list(set(range(0, len(mesh_pts))) - arm_index_set)
    hand_mesh_pts = mesh_pts[hand_indices]

    hand_mesh_tri_idx = []
    hand_mesh_vn = []

    if mesh_tri_idx.size <= 1:
        return hand_mesh_pts, hand_mesh_vn, hand_mesh_tri_idx

    def _index_shift(ind):
        if ind >= arm_index_range[1]:
            return ind - (arm_index_range[1] - arm_index_range[0])
        else:
            return ind

    for i in range(mesh_tri_idx.shape[0]):
        if (
            (mesh_tri_idx[i][0] not in arm_index_set)
            and (mesh_tri_idx[i][1] not in arm_index_set)
            and (mesh_tri_idx[i][2] not in arm_index_set)
        ):
            hand_mesh_tri_idx.append(list(map(_index_shift, mesh_tri_idx[i])))
            hand_mesh_vn.append(mesh_vn[i])

    hand_mesh_tri_idx = np.array(hand_mesh_tri_idx)
    hand_mesh_vn = np.array(hand_mesh_vn)
    return hand_mesh_pts, hand_mesh_vn, hand_mesh_tri_idx


def init_pose3d_labels(cam_param_path, pose3d_gt_path):
    all_camera_params = load_camera_param(cam_param_path)
    all_global_pose3d_gt = load_global_pose3d_gt(pose3d_gt_path)
    return all_camera_params, all_global_pose3d_gt


def read_data(im_path, all_camera_params, all_global_pose3d_gt, global_mesh_gt_dir):
    """
    read the corresponding pose and mesh ground truth of the image sample, and camera parameters
    :param im_path:
    :param all_camera_params: (N_pose, N_cam, 7) focal_length, 3 translation val; 3 euler angles (degree)
    :param all_global_pose3d_gt: (N_pose, 21, 3)
    :param global_mesh_gt_dir:
    :return:
    """
    pose_id, camera_id = extract_pose_camera_id(osp.basename(im_path))

    cam_param = all_camera_params[pose_id][camera_id]  # (7, )

    # get ground truth of 3D hand pose
    global_pose3d_gt = all_global_pose3d_gt[pose_id]  # (21, 3)
    local_pose3d_gt = transform_global_to_cam(global_pose3d_gt, cam_param)  # (21, 3)
    return local_pose3d_gt, cam_param


def visualize_data(im_path, local_pose3d_gt, local_mesh_pts_gt, cam_param, mesh_tri_idx):
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    img_rgb = img[:, :, :3]

    im_height, im_width = img.shape[:2]
    fig = plt.figure()
    fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

    fl = cam_param[0]  # focal length
    cam_proj_mat = np.array(
        [[fl, 0.0, im_width / 2.0], [0.0, fl, im_height / 2.0], [0.0, 0.0, 1.0]]
    )
    pose_2d = cam_projection(local_pose3d_gt, cam_proj_mat)

    # plt = plt.figure()
    skeleton_overlay = draw_2d_skeleton(img_rgb, pose_2d)
    plt.imshow(skeleton_overlay)
    plt.title("Image With GT 2D joints")

    ret = fig2data(fig)
    plt.close(fig)

    cv2.imwrite("./hand_pose_2d.jpg", ret)

    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.axes(projection="3d")
    draw_3d_skeleton_on_ax(local_pose3d_gt, ax)
    ax.set_title("GT 3D joints")

    ret = fig2data(fig)
    plt.close(fig)

    cv2.imwrite("./hand_pose_3d.jpg", ret)
