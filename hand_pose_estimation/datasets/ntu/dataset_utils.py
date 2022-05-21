import glob
import os.path as osp

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.ntu.data_utils import cam_projection, init_pose3d_labels, read_data


def vector_to_heatmaps(keypoints, im_width, im_height, n_keypoints, model_img_size):
    """
    Creates 2D heatmaps from keypoint locations for a single image
    Input: array of size N_KEYPOINTS x 2
    Output: array of size N_KEYPOINTS x MODEL_IMG_SIZE x MODEL_IMG_SIZE
    """
    keypoints_norm = keypoints.copy()
    keypoints_norm[:, 0] = keypoints_norm[:, 0] / im_width
    keypoints_norm[:, 1] = keypoints_norm[:, 1] / im_height

    heatmaps = np.zeros([n_keypoints, model_img_size, model_img_size])
    visibility_vector = np.zeros([n_keypoints])

    for k, (x, y) in enumerate(keypoints_norm):
        x = 0 if x < 0 else x
        x = 0.999 if x >= 1 else x
        y = 0 if y < 0 else y
        y = 0.999 if y >= 1 else y
        assert x >= 0 and x <= 1 and y >= 0 and y <= 1
        heatmap = compute_heatmap(x, y, model_img_size)
        heatmaps[k] = heatmap
        visibility_vector[k] = 1

    return heatmaps, visibility_vector


def compute_heatmap(x, y, model_img_size, kernel_size=5):
    # Create joint heatmap
    heatmap = np.zeros([model_img_size, model_img_size])
    heatmap[int(y * model_img_size), int(x * model_img_size)] = 1
    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    heatmap = heatmap / np.max(heatmap)
    return heatmap


def heatmaps_to_coordinates(joint_heatmaps, model_img_size):
    keypoints = np.array(
        [np.unravel_index(heatmap.argmax(), heatmap.shape) for heatmap in joint_heatmaps]
    )
    keypoints[:, (0, 1)] = keypoints[:, (1, 0)]
    keypoints_norm = keypoints / model_img_size
    return keypoints_norm


def get_train_val_image_paths(image_dir, val_set_path, is_training):
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
            if is_training:
                if cam_name not in val_cameras:
                    image_paths.extend(glob.glob(f"{cam_folder}/*.png"))
            else:
                if cam_name in val_cameras:
                    image_paths.extend(glob.glob(f"{cam_folder}/*.png"))

    return image_paths


class HandPoseDataset(Dataset):
    """
    Class to load hand pose dataset.

    Link to dataset:
    https://github.com/3d-hand-shape/hand-graph-cnn/tree/master/data
    """

    def __init__(self, config, set_type="train"):
        self.device = config["training_details"]["device"]
        self.camera_param_path = config["dataset"]["camera_param_file"]
        self.global_pose3d_gt_path = config["dataset"]["global_pose3d_gt_file"]
        self.global_mesh_gt_dir = config["dataset"]["global_mesh_gt_dir"]
        self.n_keypoints = config["model"]["n_keypoints"]
        self.raw_image_size = config["model"]["raw_image_size"]
        self.model_img_size = config["model"]["model_img_size"]

        self.is_training = set_type == "train"

        self.data_dir = config["dataset"]["images_dir"]
        self.val_cams_file = config["dataset"]["val_cams_file"]
        # glob.glob(f"{self.data_dir}/**/*.png", recursive=True)
        self.image_names = get_train_val_image_paths(
            self.data_dir, self.val_cams_file, is_training=self.is_training
        )
        print("Total Images:", len(self.image_names))

        self.all_camera_params, self.all_global_pose3d_gt = init_pose3d_labels(
            self.camera_param_path, self.global_pose3d_gt_path
        )

        self.image_transform = transforms.Compose(
            [transforms.Resize(self.raw_image_size), transforms.ToTensor(),]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get Labels
        """
        Close to the edge
        data/synthetic_train_val/images/l01/cam05/handV2_rgt01_specTest5_gPoses_ren_25cRrRs_l01_cam05_.0235.png"
        data/synthetic_train_val/images/l01/cam08/handV2_rgt01_specTest5_gPoses_ren_25cRrRs_l01_cam08_.0155.png"
        """
        image_name = self.image_names[idx]
        (local_pose3d_gt, cam_param,) = read_data(
            image_name, self.all_camera_params, self.all_global_pose3d_gt, self.global_mesh_gt_dir
        )

        drot = -1
        brightness_factor = -1
        contrast_factor = -1
        sharpness_factor = -1
        is_mirror = False
        is_flip = False

        # Get RGB Image
        image = Image.open(image_name).convert("RGB")
        im_width, im_height = image.size

        if self.is_training:
            drot = np.random.choice([0, 90, 180, 270])
            brightness_factor = 1 + np.random.rand() * 4 / 10 - 0.2
            contrast_factor = 1 + np.random.rand() * 4 / 10 - 0.2
            sharpness_factor = 1 + np.random.rand() * 4 / 10 - 0.2
            is_mirror = True if np.random.rand() > 0.5 else False
            is_flip = True if np.random.rand() > 0.5 else False
            z_rot = np.array(
                [
                    [np.cos(np.deg2rad(drot)), -np.sin(np.deg2rad(drot)), 0],
                    [np.sin(np.deg2rad(drot)), np.cos(np.deg2rad(drot)), 0],
                    [0, 0, 1],
                ]
            )
            fl = cam_param[0]  # focal length
            local_pose3d_gt = local_pose3d_gt @ z_rot
            image = image.rotate(drot)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
            image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

            if is_mirror:
                image = ImageOps.mirror(image)
                local_pose3d_gt[:, 0] = -local_pose3d_gt[:, 0]

            if is_flip:
                image = ImageOps.flip(image)
                local_pose3d_gt[:, 1] = -local_pose3d_gt[:, 1]

        # Get 2D Poses
        fl = cam_param[0]  # focal length
        cam_proj_mat = np.array(
            [[fl, 0.0, im_width / 2.0], [0.0, fl, im_height / 2.0], [0.0, 0.0, 1.0]]
        )
        kpt_2d_gt = cam_projection(local_pose3d_gt, cam_proj_mat)

        # Preprocess
        image_inp = self.image_transform(image)
        heatmaps_gt, _ = vector_to_heatmaps(
            kpt_2d_gt, im_width, im_height, self.n_keypoints, self.model_img_size
        )
        kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] / im_width
        kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] / im_height
        kpt_3d_gt = (local_pose3d_gt - local_pose3d_gt[9]) / 100

        return {
            "image_name": image_name,
            "image_inp": image_inp,  # img to 2d
            "heatmaps_gt": heatmaps_gt,  # img to 2d
            "kpt_2d_gt": kpt_2d_gt,  # 2d to 3d
            "kpt_3d_gt": kpt_3d_gt,  # 2d to 3d
            "drot": drot,
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "sharpness_factor": sharpness_factor,
            "is_mirror": is_mirror,
            "is_flip": is_flip,
        }
