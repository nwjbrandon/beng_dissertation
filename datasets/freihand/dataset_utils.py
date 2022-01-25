import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.freihand.data_utils import cam_projection, init_pose3d_labels, read_data


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


def get_train_val_image_paths(data_dir, is_training):
    image_paths = []

    n_start, n_end = 0, 130239  # 32560
    for idx in range(n_start, n_end):
        image_paths.append(
            (os.path.join(data_dir, "train", "training", "rgb", "%08d.jpg" % idx), idx, True)
        )

    n_start, n_end = 0, 3960
    for idx in range(n_start, n_end):
        image_paths.append(
            (os.path.join(data_dir, "val", "evaluation", "rgb", "%08d.jpg" % idx), idx, False)
        )

    train, test = train_test_split(image_paths, test_size=0.1, shuffle=True, random_state=42)
    if is_training:
        return train
    else:
        return test


class HandPoseDataset(Dataset):
    """
    Class to load hand pose dataset.

    Link to dataset:
    https://github.com/3d-hand-shape/hand-graph-cnn/tree/master/data
    """

    def __init__(self, config, set_type="train"):
        self.device = config["training_details"]["device"]
        self.n_keypoints = config["model"]["n_keypoints"]
        self.raw_image_size = config["model"]["raw_image_size"]
        self.model_img_size = config["model"]["model_img_size"]
        self.data_dir = config["dataset"]["data_dir"]

        self.is_training = set_type == "train"

        self.all_camera_params_train, self.all_global_pose3d_gt_train = init_pose3d_labels(
            self.data_dir, True
        )
        self.all_camera_params_val, self.all_global_pose3d_gt_val = init_pose3d_labels(
            self.data_dir, False
        )

        self.image_names = get_train_val_image_paths(self.data_dir, is_training=self.is_training)
        print("Total Images:", len(self.image_names))

        self.image_transform = transforms.Compose(
            [transforms.Resize(self.raw_image_size), transforms.ToTensor(),]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get Labels
        image_name, image_idx, is_train_img = self.image_names[idx]
        cam_param, local_pose3d_gt = read_data(
            image_idx,
            is_train_img,
            self.all_camera_params_train,
            self.all_global_pose3d_gt_train,
            self.all_camera_params_val,
            self.all_global_pose3d_gt_val,
        )
        kpt_2d_gt = cam_projection(local_pose3d_gt, cam_param)
        brightness_factor = -1
        contrast_factor = -1
        sharpness_factor = -1

        # Get RGB Image
        image = Image.open(image_name).convert("RGB")
        im_width, im_height = image.size

        if self.is_training:
            brightness_factor = 1 + np.random.rand() * 4 / 10 - 0.2
            contrast_factor = 1 + np.random.rand() * 4 / 10 - 0.2
            sharpness_factor = 1 + np.random.rand() * 4 / 10 - 0.2

            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
            image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

        # Preprocess
        image_inp = self.image_transform(image)
        heatmaps_gt, _ = vector_to_heatmaps(
            kpt_2d_gt, im_width, im_height, self.n_keypoints, self.model_img_size
        )
        kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] / im_width
        kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] / im_height
        kpt_3d_gt = local_pose3d_gt

        return {
            "image_name": image_name,
            "image_inp": image_inp,  # img to 2d
            "heatmaps_gt": heatmaps_gt,  # img to 2d
            "kpt_2d_gt": kpt_2d_gt,  # 2d to 3d
            "kpt_3d_gt": kpt_3d_gt,  # 2d to 3d
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "sharpness_factor": sharpness_factor,
        }
