import json
import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.hanco.data_utils import cam_projection, read_data


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
    keypoints_norm = keypoints / model_img_size
    return keypoints_norm


def get_train_val_image_paths(data_dir, is_training):
    """
        sid: Sequence id: int, in [0, 1517]
    """
    meta_file = os.path.join(data_dir, "meta.json")
    with open(meta_file, "r") as fi:
        meta_data = json.load(fi)

    sids = 1517
    fids = len(meta_data["is_train"])
    cids = 8

    img_paths_train = []
    img_paths_val = []
    for sid in range(sids):
        for cid in range(cids):
            for fid in range(fids):
                sid = 110
                is_train = meta_data["is_train"][sid][fid]
                if is_train:
                    img_paths_train.append((sid, cid, fid))
                else:
                    img_paths_val.append((sid, cid, fid))

    if is_training:
        return img_paths_train
    else:
        return img_paths_val


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

        self.image_names = get_train_val_image_paths(self.data_dir, is_training=self.is_training)
        print("Total Images:", len(self.image_names))

        self.image_transform = transforms.Compose(
            [transforms.Resize(self.raw_image_size), transforms.ToTensor(),]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get Labels
        """
        Close to the edge
        """
        sid, cid, fid = self.image_names[idx]

        image_name = os.path.join(self.data_dir, f"rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg")
        image = Image.open(image_name).convert("RGB")
        im_width, im_height = image.size
        local_pose3d_gt, K = read_data(self.data_dir, sid, fid, cid)

        kpt_2d_gt = cam_projection(local_pose3d_gt, K)

        # Preprocess
        image_inp = self.image_transform(image)
        heatmaps_gt, _ = vector_to_heatmaps(
            kpt_2d_gt, im_width, im_height, self.n_keypoints, self.model_img_size
        )
        kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] / im_width
        kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] / im_height
        kpt_2d_gt[:, (0, 1)] = kpt_2d_gt[:, (1, 0)]
        kpt_3d_gt = local_pose3d_gt

        return {
            "image_name": image_name,
            "image_inp": image_inp,  # img to 2d
            "heatmaps_gt": heatmaps_gt,  # img to 2d
            "kpt_2d_gt": kpt_2d_gt,  # 2d to 3d
            "kpt_3d_gt": kpt_3d_gt,  # 2d to 3d
        }
