import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.ho3d.data_utils import cam_projection, pad_to_square, read_data


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
    data_dir = os.path.join(data_dir, "train")

    for subject in os.listdir(os.path.join(data_dir)):
        s_path = os.path.join(data_dir, subject)
        rgb = os.path.join(s_path, "rgb")
        meta = os.path.join(s_path, "meta")
        for rgb_file in os.listdir(rgb):
            file_number = rgb_file.split(".")[0]
            meta_file = os.path.join(meta, file_number + ".pkl")
            img_path = os.path.join(rgb, rgb_file)
            if is_training:
                if subject != "MC6":
                    image_paths.append((img_path, meta_file))
            else:
                if subject == "MC6":
                    image_paths.append((img_path, meta_file))
    return image_paths


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

        self.image_names = get_train_val_image_paths(self.data_dir, is_training=self.is_training,)
        print("Total Images:", len(self.image_names))

        self.image_transform = transforms.Compose(
            [transforms.Resize((self.raw_image_size, self.raw_image_size)), transforms.ToTensor(),]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get Labels
        """
        data/ho3d/train/MDF12/rgb/0061.png data/ho3d/train/MDF12/meta/0061.pkl
        """
        image_name, meta_name = self.image_names[idx]

        cam_param, local_pose3d_gt = read_data(meta_name)

        brightness_factor = -1
        contrast_factor = -1
        sharpness_factor = -1
        red = int(np.random.rand() * 255)
        green = int(np.random.rand() * 255)
        blue = int(np.random.rand() * 255)

        # Get RGB Image
        image = Image.open(image_name).convert("RGB")
        image = pad_to_square(image, (red, green, blue))
        im_width, im_height = image.size

        if self.is_training:
            brightness_factor = 1 + np.random.rand() * 4 / 10 - 0.2
            contrast_factor = 1 + np.random.rand() * 4 / 10 - 0.2
            sharpness_factor = 1 + np.random.rand() * 4 / 10 - 0.2

            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
            image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

        # Preprocess
        kpt_2d_gt = cam_projection(local_pose3d_gt, cam_param)
        image_inp = self.image_transform(image)
        try:
            heatmaps_gt, _ = vector_to_heatmaps(
                kpt_2d_gt, im_width, im_height, self.n_keypoints, self.model_img_size
            )
        except:
            print(idx, image_name, meta_name)
            raise
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
            "red": red,
            "green": green,
            "blue": blue,
        }
