import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.data_utils import draw_2d_skeleton, draw_3d_skeleton_on_ax, read_data


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


def compute_heatmap(x, y, model_img_size, kernel_size=9):
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
    data = []
    for exp in os.listdir(data_dir):
        exp_dir = os.path.join(data_dir, exp)
        labels = os.listdir(exp_dir)
        for label in labels:
            label_dir = os.path.join(exp_dir, label)
            img_path_regexp = os.path.join(label_dir, "*.jpg")
            img_paths = glob.glob(img_path_regexp)
            for img_path in img_paths:
                img_name = img_path.split("/")[-1].split(".")[0]
                img_index = img_name.split("_")[-1]

                pose2d_path = os.path.join(label_dir, f"pose2d_{img_index}.npy")
                pose3d_path = os.path.join(label_dir, f"pose3d_{img_index}.npy")
                data_point = (img_path, pose2d_path, pose3d_path)
                data.append(data_point)
    return data


def adjust_image(image, brightness_factor, contrast_factor, sharpness_factor):
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)
    return image


def rotate_image(image, kpt_2d_gt, kpt_3d_gt, drot):
    image = image.rotate(drot)

    z_rot = np.array(
        [
            [np.cos(np.deg2rad(drot)), -np.sin(np.deg2rad(drot)), 0],
            [np.sin(np.deg2rad(drot)), np.cos(np.deg2rad(drot)), 0],
            [0, 0, 1],
        ]
    )
    kpt_3d_gt = kpt_3d_gt @ z_rot

    z_rot = np.array(
        [
            [np.cos(np.deg2rad(drot)), -np.sin(np.deg2rad(drot))],
            [np.sin(np.deg2rad(drot)), np.cos(np.deg2rad(drot))],
        ]
    )
    kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] - 240
    kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] - 240
    kpt_2d_gt = kpt_2d_gt @ z_rot
    kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] + 240
    kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] + 240

    return image, kpt_2d_gt, kpt_3d_gt


def mirror_image(image, kpt_2d_gt, kpt_3d_gt, is_mirror):
    if is_mirror:
        image = ImageOps.mirror(image)
        kpt_2d_gt[:, 0] = -kpt_2d_gt[:, 0] + 480
        kpt_3d_gt[:, 0] = -kpt_3d_gt[:, 0]
    return image, kpt_2d_gt, kpt_3d_gt


def flip_image(image, kpt_2d_gt, kpt_3d_gt, is_flip):
    if is_flip:
        image = ImageOps.flip(image)
        kpt_2d_gt[:, 1] = -kpt_2d_gt[:, 1] + 480
        kpt_3d_gt[:, 1] = -kpt_3d_gt[:, 1]
    return image, kpt_2d_gt, kpt_3d_gt


def translate_image(image, kpt_2d_gt, dx, dy):
    image = image.transform(image.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy))
    kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] + dx
    kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] + dy
    return image, kpt_2d_gt


def random_drot_dx_dy(kpt_2d, im_width, im_height):
    drot = 0
    dx = 0
    dy = 0

    for _ in range(5):
        kpt_2d_gt = kpt_2d.copy()

        drot = np.random.choice([0, 90, 180, 270])
        # drot = np.random.rand() * 360
        dx, dy = 0, 0
        # dx = int(np.random.rand() * 200) - 100
        # dy = int(np.random.rand() * 200) - 100

        # rotate
        z_rot = np.array(
            [
                [np.cos(np.deg2rad(drot)), -np.sin(np.deg2rad(drot))],
                [np.sin(np.deg2rad(drot)), np.cos(np.deg2rad(drot))],
            ]
        )
        kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] - 240
        kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] - 240
        kpt_2d_gt = kpt_2d_gt @ z_rot
        kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] + 240
        kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] + 240

        # translate
        kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] + dx
        kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] + dy

        keypoints_norm = kpt_2d_gt.copy()
        keypoints_norm[:, 0] = keypoints_norm[:, 0] / im_width
        keypoints_norm[:, 1] = keypoints_norm[:, 1] / im_height

        if (
            np.all(keypoints_norm[:, 0] < 1)
            and np.all(keypoints_norm[:, 0] > 0)
            and np.all(keypoints_norm[:, 1] < 1)
            and np.all(keypoints_norm[:, 1] > 0)
        ):
            return drot, dx, dy
    return 0, 0, 0


def fill_holes(image, bg_imgs, im_width, im_height):
    image = np.asarray(image)
    black_px_mask = np.all(image < 1, axis=2)

    bg_img = Image.open(np.random.choice(bg_imgs)).convert("RGB").resize((im_width, im_height))
    bg_img = np.asarray(bg_img)

    image[black_px_mask] = bg_img[black_px_mask]
    image = Image.fromarray(image)
    return image


class BodyPoseDataset(Dataset):
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

        self.bg_imgs_dir = config["training_details"]["bg_imgs_dir"]
        self.bg_imgs = glob.glob(os.path.join(self.bg_imgs_dir, "*.jpg"))

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
        image_name, pose2d_path, pose3d_path = self.image_names[idx]

        image, kpt_2d_gt, kpt_3d_gt = read_data(image_name, pose2d_path, pose3d_path)

        drot = -1
        dx = -1
        dy = -1
        brightness_factor = -1
        contrast_factor = -1
        sharpness_factor = -1
        is_mirror = False
        is_flip = False

        kpt_2d_gt = kpt_2d_gt[:, :2].astype(float)
        kpt_3d_gt = kpt_3d_gt.astype(float)
        im_width, im_height = image.size

        if self.is_training:
            drot, dx, dy = random_drot_dx_dy(kpt_2d_gt.copy(), im_width, im_height)
            brightness_factor = 1 + np.random.rand() * 5 / 10 - 0.25
            contrast_factor = 1 + np.random.rand() * 5 / 10 - 0.25
            sharpness_factor = 1 + np.random.rand() * 5 / 10 - 0.25
            is_mirror = True if np.random.rand() > 0.5 else False
            is_flip = True if np.random.rand() > 0.5 else False

            image = adjust_image(image, brightness_factor, contrast_factor, sharpness_factor)
            image, kpt_2d_gt, kpt_3d_gt = rotate_image(image, kpt_2d_gt, kpt_3d_gt, drot)
            image, kpt_2d_gt, kpt_3d_gt = mirror_image(image, kpt_2d_gt, kpt_3d_gt, is_mirror)
            image, kpt_2d_gt, kpt_3d_gt = flip_image(image, kpt_2d_gt, kpt_3d_gt, is_flip)
            image, kpt_2d_gt = translate_image(image, kpt_2d_gt, dx, dy)
            # image = fill_holes(image, self.bg_imgs, im_width, im_height)

        # TODO: set to false during training
        if False:
            skeleton_overlay = draw_2d_skeleton(np.asarray(image), kpt_2d_gt)
            plt.imshow(skeleton_overlay)

            plt.figure(figsize=(10, 10), dpi=80)
            ax = plt.axes(projection="3d")
            draw_3d_skeleton_on_ax(kpt_3d_gt, ax)
            ax.set_title("GT 3D joints")
            plt.show()

        heatmaps_gt, _ = vector_to_heatmaps(
            kpt_2d_gt, im_width, im_height, self.n_keypoints, self.model_img_size
        )

        image_inp = self.image_transform(image)
        kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] / im_width
        kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] / im_height
        kpt_3d_gt = kpt_3d_gt - kpt_3d_gt[0]

        return {
            "image_name": image_name,
            "image_inp": image_inp,  # img to 2d
            "heatmaps_gt": heatmaps_gt,  # img to 2d
            "kpt_2d_gt": kpt_2d_gt,  # 2d to 3d
            "kpt_3d_gt": kpt_3d_gt,  # 2d to 3d
        }
