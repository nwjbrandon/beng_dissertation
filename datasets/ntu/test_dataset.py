import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ntu.data_utils import (
    draw_2d_skeleton,
    draw_3d_skeleton_on_ax,
    init_pose3d_labels,
    read_data,
    visualize_data
)
from datasets.ntu.dataset_utils import get_train_val_image_paths, heatmaps_to_coordinates
from utils.io import import_module


class TestDataset:
    def __init__(self, config):
        self.config = config

    def evaluate(self):
        Dataset = import_module(self.config["dataset"]["dataset_name"])
        test_dataset = Dataset(config=self.config, set_type="test")
        test_dataloader = DataLoader(test_dataset, 1, num_workers=0,)

        Model = import_module(self.config["model"]["model_name"])
        model = Model(self.config)
        model = model.to(self.config["test"]["device"])
        assert self.config["test"]["model_file"] != ""
        model.load_state_dict(
            torch.load(
                self.config["test"]["model_file"],
                map_location=torch.device(self.config["test"]["device"]),
            )
        )
        model.eval()

        gt_3d = []
        pred_3d = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_dataloader), 0):
                image_inp = data["image_inp"]
                heatmaps_gt = data["heatmaps_gt"]
                kpt_3d_gt = data["kpt_3d_gt"]
                image_inp = image_inp.to(torch.device(self.config["test"]["device"]))
                heatmaps_gt = heatmaps_gt.to(torch.device(self.config["test"]["device"]))
                kpt_3d_gt = kpt_3d_gt.to(torch.device(self.config["test"]["device"]))

                pred = model(image_inp)
                kpt_3d_pred = pred[1].cpu().numpy()[0]
                kpt_3d_gt = kpt_3d_gt.cpu().numpy()[0]

                gt_3d.append(kpt_3d_gt)
                pred_3d.append(kpt_3d_pred)

        gt_3d = np.array(gt_3d)
        pred_3d = np.array(pred_3d)

        pjpe_3d = np.linalg.norm(pred_3d - gt_3d, axis=2) * 1000
        pck5_3d = pjpe_3d < 5
        pck15_3d = pjpe_3d < 15

        mpjpe_3d = np.mean(pjpe_3d, axis=0)
        mpjpe_3d = np.concatenate([mpjpe_3d, [np.mean(mpjpe_3d)]])

        mpck5_3d = np.mean(pck5_3d, axis=0)
        mpck5_3d = np.concatenate([mpck5_3d, [np.mean(mpck5_3d)]])

        mpck15_3d = np.mean(pck15_3d, axis=0)
        mpck15_3d = np.concatenate([mpck15_3d, [np.mean(mpck15_3d)]])

        df = pd.DataFrame({"mpjpe": mpjpe_3d, "pck@5mm": mpck5_3d, "pck@15mm": mpck15_3d})
        print(df.head(22))

    def validate(self):
        Dataset = import_module(self.config["dataset"]["dataset_name"])
        test_dataset = Dataset(config=self.config, set_type="test")
        test_dataloader = DataLoader(
            test_dataset,
            self.config["training_details"]["batch_size"],
            num_workers=self.config["training_details"]["num_workers"],
        )

        Model = import_module(self.config["model"]["model_name"])
        model = Model(self.config)
        model = model.to(self.config["test"]["device"])
        assert self.config["test"]["model_file"] != ""
        model.load_state_dict(
            torch.load(
                self.config["test"]["model_file"],
                map_location=torch.device(self.config["test"]["device"]),
            )
        )
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                image_name = data["image_name"]
                image_inp = data["image_inp"]
                heatmaps_gt = data["heatmaps_gt"]
                image_inp = image_inp.to(torch.device(self.config["test"]["device"]))
                heatmaps_gt = heatmaps_gt.to(torch.device(self.config["test"]["device"]))

                heatmaps_pred = model(image_inp)
                heatmaps_pred = heatmaps_pred[0].numpy()[0]
                heatmaps_gt = heatmaps_gt.numpy()[0]

                kpt_2d_pred = heatmaps_to_coordinates(
                    heatmaps_pred, self.config["model"]["model_img_size"]
                )
                image = Image.open(image_name[0]).convert("RGB")
                im_width, im_height = image.size
                kpt_2d_pred[:, 0] = kpt_2d_pred[:, 0] * im_width
                kpt_2d_pred[:, 1] = kpt_2d_pred[:, 1] * im_height

                skeleton_overlay = draw_2d_skeleton(image, kpt_2d_pred)
                plt.imshow(skeleton_overlay)

                fig = plt.figure(figsize=(10, 10))
                for j in range(self.config["model"]["n_keypoints"]):
                    fig.add_subplot(5, 5, j + 1)
                    plt.title(f"kpt {j}")
                    plt.imshow(heatmaps_pred[j])

                fig = plt.figure(figsize=(10, 10))
                for j in range(self.config["model"]["n_keypoints"]):
                    fig.add_subplot(5, 5, j + 1)
                    plt.title(f"kpt {j}")
                    plt.imshow(heatmaps_gt[j])

                plt.show()

    def visualize(self):
        image_file = self.config["dataset"]["sample_image_file"]
        camera_param_file = self.config["dataset"]["camera_param_file"]
        global_pose3d_gt_file = self.config["dataset"]["global_pose3d_gt_file"]
        global_mesh_gt_dir = self.config["dataset"]["global_mesh_gt_dir"]

        all_camera_params, all_global_pose3d_gt = init_pose3d_labels(
            camera_param_file, global_pose3d_gt_file
        )

        (
            local_pose3d_gt,
            local_mesh_pts_gt,
            local_mesh_normal_gt,
            cam_param,
            mesh_tri_idx,
        ) = read_data(image_file, all_camera_params, all_global_pose3d_gt, global_mesh_gt_dir)

        visualize_data(image_file, local_pose3d_gt, local_mesh_pts_gt, cam_param, mesh_tri_idx)

    def sample(self):
        images_dir = self.config["dataset"]["images_dir"]
        val_cams_file = self.config["dataset"]["val_cams_file"]

        data = get_train_val_image_paths(images_dir, val_cams_file, True)
        print(len(data), data[0])

        data = get_train_val_image_paths(images_dir, val_cams_file, False)
        print(len(data), data[0])

    def check(self):
        Dataset = import_module(self.config["dataset"]["dataset_name"])
        train_dataset = Dataset(config=self.config, set_type="train")
        train_dataloader = DataLoader(
            train_dataset,
            self.config["training_details"]["batch_size"],
            shuffle=self.config["training_details"]["shuffle"],
            num_workers=self.config["training_details"]["num_workers"],
        )

        for data in train_dataloader:
            # image_inp = data["image_inp"].to(torch.device(config["device"]))
            heatmaps_gt = data["heatmaps_gt"].to(torch.device(self.config["test"]["device"]))
            kpt_2d_gt = data["kpt_2d_gt"].to(torch.device(self.config["test"]["device"]))
            kpt_3d_gt = data["kpt_3d_gt"].to(torch.device(self.config["test"]["device"]))
            drot = data["drot"].to(torch.device(self.config["test"]["device"]))
            dx = data["dx"].to(torch.device(self.config["test"]["device"]))
            dy = data["dy"].to(torch.device(self.config["test"]["device"]))
            brightness_factor = data["brightness_factor"].to(
                torch.device(self.config["test"]["device"])
            )
            contrast_factor = data["contrast_factor"].to(
                torch.device(self.config["test"]["device"])
            )
            sharpness_factor = data["sharpness_factor"].to(
                torch.device(self.config["test"]["device"])
            )
            is_mirror = data["is_mirror"].to(torch.device(self.config["test"]["device"]))
            is_flip = data["is_flip"].to(torch.device(self.config["test"]["device"]))

            # Assume heatmaps_gt is output of the model
            image_name = data["image_name"][0]
            kpt_2d_gt = kpt_2d_gt.cpu().numpy()[0]
            heatmaps_gt = heatmaps_gt.cpu().numpy()[0]
            kpt_3d_gt = kpt_3d_gt.cpu().numpy()[0]
            drot = drot.cpu().numpy()[0]
            dx = dx.cpu().numpy()[0]
            dy = dy.cpu().numpy()[0]
            brightness_factor = brightness_factor.cpu().numpy()[0]
            contrast_factor = contrast_factor.cpu().numpy()[0]
            sharpness_factor = sharpness_factor.cpu().numpy()[0]
            is_mirror = is_mirror.cpu().numpy()[0]
            is_flip = is_flip.cpu().numpy()[0]

            kpt_2d_pred = heatmaps_to_coordinates(
                heatmaps_gt, self.config["model"]["model_img_size"]
            )

            print(
                "name:",
                image_name,
                "drot:",
                drot,
                "dx:",
                dx,
                "dy:",
                dy,
                "brightness_factor:",
                brightness_factor,
                "contrast_factor:",
                contrast_factor,
                "sharpness_factor:",
                sharpness_factor,
                "is_mirror:",
                is_mirror,
                "is_flip:",
                is_flip,
            )
            # Recover original 2d pose
            image = Image.open(image_name).convert("RGB")
            if drot != -1:
                image = image.rotate(drot)
                image = image.transform(image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
                image = ImageEnhance.Brightness(image).enhance(brightness_factor)
                image = ImageEnhance.Contrast(image).enhance(contrast_factor)
                image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)
                if is_mirror:
                    image = ImageOps.mirror(image)
                if is_flip:
                    image = ImageOps.flip(image)

            im_width, im_height = image.size
            kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] * im_width
            kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] * im_height
            kpt_2d_pred[:, 0] = kpt_2d_pred[:, 0] * im_width
            kpt_2d_pred[:, 1] = kpt_2d_pred[:, 1] * im_height

            # Draw 2d pose on image from heatmap
            fig = plt.figure(figsize=(5, 5))
            skeleton_overlay = draw_2d_skeleton(image, kpt_2d_pred)
            plt.imshow(skeleton_overlay)

            # Draw 2d pose on image from ground truth
            fig = plt.figure(figsize=(5, 5))
            skeleton_overlay = draw_2d_skeleton(image, kpt_2d_gt)
            plt.imshow(skeleton_overlay)

            # Plot heatmaps
            fig = plt.figure(figsize=(10, 10))
            for j in range(self.config["model"]["n_keypoints"]):
                fig.add_subplot(5, 5, j + 1)
                plt.title(f"kpt {j}")
                plt.imshow(heatmaps_gt[j])

            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection="3d")
            draw_3d_skeleton_on_ax(kpt_3d_gt, ax)
            ax.set_title("GT 3D joints")

            plt.show()
