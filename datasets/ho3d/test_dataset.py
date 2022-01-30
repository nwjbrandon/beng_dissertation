import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader

from datasets.ho3d.data_utils import (
    draw_2d_skeleton,
    draw_3d_skeleton_on_ax,
    pad_to_square,
    visualize_data
)
from datasets.ho3d.dataset_utils import get_train_val_image_paths, heatmaps_to_coordinates
from utils.io import import_module


class TestDataset:
    def __init__(self, config):
        self.config = config

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
        sample_image = self.config["dataset"]["sample_image"]
        sample_meta = self.config["dataset"]["sample_meta"]

        visualize_data(sample_image, sample_meta)

    def sample(self):
        data_dir = self.config["dataset"]["data_dir"]

        data = get_train_val_image_paths(data_dir, is_training=True)
        print(len(data), data[0])

        data = get_train_val_image_paths(data_dir, is_training=False)
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
            brightness_factor = data["brightness_factor"].to(
                torch.device(self.config["test"]["device"])
            )
            contrast_factor = data["contrast_factor"].to(
                torch.device(self.config["test"]["device"])
            )
            sharpness_factor = data["sharpness_factor"].to(
                torch.device(self.config["test"]["device"])
            )
            red = data["red"].to(torch.device(self.config["test"]["device"]))
            green = data["green"].to(torch.device(self.config["test"]["device"]))
            blue = data["blue"].to(torch.device(self.config["test"]["device"]))

            # Assume heatmaps_gt is output of the model
            image_name = data["image_name"][0]
            kpt_2d_gt = kpt_2d_gt.cpu().numpy()[0]
            heatmaps_gt = heatmaps_gt.cpu().numpy()[0]
            kpt_3d_gt = kpt_3d_gt.cpu().numpy()[0]

            kpt_2d_pred = heatmaps_to_coordinates(
                heatmaps_gt, self.config["model"]["model_img_size"]
            )
            brightness_factor = brightness_factor.cpu().numpy()[0]
            contrast_factor = contrast_factor.cpu().numpy()[0]
            sharpness_factor = sharpness_factor.cpu().numpy()[0]
            red = red.cpu().numpy()[0]
            green = green.cpu().numpy()[0]
            blue = blue.cpu().numpy()[0]

            print(
                "name:",
                image_name,
                "brightness_factor:",
                brightness_factor,
                "contrast_factor:",
                contrast_factor,
                "sharpness_factor:",
                sharpness_factor,
                "red:",
                red,
                "green:",
                green,
                "blue:",
                blue,
            )
            # Recover original 2d pose
            image = Image.open(image_name).convert("RGB")
            image = pad_to_square(image, (red, green, blue))

            im_width, im_height = image.size
            kpt_2d_gt[:, 0] = kpt_2d_gt[:, 0] * im_width
            kpt_2d_gt[:, 1] = kpt_2d_gt[:, 1] * im_height
            kpt_2d_pred[:, 0] = kpt_2d_pred[:, 0] * im_width
            kpt_2d_pred[:, 1] = kpt_2d_pred[:, 1] * im_height

            if brightness_factor != -1:
                image = ImageEnhance.Brightness(image).enhance(brightness_factor)
                image = ImageEnhance.Contrast(image).enhance(contrast_factor)
                image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

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
            plt.title(f"kpt {22}")
            plt.imshow(1 - heatmaps_gt.max(axis=0))

            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection="3d")
            kpt_3d_gt = kpt_3d_gt - kpt_3d_gt[0]
            draw_3d_skeleton_on_ax(kpt_3d_gt * 10, ax)
            ax.set_title("GT 3D joints")

            plt.show()
