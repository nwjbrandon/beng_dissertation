import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.ntu_real.data_utils import (
    draw_2d_skeleton,
    draw_3d_skeleton_on_ax,
    init_pose3d_labels,
    read_data,
    visualize_data
)
from datasets.ntu_real.dataset_utils import get_train_val_image_paths, heatmaps_to_coordinates
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
                heatmaps_pred = heatmaps_pred.numpy()[0]
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

            # Assume heatmaps_gt is output of the model
            image_name = data["image_name"][0]
            kpt_2d_gt = kpt_2d_gt.cpu().numpy()[0]
            heatmaps_gt = heatmaps_gt.cpu().numpy()[0]
            kpt_3d_gt = kpt_3d_gt.cpu().numpy()[0]
            drot = drot.cpu().numpy()[0]

            kpt_2d_pred = heatmaps_to_coordinates(
                heatmaps_gt, self.config["model"]["model_img_size"]
            )

            print(
                "name:", image_name, "drot:", drot,
            )
            # Recover original 2d pose
            image = Image.open(image_name).convert("RGB")
            if drot != -1:
                image = image.rotate(drot)
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
            draw_3d_skeleton_on_ax(kpt_3d_gt * 100, ax)
            ax.set_title("GT 3D joints")

            plt.show()
