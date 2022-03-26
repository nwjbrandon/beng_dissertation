import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.losses import IoULoss


class Pose3DTrainer:
    def __init__(self, model, optimizer, config, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epochs = config["training_details"]["epochs"]
        self.device = config["training_details"]["device"]
        self.loss_file = config["training_details"]["loss_file"]
        self.model_file = config["training_details"]["model_file"]
        self.is_freeze_pose2d = config["training_details"]["is_freeze_pose2d"]

        self.scheduler = scheduler
        self.checkpoint_frequency = 1
        self.criterion1 = IoULoss()
        self.criterion2 = nn.MSELoss(reduction="mean")
        self.criterion3 = nn.MSELoss(reduction="mean")

        if self.is_freeze_pose2d:
            for _, param in self.model.pose_2d.named_parameters():
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

    def train(self, train_dataloader, val_dataloader):
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            log = "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
                epoch + 1,
                self.epochs,
                np.round(self.loss["train"][-1], 10),
                np.round(self.loss["val"][-1], 10),
            )
            print(log)
            with open(self.loss_file, "a") as f:
                f.write(f"{log}\n")

            # reducing LR if no improvement
            if self.scheduler is not None:
                self.scheduler.step(self.loss["val"][-1])

            # saving model
            if (epoch + 1) % self.checkpoint_frequency == 0:
                torch.save(self.model.state_dict(), f"{self.model_file}_{epoch+1}.pth")

        torch.save(self.model.state_dict(), "model_final")
        return self.model

    def _epoch_train(self, dataloader):
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(dataloader), 0):
            image_inp = data["image_inp"].float()
            heatmaps_gt = data["heatmaps_gt"].float()
            kpt_all_gt = data["kpt_all_gt"].float()

            image_inp = image_inp.to(self.device)
            heatmaps_gt = heatmaps_gt.to(self.device)
            kpt_all_gt = kpt_all_gt.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(image_inp)

            losses = self.criterion(pred[0], heatmaps_gt, pred[1], kpt_all_gt)

            losses.backward()
            self.optimizer.step()
            running_loss.append(losses.item())

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _epoch_eval(self, dataloader):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader), 0):
                image_inp = data["image_inp"].float()
                heatmaps_gt = data["heatmaps_gt"].float()
                kpt_all_gt = data["kpt_all_gt"].float()
                image_inp = image_inp.to(self.device)
                heatmaps_gt = heatmaps_gt.to(self.device)
                kpt_all_gt = kpt_all_gt.to(self.device)

                pred = self.model(image_inp)

                losses = self.criterion(pred[0], heatmaps_gt, pred[1], kpt_all_gt)

                running_loss.append(losses.item())

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def criterion(self, heatmaps_pred, heatmaps_gt, kpt_all_pred, kpt_all_gt):
        kpt_2d_gt = kpt_all_gt[:,:,:2]
        kpt_3d_gt = kpt_all_gt[:,:,2:]

        kpt_2d_pred = kpt_all_pred[:,:,:2]
        kpt_3d_pred = kpt_all_pred[:,:,2:]

        loss2 = self.criterion2(kpt_2d_gt, kpt_2d_pred)
        loss3 = self.criterion2(kpt_3d_gt, kpt_3d_pred)

        return loss2 + loss3
