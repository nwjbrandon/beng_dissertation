import numpy as np
import torch
from tqdm import tqdm

from utils.losses import HeatmapLoss


class Pose2DTrainer:
    def __init__(self, model, optimizer, config, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epochs = config["training_details"]["epochs"]
        self.device = config["training_details"]["device"]
        self.loss_file = config["training_details"]["loss_file"]
        self.model_file = config["training_details"]["model_file"]
        self.scheduler = scheduler
        self.checkpoint_frequency = 1
        self.criterion1 = HeatmapLoss()

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

            image_inp = image_inp.to(self.device)
            heatmaps_gt = heatmaps_gt.to(self.device)

            self.optimizer.zero_grad()

            heatmaps_pred = self.model(image_inp)

            losses = self.criterion(heatmaps_pred[0], heatmaps_gt)

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
                image_inp = image_inp.to(self.device)
                heatmaps_gt = heatmaps_gt.to(self.device)

                heatmaps_pred = self.model(image_inp)

                losses = self.criterion(heatmaps_pred[0], heatmaps_gt)

                running_loss.append(losses.item())

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def criterion(self, heatmaps_pred, heatmaps_gt):
        loss1 = self.criterion1(heatmaps_pred, heatmaps_gt)
        losses = loss1
        return losses
