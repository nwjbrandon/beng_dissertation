import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.io import import_module, load_config

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    cudnn.enabled = True


np.random.seed(42)


def train(config_file):
    config = load_config(config_file)
    print(config)

    Dataset = import_module(config["dataset"]["dataset_name"])
    train_dataset = Dataset(config=config, set_type="train")
    train_dataloader = DataLoader(
        train_dataset,
        config["training_details"]["batch_size"],
        num_workers=config["training_details"]["num_workers"],
    )

    val_dataset = Dataset(config=config, set_type="val")
    val_dataloader = DataLoader(
        val_dataset,
        config["training_details"]["batch_size"],
        shuffle=config["training_details"]["shuffle"],
        num_workers=config["training_details"]["num_workers"],
    )

    Model = import_module(config["model"]["model_name"])
    model = Model(config)
    model = model.to(config["training_details"]["device"])
    if config["model"]["model_file"] != "":
        print("Loading:", config["model"]["model_file"])
        model.load_state_dict(
            torch.load(
                config["model"]["model_file"],
                map_location=torch.device(config["training_details"]["device"]),
            )
        )
    Optimizer = import_module(config["training_details"]["optimizer"])
    optimizer = Optimizer(model.parameters(), lr=config["training_details"]["learning_rate"])
    Scheuler = import_module(config["training_details"]["scheduler"])
    scheduler = Scheuler(
        optimizer=optimizer,
        factor=config["training_details"]["scheduler_factor"],
        patience=config["training_details"]["scheduler_patience"],
        verbose=config["training_details"]["scheduler_verbose"],
        threshold=config["training_details"]["scheduler_threshold"],
    )
    Trainer = import_module(config["training_details"]["trainer"])
    trainer = Trainer(model, optimizer, config, scheduler)

    trainer.train(train_dataloader, val_dataloader)


def test(config_file, mode):
    config = load_config(config_file)
    print(config)
    TestDataset = import_module(config["test"]["test_class"])
    test_dataset = TestDataset(config)

    if mode == "sample":
        test_dataset.sample()
    elif mode == "visualize":
        test_dataset.visualize()
    elif mode == "check":
        test_dataset.check()
    elif mode == "validate":
        test_dataset.validate()
    elif mode == "evaluate":
        test_dataset.evaluate()
    else:
        print("Not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Pose Estimator Train And Val")
    parser.add_argument(
        "--cfg", type=str, default="cfgs/unet_v1.yml", help="specify config file in cfgs"
    )
    parser.add_argument("--mode", type=str, default="train", help="specify mode to run")

    args = parser.parse_args()
    print(args)
    config_file = args.cfg
    mode = args.mode

    if mode == "train":
        train(config_file)
    else:
        test(config_file, mode=mode)
