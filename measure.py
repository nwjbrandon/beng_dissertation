import time

import torch

from models.blazenet_model_3d_v1 import Pose3dModel


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


config = {
    "model": {
        "n_keypoints": 21,
        "model_file": "blazenet3d_combined_v1.pth",
        "device": "cpu",
        "raw_image_size": 256,
        "model_img_size": 128,
    }
}
model = Pose3dModel(config)
model = model.to(config["model"]["device"])

params = get_num_params(model.pose_3d)

image_inp = torch.rand(1, 3, 256, 256).to(config["model"]["device"])
print(image_inp.shape)

for i in range(100):
    _ = model(image_inp)


start_time = time.time()
for _ in range(300):
    _ = model(image_inp)
end_time = time.time()

duration = end_time - start_time
print("duration:", duration)
print("fps:", 300 / duration)
print("params:", params)
