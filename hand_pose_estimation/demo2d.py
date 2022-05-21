import cv2
import torch
from PIL import Image
from torchvision import transforms

from datasets.freihand.data_utils import draw_2d_skeleton
from datasets.freihand.dataset_utils import heatmaps_to_coordinates
from models.blazenet_model_v5 import Pose2dModel

config = {
    "model": {
        "n_keypoints": 21,
        "model_file": "exp/model_1.pth",
        "device": "cuda",
        "raw_image_size": 256,
        "model_img_size": 128,
    }
}
model = Pose2dModel(config)
model = model.to(config["model"]["device"])
model.load_state_dict(
    torch.load(config["model"]["model_file"], map_location=torch.device(config["model"]["device"]),)
)

image_transform = transforms.Compose(
    [transforms.Resize(config["model"]["raw_image_size"]), transforms.ToTensor(),]
)

# define a video capture object
vid = cv2.VideoCapture(0)

with torch.no_grad():
    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:480, :480]
        frame = Image.fromarray(frame).convert("RGB")
        im_width, im_height = frame.size

        image_inp = image_transform(frame)
        image_inp = image_inp.unsqueeze(0)
        image_inp = image_inp.float().to(config["model"]["device"])
        pred = model(image_inp)
        heatmaps_pred = pred[0].cpu().numpy()[0]

        kpt_2d_pred = heatmaps_to_coordinates(heatmaps_pred, config["model"]["model_img_size"])
        kpt_2d_pred[:, 0] = kpt_2d_pred[:, 0] * im_width
        kpt_2d_pred[:, 1] = kpt_2d_pred[:, 1] * im_height
        skeleton_overlay = draw_2d_skeleton(frame, kpt_2d_pred)

        skeleton_overlay = cv2.cvtColor(skeleton_overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", skeleton_overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

vid.release()
cv2.destroyAllWindows()
