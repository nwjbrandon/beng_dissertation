import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.datasets import LoadStreams
from utils.general_utils import (check_img_size, non_max_suppression,
                                 scale_coords)
from utils.plot_utils import Annotator, Colors

from models.common import DetectMultiBackend


class HandObjectDetection:
    def __init__(
        self,
        weights=["./best.pt"],
        source="0",
        data="data/hand.yaml",
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device="cpu",
    ):
        self.weights = weights
        self.source = source
        self.data = data
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device

        self.imgsz = [640, 640]
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.visualize = False
        self.line_thickness = 3
        self.hide_labels = False
        self.hide_conf = False
        self.half = False
        self.dnn = False

        self.colors = Colors()

        # Load model
        self.device = torch.device(self.device)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, dnn=self.dnn, data=self.data
        )
        self.stride, self.names, self.pt = (
            self.model.stride,
            self.model.names,
            self.model.pt,
        )
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Run inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)  # warmup

    def detect(self, path, im, im0s, vid_cap, s):
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0

        # Inference
        pred = self.model(im, augment=self.augment, visualize=self.visualize)

        # NMS
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            self.classes,
            self.agnostic_nms,
            max_det=self.max_det,
        )

        detected_hands = []

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=self.colors(c, True))
                    detected_hands.append((xyxy, c, conf))

        im0 = annotator.result()
        return im0, detected_hands


if __name__ == "__main__":
    weights = ["best.pt"]
    source = "0"
    data = "data/hand.yaml"
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 1000
    device = "cpu"
    detector = HandObjectDetection(
        weights=weights,
        source=source,
        data=data,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        device=device,
    )
    dataset = LoadStreams(
        detector.source, img_size=detector.imgsz, stride=detector.stride, auto=detector.pt
    )
    with torch.no_grad():
        for path, im, im0s, vid_cap, s in dataset:
            im0, _ = detector.detect(path, im, im0s, vid_cap, s)

            # Stream results
            cv2.imshow("demo", im0)
            if cv2.waitKey(1) == ord("q"):
                break
