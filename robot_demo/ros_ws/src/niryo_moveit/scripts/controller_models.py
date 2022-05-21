#!/usr/bin/env python3
from __future__ import division, print_function

import sys

sys.path.insert(0, "/home/nwjbrandon/fyp/mycobot_teleop/models")
import math
import threading
import time

import cv2
import niryo_moveit.msg as niryo_moveit_msgs
import numpy as np
import rospy
import torch
import transforms3d
from camera_stream import LoadStreams
from hand_object_detection import HandObjectDetection
from hand_pose_estimation import HandPoseEstimation
from tf.transformations import quaternion_from_euler
from pymycobot.mycobot import MyCobot


class Controller:
    def __init__(self, config):
        self.config = config

        # Create ros nodes
        rospy.init_node("controller_models", anonymous=True)
        self._end_effector_pose_pub = rospy.Publisher(
            "niryo_end_effector_pose", niryo_moveit_msgs.EndEffectorPose, queue_size=1
        )

        # Store hand position
        self.end_effector_position = np.array([0.235, 0.0, 0.115])
        self.hand_position_displacement = np.array([0.0, 0.0, 0.0])

        # Store hand orientation
        self.end_effector_orientation = np.array([3.15, 0.0, 0.0])
        self.hand_orientation_displacement = np.array([0.0, 0.0, 0.0])

        # Store gripper state
        self.gripper_state = 1

        self.r = rospy.Rate(30)

        # Hand object detection model
        self.hand_object_detection = HandObjectDetection(
            weights=self.config["weights"]["hand_object_detection"],
            source=self.config["sources"],
            data=self.config["hand_object_detection_data"],
            conf_thres=self.config["conf_thres"],
            iou_thres=self.config["iou_thres"],
            max_det=self.config["max_det"],
            device=self.config["device"],
        )

        # Hand pose estimation model
        self.hand_pose_estimation = HandPoseEstimation(
            model_file=self.config["weights"]["hand_pose_estimation"]
        )

        # Camera
        self.camera = LoadStreams(
            self.config["sources"],
            img_size=self.hand_object_detection.imgsz,
            stride=self.hand_object_detection.stride,
            auto=self.hand_object_detection.pt,
        )

        self.port = "/dev/ttyUSB0"
        self.baud = 115200
        self.mycobot = MyCobot(self.port, self.baud)
        time.sleep(1)

    def run(self):
        # Send initial pose
        self.send_pose()
        warmup_counter = 0
        with torch.no_grad():
            while not rospy.is_shutdown():
                # Visualise for 10 seconds
                output_imgs = self.detect_hand_pose()
                img = np.vstack(output_imgs)

                if warmup_counter <= 150:
                    warmup_counter += 1
                    img = cv2.putText(
                        img=img,
                        text="Ready",
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=2,
                        color=(0, 0, 255),
                        thickness=3,
                    )
                else:
                    self.send_pose()
                    img = cv2.putText(
                        img=img,
                        text="Start",
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=2,
                        color=(0, 255, 0),
                        thickness=3,
                    )

                cv2.imshow("demo", img)
                if cv2.waitKey(1) == ord("q"):
                    break
                self.r.sleep()

    def detect_height(self, detected_hands, output_imgs):
        # Front camera
        if detected_hands[0] is not None:
            height, width = output_imgs[0].shape[:2]
            xyxy, _, _ = detected_hands[0]
            x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            dx = x - width // 2
            dy = y - height // 2

            self.hand_position_displacement[2] = dy / 1000

    def detect_width(self, detected_hands, output_imgs):
        # Front camera
        if detected_hands[0] is not None:
            height, width = output_imgs[0].shape[:2]
            xyxy, _, _ = detected_hands[0]
            x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            dx = x - width // 2
            dy = y - height // 2

            self.hand_position_displacement[1] = dx / 1000

    def detect_depth(self, detected_hands, output_imgs):
        # Bottom camera
        if detected_hands[1] is not None:
            height, width = output_imgs[1].shape[:2]
            xyxy, _, _ = detected_hands[1]
            x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            dx = x - width // 2
            dy = y - height // 2

            self.hand_position_displacement[0] = dy / 1000

    def detect_orientation(self, detected_hands, output_imgs, im0s, cropped_length=480):
        if detected_hands[1] is not None:
            height, width = im0s[1].shape[:2]

            xyxy, _, _ = detected_hands[1]
            x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            x1, y1, x2, y2 = (
                int(x - cropped_length / 2),
                int(y - cropped_length / 2),
                int(x + cropped_length / 2),
                int(y + cropped_length / 2),
            )
            if x1 <= 0:
                x1, x2 = 0, cropped_length
            if y1 <= 0:
                y1, y2 = 0, cropped_length
            if x2 >= width:
                x1, x2 = width - cropped_length, width
            if y2 >= height:
                y1, y2 = height - cropped_length, height

            cropped = im0s[1][y1:y2, x1:x2]
            inp_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            orientation_vector, grasp_dist, kpt_3d_pred = self.hand_pose_estimation(inp_img)
            x, y, z = orientation_vector
            R = np.concatenate([[x], [y], [z]]).T
            yaw, pitch, roll = transforms3d.euler.mat2euler(R, "sxyz")

            if yaw < 0:
                yaw = math.pi - (math.pi - abs(yaw))
            else:
                yaw = math.pi + (math.pi - abs(yaw)) * 1.5

            if pitch < 0:
                pitch *= 1.5
            else:
                pitch = pitch

            print(grasp_dist)
            if grasp_dist < 0.05:
                self.gripper_state = 0
                self.mycobot.set_gripper_state(1, 100)
            else:
                self.gripper_state = 1
                self.mycobot.set_gripper_state(0, 100)

            # TODO: more fixing here
            # if roll < 0:
            #     roll = math.pi - (math.pi - abs(roll))
            # else:
            #     roll = math.pi + (math.pi - abs(roll))

            # TODO: make it more sensitive
            self.end_effector_orientation[0] = yaw
            self.end_effector_orientation[1] = pitch
            # self.end_effector_orientation[2] = roll

    def detect_hand_pose(self):
        path, im, im0s, vid_cap, s = next(iter(self.camera))
        output_imgs, detected_hands = self.hand_object_detection.detect(path, im, im0s, vid_cap, s)
        self.detect_orientation(detected_hands, output_imgs, im0s)

        self.detect_height(detected_hands, output_imgs)
        self.detect_width(detected_hands, output_imgs)
        self.detect_depth(detected_hands, output_imgs)

        return output_imgs

    def send_pose(self):
        pose = niryo_moveit_msgs.EndEffectorPose()

        # pos
        current_end_effector_position = self.end_effector_position - self.hand_position_displacement
        pose.target.position.x = current_end_effector_position[0]
        pose.target.position.y = current_end_effector_position[1]
        pose.target.position.z = current_end_effector_position[2]

        # orientation
        current_end_effector_orientation = (
            self.end_effector_orientation - self.hand_orientation_displacement
        )
        current_end_effector_orientation = self.end_effector_orientation
        yaw = current_end_effector_orientation[0]
        pitch = current_end_effector_orientation[1]
        roll = current_end_effector_orientation[2]
        q = quaternion_from_euler(yaw, pitch, roll)
        pose.target.orientation.x = q[0]
        pose.target.orientation.y = q[1]
        pose.target.orientation.z = q[2]
        pose.target.orientation.w = q[3]

        # gripper
        pose.gripper_state = self.gripper_state

        print(pose)
        self._end_effector_pose_pub.publish(pose)


if __name__ == "__main__":
    config = {
        "weights": {
            "hand_object_detection": [
                "/home/nwjbrandon/fyp/mycobot_teleop/models/hand_object_detection.pt"
            ],
            "hand_pose_estimation": "/home/nwjbrandon/fyp/mycobot_teleop/models/model_v3_1_6.pth",
        },
        "sources": ["/dev/video0", "/dev/video2"],
        "hand_object_detection_data": "/home/nwjbrandon/fyp/mycobot_teleop/models/yolov5/data/hand.yaml",
        "conf_thres": 0.25,
        "iou_thres": 0.45,
        "max_det": 1,
        "device": "cuda",
    }
    controller = Controller(config=config)
    controller.run()
