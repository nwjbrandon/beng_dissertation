import sys
from queue import Queue

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from camera import RS
from pose2d.inference import Pose2D
from vis import draw_kpts_2d, draw_kpts_3d, vis_inference

MIN_DEPTH_THRESHOLD = 0.3
MAX_DEPTH_THRESHOLD = 3
INVALID_DEPTH_CHANGE_THRESHOLD = 0.06

START_PIXEL = 80
END_PIXEL = 80 + 480


def main():
    prev_2d_pose = None
    prev_3d_pose = None

    rs_camera = RS()
    pose2d = Pose2D("./body_pose.pth", False)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=480, height=480, left=0, top=600)

    # Test 3s
    for _ in range(90):
        color_frame, depth_frame = rs_camera.get_rs_frame()
        curr_2d_pose, curr_3d_pose, images = process_frame(
            pose2d, vis, color_frame, depth_frame, prev_2d_pose, prev_3d_pose
        )
        cv2.putText(
            images, "Ready", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", images)

        prev_2d_pose = curr_2d_pose
        prev_3d_pose = curr_3d_pose

        if cv2.waitKey(1) == ord("q"):
            sys.exit()

    # Save 10s
    for i in range(500):
        color_frame, depth_frame = rs_camera.get_rs_frame()
        color_image = np.asanyarray(color_frame.get_data()).copy()[:, START_PIXEL:END_PIXEL, :]
        curr_2d_pose, curr_3d_pose, images = process_frame(
            pose2d, vis, color_frame, depth_frame, prev_2d_pose, prev_3d_pose
        )
        cv2.putText(
            images, "Start", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA
        )
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", images)

        prev_2d_pose = curr_2d_pose
        prev_3d_pose = curr_3d_pose

        if cv2.waitKey(1) == ord("q"):
            sys.exit()

        final_2d_pose = curr_2d_pose.copy()
        final_2d_pose[:, 0] -= START_PIXEL
        final_3d_pose = curr_3d_pose.copy()

        cv2.imwrite(f"data/frame_{i}.jpg", color_image)
        np.save(f"data/pose2d_{i}.npy", final_2d_pose)
        np.save(f"data/pose3d_{i}.npy", final_3d_pose)

    # finally:
    vis.close()
    cv2.destroyAllWindows()
    rs_camera.stop()


def process_frame(pose2d, vis, color_frame, depth_frame, prev_2d_pose, prev_3d_pose):
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Get 2D poses
    kpts_2d = pose2d.inference(color_image)

    if kpts_2d is None:
        if prev_2d_pose is None or prev_3d_pose is None:
            raise "Starting pose is not detected"

    # Get 3D poses
    camera_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    curr_2d_pose, curr_3d_pose = compute_current_pose(
        color_image, kpts_2d, depth_frame, camera_intrinsics, prev_2d_pose, prev_3d_pose
    )

    # Draw poses
    color_image = draw_kpts_2d(color_image, curr_2d_pose[:, :2])
    color_image = cv2.line(
        color_image, (START_PIXEL, 0), (START_PIXEL, 480), (0, 0, 255), thickness=3
    )
    color_image = cv2.line(color_image, (END_PIXEL, 0), (END_PIXEL, 480), (0, 0, 255), thickness=3)

    draw_kpts_3d(vis, curr_3d_pose)
    images = vis_inference(color_image, depth_image)

    return curr_2d_pose, curr_3d_pose, images


def compute_current_pose(
    color_image, kpts_2d, depth_frame, camera_intrinsics, prev_2d_pose, prev_3d_pose
):
    curr_2d_pose = []
    curr_3d_pose = []
    for kpt_idx, kpt_2d in enumerate(kpts_2d):
        x, y = int(kpt_2d[0]), int(kpt_2d[1])
        if x == -1 or y == -1:
            kpt_2d, kpt_3d = compute_hidden_kpt(prev_2d_pose, prev_3d_pose, kpt_idx)
        else:
            kpt_2d, kpt_3d = compute_visible_kpt(
                depth_frame, x, y, camera_intrinsics, kpt_idx, prev_3d_pose
            )

        color_image = cv2.circle(color_image, (kpt_2d[0], kpt_2d[1]), 1, (0, 255, 0), 2)
        curr_2d_pose.append(kpt_2d)
        curr_3d_pose.append(kpt_3d)

    curr_2d_pose = np.array(curr_2d_pose)
    curr_3d_pose = np.array(curr_3d_pose)
    return curr_2d_pose, curr_3d_pose


def compute_hidden_kpt(prev_2d_pose, prev_3d_pose, kpt_idx):
    valid = 0
    x, y = prev_2d_pose[kpt_idx][:2]
    kpt_2d = [x, y, valid]
    kpt_3d = prev_3d_pose[kpt_idx]
    return kpt_2d, kpt_3d


def compute_visible_kpt(depth_frame, x, y, camera_intrinsics, kpt_idx, prev_3d_pose):
    kpt_2d = [x, y, 1]
    if prev_3d_pose is None:
        kpt_3d, _ = compute_3d_kpt(depth_frame, x, y, camera_intrinsics)
    else:
        prev_kpt_3d = prev_3d_pose[kpt_idx]
        kpt_3d = get_nearest_valid_3d_kpt(depth_frame, x, y, camera_intrinsics, prev_kpt_3d)
        if kpt_3d is None:
            kpt_3d = prev_kpt_3d
    return kpt_2d, kpt_3d


def get_nearest_valid_3d_kpt(depth_frame, x, y, camera_intrinsics, prev_kpt_3d, max_cost=3):
    steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    q = Queue()
    node = (x, y, 0)
    q.put(node)

    while q.qsize():
        node = q.get()
        x, y, cost = node
        curr_kpt_3d, depth = compute_3d_kpt(depth_frame, x, y, camera_intrinsics)

        if depth < MIN_DEPTH_THRESHOLD or depth > MAX_DEPTH_THRESHOLD:
            for step in steps:
                x_, y_, cost_ = x + step[0], y + step[1], cost + 1
                if cost_ > max_cost:
                    continue
                node_ = (x_, y_, cost_)
                q.put(node_)
        else:
            depth_change = np.abs(curr_kpt_3d[2] - prev_kpt_3d[2])
            if depth_change < INVALID_DEPTH_CHANGE_THRESHOLD:
                return curr_kpt_3d
    return None


def compute_3d_kpt(depth_frame, x, y, camera_intrinsics):
    depth = depth_frame.as_depth_frame().get_distance(x, y)
    kpt_3d = rs.rs2_deproject_pixel_to_point(camera_intrinsics, [x, y], depth)
    return kpt_3d, depth


if __name__ == "__main__":
    main()
