import cv2
import numpy as np
import open3d as o3d

JOINTS_RIGHT = [5, 6, 7]
JOINTS_LEFT = [2, 3, 4]
PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    1,
    5,
    6,
]
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
RADIUS = 2


def draw_kpts_2d(img, keypoints):
    for idx, keypoint in enumerate(keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        if idx in JOINTS_LEFT:
            cv2.circle(img, (x, y), RADIUS, RED, -1)
            cv2.putText(
                img, str(idx), (x - 40, y - 30), FONT, FONT_SCALE, RED, THICKNESS,
            )
        elif idx in JOINTS_RIGHT:
            cv2.circle(img, (x, y), RADIUS, BLUE, -1)
            cv2.putText(
                img, str(idx), (x - 40, y - 30), FONT, FONT_SCALE, BLUE, THICKNESS,
            )
        else:
            cv2.circle(img, (x, y), RADIUS, GREEN, -1)
            cv2.putText(
                img, str(idx), (x - 40, y - 30), FONT, FONT_SCALE, GREEN, THICKNESS,
            )

    for j, parent in enumerate(PARENTS):
        if parent == -1:
            continue
        if j in JOINTS_RIGHT:
            col = BLUE
        elif j in JOINTS_LEFT:
            col = RED
        else:
            col = GREEN
        pt1 = int(keypoints[j][0]), int(keypoints[j][1])
        pt2 = int(keypoints[parent][0]), int(keypoints[parent][1])
        img = cv2.line(img, pt1, pt2, col, thickness=2)

    return img


def vis_inference(color_image, depth_image):

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(
            color_image,
            dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
            interpolation=cv2.INTER_AREA,
        )
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    return images


def draw_kpts_3d(vis, points):
    lines = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.clear_geometries()
    vis.add_geometry(line_set)
    vis.update_geometry(line_set)
    vis.poll_events()
