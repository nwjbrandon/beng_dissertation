import matplotlib.pyplot as plt
import numpy as np

from datasets.ntu.data_utils import draw_3d_skeleton_on_ax


# https://itectec.com/matlab/matlab-how-to-calculate-roll-pitch-and-yaw-from-xyz-coordinates-of-3-planar-points/
def compute_hand_orientation(p1, p2, p3):
    p1, p2, p3 = p1[:3], p2[:3], p3[:3]
    x = (p1 + p2) / 2 - p3
    v1, v2 = p2 - p1, p3 - p1
    z = np.cross(v1, v2)
    z = z / np.linalg.norm(z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z
    # R = np.concatenate([[x], [y], [z]]).T
    # yaw, pitch, roll = transforms3d.euler.mat2euler(R, "sxyz")
    # print("yaw:", yaw)
    # print("pitch:", pitch)
    # print("roll:", roll)
    # return yaw, pitch, roll


kpt_3d_gt = np.array(
    [
        [-0.02857826, -0.00654114, -0.05728105],
        [-0.01741742, 0.0208489, -0.05616114],
        [0.00557601, 0.04172446, -0.05246815],
        [0.03181977, 0.03519288, -0.04295793],
        [0.04097156, 0.01220687, -0.03509707],
        [-0.0010074, 0.02179615, -0.00438282],
        [0.03072128, 0.03354137, -0.02398562],
        [0.01985451, 0.03186024, -0.03728086],
        [0.00673957, 0.02734193, -0.02137811],
        [0.0, 0.0, 0.0],
        [0.03384184, 0.01630178, -0.02400598],
        [0.01868615, 0.01590482, -0.04057344],
        [0.00156216, 0.00828345, -0.0315297],
        [0.00579437, -0.01687093, -0.00284027],
        [0.03431129, 0.00039458, -0.02352602],
        [0.01737777, 0.00120126, -0.03985255],
        [-0.00027849, -0.00534471, -0.03752548],
        [0.01559031, -0.03018046, -0.00491233],
        [0.03315909, -0.03092016, 0.01552987],
        [0.04336386, -0.03063927, 0.02779464],
        [0.05008544, -0.0323694, 0.04195965],
    ]
)

print(kpt_3d_gt)


p1 = kpt_3d_gt[5]
p2 = kpt_3d_gt[17]
p3 = kpt_3d_gt[0]
print(p1)
x, y, z = compute_hand_orientation(p1, p2, p3)
print(x, y, z)

p4 = kpt_3d_gt[9]
p4x = p4 + x * 0.1
p4y = p4 + y * 0.1
p4z = p4 + z * 0.1




fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection="3d")
draw_3d_skeleton_on_ax(kpt_3d_gt, ax)
ax.set_title("GT 3D joints")
ax.plot(
    [p4[0], p4x[0]],
    [p4[1], p4x[1]],
    [p4[2], p4x[2]],
    zdir="z",
    c="red",
)
ax.plot(
    [p4[0], p4y[0]],
    [p4[1], p4y[1]],
    [p4[2], p4y[2]],
    zdir="z",
    c="green",
)
ax.plot(
    [p4[0], p4z[0]],
    [p4[1], p4z[1]],
    [p4[2], p4z[2]],
    zdir="z",
    c="blue",
)

plt.show()
