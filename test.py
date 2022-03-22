import torch

kpt3d = torch.rand(2, 4, 3)

joint1 = torch.tensor([0, 1, 0])
joint2 = torch.tensor([1, 2, 3])
bones = kpt3d[:, joint1, :] - kpt3d[:, joint2, :]

# print(kpt3d.shape)

camera_instr = torch.tensor([
    [
        [120, 0, 0],
        [0, 120, 0],
        [0, 0, 1]
    ],
    [
        [120, 0, 0],
        [0, 120, 0],
        [0, 0, 1]
    ]
]).float()
# print(camera_instr.shape)
kpt2d = torch.bmm(kpt3d, camera_instr)
print(kpt2d.shape)

inv = kpt2d[:, :, 2].unsqueeze(1)
print(inv.shape)
kpt2d = kpt2d / inv
# kpt2d = torch.div(kpt2d, inv)
print(kpt2d)