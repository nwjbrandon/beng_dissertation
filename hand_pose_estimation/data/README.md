# Datasets

## NTU
- https://github.com/3d-hand-shape/hand-graph-cnn
```
(fyp) nwjbrandon@chloe:~/fyp/hand_pose_estimation$ tree -L 1 -f data/synthetic_train_val/
data/synthetic_train_val
├── data/synthetic_train_val/3D_labels
├── data/synthetic_train_val/hand_3D_mesh
└── data/synthetic_train_val/images

3 directories, 0 files
```

## Freihand
- https://github.com/lmb-freiburg/freihand
```
(fyp) nwjbrandon@chloe:~/fyp/hand_pose_estimation$ tree -L 3 -f data/freihand/
data/freihand
├── data/freihand/train
│   ├── data/freihand/train/evaluation
│   │   └── data/freihand/train/evaluation/rgb
│   ├── data/freihand/train/evaluation_K.json
│   ├── data/freihand/train/evaluation_scale.json
│   ├── data/freihand/train/training
│   │   ├── data/freihand/train/training/mask
│   │   └── data/freihand/train/training/rgb
│   ├── data/freihand/train/training_K.json
│   ├── data/freihand/train/training_mano.json
│   ├── data/freihand/train/training_scale.json
│   ├── data/freihand/train/training_verts.json
│   └── data/freihand/train/training_xyz.json
└── data/freihand/val
    ├── data/freihand/val/evaluation
    │   ├── data/freihand/val/evaluation/anno
    │   ├── data/freihand/val/evaluation/colormap
    │   ├── data/freihand/val/evaluation/facemap
    │   ├── data/freihand/val/evaluation/rgb
    │   ├── data/freihand/val/evaluation/segmap
    │   └── data/freihand/val/evaluation/verts_offset_map
    ├── data/freihand/val/evaluation_errors.json
    ├── data/freihand/val/evaluation_K.json
    ├── data/freihand/val/evaluation_mano.json
    ├── data/freihand/val/evaluation_scale.json
    ├── data/freihand/val/evaluation_verts.json
    └── data/freihand/val/evaluation_xyz.json

14 directories, 13 files
```

# HO3D
- https://github.com/bardiadoosti/HOPE/blob/master/datasets/ho/make_data.py
```
(fyp) nwjbrandon@chloe:~/fyp/hand_pose_estimation$ tree -L 1 -f data/ho3d/
data/ho3d
├── data/ho3d/evaluation
├── data/ho3d/evaluation.txt
├── data/ho3d/train
└── data/ho3d/train.txt

2 directories, 2 files
```