# Hand Pose Dataset

## Setup (Ubuntu 18 Python3.8)
- Install dependencies
```
pip install -r requirements.txt
```
- For RTX 3070
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
- Setup the dataset in the folder `data/` 

## Sample Visualization
- Sample name of image file
```
python3 scripts/sample_image_names.py
```
- Sample visualization of pose and image
```
python3 scripts/sample_visualization.py
```
- Sample dataset for training and testing
```
python3 scripts/sample_dataset.py
```

## Train
- Train 2D Pose
```
python3 train_2d.py
```
- Train 3D Pose
```
python3 train_3d.py
```
- Train End To End
```
```

## Test
- Test 2D Pose
```
python3 test_2d.py
```
- Test 3D Pose
```
python3 test_3d.py
```
- Test End To End
```
```

## Demo
- Demo with webcam
```
```

## References
- https://github.com/3d-hand-shape/hand-graph-cnn
- https://github.com/OlgaChernytska/2D-Hand-Pose-Estimation-RGB
- https://github.com/garyzhao/SemGCN