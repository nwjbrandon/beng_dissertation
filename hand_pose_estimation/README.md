# Hand Pose Estimation

## Real-Time Inference
<p align="center">
  <img alt="inference 1" src="assets/pose2d.png" width="19.5%">
  <img alt="inference 2" src="assets/pose3d.png" width="20%">
</p>

- Watch [live demo](https://www.youtube.com/watch?v=uNvQX3RaTe0)

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

## Train 2D Model
- Train combined dataset
```
python3 train_v2.py --cfg cfgs/combined_embedding.yml
```
- Train combined dataset
```
python3 train.py --cfg cfgs/ntu_embedding.yml
```

## Train 3D Model
- Train model
```
python3 train.py --cfg cfgs/ntu/ntu_v3_1_6.yml
```
- Sample dataset
```
python3 train.py --cfg cfgs/ntu/ntu_v3_1_6.yml --mode sample
```
- Check dataset preprocessing
```
python3 train.py --cfg cfgs/ntu/ntu_v3_1_6.yml --mode check
```
- Save dataset preprocessing 
```
python3 train.py --cfg cfgs/ntu/ntu_v3_1_6.yml --mode visualize
```
- Validate model inference
```
python3 train.py --cfg cfgs/ntu/ntu_v3_1_6.yml --mode validate
```
- evaluate model inference
```
python3 train.py --cfg cfgs/ntu/ntu_v3_1_6.yml --mode evaluate
```
