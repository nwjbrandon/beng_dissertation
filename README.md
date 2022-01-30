# Hand Pose Estimation

<p align="center">
  <img alt="inference 1" src="assets/1.png" width="15%">
  <img alt="inference 2" src="assets/2.png" width="15%">
  <img alt="inference 3" src="assets/3.png" width="15%">
  <img alt="inference 4" src="assets/4.png" width="15%">
  <img alt="inference 5" src="assets/5.png" width="15%">
  <img alt="inference 6" src="assets/6.png" width="15%">
</p>

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

## Train
- Train blazenet
```
python3 train.py --cfg cfgs/blazenet_freihand_v1.yml
```
- Sample dataset
```
python3 train.py --cfg cfgs/blazenet_freihand_v1.yml --mode sample
```
- Check dataset preprocessing
```
python3 train.py --cfg cfgs/blazenet_freihand_v1.yml --mode check
```
- Save dataset preprocessing 
```
python3 train.py --cfg cfgs/blazenet_freihand_v1.yml --mode visualize
```
- Validate model inference
```
python3 train.py --cfg cfgs/blazenet_freihand_v1.yml --mode validate
```