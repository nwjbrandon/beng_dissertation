# CG4001 FYP - Telerobots

## Setup (Ubuntu 20 Noetic)
- Install ROS dependencies
```
rosdep update
sudo apt update
sudo apt dist-upgrade
sudo apt install ros-noetic-catkin python3-catkin-tools python3-osrf-pycommon
sudo apt install python3-wstool
```
- Install ROS Moveit
```
mkdir -p ~/ws_moveit/src # Install in the
cd ~/ws_moveit/src
wstool init .
wstool merge -t . https://raw.githubusercontent.com/ros-planning/moveit/master/moveit.rosinstall
wstool remove  moveit_tutorials  # this is cloned in the next section
wstool update -t .
```
- Install Unity
```
./UnityHub.AppImage unityhub://2020.3.11f1/99c7afb366b3
```
- Install ROS dependencies
```
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python3-pip ros-noetic-robot-state-publisher ros-noetic-moveit ros-noetic-rosbridge-suite ros-noetic-joy ros-noetic-ros-control ros-noetic-ros-controllers
sudo -H pip3 install rospkg jsonpickle
sudo apt install python3-tk
```
- Clone and build repo
```
cd ~/ros_ws
catkin build
```
- Install Python dependencies
```
pip3 install -r requirements.txt
```

## Run
- Start unity simulator
- Select Demo Scene
- Build ROS messages (if necessary)
- Start planner node using Docker (Refer to Docker section to build images)
```
sudo bash build.sh
sudo bash run.sh
```
- Start controller node (Select one of the mode)
```
roslaunch niryo_moveit controller_gui.launch # control on local machine using gui
roslaunch niryo_moveit controller_models.launch # control on local machine using mediapipe models
roslaunch niryo_moveit controller_remote.launch # control remotely by receiving sqs messages
```
- Run remotely with AWS SQS
```
python3 controller_gui.py
python3 controller_models.py
```

## Docker
- Install docker
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
```
sudo apt-get update
```
```
sudo apt-get install -y nvidia-docker2
```
```
sudo systemctl restart docker
```
- Build image
```
sudo bash build.sh
```
- Run image
```
sudo bash run.sh
```