FROM osrf/ros:melodic-desktop-full

SHELL ["/bin/bash", "-c"]

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# set enviroment    
ENV ROS_MASTER_URI=http://192.168.50.150:11311
ENV ROS_IP=192.168.50.150
ENV ROS_WS=/ros_ws

# install moveit packages
RUN source /opt/ros/melodic/setup.sh && rosdep update
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install ros-melodic-catkin python-catkin-tools -y
RUN apt install ros-melodic-moveit -y

RUN apt-get update && sudo apt-get upgrade -y
RUN apt-get install python-pip ros-melodic-robot-state-publisher ros-melodic-moveit ros-melodic-rosbridge-suite ros-melodic-joy ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-tf2-web-republisher -y
RUN pip install rospkg jsonpickle
RUN pip install pymycobot 

# build custom ros packages
RUN mkdir -p $ROS_WS/src
WORKDIR $ROS_WS
COPY ros_ws/src src
RUN source /opt/ros/melodic/setup.sh && catkin build

# run ros nodes
CMD source devel/setup.bash && roslaunch niryo_moveit main.launch