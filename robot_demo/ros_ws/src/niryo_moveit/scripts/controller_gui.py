#!/usr/bin/env python
from __future__ import division, print_function

import math
import threading
import tkinter as tk

import niryo_moveit.msg as niryo_moveit_msgs
import rospy
from tf.transformations import quaternion_from_euler

"""
target: 
  position: 
    x: 0.235
    y: 0.0
    z: 0.115
  orientation: 
    x: 0.9999996829318346
    y: 0.0
    z: 0.0
    w: 0.0007963267107332633
gripper_state: 1
"""


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        rospy.init_node("controller_gui", anonymous=True)
        self._end_effector_pose_pub = rospy.Publisher(
            "niryo_end_effector_pose", niryo_moveit_msgs.EndEffectorPose, queue_size=1
        )
        self.pos_slider_start_pos = (75, 100)
        self.pos_control_values = [0] * 6

        # cm
        self.min_pos = -100
        self.max_pos = 100

        # radian scale by 100
        self.min_rad = -int(math.pi * 100)
        self.max_rad = int(math.pi * 100)

        # pose
        self.start_pose = [47, 0, 23, 315, 0, 0]
        self.gripper_state = 1

        self.create_widgets()

        # threading to publish pose
        self.r = rospy.Rate(10)
        self.worker = threading.Thread(target=self.publish)
        self.worker.daemon = True
        self.worker.start()

    def create_widgets(self):
        self.x = tk.Scale(self.master, variable=tk.IntVar(), from_=self.min_pos, to=self.max_pos)
        self.x["command"] = lambda x: self.set_pos_values(0, x)
        self.x.place(x=self.pos_slider_start_pos[0] + 50, y=self.pos_slider_start_pos[1])

        self.y = tk.Scale(self.master, variable=tk.IntVar(), from_=self.min_pos, to=self.max_pos)
        self.y["command"] = lambda x: self.set_pos_values(1, x)
        self.y.place(x=self.pos_slider_start_pos[0] + 100, y=self.pos_slider_start_pos[1])

        self.z = tk.Scale(self.master, variable=tk.IntVar(), from_=self.min_pos, to=self.max_pos)
        self.z["command"] = lambda x: self.set_pos_values(2, x)
        self.z.place(x=self.pos_slider_start_pos[0] + 150, y=self.pos_slider_start_pos[1])

        self.yaw = tk.Scale(self.master, variable=tk.IntVar(), from_=self.min_rad, to=self.max_rad)
        self.yaw["command"] = lambda x: self.set_pos_values(3, x)
        self.yaw.place(x=self.pos_slider_start_pos[0] + 200, y=self.pos_slider_start_pos[1])

        self.pitch = tk.Scale(
            self.master, variable=tk.IntVar(), from_=self.min_rad, to=self.max_rad
        )
        self.pitch["command"] = lambda x: self.set_pos_values(4, x)
        self.pitch.place(x=self.pos_slider_start_pos[0] + 250, y=self.pos_slider_start_pos[1])

        self.row = tk.Scale(self.master, variable=tk.IntVar(), from_=self.min_rad, to=self.max_rad)
        self.row["command"] = lambda x: self.set_pos_values(5, x)
        self.row.place(x=self.pos_slider_start_pos[0] + 300, y=self.pos_slider_start_pos[1])

        self.start_btn = tk.Button(self.master, text="Return to Start")
        self.start_btn["command"] = lambda: self.set_pose(self.start_pose)
        self.start_btn.place(x=100, y=250)

        self.start_btn = tk.Button(self.master, text="Open Gripper")
        self.start_btn["command"] = lambda: self.set_gripper_state(1)
        self.start_btn.place(x=250, y=250)

        self.start_btn = tk.Button(self.master, text="Close Gripper")
        self.start_btn["command"] = lambda: self.set_gripper_state(0)
        self.start_btn.place(x=400, y=250)

    def set_pos_values(self, joint_idx, pos):
        self.pos_control_values[joint_idx] = int(pos)

    def set_pose(self, pose):
        self.x.set(pose[0])
        self.y.set(pose[1])
        self.z.set(pose[2])
        self.yaw.set(pose[3])
        self.pitch.set(pose[4])
        self.row.set(pose[5])
        self.pos_control_values = pose[:]

    def set_gripper_state(self, gripper_state):
        self.gripper_state = gripper_state

    def publish(self):
        self.set_pose(self.start_pose)
        while not rospy.is_shutdown():
            pose = niryo_moveit_msgs.EndEffectorPose()

            # pos
            pose.target.position.x = self.pos_control_values[0] / 200
            pose.target.position.y = self.pos_control_values[1] / 200
            pose.target.position.z = self.pos_control_values[2] / 200
            print(pose.target.position)
            # orientation
            yaw = self.pos_control_values[3] / 100
            pitch = self.pos_control_values[4] / 100
            roll = self.pos_control_values[5] / 100
            # print(yaw, pitch, roll)
            q = quaternion_from_euler(yaw, pitch, roll)
            pose.target.orientation.x = q[0]
            pose.target.orientation.y = q[1]
            pose.target.orientation.z = q[2]
            pose.target.orientation.w = q[3]

            # gripper
            pose.gripper_state = self.gripper_state
            self._end_effector_pose_pub.publish(pose)

            self.r.sleep()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")
    root.resizable(width=False, height=False)
    app = Application(master=root)
    app.mainloop()
