#!/usr/bin/env python

from __future__ import print_function

import sys
import time

import actionlib
import niryo_moveit.msg as niryo_moveit_msgs
import rospy
from Queue import Queue

from pymycobot.mycobot import MyCobot

joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


class Robot:
    def __init__(self):
        rospy.init_node("niryo_robot")
        self.port = "/dev/ttyUSB0"
        self.baud = 115200
        self.mycobot = MyCobot(self.port, self.baud)
        time.sleep(1)

        self.q = Queue()
        self.r = rospy.Rate(30)

        self._niryo_trajectory_server = actionlib.SimpleActionServer(
            "niryo_trajectory",
            niryo_moveit_msgs.ExecuteTrajectoryAction,
            execute_cb=self.execute_niryo_trajectory,
            auto_start=True,
        )
        self._niryo_trajectory_waypoint_pub = rospy.Publisher(
            "niryo_trajectory_waypoint", niryo_moveit_msgs.TrajectoryWaypoint, queue_size=10
        )

    def execute_niryo_trajectory(self, msg):
        waypoints = msg.trajectory.joint_trajectory.points

        for waypoint_idx, waypoint in enumerate(waypoints):
            feedback = niryo_moveit_msgs.ExecuteTrajectoryFeedback(waypoint_idx=waypoint_idx)
            self._niryo_trajectory_server.publish_feedback(feedback)

            msg = niryo_moveit_msgs.TrajectoryWaypoint(positions=waypoint.positions)
            self._niryo_trajectory_waypoint_pub.publish(msg)
            self._execute_physical_joints(positions=waypoint.positions)

            self.r.sleep()

        result = niryo_moveit_msgs.ExecuteTrajectoryResult(is_done=True)
        self._niryo_trajectory_server.set_succeeded(result)

    def _execute_physical_joints(self, positions):
        self.mycobot.send_radians(positions, 100)
        return

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    robot = Robot()
    robot.run()
