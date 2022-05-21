#!/usr/bin/env python

import actionlib
import niryo_moveit
import rospy
from niryo_moveit.msg import (NiryoJointTrajectoryPoint, TrajectoryFeedback,
                              TrajectoryGoal, TrajectoryResult)


class RobotMoveActionServer:
    def __init__(self):
        rospy.init_node("server")

        self._feedback = niryo_moveit.msg.TrajectoryFeedback()
        self._result = niryo_moveit.msg.TrajectoryResult()

        self.r = rospy.Rate(30)

        self._robot_action_pub = rospy.Publisher(
            "robot_action", NiryoJointTrajectoryPoint, queue_size=10
        )
        self._action_server = actionlib.SimpleActionServer(
            "mycobot/robot_action",
            niryo_moveit.msg.TrajectoryAction,
            execute_cb=self.execute_robot_action,
            auto_start=False,
        )
        self._action_server.start()

    def execute_robot_action(self, msg):
        waypoints = msg.trajectory.joint_trajectory.points

        for waypoint_idx, waypoint in enumerate(waypoints):
            self._feedback.waypoint = waypoint
            self._feedback.waypoint_idx = waypoint_idx
            self._feedback.is_done = False
            self._action_server.publish_feedback(self._feedback)
            self._robot_action_pub.publish(NiryoJointTrajectoryPoint(positions=waypoint.positions))

            self.r.sleep()

        self._result.waypoint_idx = len(waypoints)
        self._result.is_done = True
        self._action_server.set_succeeded(self._result)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    action_server = RobotMoveActionServer()
    action_server.run()
