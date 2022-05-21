#!/usr/bin/env python

from __future__ import print_function

import sys

import actionlib
import geometry_msgs.msg as geometry_msgs
import moveit_commander
import moveit_msgs.msg as moveit_msgs
import niryo_moveit.msg as niryo_moveit_msgs
import rospy
import sensor_msgs.msg as sensor_msgs
from tf.transformations import euler_from_quaternion

joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


class Mover:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("niryo_mover")

        self.group_name = "arm"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.move_group.set_planning_time(0.1)
        self.move_group.allow_looking(False)
        self.move_group.allow_replanning(False)
        self.planning_rate = rospy.Rate(5)

        self._niryo_robot_joints_sub = rospy.Subscriber(
            "niryo_robot_joints",
            niryo_moveit_msgs.NiryoRobotJoints,
            self.set_robot_joints,
            queue_size=1,
        )
        self._niryo_end_effector_pose_sub = rospy.Subscriber(
            "niryo_end_effector_pose",
            niryo_moveit_msgs.EndEffectorPose,
            self.set_end_effector_pose,
            queue_size=1,
        )
        self._niryo_trajectory_client = actionlib.SimpleActionClient(
            "niryo_trajectory", niryo_moveit_msgs.ExecuteTrajectoryAction,
        )

        self.robot_joints = None
        self.end_effector_pose = None
        self.hand_pose = None

        self._niryo_trajectory_client.wait_for_server()

    def set_robot_joints(self, msg):
        self.robot_joints = msg.joints

    def set_end_effector_pose(self, msg):
        self.end_effector_pose = msg.target

    def run(self):
        while not rospy.is_shutdown():
            self.planning_rate.sleep()

            # Receive robot joints and end effector pose before planning trajectory
            is_plan_info_none = self.robot_joints is None or self.end_effector_pose is None
            if is_plan_info_none:
                continue

            # Receive hand pose from user before planning trajectory
            if self.hand_pose is not None and self.are_poses_equal(
                self.hand_pose, self.end_effector_pose
            ):
                continue

            self.move()

    def move(self):
        robot_joints, end_effector_pose = self.get_trajectory_plan_info()
        trajectory_plan = self.get_trajectory_plan(robot_joints, end_effector_pose)
        self.execute_trajectory_plan(trajectory_plan, end_effector_pose)

    def get_trajectory_plan_info(self):
        return self.robot_joints, self.end_effector_pose

    def get_trajectory_plan(self, robot_joints, end_effector_pose):
        current_joint_state = sensor_msgs.JointState()
        current_joint_state.name = joint_names
        current_joint_state.position = robot_joints

        moveit_robot_state = moveit_msgs.RobotState()
        moveit_robot_state.joint_state = current_joint_state
        self.move_group.set_start_state(moveit_robot_state)

        self.move_group.set_pose_target(end_effector_pose)
        plan = self.move_group.plan()
        self.move_group.clear_pose_targets()

        if not plan:
            exception_str = """
                Trajectory could not be planned for a destination of {} with starting joint angles {}.
                Please make sure target and destination are reachable by the robot.
            """.format(
                end_effector_pose, robot_joints
            )
            raise Exception(exception_str)

        return plan

    def execute_trajectory_plan(self, trajectory_plan, end_effector_pose):
        # If the trajectory has no points, planning has failed and we return an empty response
        if not trajectory_plan.joint_trajectory.points:
            return

        self.hand_pose = end_effector_pose
        self._niryo_trajectory_client.send_goal(
            niryo_moveit_msgs.ExecuteTrajectoryGoal(trajectory=trajectory_plan),
            active_cb=self._trajectory_start,
            feedback_cb=self._trajectory_feedback,
            done_cb=self._trajectory_done,
        )
        self._niryo_trajectory_client.wait_for_result()

    def are_poses_equal(self, p1, p2):
        pos1 = p1.position
        pos2 = p2.position
        o1 = p1.orientation
        o2 = p2.orientation
        r1, p1, y1 = euler_from_quaternion([o1.x, o1.y, o1.z, o1.w])
        r2, p2, y2 = euler_from_quaternion([o2.x, o2.y, o2.z, o2.w])
        is_pos_equal = (
            self.isclose(pos1.x, pos2.x)
            and self.isclose(pos1.y, pos2.y)
            and self.isclose(pos1.z, pos2.z)
        )
        is_ori_equal = self.isclose(r1, r2) and self.isclose(p1, p2) and self.isclose(y1, y2)
        if is_pos_equal and is_ori_equal:
            return True
        else:
            return False

    def isclose(self, n1, n2, tol=0.001):
        return abs(n1 - n2) <= tol

    def _trajectory_start(self):
        return

    def _trajectory_feedback(self, feedback):
        return

    def _trajectory_done(self, state, result):
        return


if __name__ == "__main__":
    mover = Mover()
    mover.run()
