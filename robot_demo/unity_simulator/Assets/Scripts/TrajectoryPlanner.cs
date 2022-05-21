using System;
using System.Collections;
using System.Linq;
using RosMessageTypes.Geometry;
using RosMessageTypes.NiryoMoveit;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using UnityEngine;

public class TrajectoryPlanner : MonoBehaviour
{
    // Game Objects
    [SerializeField]
    GameObject m_NiryoOne;
    public GameObject NiryoOne { get => m_NiryoOne; set => m_NiryoOne = value; }
    [SerializeField]
    GameObject m_Target;
    public GameObject Target { get => m_Target; set => m_Target = value; }
    [SerializeField]
    GameObject m_TargetPlacement;
    public GameObject TargetPlacement { get => m_TargetPlacement; set => m_TargetPlacement = value; }

    // Variables required for ROS communication
    private string trajectoryWaypointTopic = "niryo_trajectory_waypoint";
    private string endEffectorPoseTopic = "niryo_end_effector_pose";
    private string robotJointsTopic = "niryo_robot_joints";
    ROSConnection m_Ros;    

    // Articulation Bodies
    const int k_NumRobotJoints = 6;
    ArticulationBody[] m_JointArticulationBodies;
    ArticulationBody m_LeftGripper;
    ArticulationBody m_RightGripper;

    // Game Update
    private double interval = 0.01; 
    private double nextTime = 0;

    void Start()
    {
        // Get ROS connection static instance
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.Subscribe<TrajectoryWaypointMsg>(trajectoryWaypointTopic, ExecuteTrajectoryWaypoint);
        m_Ros.Subscribe<EndEffectorPoseMsg>(endEffectorPoseTopic, ExecuteGripperState);
        m_Ros.RegisterPublisher<NiryoRobotJointsMsg>(robotJointsTopic);

        m_JointArticulationBodies = new ArticulationBody[k_NumRobotJoints];

        var linkName = string.Empty;
        for (var i = 0; i < k_NumRobotJoints; i++)
        {
            linkName += SourceDestinationPublisher.LinkNames[i];
            m_JointArticulationBodies[i] = m_NiryoOne.transform.Find(linkName).GetComponent<ArticulationBody>();
        }

        // Find left and right fingers
        var rightGripper = linkName + "/gripper_base/servo_head/control_rod_right/right_gripper";
        var leftGripper = linkName + "/gripper_base/servo_head/control_rod_left/left_gripper";
        Debug.Log(rightGripper);
        Debug.Log(leftGripper);

        m_RightGripper = m_NiryoOne.transform.Find(rightGripper).GetComponent<ArticulationBody>();
        m_LeftGripper = m_NiryoOne.transform.Find(leftGripper).GetComponent<ArticulationBody>();
    }

    private void Update() 
    {
        if (Time.time >= nextTime) 
        {
            PublishJoints();
            nextTime += interval; 
        }
    }

    void CloseGripper()
    {
        var leftDrive = m_LeftGripper.xDrive;
        var rightDrive = m_RightGripper.xDrive;

        leftDrive.target = -0.01f;
        rightDrive.target = 0.01f;

        m_LeftGripper.xDrive = leftDrive;
        m_RightGripper.xDrive = rightDrive;
    }

    void OpenGripper()
    {
        var leftDrive = m_LeftGripper.xDrive;
        var rightDrive = m_RightGripper.xDrive;

        leftDrive.target = 0.01f;
        rightDrive.target = -0.01f;

        m_LeftGripper.xDrive = leftDrive;
        m_RightGripper.xDrive = rightDrive;
    }

    NiryoRobotJointsMsg GetNiryoRobotJoints()
    {
        var joints = new NiryoRobotJointsMsg();

        for (var i = 0; i < k_NumRobotJoints; i++)
        {
            joints.joints[i] = m_JointArticulationBodies[i].jointPosition[0];
        }

        return joints;
    }

    public void PublishJoints()
    {
        m_Ros.Publish(robotJointsTopic, GetNiryoRobotJoints());
    }

    void ExecuteTrajectoryWaypoint(TrajectoryWaypointMsg msg)
    {
        var jointPositions = msg.positions;
        var result = jointPositions.Select(r => (float)r * Mathf.Rad2Deg).ToArray();

        // Set the joint values for every joint
        for (var joint = 0; joint < m_JointArticulationBodies.Length; joint++)
        {
            var joint1XDrive = m_JointArticulationBodies[joint].xDrive;
            joint1XDrive.target = result[joint];
            m_JointArticulationBodies[joint].xDrive = joint1XDrive;
        }
    }

    void ExecuteGripperState(EndEffectorPoseMsg msg)
    {
        if (msg.gripper_state > 0.5) {
            OpenGripper();
        } else {
            CloseGripper();
        }
    }
}