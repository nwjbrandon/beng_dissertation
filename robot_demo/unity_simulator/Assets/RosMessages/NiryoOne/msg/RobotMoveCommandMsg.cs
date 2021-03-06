//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.NiryoOne
{
    [Serializable]
    public class RobotMoveCommandMsg : Message
    {
        public const string k_RosMessageName = "niryo_one_msgs/RobotMoveCommand";
        public override string RosMessageName => k_RosMessageName;

        public int cmd_type;
        public double[] joints;
        public Geometry.PointMsg position;
        public RPYMsg rpy;
        public ShiftPoseMsg shift;
        public TrajectoryPlanMsg Trajectory;
        public Geometry.PoseMsg pose_quat;
        public string saved_position_name;
        public int saved_trajectory_id;
        public ToolCommandMsg tool_cmd;
        //  In the future, allow a tool command to be launched at the same time as an Arm command
        //  3 choices : arm only, arm + tool, tool only
        //  bool use_tool 

        public RobotMoveCommandMsg()
        {
            this.cmd_type = 0;
            this.joints = new double[0];
            this.position = new Geometry.PointMsg();
            this.rpy = new RPYMsg();
            this.shift = new ShiftPoseMsg();
            this.Trajectory = new TrajectoryPlanMsg();
            this.pose_quat = new Geometry.PoseMsg();
            this.saved_position_name = "";
            this.saved_trajectory_id = 0;
            this.tool_cmd = new ToolCommandMsg();
        }

        public RobotMoveCommandMsg(int cmd_type, double[] joints, Geometry.PointMsg position, RPYMsg rpy, ShiftPoseMsg shift, TrajectoryPlanMsg Trajectory, Geometry.PoseMsg pose_quat, string saved_position_name, int saved_trajectory_id, ToolCommandMsg tool_cmd)
        {
            this.cmd_type = cmd_type;
            this.joints = joints;
            this.position = position;
            this.rpy = rpy;
            this.shift = shift;
            this.Trajectory = Trajectory;
            this.pose_quat = pose_quat;
            this.saved_position_name = saved_position_name;
            this.saved_trajectory_id = saved_trajectory_id;
            this.tool_cmd = tool_cmd;
        }

        public static RobotMoveCommandMsg Deserialize(MessageDeserializer deserializer) => new RobotMoveCommandMsg(deserializer);

        private RobotMoveCommandMsg(MessageDeserializer deserializer)
        {
            deserializer.Read(out this.cmd_type);
            deserializer.Read(out this.joints, sizeof(double), deserializer.ReadLength());
            this.position = Geometry.PointMsg.Deserialize(deserializer);
            this.rpy = RPYMsg.Deserialize(deserializer);
            this.shift = ShiftPoseMsg.Deserialize(deserializer);
            this.Trajectory = TrajectoryPlanMsg.Deserialize(deserializer);
            this.pose_quat = Geometry.PoseMsg.Deserialize(deserializer);
            deserializer.Read(out this.saved_position_name);
            deserializer.Read(out this.saved_trajectory_id);
            this.tool_cmd = ToolCommandMsg.Deserialize(deserializer);
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.cmd_type);
            serializer.WriteLength(this.joints);
            serializer.Write(this.joints);
            serializer.Write(this.position);
            serializer.Write(this.rpy);
            serializer.Write(this.shift);
            serializer.Write(this.Trajectory);
            serializer.Write(this.pose_quat);
            serializer.Write(this.saved_position_name);
            serializer.Write(this.saved_trajectory_id);
            serializer.Write(this.tool_cmd);
        }

        public override string ToString()
        {
            return "RobotMoveCommandMsg: " +
            "\ncmd_type: " + cmd_type.ToString() +
            "\njoints: " + System.String.Join(", ", joints.ToList()) +
            "\nposition: " + position.ToString() +
            "\nrpy: " + rpy.ToString() +
            "\nshift: " + shift.ToString() +
            "\nTrajectory: " + Trajectory.ToString() +
            "\npose_quat: " + pose_quat.ToString() +
            "\nsaved_position_name: " + saved_position_name.ToString() +
            "\nsaved_trajectory_id: " + saved_trajectory_id.ToString() +
            "\ntool_cmd: " + tool_cmd.ToString();
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register()
        {
            MessageRegistry.Register(k_RosMessageName, Deserialize);
        }
    }
}
