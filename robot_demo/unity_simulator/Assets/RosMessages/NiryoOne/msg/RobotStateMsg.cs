//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.NiryoOne
{
    [Serializable]
    public class RobotStateMsg : Message
    {
        public const string k_RosMessageName = "niryo_one_msgs/RobotState";
        public override string RosMessageName => k_RosMessageName;

        public Geometry.PointMsg position;
        public RPYMsg rpy;

        public RobotStateMsg()
        {
            this.position = new Geometry.PointMsg();
            this.rpy = new RPYMsg();
        }

        public RobotStateMsg(Geometry.PointMsg position, RPYMsg rpy)
        {
            this.position = position;
            this.rpy = rpy;
        }

        public static RobotStateMsg Deserialize(MessageDeserializer deserializer) => new RobotStateMsg(deserializer);

        private RobotStateMsg(MessageDeserializer deserializer)
        {
            this.position = Geometry.PointMsg.Deserialize(deserializer);
            this.rpy = RPYMsg.Deserialize(deserializer);
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.position);
            serializer.Write(this.rpy);
        }

        public override string ToString()
        {
            return "RobotStateMsg: " +
            "\nposition: " + position.ToString() +
            "\nrpy: " + rpy.ToString();
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
