//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Std;

namespace RosMessageTypes.Moveit
{
    [Serializable]
    public class PositionConstraintMsg : Message
    {
        public const string k_RosMessageName = "moveit_msgs/PositionConstraint";
        public override string RosMessageName => k_RosMessageName;

        //  This message contains the definition of a position constraint.
        public HeaderMsg header;
        //  The robot link this constraint refers to
        public string link_name;
        //  The offset (in the link frame) for the target point on the link we are planning for
        public Geometry.Vector3Msg target_point_offset;
        //  The volume this constraint refers to 
        public BoundingVolumeMsg constraint_region;
        //  A weighting factor for this constraint (denotes relative importance to other constraints. Closer to zero means less important)
        public double weight;

        public PositionConstraintMsg()
        {
            this.header = new HeaderMsg();
            this.link_name = "";
            this.target_point_offset = new Geometry.Vector3Msg();
            this.constraint_region = new BoundingVolumeMsg();
            this.weight = 0.0;
        }

        public PositionConstraintMsg(HeaderMsg header, string link_name, Geometry.Vector3Msg target_point_offset, BoundingVolumeMsg constraint_region, double weight)
        {
            this.header = header;
            this.link_name = link_name;
            this.target_point_offset = target_point_offset;
            this.constraint_region = constraint_region;
            this.weight = weight;
        }

        public static PositionConstraintMsg Deserialize(MessageDeserializer deserializer) => new PositionConstraintMsg(deserializer);

        private PositionConstraintMsg(MessageDeserializer deserializer)
        {
            this.header = HeaderMsg.Deserialize(deserializer);
            deserializer.Read(out this.link_name);
            this.target_point_offset = Geometry.Vector3Msg.Deserialize(deserializer);
            this.constraint_region = BoundingVolumeMsg.Deserialize(deserializer);
            deserializer.Read(out this.weight);
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.header);
            serializer.Write(this.link_name);
            serializer.Write(this.target_point_offset);
            serializer.Write(this.constraint_region);
            serializer.Write(this.weight);
        }

        public override string ToString()
        {
            return "PositionConstraintMsg: " +
            "\nheader: " + header.ToString() +
            "\nlink_name: " + link_name.ToString() +
            "\ntarget_point_offset: " + target_point_offset.ToString() +
            "\nconstraint_region: " + constraint_region.ToString() +
            "\nweight: " + weight.ToString();
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
