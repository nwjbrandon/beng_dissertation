//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.NiryoOne
{
    [Serializable]
    public class GetPositionListResponse : Message
    {
        public const string k_RosMessageName = "niryo_one_msgs/GetPositionList";
        public override string RosMessageName => k_RosMessageName;

        public PositionMsg[] positions;

        public GetPositionListResponse()
        {
            this.positions = new PositionMsg[0];
        }

        public GetPositionListResponse(PositionMsg[] positions)
        {
            this.positions = positions;
        }

        public static GetPositionListResponse Deserialize(MessageDeserializer deserializer) => new GetPositionListResponse(deserializer);

        private GetPositionListResponse(MessageDeserializer deserializer)
        {
            deserializer.Read(out this.positions, PositionMsg.Deserialize, deserializer.ReadLength());
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.WriteLength(this.positions);
            serializer.Write(this.positions);
        }

        public override string ToString()
        {
            return "GetPositionListResponse: " +
            "\npositions: " + System.String.Join(", ", positions.ToList());
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register()
        {
            MessageRegistry.Register(k_RosMessageName, Deserialize, MessageSubtopic.Response);
        }
    }
}
