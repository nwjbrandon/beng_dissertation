//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.NiryoOne
{
    [Serializable]
    public class ManageProcessRequest : Message
    {
        public const string k_RosMessageName = "niryo_one_msgs/ManageProcess";
        public override string RosMessageName => k_RosMessageName;

        //  start, stop, restart, kill, start_all, stop_all
        public byte action;
        public string name;

        public ManageProcessRequest()
        {
            this.action = 0;
            this.name = "";
        }

        public ManageProcessRequest(byte action, string name)
        {
            this.action = action;
            this.name = name;
        }

        public static ManageProcessRequest Deserialize(MessageDeserializer deserializer) => new ManageProcessRequest(deserializer);

        private ManageProcessRequest(MessageDeserializer deserializer)
        {
            deserializer.Read(out this.action);
            deserializer.Read(out this.name);
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.action);
            serializer.Write(this.name);
        }

        public override string ToString()
        {
            return "ManageProcessRequest: " +
            "\naction: " + action.ToString() +
            "\nname: " + name.ToString();
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
