import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist

def pose_publisher():
    pub = rospy.Publisher('gazebo/set_model_states', ModelStates, queue_size=1)
    poses_msg = ModelStates()
    poses_msg.name = [None] * 2
    poses_msg.pose = [Pose()for i in range(2)]
    poses_msg.twist = [Twist()for i in range(2)]
    poses_msg.name[0] = 'rover_static'
    poses_msg.pose[0].position.x  = 70
    poses_msg.pose[0].position.y = 1
    poses_msg.pose[0].position.z = 0
    poses_msg.name[1] = 'rover_static_clone'
    poses_msg.pose[1].position.x  = 70
    poses_msg.pose[1].position.y = 5
    poses_msg.pose[1].position.z = 0
    f = 20
    v = 0.5
    rate = rospy.Rate(f)
    while not rospy.is_shutdown():
        for i in range(2):
            poses_msg.pose[i].position.x = poses_msg.pose[i].position.x + v / f
        pub.publish(poses_msg)
        rate.sleep()
if __name__ == '__main__':
      rospy.init_node('pose_publisher')
      try:
          pose_publisher()
      except rospy.ROSInterruptException:
          pass