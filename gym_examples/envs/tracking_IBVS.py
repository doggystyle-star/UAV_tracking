''' 调试日志
1. 适配版本
1)change the acquirement depth_distancce 
2)change the control method of IBVS
'''
import math
import rospy
import random
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from pyquaternion import Quaternion
from detection_msgs.msg import BoundingBoxes
from std_msgs.msg import String,Header, Float32
from geometry_msgs.msg import Twist,TwistStamped,PoseStamped
from controller import AccelerateController, PIDController
from nine_msgs.msg import control
import sys 
from pyquaternion import Quaternion
sys.path.append("/home/robot/firmware/catkin_ws/devel/lib/python3/dist-packages") 

bridge = CvBridge()

def camera_info_callback(msg):
    global camera_info
    camera_info = msg

def color_img_callback(msg):
    global color_img
    color_img = bridge.imgmsg_to_cv2(msg, "bgr8")

def depth_img_callback(msg):
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(msg, "32FC1")
    depth_img = np.nan_to_num(depth_img)

def detact_distance2(box):
    mid_pos = [(box.xmin + box.xmax)//2, (box.ymin + box.ymax)//2] 

    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    depth = depth_img[mid_pos[1], mid_pos[0]]  
    if depth > 0:  
        point3d = np.array([(mid_pos[0] - cx) * depth / fx, (mid_pos[1] - cy) * depth / fy, depth])
        print(f"像素坐标({mid_pos[0]}, {mid_pos[1]}) 对应的三维坐标为：{point3d}")  

    # # Weighted depth distance
    # raw_distance = round(np.mean(distance_list),4)
    # distance = abs(0.9*raw_distance + 0.01*target_distance*((43**2+105**2)/((box.xmin-box.xmax)//2**2+(box.ymin-box.ymax)//2**2))**(1/2))
    return point3d
    
def local_velocity_callback(msg):
    global twiststamped
    #twiststamped.header = header
    twiststamped.twist.linear.x = msg.twist.linear.x
    twiststamped.twist.linear.y = msg.twist.linear.y
    twiststamped.twist.linear.z = msg.twist.linear.z
    twiststamped.twist.angular.x = msg.twist.angular.x
    twiststamped.twist.angular.y = msg.twist.angular.y
    twiststamped.twist.angular.z = msg.twist.angular.z
    #print (twiststamped)
def parameter_callback(msg):
    global x_accelerate, y_accelerate, z_accelerate,z_angvelocity
    Kp_yaw = msg.Kp_yaw
    Ki_yaw = msg.Ki_yaw
    Kd_yaw = msg.Kd_yaw
    x_tao1 = msg.x_tao1
    x_tao2 = msg.x_tao2
    y_tao1 = msg.y_tao1
    y_tao2 = msg.y_tao2
    z_tao1 = msg.z_tao1
    z_tao2 = msg.z_tao2
    print("robust", robust)
    '''tao1为-1'''
    x_accelerate = AccelerateController(-x_tao1,x_tao2,ET)
    y_accelerate = AccelerateController(y_tao1,y_tao2,ET)
    z_accelerate = AccelerateController(z_tao1,z_tao2,ET)
    z_angvelocity = PIDController(Kp_yaw,Ki_yaw,Kd_yaw)

def darknet_callback(data):
    global find_cnt, cmd, get_time, eval_distance, twist, depth_distance,u
    for box in data.bounding_boxes:
        if(box.Class == "person" ):
        #if(box.id == 56 ):
            print("find human")
            points = detact_distance2(box)
            eval_distance = points[2]
            # virtual phase plane
            # calcualte: cos(pitch_angle)
            eval_distance = eval_distance * math.cos(pitch) + height * math.sin(pitch)
            depth_distance = eval_distance 
            q_x = eval_distance - target_distance
            u = (box.xmax+box.xmin)/2
            v = (box.ymax+box.ymin)/2
            twist.angular.z = z_angvelocity.update(ppx,u,Dt)

            q_y = points[0]
            q_z = points[1]
            #WL = twiststamped.twist.angular.z
            q_zz = height - target_height
            XVL=twiststamped.twist.linear.x
            YVL=twiststamped.twist.linear.y
            ZVL=twiststamped.twist.linear.z
            #print("x_velocity",VL,'\t')
            twist.linear.x = x_accelerate.update_X(U_=0.5,dt=dDt,VL=XVL,PL=q_x) 
            twist.linear.y = y_accelerate.update_Y(U_=0.5,dt=dDt,VL=YVL,PL=q_y)#Left -hand
            # Limiting
            twist.linear.x = 0.5 if twist.linear.x > 0.5 else twist.linear.x 
            twist.linear.x = -0.5 if twist.linear.x < -0.5 else twist.linear.x
            twist.linear.y = 0.5 if twist.linear.y > 0.5 else twist.linear.y 
            twist.linear.y = -0.5 if twist.linear.y < -0.5 else twist.linear.y

def local_pose_callback(msg):
    global height, target_set, pitch, is_over, is_bottom
    is_over = False
    is_bottom = False
    height = msg.pose.position.z 
    pitch = q2pitch(msg.pose.orientation)
    if height > 1.8:
        is_over = True
    if height < 0.2:
        is_bottom = True

def q2pitch(q):
    # if the type of input is Quaternion，directly get angle
    if isinstance(q, Quaternion):
        pitch_angle_rad = q.yaw_pitch_roll[1]
    else:
        # firstly turn to type of Quaternion
        q_ = Quaternion(q.w, q.x, q.y, q.z)
        # then get angle
        pitch_angle_rad = q_.yaw_pitch_roll[1]
    return pitch_angle_rad

if __name__ == "__main__":
    header = Header()
    cmd = String()
    twiststamped =TwistStamped()
    Altitude = PoseStamped()
    twist = Twist()
    robust = control()
    toast = TwistStamped()
    depth_distance = Float32()
    u = Float32()
    target_distance = 3.8
    target_height = 1
    ET = 0.1
    ppx=318.482
    ppy=241.167
    fx = 616.591
    fy = 616.765

    x_accelerate = AccelerateController(0,0,ET)
    y_accelerate = AccelerateController(0,0,ET)
    z_accelerate = AccelerateController(0,0,ET)
    z_angvelocity = PIDController(0,0,0)

    target_set = True
    find_cnt_last = 0
    not_find_time = 0
    get_time = False

    find_cnt = 0
    height = 0
    Dt = 0.1
    dDt = 0.1
    eval_distance = 0

    rospy.init_node("yolo_human_tracking")

    rospy.Subscriber("/iris_0/realsense/depth_camera/color/image_raw",Image,callback = color_img_callback, queue_size=1)

    rospy.Subscriber("/iris_0/realsense/depth_camera/depth/image_raw", Image, callback = depth_img_callback, queue_size=1)

    rospy.Subscriber("/iris_0/mavros/local_position/velocity_local",TwistStamped, callback = local_velocity_callback,queue_size=1)

    rospy.Subscriber("/iris_0/mavros/local_position/pose", PoseStamped, callback = local_pose_callback,queue_size=1)

    rospy.Subscriber("/yolov5/detections", BoundingBoxes, callback = darknet_callback,queue_size=1)

    rospy.Subscriber("/iris_0/Gazeboworld/parameter_9", control,callback= parameter_callback, queue_size=1)

    rospy.Subscriber("/iris_0/realsense/depth_camera/color/camera_info",CameraInfo, callback= camera_info_callback, queue_size=1)

    cmd_vel_pub = rospy.Publisher('/xtdrone/iris_0/cmd_vel_flu', Twist, queue_size=1)
    cmd_pub = rospy.Publisher('/xtdrone/iris_0/cmd', String, queue_size=1)
    error_pub = rospy.Publisher('/xtdrone/iris_0/cmd_error',TwistStamped,queue_size=1)

    distance_pub = rospy.Publisher('/iris_0/yolo/depth_distance', Float32, queue_size=1)

    axis_pub = rospy.Publisher('iris_0/yolo/axis_u', Float32, queue_size=1)

    height_pub = rospy.Publisher('/iris_0/mavros/setpoint_raw/local', PoseStamped, queue_size=1)
    rate = rospy.Rate(30) 

    header.frame_id = "base_link"
    Altitude.header = header
    while not rospy.is_shutdown():
        Altitude.header.stamp = rospy.Time.now()  # Setting the timestamp
        
        Altitude.pose.position.z = 0.8
        height_pub.publish(Altitude)
        cmd_vel_pub.publish(twist)
        cmd_pub.publish(cmd)           
        error_pub.publish(toast)
        rate.sleep()

        if (depth_distance == "nan"):
            depth_distance = 9.6
        distance_pub.publish(depth_distance)
        axis_pub.publish(u)
        if find_cnt - find_cnt_last == 0:
            if not get_time:
                not_find_time = rospy.get_time()
                get_time = True

            if (rospy.get_time() - not_find_time > 3.0):
                twist.linear.x = 0.0
                twist.linear.y = 0.0
                cmd = "HOVER"
                print(cmd)
                
                get_time = False
            cmd = ''
        find_cnt_last = find_cnt