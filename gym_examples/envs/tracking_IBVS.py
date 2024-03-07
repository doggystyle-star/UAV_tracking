import gym
from gym import spaces
import math
import rospy
import random
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
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

def color_img_callback(msg):
    global color_img
    color_img = bridge.imgmsg_to_cv2(msg, "bgr8")

def depth_img_callback(msg):
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(msg, "32FC1")
    depth_img = np.nan_to_num(depth_img)

def detact_distance(box,randnum):
    distance_list = []
    mid_pos = [(box.xmin + box.xmax)//2, (box.ymin + box.ymax)//2] #确定索引深度的中心像素位置
    min_val = min(abs(box.xmax - box.xmin), abs(box.ymax - box.ymin))#确定深度搜索范围
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_img[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        # 绘制中心像素位置，仅用于调试
        # cv2.circle(color_img (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #Timsort排序+中值滤波
    #print(distance_list, np.mean(distance_list))

    # 加权深度距离
    raw_distance = round(np.mean(distance_list),4)
    distance = abs(0.9*raw_distance + 0.01*target_distance*((43**2+105**2)/((box.xmin-box.xmax)//2**2+(box.ymin-box.ymax)//2**2))**(1/2))
    return distance

def get_depth_frame():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # 创建对齐对象（深度对齐颜色）
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # 获取对齐帧集
            aligned_frames = align.process(frames)
            
            # 获取对齐后的深度帧和彩色帧
            aligned_depth_frame = aligned_frames.get_depth_frame()
    except:
        print('cant get picture')
        return aligned_depth_frame
    
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
    global robust
    robust.Kp_yaw = msg.Kp_yaw
    robust.Ki_yaw = msg.Ki_yaw
    robust.Kd_yaw = msg.Kd_yaw
    robust.x_tao1 = msg.x_tao1
    robust.x_tao2 = msg.x_tao2
    robust.y_tao1 = msg.y_tao1
    robust.y_tao2 = msg.y_tao2
    robust.z_tao1 = msg.z_tao1
    robust.z_tao2 = msg.z_tao2


#返回z轴角速度和x加速度
def darknet_callback(data):
    global find_cnt, cmd, get_time, eval_distance, twist, depth_distance,u
    for box in data.bounding_boxes:
        if(box.Class == "person" ):
        #if(box.id == 56 ):
            print("find human")
            eval_distance = detact_distance(box,48)

            # 加入虚拟相平面
            # 计算cos(pitch_angle)
            eval_distance = eval_distance * math.cos(pitch) + height * math.sin(pitch)
            
            depth_distance = eval_distance 
            q_x = eval_distance - target_distance
            print("深度距离为:",eval_distance)
            u = (box.xmax+box.xmin)/2
            #print("中心像素距离为:",u - ppx)
            v = (box.ymax+box.ymin)/2
            twist.angular.z = z_angvelocity.update(ppx,u,Dt)
            q_y = eval_distance*(u- ppx)/fx
            q_z = eval_distance*(v - ppy)/fy
            #WL = twiststamped.twist.angular.z
            q_zz = height - target_height
            XVL=twiststamped.twist.linear.x
            YVL=twiststamped.twist.linear.y
            ZVL=twiststamped.twist.linear.z
            #print("x_velocity",VL,'\t')
            twist.linear.x = x_accelerate.update_X(U_=0.5,dt=dDt,VL=XVL,PL=q_x) 
            twist.linear.y = y_accelerate.update_Y(U_=0.5,dt=dDt,VL=YVL,PL=q_y)#左手系
            #print(twiststamped.twist.linear.x)
            VZ = z_accelerate.update_Z(U_=0.5,dt=Dt,VL=ZVL,PL=q_zz)
            if not (is_over):
                if VZ == "nan":
                    twist.linear.z = -0.02
                elif (is_bottom): twist.linear.z = 0.02

                else:
                    twist.linear.z = VZ
            else:
                twist.linear.z = -0.02
            #录制bag包  
            toast.twist.linear.x = u - ppx
            toast.twist.linear.y = v - ppy
            toast.twist.linear.z = q_x

        else:
            twist.linear.x = 0
            twist.linear.y = 0
            twist.linear.z = 0.02
            #twist.angular.z = 0.5

#返回z加速度
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
    # print('高度为： ',height)
    # print('pitch: ', pitch)

# 返回俯仰角
def q2pitch(q):
    # 如果输入是 Quaternion 类型，则直接获取俯仰角
    if isinstance(q, Quaternion):
        pitch_angle_rad = q.yaw_pitch_roll[1]
    else:
        # 如果输入不是 Quaternion 类型，先将其转换为 Quaternion 类型
        q_ = Quaternion(q.w, q.x, q.y, q.z)
        # 获取俯仰角
        pitch_angle_rad = q_.yaw_pitch_roll[1]
    return pitch_angle_rad

if __name__ == "__main__":

    cmd = String()
    twiststamped =TwistStamped()
    twist = Twist()
    robust = control()
    toast = TwistStamped()
    depth_distance = Float32()
    u = Float32()
    target_distance = 3
    target_height = 0.6
    #env = RelativePosition(env) 
    ppx=318.482
    ppy=241.167
    # fx = 384.39654541015625
    # fy = 384.39654541015625
    fx = 616.591
    fy = 616.765

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

    cmd_vel_pub = rospy.Publisher('/xtdrone/iris_0/cmd_vel_flu', Twist, queue_size=1)
    cmd_pub = rospy.Publisher('/xtdrone/iris_0/cmd', String, queue_size=1)
    error_pub = rospy.Publisher('/xtdrone/iris_0/cmd_error',TwistStamped,queue_size=1)

    distance_pub = rospy.Publisher('/iris_0/yolo/depth_distance', Float32, queue_size=1)

    axis_pub = rospy.Publisher('iris_0/yolo/axis_u', Float32, queue_size=1)
    rate = rospy.Rate(30) 

        #PID control
    # Kp_yaw = -0.0005
    # Ki_yaw = 0.01
    # Kd_yaw = 0.00002
    z_angvelocity = PIDController(robust.Kp_yaw,robust.Ki_yaw,robust.Kd_yaw)
    #ACC control
    ET = 0.1
    # x_tao1 = 2
    # x_tao2 = 0.6

    # y_tao1 = 2
    # y_tao2 = 4

    # z_tao1 = 0.05
    # z_tao2 = 0.01

    x_accelerate = AccelerateController(robust.x_tao1,robust.x_tao2,ET)
    y_accelerate = AccelerateController(robust.y_tao1,robust.y_tao2,ET)
    z_accelerate = AccelerateController(robust.z_tao1,robust.z_tao2,ET)

    header = Header()
    header.frame_id = "base_link"
    while not rospy.is_shutdown():
        rate.sleep()
        cmd_vel_pub.publish(twist)
        # print("twist_x:",twist.linear.x)
        cmd_pub.publish(cmd)           
        error_pub.publish(toast)
        # print ("强化学习是否生效",robust.x_tao1)
        if (depth_distance == "nan"):
            depth_distance = 9.6
        distance_pub.publish(depth_distance)
        axis_pub.publish(u)
        if find_cnt - find_cnt_last == 0:
            if not get_time:
                not_find_time = rospy.get_time()
                get_time = True
            # if (rospy.get_time() - not_find_time > 3.0)or eval_distance =="nan":
            if (rospy.get_time() - not_find_time > 3.0):
                twist.linear.x = 0.0
                twist.linear.y = 0.0
                cmd = "HOVER"
                print(cmd)
                
                get_time = False

        find_cnt_last = find_cnt