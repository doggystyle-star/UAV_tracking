import gym
import math
import sys
import rospy
import random
import numpy as np
from gym import spaces
from nine_msgs.msg import control
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped
#gym.logger.set_level(40)
sys.path.append("/home/robot/firmware/xtdrone/APF_RL/gym_examples") 

class GazeboWorldEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 1
    }
    
    def __init__(self, render_mode=None):
        self.robust = control()
        #######
        #######
        #######
        self.reward = 0
        self.step_length = 0
        self.distance = np.array([3.5])
        self.target_height = np.array([1.6])
        self.axis = np.array([320.0])
        self.eval_distance = np.array([4.0])
        self.uav_height = np.array([1.0])
        self.target_axis = np.array([400.0])
        # self._agent = np.stack([self.eval_distance, self.uav_height, 0.01*self.target_axis], axis=0)
        self._target = np.stack([self.distance, self.target_height, 0.01*self.axis], axis=0)
        #observation space
        #deep distances
        self.low_x_tao1 = self.low_x_tao2 = self.low_y_tao1 = self.low_y_tao2 = self.low_z_tao1 = self.low_z_tao2 = self.low_Kp_yaw = self.low_Ki_yaw = self.low_Kd_yaw = 0.0

        self.high_x_tao1 = 10; self.high_x_tao2 = 1; self.high_y_tao1 = 5; self.high_y_tao2 = 4.9; self.high_z_tao1 = 2;self.high_z_tao2 = 1;self. high_Kp_yaw = 0.5; self.high_Ki_yaw = 0.1;self.high_Kd_yaw = 0.1
        self.observation_space = spaces.Dict(
            {
                "absolute_distance": spaces.Box(0, 8, shape=(1,), dtype=np.float32),
                "absolute_height": spaces.Box(0, 2, shape=(1,), dtype=np.float32),#x、z变化的绝对位置范围
                "absolute_axis": spaces.Box(0,640, shape=(1,),dtype=np.float32)
            }
                )

        #\Gamma1(x1,y1,z1) \Gamma2(x2,y2,z2) 先调整\Gamma2
        # 定义九维连续动作空间的上下界
        action_low = np.array([self.low_x_tao1, self.low_x_tao2, self.low_y_tao1, self.low_y_tao2, self.low_z_tao1, self.low_z_tao2, self.low_Kp_yaw, self.low_Ki_yaw, self.low_Kd_yaw])
        action_high = np.array([self.high_x_tao1, self.high_x_tao2, self.high_y_tao1, self.high_y_tao2,self.high_z_tao1, self.high_z_tao2, self.high_Kp_yaw, self.high_Ki_yaw, self.high_Kd_yaw])

        # 创建九维连续动作空间
        self.action_space = spaces.Box(low=action_low, high=action_high,dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        #收
        rospy.init_node("gazebo_world")
        rospy.Subscriber("/iris_0/mavros/local_position/pose", PoseStamped, callback = self._obs_local_pose_callback,queue_size=1)

        rospy.Subscriber("/iris_0/yolo/depth_distance", Float32, callback = self._obs_yolo_callback,queue_size=1)

        rospy.Subscriber('/iris_0/yolo/axis_u', Float32, callback=self._obs_local_axis_callback, queue_size=1)

        #发
        self.robust_pub = rospy.Publisher("/iris_0/Gazeboworld/parameter_9", control, queue_size=1)

    #返回z轴角速度和x加速度
    def _obs_yolo_callback(self,data):
        self.eval_distance = np.array([data.data])
        # print("self.eval_distance nmupy",self.eval_distance)
    def _obs_local_pose_callback(self, data):
        self.uav_height = np.array([data.pose.position.z])
    def _obs_local_axis_callback(self, data): 
        self.target_axis = np.array([data.data])

    def run(self):
        # get_time = False
        #header = Header()
        #header.frame_id = "base_link"
        # rate = rospy.Rate(30) 
        # while not rospy.is_shutdown():    
        #     self.robust_pub.publish(self.robust)
        #     print ("强化学习是否生效", self.robust.x_tao1)
        #     rate.sleep()
        self.robust_pub.publish(self.robust)

    #施加动作    
    def _apply_action(self, action):
        #发送七个参数
        #assert isinstance(action, list) or isinstance(action, np.ndarray)
        x_tao1,x_tao2,y_tao1,y_tao2,z_tao1,z_tao2,Kp_yaw,Ki_yaw,Kd_yaw = action
        # 裁剪防止输入动作超出动作空间
        self.robust.x_tao1 = np.clip(x_tao1, self.low_x_tao1, self.high_x_tao1)
        self.robust.x_tao2 = np.clip(x_tao2, self.low_x_tao2, self.high_x_tao2)
        self.robust.y_tao1 = np.clip(y_tao1, self.low_y_tao1, self.high_y_tao1)
        self.robust.y_tao2 = np.clip(y_tao2, self.low_y_tao2, self.high_y_tao2)
        self.robust.z_tao1 = np.clip(z_tao1, self.low_z_tao1, self.high_z_tao1)
        self.robust.z_tao2 = np.clip(z_tao2, self.low_z_tao2, self.high_z_tao2)
        self.robust.Kp_yaw = np.clip(Kp_yaw, self.low_Kp_yaw, self.high_Kp_yaw)
        self.robust.Ki_yaw = np.clip(Ki_yaw, self.low_Ki_yaw, self.high_Ki_yaw)
        self.robust.Kd_yaw = np.clip(Kd_yaw, self.low_Kd_yaw, self.high_Kd_yaw)
        #发给无人机
    
    def _get_obs(self):
        return {"absolute_distance": self.eval_distance,"absolute_height": self.uav_height, "absolute_axis":self.target_axis}
    
    def _get_info(self):
        return {
            "relative_distance":
                np.linalg.norm(self.eval_distance - self.distance, ord=1),
            "relative_height":
                np.linalg.norm(self.uav_height - self.target_height, ord=1),
            "relative_axis":
                np.linalg.norm(self.target_axis- self.axis, ord=1),
            "episode": {
                "r": self.reward, "l": self.step_length,
                "achieve_target": (self.eval_distance[0] - self.distance[0]) < 0.5
            }
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.reward = 0
        self.step_length = 0
        
        # 参数初始化
        self.robust.x_tao1 = 2
        self.robust.x_tao2 = 0.6
        self.robust.y_tao1 = 2
        self.robust.y_tao2 = 4
        self.robust.z_tao1 = 0.05
        self.robust.z_tao2 = 0.01
        self.robust.Kp_yaw = 0.0005
        self.robust.Ki_yaw = 0.01
        self.robust.Kd_yaw = 0.00002

        while np.abs(self.eval_distance-self.distance)>3 or np.abs(self.uav_height - self.target_height) > 1.5:
        #换个停止条件？
            self.robust.x_tao1 = 0
            self.robust.x_tao2 = 0
            self.robust.y_tao1 = 0
            self.robust.y_tao2 = 0
            self.robust.z_tao1 = 0
            self.robust.z_tao2 = 0
            self.robust.Kp_yaw = 0
            self.robust.Ki_yaw = 0
            self.robust.Kd_yaw = 0
            #写在控制里面
            # print ('landing')

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        min_distance = 0
        max_distance = 2
        min_height = 0
        max_height = 1
        min_axis = 0
        max_axis = 200
        self.step_length += 1

        # 分配权重
        weight_distance = 0.3  # 适当调整权重值
        weight_height = 0.5
        weight_axis = 0.2

        self._apply_action(action)
        observation = self._get_obs() #获取观察量距离和高度

        # reward = -np.abs(self._agent - self._target).mean()
        # if (np.abs(self.eval_distance - self.distance)<0.1 or (np.abs(self.uav_height - self.target_height))<0.1) or (np.abs(self.axis - self.target_axis))<20:
        #     reward += 1
        
        # elif (np.abs(self.eval_distance - self.distance)<0.4 or (np.abs(self.uav_height - self.target_height))<0.3) or (np.abs(self.axis - self.target_axis))<50:
        #     reward += 0.5
        # if (np.abs(self.eval_distance - self.distance)>1.5 or (np.abs(self.uav_height - self.target_height))>0.7) or (np.abs(self.axis - self.target_axis))>200:
        #     reward += 0.5
        # self.reward += reward


        distance_diff = self.eval_distance[0] - self.distance[0]
        print (distance_diff)
        height_diff = self.uav_height[0] - self.target_height[0]
        axis_diff = self.axis[0] - self.target_axis[0]

        # 归一化
        normalized_distance = (distance_diff - min_distance) / (max_distance - min_distance)
        normalized_height = (height_diff - min_height) / (max_height - min_height)
        normalized_axis = (axis_diff - min_axis) / (max_axis - min_axis)

        # 计算加权归一化量
        weighted_normalized_distance = weight_distance * normalized_distance
        weighted_normalized_height = weight_height * normalized_height
        weighted_normalized_axis = weight_axis * normalized_axis

        # 计算最终奖励，这里采用负指数函数来加强对差异的惩罚
        reward = -np.exp(weighted_normalized_distance) - np.exp(weighted_normalized_height) - np.exp(weighted_normalized_axis)

        if distance_diff < 0.1 or axis_diff < 20:
            reward += 1
        elif distance_diff < 0.4 or axis_diff < 50:
            reward += 0.5
        elif distance_diff > 1.5 or height_diff >0.3:   
            reward -= 5
        elif height_diff >= 0.5 or self.uav_height[0]<0.4 :
            print("height_dff",height_diff)
            reward -= 10
        self.reward += float(reward)
        info = self._get_info()
        # An episode is done iff the agent has reached the target or obstacle
        # 超过指定区间
        terminated = \
            abs(self.eval_distance[0] - self.distance[0])<0.1 or abs(self.uav_height[0] - self.target_height[0])<0.1 or abs(self.axis[0] - self.target_axis[0])<20 or\
            abs(self.eval_distance[0] - self.distance[0])>1 or \
            abs(self.uav_height[0] - self.target_height[0])>1 or\
            abs(self.axis[0] - self.target_axis[0])>200 or\
            abs(self.uav_height[0] < 0.2)

        
        if self.step_length >= 200:
            return observation, reward, True, False, info
        return observation, reward, terminated,False, info

    def close(self):
        return None
            
if __name__ == "__main__":
    my_node = GazeboWorldEnv()
    while not rospy.is_shutdown():
        my_node.run()