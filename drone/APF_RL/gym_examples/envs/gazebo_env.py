import gym
import rospy
import random
from gym import spaces
import numpy as np
import math

from darknet_ros_msgs.msg import BoundingBoxes
from control.tracking_IBVS import  detact_distance
from geometry_msgs.msg import TwistStamped

low_x_tao1 = low_x_tao2 = low_y_tao1 = low_y_tao2 = low_z_tao1 = low_z_tao2 = low_Kp_yaw = low_Ki_yaw = low_Kd_yaw = 0.0

high_x_tao1 = 5; high_x_tao2 = 1; high_y_tao1 = 5; high_y_tao2 = 4.9; high_z_tao1 = 0.05;high_z_tao2 = 0.01; high_Kp_yaw = 5; high_Ki_yaw = 0.5;high_Kd_yaw = 0.1



class GazeboWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'render_fps': 2
    }
    
    def __init__(self, render_mode=None):
        self.reward = 0
        self.step_length = 0
        self.distance = 4.0 
        #observation space
        #deep distances
        self.observation_space = spaces.Dict(
            {
                 "relative": spaces.Box(0, 8, shape=(1,), dtype=float)
            }
                )

        #\Gamma1(x1,y1,z1) \Gamma2(x2,y2,z2) 先调整\Gamma2
        # 定义九维连续动作空间的上下界
        action_low = np.array([low_x_tao1, low_x_tao2, low_y_tao1, low_y_tao2, low_z_tao1, low_z_tao2, low_Kp_yaw, low_Ki_yaw, low_Kd_yaw])
        action_high = np.array([high_x_tao1, high_x_tao2, high_y_tao1, high_y_tao2, high_z_tao1, high_z_tao2, high_Kp_yaw, high_Ki_yaw, high_Kd_yaw])

        # 创建七维连续动作空间
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    #施加动作    
    def apply_action(self, action):
        #发送七个参数(是否改为twist)
        #assert isinstance(action, list) or isinstance(action, np.ndarray)
        x_tao1,x_tao1,y_tao1,y_tao2,Kp_yaw,Ki_yaw,Kd_yaw = action
        # 裁剪防止输入动作超出动作空间
        x_tao1 = np.clip(x_tao1, low_x_tao1, high_x_tao1)
        x_tao2 = np.clip(x_tao2, low_x_tao2, high_x_tao2)
        y_tao1 = np.clip(y_tao1, low_y_tao1, high_y_tao1)
        y_tao2 = np.clip(y_tao2, low_y_tao2, high_y_tao2)
        z_tao1 = np.clip(z_tao1, low_z_tao1, high_z_tao1)
        z_tao2 = np.clip(z_tao2, low_z_tao2, high_z_tao2)
        Kp_yaw = np.clip(Kp_yaw, low_Kp_yaw, high_Kp_yaw)
        Ki_yaw = np.clip(Ki_yaw, low_Ki_yaw, high_Ki_yaw)
        Kd_yaw = np.clip(Kd_yaw, low_Kd_yaw, high_Kd_yaw)
        #发给无人机
        return x_tao1, x_tao2, y_tao1, y_tao2, z_tao1, z_tao2, Kp_yaw, Ki_yaw, Kd_yaw

        
    def boundingboxs_callback(data):
        global eval_distance
        for box in data.bounding_boxes:
            if(box.id == 0):
                eval_distance = detact_distance(box,48)

    #获取深度信息(和twist类型行不行)
    def _relative_position(self):
        return eval_distance
    
    #这么写可能为了后续避障
    def __get_observation(self):
        return {"relative": self._relative_position}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.reward = 0
        self.step_length = 0

        while abs(self._relative_position-self.distance) > 1.0:
           #换个停止条件？
           TwistStamped.twist.linear.x = 0
           TwistStamped.twist.linear.y = 0
           TwistStamped.twist.linear.z = 0
           print ('landing')

    def step(self, action):
            self.step_length += 1
            # 随机选择一个动作（仅作示例，实际情况中智能体会使用特定策略来选择动作）
            #parameter = self.action_space.sample()
            #parameter = self.__apply_action(action)
            
            
            observation = self.__get_observation()

            reward = -abs(self._relative_position - self.distance)
            if (self._relative_position == self.distance):
                reward += 1
            self.reward += reward

            info = {}
            # An episode is done iff the agent has reached the target or obstacle
            terminated = (self._relative_position == self.distance)
            
            if self.step_length >= 200:
                return observation, reward, True, False, info
            return observation, reward, terminated,False, info

    def close(self):
        return None
    
    rospy.init_node("gazebo_world")
    rospy.Subscriber("/iris_0/darknet_ros/bounding_boxes", BoundingBoxes, callback = boundingboxs_callback,queue_size=1)
    #envINIT
