import gym
from gym import spaces
import numpy as np


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(shape=(3,), low=-np.inf, high=np.inf)


    def observation(self, obs):
        distance = np.expand_dims(obs["absolute_distance"], axis=0) if np.isscalar(obs["absolute_distance"]) else obs["absolute_distance"]
        height = np.expand_dims(obs["absolute_height"], axis=0) if np.isscalar(obs["absolute_height"]) else obs["absolute_height"]
        u_axis = np.expand_dims(obs["absolute_axis"], axis=0) if np.isscalar(obs["absolute_axis"]) else obs["absolute_axis"]
        return np.concatenate((distance, height, u_axis), axis=0) # (3,)

