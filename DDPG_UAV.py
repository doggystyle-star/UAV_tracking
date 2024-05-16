import os
import gym
import time
import rospy
import random
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorlayer as tl
from gym_examples.wrappers import RelativePosition
import matplotlib.pyplot as plt
# Parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='train or test', default='train')
parser.add_argument('--save_path', default='ddpg_model', help='folder to save if mode == train else model path')
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--noisy_scale', type=float, default=1e-2)
args = parser.parse_args()

if args.mode == 'train':
    os.makedirs(args.save_path, exist_ok=True)
    actor1 = os.path.join(args.save_path, 'actor')
    os.makedirs(actor1, exist_ok=True)
    actor2 = os.path.join(args.save_path, 'target_actor')
    os.makedirs(actor2, exist_ok=True)
    critic1 = os.path.join(args.save_path, 'critic')
    os.makedirs(critic1, exist_ok=True)
    critic2 = os.path.join(args.save_path, 'target_critic')
    os.makedirs(critic2, exist_ok=True)
# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
noise_scale = args.noisy_scale

# Environment setup
env = gym.make('gym_examples/GazeboWorld-v0', render_mode='human')
env = env.unwrapped
env = RelativePosition(env)

# Hyperparameters
number_timesteps = 50000
test_number_timesteps = 1000
explore_timesteps = number_timesteps
epsilon = lambda i_iter: (1 - 0.99 * min(1, i_iter / explore_timesteps)) * 0.8
buffer_size = explore_timesteps // 10 * 200
target_q_update_freq = 100

o_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
noise_update_freq = 50

LR_A = 0.001                # learning rate for actor
LR_C = 0.002                # learning rate for critic
GAMMA = 0.9                 # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 1000      # size of replay buffer
BATCH_SIZE = 128
TEST_PER_EPISODES = 10
warm_start = BATCH_SIZE *2
MAX_EP_STEPS = 200

W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
b_init = tf.constant_initializer(0.1)

class ActorNetwork(tl.models.Model):
    def __init__(self, name, input_state_shape):
        super(ActorNetwork, self).__init__(name=name)
        hidden_dim = 256
        inputs = tl.layers.Input(input_state_shape, name='A_input')
        x = tl.layers.Dense(hidden_dim, tf.nn.relu, name='A_l1')(inputs)
        x = tl.layers.Dense(n_units=a_dim, name='A_a')(x)
        x = tl.layers.Lambda(lambda x: x * np.array(a_bound))(x)  # Lambda function is used for zooming output
        self.actor_model = tl.models.Model(inputs=inputs, outputs=x, name= 'Actor' + name)
    def get_actor_model(self):
        return self.actor_model
    
class CriticNetwork(tl.models.Model):
    def __init__(self, name, input_state_shape,input_action_shape):
        super(CriticNetwork, self).__init__(name=name)
        hidden_dim = 256
        o = tl.layers.Input(input_state_shape, name = 'C_s_input')
        a = tl.layers.Input(input_action_shape, name= 'C_a_input')
        x = tl.layers.Concat(concat_dim=1)([o, a])
        x = tl.layers.Dense(n_units=hidden_dim, act= tf.nn.tanh, name='C_l1')(x) 
        x = tl.layers.Dense(n_units=1, name='C_out')(x)
        self.Critic_model = tl.models.Model(inputs=[o, a], outputs=x, name='Critic'+name)
    def get_critic_model(self):
        return self.Critic_model
    
def sync(net, net_tar):
    """Copy q network to target q network"""
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)

def log_softmax(x, dim):
    temp = x - np.max(x, dim, keepdims=True)
    return temp - np.log(np.exp(temp).sum(dim, keepdims=True))


def softmax(x, dim):
    temp = np.exp(x - np.max(x, dim, keepdims=True))
    return temp / temp.sum(dim, keepdims=True)

# DDPG agent
class DDPG(object):
    def __init__(self):
        # memory is used for store：
        # MEMORY_CAPACITY，o_dim * 2 + a_dim + 1：2*state，1*action，1*reward
        self.memory = np.zeros((MEMORY_CAPACITY, o_dim * 2 + a_dim + 1), dtype=np.float32)

        self.actor_net = ActorNetwork('actor',[None, o_dim]).get_actor_model()
        self.critic_net = CriticNetwork('critic',[None, o_dim], [None, a_dim]).get_critic_model()
        
        self.pointer = 0
        if args.mode == 'train':
            self.actor_net.train()
            self.critic_net.train()
            self.target_actor_net = ActorNetwork('target_actor',[None, o_dim]).get_actor_model()
            self.target_critic_net = CriticNetwork('target_critic', [None, o_dim], [None, a_dim]).get_critic_model()
            sync(self.actor_net, self.target_actor_net)
            sync(self.critic_net, self.target_critic_net)
            self.target_actor_net.eval()
            self.target_critic_net.eval()
        else:
            self.actor_net.train()
            self.critic_net.train()
            self.target_actor_net = ActorNetwork('target_actor',[None, o_dim]).get_actor_model()
            self.target_critic_net = CriticNetwork('target_critic', [None, o_dim], [None, a_dim]).get_critic_model()
            sync(self.actor_net, self.target_actor_net)
            sync(self.critic_net, self.target_critic_net)
            self.target_actor_net.eval()
            self.target_critic_net.eval()
            print("Begin loading...\n")

            path_a = os.path.join(args.save_path,'actor/600.npz')
            path_at = os.path.join(args.save_path,'target_actor/600.npz')
            path_c = os.path.join(args.save_path,'critic/600.npz')
            path_ct = os.path.join(args.save_path,'target_critic/600.npz')

            tl.files.load_and_assign_npz_dict(name=path_a, network=self.actor_net)
            tl.files.load_and_assign_npz_dict(name=path_at, network=self.target_actor_net)
            tl.files.load_and_assign_npz_dict(name=path_c, network=self.critic_net)
            tl.files.load_and_assign_npz_dict(name=path_ct, network=self.target_critic_net)
            print("Successfully loaded...\n")
        self.niter = 0            #save.npz
        #ema，sliding average value method
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=LR_A)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=LR_C)
        self.noise_scale = noise_scale

    def ema_update(self):
        """
        sliding average update
        """
        paras = self.actor_net.trainable_weights + self.critic_net.trainable_weights    
        self.ema.apply(paras)                                                  
        for i, j in zip(self.target_actor_net.trainable_weights + self.target_critic_net.trainable_weights, paras):
            i.assign(self.ema.average(j))                                      

    def get_action(self, obv):
        low_limits = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        high_limits = np.array([10.0, 1.0, 5.0, 4.9, 2, 1, 0.05, 0.01, 0.005])
        eps = epsilon(self.niter)
        if args.mode == 'train':
            if random.random() < eps:
                return np.random.uniform(low=low_limits, high=high_limits, size=(9,))
            obv = np.expand_dims(obv, 0).astype('float32')
            if self.niter < explore_timesteps:
                self.actor_net.noise_scale = self.noise_scale
                action_ptb = self.actor_net(obv).numpy()
                self.actor_net.noise_scale = 0
                if self.niter % noise_update_freq == 0:
                    action = self.actor_net(obv).numpy()
                    kl_ptb = (log_softmax(action, 1) - log_softmax(action_ptb, 1))
                    kl_ptb = np.sum(kl_ptb * softmax(action, 1), 1).mean()
                    kl_explore = -np.log(1 - eps + eps / o_dim)
                    if kl_ptb < kl_explore:
                        self.noise_scale *= 1.01
                    else:
                        self.noise_scale /= 1.01

                return action_ptb[0]
            else:
                return self.actor_net(obv).numpy()[0]
        else:
            obv = np.expand_dims(obv, 0).astype('float32') 
            return self.actor_net(obv).numpy()[0]

    def train(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    
        bt = self.memory[indices, :]                                    
        b_o = bt[:, :o_dim]                                             
        b_a = bt[:, o_dim:o_dim + a_dim]                                
        b_r = bt[:, -o_dim - 1:-o_dim]                                  
        b_o_ = bt[:, -o_dim:]                                           

        self._critic_loss(b_o, b_a, b_r, b_o_)
        self._actor_loss(b_o)
        self.ema_update()

        self.niter += 1
        if self.niter % target_q_update_freq == 0:
            sync(self.actor_net, self.target_actor_net)
            sync(self.critic_net, self.target_critic_net)
        if self.niter % (100) == 0:
            path_a = os.path.join(args.save_path,'actor', '{}.npz'.format(self.niter))
            path_at = os.path.join(args.save_path,'target_actor', '{}.npz'.format(self.niter))
            path_c = os.path.join(args.save_path,'critic', '{}.npz'.format(self.niter))
            path_ct = os.path.join(args.save_path,'target_critic', '{}.npz'.format(self.niter))
            tl.files.save_npz_dict(self.actor_net.trainable_weights, name=path_a)
            tl.files.save_npz_dict(self.target_actor_net.trainable_weights, name=path_at)
            tl.files.save_npz_dict(self.critic_net.trainable_weights, name=path_c)
            tl.files.save_npz_dict(self.target_critic_net.trainable_weights, name=path_ct)

    def store_transition(self, o, a, r, o_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        o = o.astype(np.float32)
        o_ = o_.astype(np.float32)

        transition = np.hstack((o, a, [r], o_))

        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
    @tf.function
    def _critic_loss(self, b_o, b_a, b_r, b_o_):
        with tf.GradientTape() as tape:
            a_ = self.target_actor_net(b_o_)
            target_q = self.target_critic_net([b_o_, a_])
            y = b_r + GAMMA* target_q
            # y = tf.stop_gradient(y)
            q = self.critic_net([b_o, b_a])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic_net.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(c_grads, self.critic_net.trainable_weights))

    @tf.function
    def _actor_loss(self, b_o):
        with tf.GradientTape() as tape:
            a = self.actor_net(b_o)
            q = self.critic_net([b_o,a])
            a_loss = -tf.reduce_mean(q)
        a_grads = tape.gradient(a_loss, self.actor_net.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(a_grads, self.actor_net.trainable_weights))

    @tf.function
    def _tderror_func(self, b_o, b_a, b_r, b_o_, b_d):
        target_a = self.target_actor_net(b_o_)
        target_q = self.target_critic_net([b_o_,target_a])
        b_d = tf.expand_dims(b_d, axis=-1)  
        b_r = tf.expand_dims(b_r, axis=-1) 
        y = b_r + GAMMA * (1 - b_d) * target_q
        y = tf.stop_gradient(y)
        q = self.critic_net([b_o,target_a])
        return y - q

# Main function
if __name__ == '__main__':
    ddpg = DDPG()
    rate = rospy.Rate(20)
    plt.ion()
    while not rospy.is_shutdown():
        if args.mode == 'train':
            o, _ = env.reset()
            reward_buffer = []
            nepisode = 0  
            ep_reward = 0          
            t =time.time()
            for i in range(1, number_timesteps + 1): 
                a = ddpg.get_action(o)
                # a = clip_action(a)
                env.run()
                rate.sleep()
                o_, r, done, _, info = env.step(a)
                print("o",o)
                print("a",a)
                print("o_",o_)

                    # 保存s，a，r，s_
                ddpg.store_transition(o, a, r / 10, o_)
                    
                ep_reward += r
                if ddpg.pointer > MEMORY_CAPACITY:
                    ddpg.train()
                if done:
                    o, _ = env.reset()
                    nepisode += 1
                    reward_buffer.append(ep_reward)
                    ep_reward = 0
                else:
                    o = o_
                    
                # 每一步
                if info.get('episode'):
                    reward, length = info['episode']['r'], info['episode']['l']
                    fps = int(length / (time.time() - t))
                    print(
                        'Time steps so far: {}, episode so far: {}, '
                        'episode reward: {:.4f}, episode length: {}, FPS: {}'.format(i, nepisode, reward, length, fps))
                    t = time.time()

                if reward_buffer and done:
                    plt.ion()
                    plt.cla()
                    plt.title('DDPG')
                    plt.plot(np.array(range(len(reward_buffer))), reward_buffer)  # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.ylim(-200, 10)
                    plt.draw()
                    plt.pause(0.1)
            plt.ioff()
            plt.show()

        # test 
        else:
            nepisode = 0
            suc_epi = 0
            o, _ = env.reset()
            print(o)
            for i in tqdm(range(1, test_number_timesteps + 1)):
                a = ddpg.get_action(o)
                env.run()
                # rate.sleep()
                o_, r, done, _, info = env.step(a)
                if done:
                    if info['episode']['achieve_target']:
                        suc_epi += 1
                    o, _ = env.reset()
                    nepisode += 1
                else:
                    o = o_
                if info.get('episode'):
                    reward, length = info['episode']['r'], info['episode']['l']
                    print(
                        'Time steps so far: {}, episode so far: {},' 
                        'episode reward: {:.4f}, episode length: {}'.format(i, nepisode, reward, length))
            print(f'Successful episode: {suc_epi} / {nepisode}')