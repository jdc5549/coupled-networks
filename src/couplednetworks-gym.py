import couplednetworks as cn
import multiprocessing as mlt
import networkx as nx
import numpy as np
import math
import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from collections import defaultdict, deque

class GraphCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(GraphCNN, self).__init__(observation_space, features_dim)
        self.conv1 = GCNConv(observation_space.size[0], 64)
        self.conv2 = GCNConv(64, action_space.size[0])
        
        
class CoupledNetsEnv(gym.Env):
    def __init__(self):
        super(CoupledNetsEnv,self).__init__()
        p = 0.9
        random_removal_fraction = 1 - p  # Fraction of nodes to remove
        self.network_type = 'CFS-SW'
        self.net_a, self.net_b = cn.create_networks(self.network_type)
        self.n = self.net_a.size() + self.net_b.size()
        self.nodes = self.net_a.number_of_nodes() + self.net_b.number_of_nodes()
        self.observation_space = spaces.Box(low=0,high=1,shape=(self.n*2,),dtype=np.int16) #TODO: adjacency on coupling
        self.num_nodes_attacked = 1#int(math.floor(random_removal_fraction * self.nodes))
        #self.action_space = spaces.MultiDiscrete([self.nodes for i in range(num_nodes_attacked)])
        self.action_space = Discrete(self.nodes)
        
    def step(self,action):
        #use self.net_a and self.net_b
        if action not in self.node_list: #TODO: Implement action mask
            self.node_list.append(action)
        y = cn.attack_network(1, [self.net_a,self.net_b],rl_attack=self.node_list)
        reward = 1-y[0][1]-last_reward
        last_reward = reward
        self.attack_count += 1
        if self.attack_count >= self.num_nodes_attacked:
            done = True

        #TODO: handle tagging in the observation
        return self.observation,reward,done,info
    
    def reset(self):
        self.attack_count = 0
        self.last_reward = 0
        self.node_list = []
        self.net_a, self.net_b = cn.create_networks(self.network_type)
        edgelist_a = nx.to_edgelist(self.net_a)
        a_nodes = self.net_a.number_of_nodes()
        edgelist_b = nx.to_edgelist(self.net_a)
        edgelist = []
        for edge in edgelist_a:
            edgelist.append([edge[0],edge[1]])
            #edgelist.append([edge[1],edge[0]])
        for edge in edgelist_b:
            edgelist.append([a_nodes + edge[0],a_nodes + edge[1]])
            #edgelist.append([a_len + edge[1],a_len + edge[0]])
        self.observation = np.asarray(edgelist).transpose().flatten()/self.nodes
        return self.observation

if __name__ == '__main__':
    env = CoupledNetsEnv()
    observation = env.reset()
    #print(observation.shape)
    #print(max(observation))
    #print(env.observation_space)
    #print(env.action_space)
    #check_env(env)
    #model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./logs/")
    #model.learn(total_timesteps=10000)
    #model.save("./models/PPO_1a")
    observation = env.reset()
    action = env.action_space.sample()
    observation,reward,done,info = env.step(action)
    # print(observation.shape)
    # print(reward)
