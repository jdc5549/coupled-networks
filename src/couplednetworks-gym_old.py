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
from collections import defaultdict, deque



class CoupledNetsEnv(gym.Env):
    def __init__(self,p):
        super(CoupledNetsEnv,self).__init__()
        random_removal_fraction = 1 - p  # Fraction of nodes to remove
        self.network_type = 'CFS-SW'
        self.net_a, self.net_b = cn.create_networks(self.network_type)
        #print(nx.utils.flatten(self.net_a.edges))
        n = self.net_a.size() + self.net_b.size()
        m = max([self.net_a.number_of_nodes(),self.net_b.number_of_nodes()])
        self.observation_space = spaces.Box(low=0,high=2383,shape=(n*2,),dtype=np.int16) #TODO: adjacency on coupling
        num_nodes_attacked = 1#int(math.floor(random_removal_fraction * self.net_a.size()))
        self.action_space = spaces.MultiDiscrete([self.net_a.size() for i in range(num_nodes_attacked)])
        
    def step(self,action):
        #use self.net_a and self.net_b
        action_list = list(action)
        y = cn.attack_network(1, [self.net_a,self.net_b],rl_attack=action_list)
        reward = 1-y[0][1]
        done = True
        info = {}
        return self.observation,reward,done,info
    
    def reset(self):
        self.net_a, self.net_b = cn.create_networks(self.network_type)
        edgelist_a = nx.to_edgelist(self.net_a)
        edgelist_b = nx.to_edgelist(self.net_a)
        edgelist = []
        for edge in edgelist_a:
            edgelist.append([edge[0],edge[1]])
        for edge in edgelist_b:
            edgelist.append([edge[0],edge[1]])
        self.observation = np.asarray(edgelist).transpose().flatten()
        return self.observation

if __name__ == '__main__':
    #action = env.action_space.sample()
    #_,reward,done,info = env.step(action)
    #print(reward)
    #check_env(env)
    model = PPO.load('./models/PPO_test')
    #print(model.policy.evaluate_actions())
    ps = np.linspace(0.65,1,70)
    #for p in ps:
    p = 0.1
    env = CoupledNetsEnv(p)
    observation = env.reset()
    actions = [i for i in range(env.net_a.size())]
    values,_,_ = model.policy.evaluate_actions(torch.tensor(observation),actions)
        #print(values.shape)
        #print(action)

    #model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./logs/")
    #model.set_parameters("./models/PPO_test", exact_match=True, device='auto')
    #model.learn(total_timesteps=20000)
    #model.save("./models/PPO_test")
    #observation = env.reset()
    # action = env.action_space.sample()
    # observation,reward,done,info = env.step(action)
    # print(observation.shape)
    # print(reward)
