import time
import couplednetworks as cn
import multiprocessing as mlt
import networkx as nx
import numpy as np
import math
import gym
from gym import spaces
import sys

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque


class GraphCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(GraphCNN, self).__init__(observation_space, features_dim)
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 8)
        self.linear1 = nn.Linear(2384*8,1024)
        self.linear2 = nn.Linear(1024,features_dim)

    def forward(self,data):
        repeat_num = data.shape[0]
        size = data.shape[1]
        data = data[0,:]
        #First preprocess data for torch_geometric by adding reverse edges
        edge_index = torch.empty(size*2,dtype=torch.int64)
        for i in range(0,size,2):
            edge_index[2*i:2*i+4] = torch.tensor([data[i],data[i+1],data[i+1],data[i]])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_index = edge_index.reshape(2,-1).to(device)
        x = torch.ones(2384).unsqueeze(1).to(device)
        #Now do graph convolutions
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x,edge_index)
        x = x.reshape(1,-1)
        x = self.linear1(x)
        x = self.linear2(x)
        if repeat_num > 1:
            x = x.repeat((repeat_num,1))
        return x

class CoupledNetsEnv(gym.Env):
    def __init__(self,p,attack_degree):
        super(CoupledNetsEnv,self).__init__()
        random_removal_fraction = 1 - p  # Fraction of nodes to remove
        self.attack_degree = attack_degree
        self.name = "CoupledNetsEnv-v0"
        self.num_envs = 1
        self.network_type = 'CFS-SW'
        self.net_a, self.net_b = cn.create_networks(self.network_type)
        self.num_nodes_attacked = int(math.floor(random_removal_fraction * self.net_a.size()))
        print('Attack degree of {} on {} nodes.'.format(self.attack_degree,self.num_nodes_attacked))
        self.action_space = spaces.MultiDiscrete([self.net_a.size() for i in range(self.attack_degree)])
        n = self.net_a.size() + self.net_b.size()
        m = max([self.net_a.number_of_nodes(),self.net_b.number_of_nodes()])
        self.observation_space = spaces.Box(low=0,high=2383,shape=(n*2,),dtype=np.int16) #TODO: adjacency on coupling

    def step(self,node_list):
        if type(node_list) == list:
            y = cn.attack_network(1, [self.net_a,self.net_b],rl_attack=node_list)
        else:
            y = cn.attack_network(1, [self.net_a,self.net_b],rl_attack=node_list.tolist())
        reward = 1-y[0][1]
        done = True
        info = {}
        return self.observation,reward,done,info
    
    def reset(self):
        self.net_a, self.net_b = cn.create_networks(self.network_type)
        edgelist_a = nx.to_edgelist(self.net_a)
        edgelist_b = nx.to_edgelist(self.net_b)
        edgelist = []
        for edge in edgelist_a:
            edgelist.append([edge[0],edge[1]])
        for edge in edgelist_b:
            edgelist.append([edge[0],edge[1]])
        self.observation = np.asarray(edgelist).transpose().flatten()
        return self.observation

def my_predict(observation,model,num_samples,env): #temporarily here until I can move it to a better place in the stable baselines code
    obs_tensor = obs_as_tensor(observation, model.device).unsqueeze(0)
    latent_pi,_,latent_sde = model.policy._get_latent(obs_tensor)
    act_dist = model.policy._get_action_dist_from_latent(latent_pi, latent_sde)
    actions = torch.stack([dist.sample(sample_shape=(num_samples,)).squeeze() for dist in act_dist.distribution],dim=1)
    nodes = []
    for i in range(actions.shape[0]):
        act = actions[i].cpu().numpy()
        overlap = any(a in nodes for a in act)
        retries = 0
        retry_lim = 100
        while overlap and retries < retry_lim:
            act = torch.stack([dist.sample() for dist in act_dist.distribution],dim=1).cpu().numpy()[0]
            overlap = any(a in nodes for a in act)
            retries += 1
        while overlap and retries >= retry_lim:
            act = env.action_space.sample()
            overlap = any(a in nodes for a in act)
            retries += 1  
        for a in act:
            nodes.append(a)
    return nodes

if __name__ == '__main__':
    train = True
    method = 'RL'
    if train:
        env = CoupledNetsEnv(0.90,1)
        observation = env.reset()
        tic = time.perf_counter()
        #model = PPO('MlpPolicy',env,verbose=1,n_steps=5,batch_size=env.num_nodes_attacked)
        # policy_kwargs = dict(
        #     features_extractor_class=GraphCNN,
        #     features_extractor_kwargs=dict(features_dim=64),
        # )
        model = PPO("MlpPolicy", env,n_steps=50,batch_size = env.num_nodes_attacked , verbose=1,tensorboard_log="./logs/")
        model.learn(total_timesteps=30000,tb_log_name = 'PPO_p10_singles',log_interval=10)
        model.save("./models/PPO_p10_pairs_4")
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
    else:
        tic = time.perf_counter()
        p_vals = np.linspace(0.65,0.995,70)
        num_runs = 25
        mean_rewards = []
        for p in p_vals:
            print('p =',p)
            env = CoupledNetsEnv(p,3)
            if method == 'RL':
                model = PPO.load('models/PPO_p05_singles_2',env=env)
                num_samples = int(env.num_nodes_attacked / env.attack_degree)
                observation = env.reset()
                reward_p = []
                for i in range(num_runs):
                    node_list = my_predict(observation,model,num_samples,env)
                    _,reward,_,_ = env.step(node_list)
                    reward_p.append(reward)
                    observation = env.reset()
            elif method == 'Random':
                observation = env.reset()
                reward_p = []
                for i in range(num_runs):
                    node_list = []
                    for j in range(env.num_nodes_attacked):
                        node_list.append(env.action_space.sample()[0])
                    _,reward,_,_ = env.step(node_list)
                    reward_p.append(reward)
                    observation = env.reset()
            elif method == 'Heuristic':
                observation = env.reset()
                reward_p = []
                for i in range(num_runs):
                    node_by_deg = sorted(env.net_a.degree, key=lambda x: x[1], reverse=True)
                    node_list = []
                    for j in range(0, env.num_nodes_attacked, 1):
                        node_list.append(node_by_deg[j][0])
                    _,reward,_,_ = env.step(node_list)
                    reward_p.append(reward)
                    observation = env.reset()
            mean_rewards.append(safe_mean(reward_p))

        data = np.stack([[1-p for p in p_vals],[r for r in mean_rewards]])
        np.save('./output/triplet_attack_gym_p05',data)
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
        import matplotlib.pyplot as plt
        # plot the result
        plt.plot([1-p for p in p_vals], [r for r in mean_rewards], 'bo')  # 'rx' for red x 'g+' for green + marker
        plt.xlabel('Percent of Nodes Attacked')
        plt.ylabel('Percent of Nodes Down After Cascade')
        #plt.plot(x, average_p_half, 'bo')  # 'rx' for red x 'g+' for green + marker
        # show the plot
        plt.ylim([0,1])
        plt.show()


