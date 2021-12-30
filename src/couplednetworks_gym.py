import time
import couplednetworks as cn
import multiprocessing as mlt
import networkx as nx
import numpy as np
import math
import gym
from gym import spaces
import sys
from typing import Callable
from networkx.algorithms.centrality import degree_centrality,katz_centrality,betweenness_centrality,harmonic_centrality
import random

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.utils import set_random_seed

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque

class CoupledNetsEnv(gym.Env):
    def __init__(self,p,attack_degree):
        super(CoupledNetsEnv,self).__init__()
        self.p = p
        self.attack_degree = attack_degree
        self.name = "CoupledNetsEnv-v1"
        self.num_envs = 1
        self.network_type = 'SF'
        #self.net_a, self.net_b = cn.create_networks(self.network_type)
        self.net_a, self.net_b = create_simple_net()
        self.num_nodes_attacked = int(math.floor(self.p * self.net_b.number_of_nodes()))
        #print('Attack degree of {} on {} nodes.'.format(self.attack_degree,self.num_nodes_attacked))
        net_feature_size = 4#[degree,degree_centrality,katz_centrality,betweenness_centrality,harmonic_centrality]
        self.action_space = spaces.MultiDiscrete([self.net_b.number_of_nodes() for i in range(self.attack_degree)])
        self.observation_space = spaces.Box(low=-1,high=1,shape=(self.net_b.number_of_nodes(),net_feature_size),dtype=np.float32)

    def step(self,node_list):
        node_list_def = []#[self.action_space.sample()[0] for i in range(self.num_nodes_attacked)]
        if type(node_list) != list:
            node_list = node_list.tolist()
        final_node_list = []
        for node in node_list:
            if node not in node_list_def:
                final_node_list.append(node)  
        y = cn.attack_network(1, [self.net_a,self.net_b],rl_attack=final_node_list)
        reward = y[0][1]
        done = True
        info = {}
        return self.observation_space.sample(),reward,done,info
    
    def reset(self):
        #self.net_a, self.net_b = cn.create_networks(self.network_type)
        self.net_a, self.net_b = create_simple_net()
        node_degrees = [node[1] for node in self.net_b.degree]
        max_degree = max(node_degrees)
        min_degree = min(node_degrees)
        node_degrees = [2*(deg-min_degree)/(max_degree-min_degree)-1 if (max_degree-min_degree) != 0 else 0 for deg in node_degrees]

        degree_centralities = degree_centrality(self.net_b).values()
        max_c = max(degree_centralities)
        min_c = min(degree_centralities)
        degree_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in degree_centralities]

        # katz_centralities = katz_centrality(self.net_b).values()
        # max_c = max(katz_centralities)
        # min_c = min(katz_centralities)
        # katz_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in katz_centralities]


        betweenness_centralities = betweenness_centrality(self.net_b).values()
        max_c = max(betweenness_centralities)
        min_c = min(betweenness_centralities)
        betweenness_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in betweenness_centralities]


        harmonic_centralities = harmonic_centrality(self.net_b).values()
        max_c = max(harmonic_centralities)
        min_c = min(harmonic_centralities)
        harmonic_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in harmonic_centralities]

        obs = np.stack([node_degrees,degree_centralities,betweenness_centralities,harmonic_centralities]).T#katz_centralities,
        return obs

def create_simple_net():
    random.seed(np.random.randint(0,10000))
    net = nx.Graph()
    nodes = [0,1,2]
    net.add_nodes_from(nodes)
    double_node = random.sample(nodes,1)[0]
    single_nodes = nodes.copy()
    single_nodes.remove(double_node)
    net.add_edges_from([(single_nodes[0],double_node),(double_node,single_nodes[1])])
    return [net,net]

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

def make_CN_env(p: float, attack_degree: int, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    :param p: 1-p is the percentage of nodes to attack
    :param attack_degree: the degree of node attack
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = CoupledNetsEnv(p,attack_degree)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    train = False
    num_cpu = 1
    p = 0.2
    if train:
        if num_cpu > 1:
            env = VecMonitor(SubprocVecEnv([make_CN_env(p,1,i) for i in range(num_cpu)]))
            model = PPO("MlpPolicy", env,n_steps=4,batch_size = env.get_attr('num_nodes_attacked')[0], verbose=1,tensorboard_log="./logs/")
        else:
            env = CoupledNetsEnv(p,1)
            obs = env.reset()
            model = A2C("MlpPolicy", env,verbose=1,tensorboard_log="./logs/",n_steps=1000)#,learning_rate=0.0007)#batch_size = env.num_nodes_attacked)
        tic = time.perf_counter()
        model.learn(total_timesteps=100e3,tb_log_name = 'p20_A2C_topo_3n',log_interval=1)
        model.save("./models/p20_A2C_topo_3n")
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
    else:
        method = 'Heuristic'
        print('Attack method: {}'.format(method))
        tic = time.perf_counter()
        p_vals = [0.4]
        #p_vals = np.linspace(0.65,0.995,70)
        #p_vals = np.linspace(0.9,0.905,1)
        num_runs = 100
        data = np.zeros((len(p_vals),num_runs+1))
        data[:,0] = p_vals
        if num_cpu > 1:
            vec_envs = []
            num_vec_envs = math.ceil(len(p_vals)/num_cpu)
            for j in range(num_vec_envs):
                print('{} of {}'.format(j+1,num_vec_envs))
                num_envs = min([num_cpu,len(p_vals)-num_cpu*j])
                env = VecMonitor(SubprocVecEnv([make_CN_env(p_vals[i],1,i) for i in range(j*num_cpu,j*num_cpu+num_envs)]))
                if method == 'RL':
                    print('Multiprocessing not supported for RL method')
                    exit()
                elif method == 'Random':
                    observation = env.reset()
                    for i in range(num_runs):
                        node_lists = []
                        for k in range(num_envs):
                            node_list = []
                            for l in range(env.get_attr('num_nodes_attacked',indices=k)[0]):
                                rand_node = np.random.randint(env.get_attr('net_b',indices=0)[0].number_of_nodes())
                                while rand_node in node_list:
                                    rand_node = np.random.randint(env.get_attr('net_b',indices=0)[0].number_of_nodes())
                                node_list.append(rand_node)
                            node_lists.append(node_list)
                        _,rewards,_,_ = env.step(node_lists)
                        for k in range(num_envs):
                            idx = np.where(p_vals == env.get_attr('p',indices=k)[0])[0][0]
                            data[idx,i+1] = rewards[k]
                            print("Run {}. p_in: {}, p_out: {}".format(i,1-p_vals[idx],data[idx,i+1]))
                elif method == 'Heuristic':
                    observation = env.reset()
                    for i in range(num_runs):
                        node_lists = []
                        for k in range(num_envs):
                            node_list = []
                            node_by_deg = sorted(env.get_attr('net_b',indices=0)[0].degree, key=lambda x: x[1], reverse=True)
                            for l in range(env.get_attr('num_nodes_attacked',indices=k)[0]):
                                node_list.append(node_by_deg[l][0])
                            node_lists.append(node_list)
                        _,rewards,_,_ = env.step(node_lists)
                        for k in range(num_envs):
                            idx = np.where(p_vals == env.get_attr('p',indices=k)[0])[0][0]
                            data[idx,i+1] = rewards[k]
                            print("Run {}. p_in: {}, p_out: {}".format(i,1-p_vals[idx],data[idx,i+1]))
        else:
            for p in p_vals:
                print('p =',p)
                env = CoupledNetsEnv(p,3)
                if method == 'RL':
                    model = PPO.load('models/p20_A2C_topo_25n',env=env)
                    num_samples = int(env.num_nodes_attacked / env.attack_degree)
                    observation = env.reset()
                    for i in range(num_runs):
                        node_list = my_predict(observation,model,num_samples,env)
                        node_by_deg = [node for node in sorted(env.net_b.degree, key=lambda x: x[1], reverse=True)]
                        print('RL nodes',node_list)
                        print('Node deg nodes',node_by_deg)
                        _,reward,_,_ = env.step(node_list)
                        print(reward)
                        idx = 0#np.where(p_vals == p)
                        data[idx,i+1] = reward   
                        observation = env.reset()
                elif method == 'Random':
                    observation = env.reset()
                    for i in range(num_runs):
                        node_list = []
                        for j in range(env.num_nodes_attacked):
                            rand_node = np.random.randint(env.net_b.number_of_nodes())
                            while rand_node in node_list:
                                rand_node = np.random.randint(env.net_b.number_of_nodes())
                            node_list.append(rand_node)
                        _,reward,_,_ = env.step(node_list)
                        data[0,i+1] = reward                        
                        observation = env.reset()
                elif method == 'Heuristic':
                    observation = env.reset()
                    for i in range(num_runs):
                        node_by_deg = sorted(env.net_b.degree, key=lambda x: x[1], reverse=True)
                        node_list = []
                        for j in range(0, env.num_nodes_attacked, 1):
                            node_list.append(node_by_deg[j][0])
                        _,reward,_,_ = env.step(node_list)
                        #idx = np.where(p_vals == p)
                        data[0,i+1] = reward                        
                        observation = env.reset()
                elif method == 'File':
                    for i in range(num_runs):
                        #get node list from file
                        observation = env.reset()
                        node_list = cn.get_nodes_from_file(3,p,env.net_a)
                        node_list = [node-1 for node in node_list] #Shift from 1-index to 0-index
                        _,reward,_,_ = env.step(node_list)
                        idx = np.where(p_vals == p)
                        data[idx,i+1] = reward

        np.save('./output/heuristic_attack_ctl_defense',data)
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
        import matplotlib.pyplot as plt
        # plot the result
        mean_rewards = [np.mean(data[i][1:]) for i in range(len(p_vals))]
        print(mean_rewards)
        plt.plot(data[:,0], [r for r in mean_rewards], 'bo')  # 'rx' for red x 'g+' for green + marker
        plt.xlabel('Percent of Nodes Attacked')
        plt.ylabel('Percent of Nodes Down After Cascade')
        #plt.plot(x, average_p_half, 'bo')  # 'rx' for red x 'g+' for green + marker
        # show the plot
        plt.ylim([0,1])
        plt.show()


