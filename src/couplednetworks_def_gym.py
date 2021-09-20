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
import json


from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.utils import set_random_seed

from couplednetworks_gym import my_predict, CoupledNetsEnv

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque

import matlab.engine
import h5py

class CoupledNetsDefEnv(gym.Env):
    def __init__(self,atk_model,p,attack_degree,config_name):
        super(CoupledNetsDefEnv,self).__init__()
        self.name = "CoupledNetsDefEnv-v0"
        self.config_name = config_name
        config = json.load(open(config_name))
        ps = h5py.File(config['power_systems_data_location'])['ps']
        self.ps = dict(ps)
        del self.ps['B']
        Pd = self.ps['shunt'][1,:]
        Pg = self.ps['gen'][1,:]
        flow = ps['branch'][11,:]
        #print(ps['shunt'][5,:].size) #factor
        #print(ps['gen'][7,:].size)   #status
        self.observation_space = spaces.Box(low=-100,high=100,shape=(Pd.size+Pg.size+flow.size,))
        self.action_space = spaces.Box(low=-1,high=1,shape=(Pd.size+Pg.size,))
        self.atk_model = atk_model
        self.p = p
        self.attack_degree = attack_degree
        self.nodes_attacked = []

    def step(self,deltas):
        #Get current loads and generations from the ps data structure from MATLAB
        Pd = np.multiply(self.ps['shunt'][:,1],self.ps['shunt'][:,5])
        Pg = np.multiply(self.ps['gen'][:,1],self.ps['gen'][:,7])
        Pg_min = np.multiply(self.ps['gen'][:,9],self.ps['gen'][:,7])
        #shift and scale deltas into allowable range of [-x,0] for each measured value x
        deltas =  deltas - 1
        delta_Pd = deltas[:Pd.size]
        delta_Pg = deltas[Pd.size:]
        with matlab.engine.start_matlab() as eng:
            eng.addpath('./src/',nargout=0)
            delta_Pd_matlab = matlab.double(delta_Pd.tolist())
            delta_Pg_matlab = matlab.double(delta_Pg.tolist())
            if self.t == 0:
                nodes_attacked_matlab = matlab.double([node+1 for node in self.nodes_attacked])
            else:
                nodes_attacked_matlab = []
            ps_matlab = {}
            for key in self.ps:
                if key != 'baseMVA':
                    ps_matlab[key] = matlab.double(self.ps[key].tolist())
                else:
                    ps_matlab[key] = self.ps[key]
            ps,self.t,done,p_out = eng.cn_step_def(delta_Pd_matlab,delta_Pg_matlab,nodes_attacked_matlab,self.t,self.it_no,ps_matlab,self.opt,nargout=4)
        for key in ps:
            if key != 'baseMVA':
                self.ps[key] = np.asarray(ps[key])
            else:
                self.ps[key] = float(ps[key])
        #calculate the overflow
        flow_max = self.ps["branch"][:,6]
        flow = self.ps["branch"][:,11]
        overload_penalty = 0
        for i in range(len(flow)):
            overload = flow[i] - flow_max[i]
            if overload > 0:
                overload_penalty += overload
        reward = np.sum(delta_Pd) - overload_penalty
        # if done:
        #     reward = -p_out
        # else:
        #     reward = 0
        info = {}
        info['p_out'] = p_out
        obs = self.obs_from_ps()
        self.it_no += 1
        return obs,reward,done,info
    
    def reset(self):
        self.t = 0
        self.it_no = 0
        env = CoupledNetsEnv(self.p,self.attack_degree)
        num_samples = int(env.num_nodes_attacked / env.attack_degree)
        if self.atk_model == 'Heuristic':
            node_by_deg = sorted(env.net_b.degree, key=lambda x: x[1], reverse=True)
            node_list = []
            for j in range(0, env.num_nodes_attacked, 1):
                node_list.append(node_by_deg[j][0])
            self.nodes_attacked = node_list
        elif self.atk_model == 'Random':
            node_list = []
            for j in range(env.num_nodes_attacked):
                rand_node = np.random.randint(env.net_b.number_of_nodes())
                while rand_node in node_list:
                    rand_node = np.random.randint(env.net_b.number_of_nodes())
                node_list.append(rand_node)
            self.nodes_attacked = node_list
        else:
            self.nodes_attacked = my_predict(env.reset(),self.atk_model,num_samples,env)
        coupled_nodes = cn.find_coupled_nodes(-1,-1)  # get the coupled nodes
        with matlab.engine.start_matlab() as eng:
            eng.addpath('./src/',nargout=0)
            nodes_attacked_matlab = matlab.double([node+1 for node in self.nodes_attacked])
            coupled_nodes_matlab = matlab.double(coupled_nodes)
            ps,opt = eng.cn_reset_def(nodes_attacked_matlab,coupled_nodes_matlab,self.config_name,nargout=2)
        self.opt = opt
        for key in ps:
            if key != 'baseMVA':
                self.ps[key] = np.asarray(ps[key])
            else:
                self.ps[key] = float(ps[key])
        obs = self.obs_from_ps()
        return obs

    def obs_from_ps(self):
        #The vars in this dictionary are transposed compared to laoding from file in __init__. Not sure why.
        Pd = np.multiply(self.ps['shunt'][:,1],self.ps['shunt'][:,5])
        Pg = np.multiply(self.ps['gen'][:,1],self.ps['gen'][:,7])
        flow = self.ps['branch'][:,11]
        #normalize Pd,Pg, and flow into the range (-1,1)
        Pd_max = 500 #just using an estimate, as there's no hard limit
        for i in range(len(Pd)):
            if abs(Pd[i]) > Pd_max:
                print('Load value larger than maximum value estimate used to normalize obs. Consider increasing max value.')
            Pd[i] = (Pd[i]/Pd_max -0.5)*2
        Pg_max = self.ps['gen'][:,8]
        Pg_min = self.ps['gen'][:,9]
        ge_status = self.ps['gen'][:,7]
        for i in range(len(Pg)):
            if not(Pg_max[i] > Pg_min[i]) or ge_status[i] == 0: #catch numerical issues resulting from cases where Pg_max[i] = Pg_min[i]
                Pg[i] = 0
            else:
                Pg[i] = ((Pg[i]-Pg_min[i])/(Pg_max[i]-Pg_min[i]) -0.5)*2
        #print(min(Pg))
        flow_max = self.ps['branch'][:,6]
        flow_min = -flow_max
        for i in range(len(flow)):
            flow[i] = ((flow[i]-flow_min[i])/(flow_max[i]-flow_min[i]) -0.5)*2
        return np.concatenate((Pd,Pg,flow))

def make_CN_def_env(atk_model: str, p: float, attack_degree: int,config: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    :param p: 1-p is the percentage of nodes to attack
    :param attack_degree: the degree of node attack
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = CoupledNetsDefEnv(atk_model,p,attack_degree,config)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    train = True
    num_cpu = 48
    if train:
        if num_cpu > 1:
            env = VecMonitor(SubprocVecEnv([make_CN_def_env('Heuristic',0.985,1,'config/config_cn_runner_test_intermediate.json',i) for i in range(num_cpu)]))
            model = PPO("MlpPolicy", env,n_steps=50,batch_size = 10, verbose=1,tensorboard_log="./logs/")
        else:
            env = CoupledNetsDefEnv('p10_singles_mp56_real',0.9,1,'config/config_cn_runner_test_intermediate.json')
            model = PPO("MlpPolicy", env,n_steps=64,batch_size = 16, verbose=1,tensorboard_log="./logs/")
        tic = time.perf_counter()
        model.learn(total_timesteps=25000,tb_log_name = 'heuristic_atk_rl_defense_mp50_p015',log_interval=1)
        model.save("./models/heuristic_atk_rl_defense_mp50_p015")
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
    else:
        tic = time.perf_counter()
        p_vals = np.linspace(0.65,0.995,70)
        #p_vals = np.linspace(0.9,0.91,2)
        num_runs = 5
        data = np.zeros((len(p_vals),num_runs+1))
        data[:,0] = [1-p_val for p_val in p_vals]
        if num_cpu > 1:
            vec_envs = []
            num_vec_envs = math.ceil(len(p_vals)/num_cpu)
            for j in range(num_vec_envs):
                print('{} of {}'.format(j+1,num_vec_envs))
                num_envs = min([num_cpu,len(p_vals)-num_cpu*j])
                env = VecMonitor(SubprocVecEnv([make_CN_def_env('Heuristic',p_vals[i],1,'config/config_cn_runner_test_intermediate.json',i) for i in range(j*num_cpu,j*num_cpu+num_envs)]))
                model = PPO.load('models/rl_atk_rl_defense_mp50',env=env)
                done_counts = np.zeros(num_envs,dtype=int)
                obs = env.reset()
                while any(done_counts < num_runs):
                    #deltas,_ = model.predict(obs)
                    deltas = np.asarray([space.sample() for space in env.get_attr('action_space')])
                    obs,_,dones,infos = env.step(deltas)
                    for i in range(len(dones)):
                        if dones[i]:
                            if done_counts[i] < num_runs:
                                idx = np.where(p_vals == env.get_attr('p',indices=i)[0])[0][0]
                                data[idx,done_counts[i]+1] = infos[i]['p_out']
                                print("Run {}. p_in: {}, p_out: {}".format(done_counts[i],1-p_vals[idx],data[idx,done_counts[i]+1]))
                            done_counts[i] += 1
        else:
            for p in p_vals:
                print('p =',p)
                env = CoupledNetsDefEnv('Random',p,1,'config/config_cn_runner_test_intermediate.json')
                model = PPO.load('models/def_test_mp50',env=env)
                reward_p = []
                for i in range(num_runs):
                    done = False
                    obs = env.reset()
                    while not done:
                        deltas,_ = model.predict(obs)
                        obs,_,done,info = env.step(deltas)
                    print('p_out:',info['p_out'])
                    reward_p.append(info['p_out'])
                mean_rewards.append(safe_mean(reward_p))
        np.save('./output/heuristic_attack_random_defense',data)
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
        import matplotlib.pyplot as plt
        # plot the result
        mean_rewards = [np.mean(data[i][1:]) for i in range(len(p_vals))]
        plt.plot(data[:,0], [r for r in mean_rewards], 'bo')  # 'rx' for red x 'g+' for green + marker
        plt.xlabel('Percent of Nodes Attacked')
        plt.ylabel('Percent of Nodes Down After Cascade')
        #plt.plot(x, average_p_half, 'bo')  # 'rx' for red x 'g+' for green + marker
        # show the plot
        plt.ylim([0,1])
        plt.show()