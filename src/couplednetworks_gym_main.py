import time
import couplednetworks as cn
import multiprocessing as mlt
import networkx as nx
from networkx.algorithms.centrality import degree_centrality,katz_centrality,betweenness_centrality,harmonic_centrality
import numpy as np
import math
import gym
from gym import spaces
import sys
from typing import Callable
import random
import os
import copy


import marl
from marl.agent import DQNAgent, DQNCriticAgent,MinimaxDQNCriticAgent,FeaturizedACAgent,MinimaxQAgent
from marl.model.qvalue import MultiQTable
from marl.exploration.eps_greedy import EpsGreedy
from marl.experience.replay_buffer import ReplayMemory
from marl.model.nn.mlpnet import CriticMlp,MultiCriticMlp,MlpNet,GumbelMlpNet
from marl.policy.policies import RandomPolicy,HeuristicPolicy,QCriticPolicy
from marl.tools import gymSpace2dim,ncr,get_combinatorial_actions
from marl.agent.agent import Agent
from marl.marl import MARL_with_exploiters
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
from stable_baselines3.common.utils import set_random_seed


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import nashpy as nash

class CoupledNetsEnv2(gym.Env):
    def __init__(self,net_size,p_atk,p_def,net_type,degree=1,filename=None,discrete_obs=False):
        super(CoupledNetsEnv2,self).__init__()
        self.net_size = net_size
        self.p_atk = p_atk
        self.p_def = p_def
        self.network_type = net_type
        self.filename = filename
        self.discrete_obs = discrete_obs
        if self.network_type == 'File' and filename is not None:
            if isinstance(self.filename,str):
                self.net_b = nx.read_edgelist(self.filename,nodetype=int)         
                self.net_a = self.net_b
                self.obs = self.get_obs()
            else:
                self.obs = []
                for fn in (self.filename):
                    self.net_b = nx.read_edgelist(fn,nodetype=int)         
                    self.net_a = self.net_b
                    self.obs.append(self.get_obs())
        else:
            self.net_a, self.net_b = cn.create_networks(self.network_type,num_nodes=self.net_size)
        #self.net_a,self.net_b = create_simple_net()
        self.num_nodes_attacked = int(math.floor(p_atk * self.net_b.number_of_nodes()))
        #print("Attacking {} of {} Nodes".format(self.num_nodes_attacked,self.net_b.number_of_nodes()))
        self.num_nodes_defended = int(math.floor(p_def * self.net_b.number_of_nodes()))
        #print("Defending {} of {} Nodes".format(self.num_nodes_defended,self.net_b.number_of_nodes()))
        self.name = "CoupledNetsEnv2-v0"
        self.num_envs = 1
        net_feature_size = 4#[degree,degree_centrality,katz_centrality,betweenness_centrality,harmonic_centrality]
        self.degree = degree
        self.action_space = spaces.Discrete(ncr(self.net_b.number_of_nodes(),degree))
        if discrete_obs:
            self.observation_space = spaces.Discrete(np.power(2,self.net_b.number_of_nodes()*(self.net_b.number_of_nodes()-1)/2))
        else:
            self.observation_space = spaces.Box(low=-1,high=1,shape=(self.net_b.number_of_nodes(),net_feature_size),dtype=np.float32)


    def step(self,node_lists):
        node_list_atk = node_lists[0]
        node_list_def = node_lists[1]
        final_node_list = []
        if type(node_list_atk) == int:
            if node_list_atk != node_list_def:
                final_node_list.append(node_list_atk)
        else:
            for node in node_list_atk:
                if node not in node_list_def:
                    final_node_list.append(node)
        y = cn.attack_network(1, [self.net_a,self.net_b],rl_attack=final_node_list)
        r = y[0][1]
        reward = [r,-r]
        done = [True,True]
        info = {}
        observation = [self.observation_space.sample(),self.observation_space.sample()] #this is a dummy obs because it's a one-shot game
        return observation,reward,done,info
    
    def get_obs(self):
        if self.discrete_obs:
            adj = nx.adjacency_matrix(self.net_b).todense()
            row,col = np.tril_indices(self.net_b.number_of_nodes(),k=-1)
            idxs = [[row[i],col[i]] for i in range(len(row))]
            adj_l = [adj[row[i],col[i]] for i in range(len(row))]
            print(adj_l)
            obs = sum(val*(2**idx) for idx,val in enumerate(reversed(adj_l)))
            print(obs)
            exit()
            return [obs,obs]
        else:
            n0 = sorted(self.net_b.nodes())[0] #recognize 0 vs 1 indexing of node names
            num_nodes = self.net_b.number_of_nodes()
            nodes = [i for i in range(num_nodes)]
            max_node = max(nodes)
            min_node = min(nodes)
            nodes = [2*(node-min_node)/(max_node-min_node)-1 if (max_node-min_node) != 0 else 0 for node in nodes]

            # node_degrees = [self.net_b.degree[i+n0] for i in range(num_nodes)]
            # max_degree = max(node_degrees)
            # min_degree = min(node_degrees)
            # node_degrees = [2*(deg-min_degree)/(max_degree-min_degree)-1 if (max_degree-min_degree) != 0 else 0 for deg in node_degrees]

            degree_centralities = [degree_centrality(self.net_b)[i+n0] for i in range(num_nodes)]
            max_c = max(degree_centralities)
            min_c = min(degree_centralities)
            degree_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in degree_centralities]

            # harmonic_centralities = [harmonic_centrality(self.net_b)[i+n0] for i in range(num_nodes)]
            # max_c = max(harmonic_centralities)
            # min_c = min(harmonic_centralities)
            # harmonic_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in harmonic_centralities]

            # k = min(self.net_b.number_of_nodes(),10)
            # betweenness_centralities = [betweenness_centrality(self.net_b,k=k)[i+n0] for i in range(num_nodes)]
            # max_c = max(betweenness_centralities)
            # min_c = min(betweenness_centralities)
            # betweenness_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in betweenness_centralities]

            obs = np.stack([nodes,degree_centralities,degree_centralities,degree_centralities]).T#katz_centralities,
            return [obs,obs]

    def reset(self,fid=None):
        if self.network_type == 'SF':
            self.net_a, self.net_b = cn.create_networks(self.network_type,num_nodes=self.net_size)
            return self.get_obs()
        elif self.network_type == 'File':
            if not isinstance(self.filename,str):
                if fid is None:
                    fid = random.choice([i for i in range(len(self.filename))])
                fn = self.filename[fid]
                self.net_b = nx.read_edgelist(fn,nodetype=int)         
                self.net_a = self.net_b
                return self.obs[fid]
            else:
                return self.obs



def make_CN2_env(p_atk: float, p_def: float, network_type: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    :param p: 1-p is the percentage of nodes to attack
    :param attack_degree: the degree of node attack
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = CoupledNetsEnv2(p_atk,p_def,network_type)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Coupled Gym Args')
    parser.add_argument("--train",default=False,type=bool,help='Trains a model when true, evaluates a model when false. Default True.')
    parser.add_argument("--training_steps",default=100e3,type=int,help='Number of steps to train the model if we are training.')
    parser.add_argument("--batch_size",default=100,type=int,help='Batch size for NN training')
    parser.add_argument("--test_freq",default=10e3,type=int,help='Frequency at which to test the agent performance during training.')
    parser.add_argument("--save_freq",default=10e3,type=int,help='Frequency at which to save the agent model during training.')
    parser.add_argument("--exploiters",default=False,type=bool,help='Whether to train exploiters. Default False.')
    parser.add_argument("--exploited_type",default='NN',type=str,help='What type of agent to train exploiters against. Valid choices are: NN,Random, and Heuristic.')
    parser.add_argument("--testing_episodes",default=1000,type=int,help='Number of testing episodes for evaluation (when train is false).')
    parser.add_argument("--ego_model_dir",default=None,type=str,help='dir where nn model to load is for the ego agents')
    parser.add_argument("--exploiter_model_dir",default=None,type=str,help='dir where nn model to load is for the exploiter agents')
    parser.add_argument("--num_cpu",default=1,type=int,help='The number of parallel environments to use. Default 1')
    parser.add_argument("--p",default=0.1,type=float,help='Fraction of total nodes to be attacked/defended. Default 0.1')
    parser.add_argument("--degree",default=1,type=int,help='Number of nodes selected by the agent policy at a time. Default 1.')
    parser.add_argument("--net_type",default='SF',type=str,help='Strategy for network creation. Use "SF" for random net, and "File" to load the network in "net_file".')
    parser.add_argument("--net_size",default=10,type=int,help='Number of nodes in the power network.')
    parser.add_argument("--net_file_train_dir",default=None,type=str,help='If "net_type" == "File", loads the network topology from this file into the environment.')
    parser.add_argument("--mlp_hidden_size",default=64,type=int,help='Hidden layer size for MLP nets used for RL agent.')
    parser.add_argument("--discrete_obs",default=False,type=bool,help='When true, uses an adjacency matrix for the observation instead of the featurized vector.')
    parser.add_argument("--tabular_q",default=False,type=bool,help='Use tabular Q-learning instead of neural network Q learning')
    parser.add_argument("--nash_eqs_dir",default=None,type=str,help='Directory where Nash EQ benchmarks are stored')
    parser.add_argument("--test_nets_dir",default=None,type=str,help='Directory where the network topologies for testing are stored')
    parser.add_argument("--cn_config",default='config/config.json',type=str,help='Config file for the coupled network simulator')
    parser.add_argument("--exp_name",default='my_exp',type=str,help='Name of experiment that will be associated with log and model files.')

    args = parser.parse_args()
    if args.nash_eqs_dir is not None:
        eqs = [np.load(os.path.join(args.nash_eqs_dir,f)) for f in os.listdir(args.nash_eqs_dir)]
    else:
        eqs = None
    if args.net_type == 'File':
        print('Initializing Environments from File...')
        env = CoupledNetsEnv2(args.net_size,args.p,args.p,'File',degree=args.degree,
            filename = [os.path.join(args.net_file_train_dir,f) for f in os.listdir(args.net_file_train_dir)])
        print('Done.')
    else:
        env = CoupledNetsEnv2(args.net_size,args.p,args.p,args.net_type,degree=args.degree,discrete_obs=args.discrete_obs,filename=args.net_file_train_dir)
    obs_sp = env.observation_space
    act_sp = env.action_space
    num_samples = int(env.num_nodes_attacked/args.degree)
    if args.train: 
        exploration = EpsGreedy(num_actions=num_samples)
        experience = ReplayMemory(capacity=int(1e6))
        agent_train = not args.exploiters
        if args.ego_model_dir is not None:
            attacker_model_file = os.path.join(args.ego_model_dir + 'Attacker/',os.listdir(args.ego_model_dir + 'Attacker/')[-1])
            defender_model_file = os.path.join(args.ego_model_dir + 'Defender/',os.listdir(args.ego_model_dir + 'Defender/')[-1])
        else:
            attacker_model_file = None
            defender_model_file = None
        if args.tabular_q:
            attacker_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=0,experience=experience,exploration=exploration, batch_size=args.batch_size*num_samples,
                name = 'Attacker',lr=0.1)
            defender_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=1,experience=experience,exploration=exploration, batch_size=args.batch_size*num_samples,
                name = 'Defender',lr=0.1)
        else: 
            qmodel = MultiCriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],hidden_size=args.mlp_hidden_size) #action feature space same as obs space
            attacker_agent = MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,act_degree=args.degree,index=0,experience=experience,exploration=exploration,
                batch_size=args.batch_size*num_samples,name='Attacker',lr=1e-3,train=agent_train,model=attacker_model_file)
            defender_agent = MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,act_degree=args.degree,index=1,experience=experience,exploration=exploration,
                batch_size=args.batch_size*num_samples, name='Defender',lr=1e-3,train=agent_train,model=defender_model_file)
        if args.exploiters:
            if args.exploited_type == 'NN':
                agent_list = [attacker_agent,defender_agent]
            elif args.exploited_type == 'Random':
                random_policy = RandomPolicy(act_sp,num_actions=num_samples,all_actions= get_combinatorial_actions(gymSpace2dim(obs_sp)[0],args.degree))
                agent_list = [copy.deepcopy(random_policy),copy.deepcopy(random_policy)]
            elif args.exploited_type == 'Heuristic':
                heuristic_policy = HeuristicPolicy(act_sp,num_actions=num_samples,all_actions= get_combinatorial_actions(gymSpace2dim(obs_sp)[0],args.degree))
                agent_list = [copy.deepcopy(heuristic_policy),copy.deepcopy(heuristic_policy)]

            qmodel_explt = CriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1]) #action feature space same as obs space
            actor = GumbelMlpNet(gymSpace2dim(obs_sp)[1],gymSpace2dim(act_sp))#,last_activ=F.softmax)
            def_expltr = FeaturizedACAgent(copy.deepcopy(qmodel_explt),copy.deepcopy(actor),obs_sp,act_sp,experience=copy.deepcopy(experience),exploration=copy.deepcopy(exploration),batch_size=args.batch_size*num_samples,name='Defender Exploiter',
               lr_critic=1e-3,lr_actor=1e-4,act_degree=args.degree)#,target_update_freq=1000,name='Attacker')
            atk_expltr = FeaturizedACAgent(copy.deepcopy(qmodel_explt),copy.deepcopy(actor),obs_sp,act_sp,experience=copy.deepcopy(experience),exploration=copy.deepcopy(exploration),batch_size=args.batch_size*num_samples,name='Attacker Exploiter',
               lr_critic=1e-3,lr_actor=1e-4,act_degree=args.degree)
            #def_expltr = DQNCriticAgent(copy.deepcopy(qmodel_explt),obs_sp,act_sp,experience=copy.deepcopy(experience),lr=0.001,batch_size=args.batch_size,name="Defender Exploiter",
            #        train=True,model=None,act_degree=args.degree)
            #atk_expltr = DQNCriticAgent(copy.deepcopy(qmodel_explt),obs_sp,act_sp,experience=copy.deepcopy(experience),lr=0.001,batch_size=args.batch_size,name="Attacker Exploiter",
            #        train=True,model=None,act_degree=args.degree)
            mas = MARL_with_exploiters(agent_list,[def_expltr,atk_expltr],log_dir='marl_logs',name=args.exp_name,obs=[ob[0] for ob in env.obs],nash_policies=eqs,
                    exploited=args.exploited_type,explt_opp_update_freq=args.test_freq)
        else:
            mas = marl.MARL([attacker_agent,defender_agent],name=args.exp_name,log_dir='marl_logs',nash_policies=eqs,act_degree=args.degree)
        if args.exploited_type == 'NN':
            attacker_agent.set_mas(mas)
            defender_agent.set_mas(mas)
        test_envs = [CoupledNetsEnv2(args.net_size,args.p,args.p,'File',discrete_obs=args.discrete_obs,degree=args.degree,
            filename = os.path.join(args.test_nets_dir,f)) for f in os.listdir(args.test_nets_dir)]  
        tic = time.perf_counter()
        mas.learn(env,nb_timesteps=args.training_steps,test_freq=args.test_freq,save_freq=args.save_freq,multi_proc=(args.num_cpu>1),verbose=2,test_envs=test_envs,exploiters=args.exploiters)
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
    else:
        if args.exploiters:
            if args.exploited_type == 'NN':
                attacker_model_file = os.path.join(args.ego_model_dir + 'Attacker/',os.listdir(args.ego_model_dir + 'Attacker/')[-1])
                defender_model_file = os.path.join(args.ego_model_dir + 'Defender/',os.listdir(args.ego_model_dir + 'Defender/')[-1])
                if args.tabular_q:
                    attacker_agent = MinimaxQAgent(1, act_sp, act_sp,act_degree=args.degree,index=0,batch_size=args.batch_size*num_samples,
                        name = 'Attacker',train=False,model=attacker_model_file)
                    defender_agent = MinimaxQAgent(1, act_sp, act_sp,act_degree=args.degree,index=1, batch_size=args.batch_size*num_samples,
                        name = 'Defender',train=False,model=defender_model_file)
                else:
                    qmodel= MultiCriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],hidden_size=args.mlp_hidden_size) #action feature space same as obs space
                    attacker_agent = MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,act_degree=args.degree,index=0,name='Attacker',train=False,model=attacker_model_file)
                    defender_agent = MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,act_degree=args.degree,index=1,name='Defender',train=False,model=defender_model_file)
                agent_list = [attacker_agent,defender_agent]
            elif args.exploited_type == 'Random':
                random_policy = RandomPolicy(act_sp,num_actions=num_samples,all_actions= get_combinatorial_actions(gymSpace2dim(obs_sp)[0],args.degree))
                agent_list = [copy.deepcopy(random_policy),copy.deepcopy(random_policy)]
            elif args.exploited_type == 'Heuristic':
                heuristic_policy = HeuristicPolicy(act_sp,num_actions=num_samples,all_actions= get_combinatorial_actions(gymSpace2dim(obs_sp)[0],args.degree))
                agent_list = [copy.deepcopy(heuristic_policy),copy.deepcopy(heuristic_policy)]
            qmodel_explt = CriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1]) #action feature space same as obs space
            actor = GumbelMlpNet(gymSpace2dim(obs_sp)[1],gymSpace2dim(act_sp))#,last_activ=F.softmax)
            def_expltr_model_file = os.path.join(args.exploiter_model_dir + 'Defender Exploiter/',os.listdir(args.exploiter_model_dir + 'Defender Exploiter/')[-1])
            def_expltr = FeaturizedACAgent(copy.deepcopy(qmodel_explt),copy.deepcopy(actor),obs_sp,act_sp,name='Defender Exploiter',act_degree=args.degree,model=def_expltr_model_file,train=False)
            atk_expltr_model_file = os.path.join(args.exploiter_model_dir + 'Attacker Exploiter/',os.listdir(args.exploiter_model_dir + 'Attacker Exploiter/')[-1])
            atk_expltr = FeaturizedACAgent(copy.deepcopy(qmodel_explt),copy.deepcopy(actor),obs_sp,act_sp,name='Attacker Exploiter',act_degree=args.degree,model=atk_expltr_model_file,train=False)
            mas = MARL_with_exploiters(agent_list,[def_expltr,atk_expltr],name=args.exp_name,obs=[ob[0] for ob in env.obs],nash_policies=eqs,exploited=args.exploited_type)
        else:
            attacker_model_file = os.path.join(args.ego_model_dir + 'Attacker/',os.listdir(args.ego_model_dir + 'Attacker/')[-1])
            defender_model_file = os.path.join(args.ego_model_dir + 'Defender/',os.listdir(args.ego_model_dir + 'Defender/')[-1])
            if args.tabular_q:
                attacker_agent = MinimaxQAgent(1, act_sp, act_sp,act_degree=args.degree,index=0,batch_size=args.batch_size*num_samples,
                    name = 'Attacker',train=False,model=attacker_model_file)
                defender_agent = MinimaxQAgent(1, act_sp, act_sp,act_degree=args.degree,index=1, batch_size=args.batch_size*num_samples,
                    name = 'Defender',train=False,model=defender_model_file)
            else:
                qmodel= MultiCriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],hidden_size=args.mlp_hidden_size) #action feature space same as obs space
                attacker_agent = MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,act_degree=args.degree,index=0,name='Attacker',train=False,model=attacker_model_file)
                defender_agent = MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,act_degree=args.degree,index=1,name='Defender',train=False,model=defender_model_file)
            agent_list = [attacker_agent,defender_agent]
            mas = marl.MARL(agent_list,name=args.exp_name,log_dir='marl_logs',nash_policies=eqs,act_degree=args.degree)
        test_envs = [CoupledNetsEnv2(args.net_size,args.p,args.p,'File',filename = os.path.join(args.test_nets_dir,f)) for f in os.listdir(args.test_nets_dir)]
        test_dict = mas.test(test_envs,nb_episodes=args.testing_episodes,nashEQ_policies=eqs,exploiters=args.exploiters,render=False)
        save_dict = {}
        save_dict['policies'] = test_dict['policies'].tolist()
        save_dict['ego_attacker_rew'] = np.reshape(test_dict['agent_rewards'][0],[len(test_envs),args.testing_episodes]).tolist()
        if args.exploiters:
            save_dict['exploitability'] =  np.reshape(test_dict['exploitability'][0],[len(test_envs),args.testing_episodes]).tolist()
            save_dict['exploiter_rew'] = np.reshape(test_dict['exploiter_rewards'][0],[len(test_envs),args.testing_episodes,2]).tolist()
        if args.nash_eqs_dir is not None:
            save_dict['nash_eqs'] = np.asarray(eqs).tolist()
            save_dict['nash_kl_div'] = np.reshape(test_dict['nash_kl_div'][0],[len(test_envs),args.testing_episodes]).tolist()
        test_results_dir = 'output/test_results/'
        import json
        save_fn = test_results_dir + args.exp_name + '.json'
        with open(save_fn,'w') as f:
            json.dump(save_dict,f)
        print('Finished Testing. Saved results to ',save_fn)
        print('Ego Attacker Reward: Mean {}, Dev. {}'.format(test_dict['agent_rewards'][1],test_dict['agent_rewards'][2]))
        if args.nash_eqs_dir is not None:
            print('Nash KL Divergence: Mean {}, Dev. {}'.format(test_dict['nash_kl_div'][1],test_dict['nash_kl_div'][2]))
        if args.exploiters:
            print('Exploitability: Mean {}, Dev. {}'.format(test_dict['exploitability'][1],test_dict['exploitability'][2]))
            print('Defender Exploiter Reward: Mean {}'.format(test_dict['exploiter_rewards'][1]))
            print('Attacker Exploiter Reward: Mean {}'.format(test_dict['exploiter_rewards'][2]))
