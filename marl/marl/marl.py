import os
import marl
import copy
from .agent import TrainableAgent, Agent
import torch
from torch.utils.tensorboard import SummaryWriter
from marl.tools import get_combinatorial_actions, gymSpace2dim
import numpy as np

class MAS(object):
    """
    The class of multi-agent "system".
    
    :param agents_list: (list) The list of agents in the MAS
    :param name: (str) The name of the system
    """
    
    def __init__(self, agents_list=[], name="mas"):
        self.name = name
        self.agents = agents_list
        
    def append(self, agent):
        """
        Add an agent to the system.

        :param agent: (Agent) The agents to be added
        """
        self.agents.append(agent)          
    
    def action(self, observation):
        """
        Return the joint action.

        :param observation: The joint observation
        """
        return [ag.greedy_action(ag, obs) for ag, obs in zip(self.agents, observation)]    
    
    def get_by_name(self, name):
        for ag in self.agents:
            if ag.name == name:
                return ag
        return None
    
    def get_by_id(self, id):
        for ag in self.agents:
            if ag.id == id:
                return ag
        return None
        
    def __len__(self):
        return len(self.agents)

class MARL(TrainableAgent, MAS):
    """
    The class for a multi-agent reinforcement learning.
    
    :param agents_list: (list) The list of agents in the MARL model
    :param name: (str) The name of the system
    """
    def __init__(self, agents_list=[],name='marl', log_dir="logs",nash_policies=None,utils=None,act_degree=1):
        MAS.__init__(self, agents_list=agents_list, name=name)
        self.experience = marl.experience.make("ReplayMemory", capacity=10000)
        self.log_dir = log_dir
        self.nash_policies = nash_policies
        self.utils = utils
        self.last_policies = None
        self.init_writer(log_dir)
        self.explt_opp_update_freq = 1000
        self.degree = [ag.degree for ag in agents_list]
        self.exploited = 'None'
        
    def reset(self):
        for ag in self.agents:
            ag.reset()
            
    def init_writer(self, log_dir):
        log_path = os.path.join(log_dir, self.name)
        self.writer = SummaryWriter(log_path)
        # for ag in self.agents:
        #     if isinstance(ag, TrainableAgent):
        #         ag.init_writer(log_path)
        
    def store_experience(self, *args):
        observation,action, reward, next_observation, done,info = args
        for i,ag in enumerate(self.agents):
            if hasattr(self.agents[0],'mas'):
                if isinstance(ag, TrainableAgent) and i < 1:
                    ag.store_experience(observation, action, reward, next_observation, done,info)
            else:
                if isinstance(ag, TrainableAgent):
                    ag.store_experience(observation[i], action[i], reward[i], next_observation[i], done[i],info[i])
            
    def update_model(self, t):
        # TrainableAgent.update_model(self, t) 
        critic_mse = []
        for i,ag in enumerate(self.agents):
            if hasattr(self.agents[0],'mas'):
                if isinstance(ag, TrainableAgent) and ag.train and i < 1:
                    critic_mse.append(ag.update_model(t))
            else:
                if isinstance(ag, TrainableAgent) and ag.train:
                    critic_mse.append(ag.update_model(t))
        return critic_mse
    
    def reset_exploration(self, nb_timesteps):
        # TrainableAgent.update_exploration(self, nb_timesteps)        
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.reset_exploration(nb_timesteps)
    
    def update_exploration(self, t):
        # TrainableAgent.update_exploration(self, t)        
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                eps = ag.exploration.update(t)
        return eps

    def get_agent_policies(self,observation):
        policies = []
        vals = []
        t_obs = torch.tensor(observation).float()
        if self.agents[0].degree > 1:
            feat_actions = torch.stack([t_obs[action].flatten() for action in self.agents[0].all_actions]).float()
        else:
            feat_actions = t_obs
        obs = torch.mean(t_obs,axis=0)
        for ag in self.agents:
            policy,val = ag.policy.get_policy(obs,feat_actions)
            #print(ag.name)
            #print(val)
            #print(ag.policy.Q(obs,feat_actions[0],feat_actions[1]))
            # if ag.name == 'Defender':
            #     exit()
            policies.append(policy)
            vals.append(val)
        return policies, vals

        
    def action(self, observation,num_actions=1):
        actions = []
        for ag, obs in zip(self.agents,observation):
            if ag.train:
                actions.append(ag.action(obs))
            else:
                actions.append(ag.greedy_action(obs))
        return actions
        
    def greedy_action(self, observation,num_actions=1):
        return [ag.greedy_action(obs,num_actions=num_actions) for ag, obs in zip(self.agents, observation)]
    
    def save_policy(self, folder='.', filename='', timestep=None):
        """
        Save the policy in a file called '<filename>-<agent_name>-<timestep>'.
        
        :param folder: (str) The path to the directory where to save the model(s)
        :param filename: (str) A specific name for the file (ex: 'test2')
        :param timestep: (int) The current timestep  
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_tmp = "{}-{}".format(filename, self.name) if filename is not '' else "{}".format(self.name)
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.save_policy(folder=folder, filename=filename_tmp, timestep=timestep)
        
    # def get_best_rew(rew1, rew2):
    #     for ind, ag in enumerate(self.agents):
    #         rew1[ind] = ag.get_best_rew(rew1[ind], rew2[ind])
    #     return rew1
        
    def save_policy_if_best(self, best_err, err, folder='.', filename=''):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_tmp = "{}-{}".format(filename, self.name) if filename is not '' else "{}".format(self.name)
        #for ind, ag in enumerate(self.agents):
        if isinstance(self.agents[0], TrainableAgent):
            best_err = self.agents[0].save_policy_if_best(best_err, err, folder=folder, filename=filename_tmp)
        else:
            if best_err < err:
                best_err = err
        return best_err
            
    def worst_err(self):
        # best_rew = []
        # for ag in self.agents:
        #     best_rew += [ag.worst_rew()]
        return np.inf
                
    def load_model(self, filename):
        for ag in self.agents:
            if isinstance(ag, TrainableAgent):
                ag.load_model(filename)
                
    def training_log(self, verbose):
        log = ""
        if verbose >= 2:
            for ag in self.agents:
                if isinstance(ag, TrainableAgent):
                    log += ag.training_log(verbose)
                else:
                    log += "#> {}\n".format(ag.name)
        return log


class MARL_with_exploiters(MARL):
    """
    The class for a multi-agent reinforcement learning.
    
    :param ego_agents_list: (list) The list of ego agents in the MARL model
    :param exploiter_agents_list: The list of exploiter agents for each ego agent in the MARL model
    :param name: (str) The name of the system
    """

    #TODO: make more general to other use cases
    def __init__(self, ego_agents_list=[],exploiter_agents_list=[], name='marl_exploiters',log_dir="logs",obs=[],nash_policies=None,explt_opp_update_freq=1000,act_degree=1,exploited='NN'):
        MAS.__init__(self, agents_list=exploiter_agents_list, name=name)
        self.explt_opp_update_freq = explt_opp_update_freq
        self.ego_agents_list = ego_agents_list
        self.exploiter_agents_list = exploiter_agents_list
        self.obs = obs
        self.exploited = exploited


        if exploited == 'NN':
            self.ego_policies = []
            print('Loading Ego Agent Policies...')
            for ob in self.obs:
                self.ego_policies.append(self.get_agent_policies(ob)[:2])
            print('Done.')

        #self.last_ego_models = [copy.deepcopy(ag.policy) for ag in self.ego_agents_list]        
        self.nash_policies = nash_policies
        self.last_policies = None
        self.degree = act_degree
        self.init_writer(log_dir)


    def action(self, observation,num_actions=1):
        agent_actions = []
        obs = observation[0]
        for ag in self.exploiter_agents_list:
            if ag.train:
                agent_actions.append(ag.action(obs))
            else:
                agent_actions.append(ag.greedy_action(obs,num_actions=num_actions))
        obs_index = -1
        for i,ob in enumerate(self.obs):
            if np.allclose(ob,observation[0]):
                obs_index = i
        if obs_index == -1:
            print("observation not recognized")
            exit()
        if self.exploited == 'NN':
            policies = self.ego_policies[obs_index]
            ego_acts = [ag.policy(obs,num_actions=num_actions,policy=policies[i]) for i,ag in enumerate(self.ego_agents_list)]
        else:
            ego_acts = [pi([obs],num_actions=num_actions) for pi in self.ego_agents_list]
            if len(ego_acts[0]) ==1:
                ego_acts = [a[0] for a in ego_acts]
        def_expltr_acts = [agent_actions[0],ego_acts[1]]
        atk_expltr_acts = [ego_acts[0],agent_actions[1]]
        actions = [ego_acts,def_expltr_acts,atk_expltr_acts]
        return actions

    def update_model(self, t):
        # TrainableAgent.update_model(self, t) 
        for ag in self.exploiter_agents_list:
            if isinstance(ag, TrainableAgent) and ag.train:
                ag.update_model(t)
        if t%self.explt_opp_update_freq == 0:
            self.last_ego_models = [copy.deepcopy(ag.policy) for ag in self.ego_agents_list]        


    def greedy_action(self, observation,num_actions=1):
        agent_actions = []
        obs = observation[0]
        for ag in self.exploiter_agents_list: 
            agent_actions.append(ag.greedy_action(obs,num_actions=num_actions))
        obs_index = -1
        for i,ob in enumerate(self.obs):
            if np.all(np.allclose(ob,observation[0])):
                obs_index = i
        if obs_index == -1:
            print("observation not recognized")
            exit()
        if self.exploited == 'NN':
            policies = self.ego_policies[obs_index]
            ego_acts = [ag.policy(obs,num_actions=num_actions,policy=policies[i]) for i,ag in enumerate(self.ego_agents_list)]
        else:
            ego_acts = [pi([obs],num_actions=num_actions) for pi in self.ego_agents_list]
            if len(ego_acts[0]) ==1:
                ego_acts = [a[0] for a in ego_acts]
        def_expltr_acts = [agent_actions[0],ego_acts[1]]
        atk_expltr_acts = [ego_acts[0],agent_actions[1]]
        actions = [ego_acts,def_expltr_acts,atk_expltr_acts]
        return actions

    def store_experience(self, *args):
        observation,action, reward, next_observation, done = args
        self.exploiter_agents_list[0].store_experience(observation[1][0],action[1][0],reward[1][0],next_observation[1][0],done[1][0])
        self.exploiter_agents_list[1].store_experience(observation[2][1],action[2][1],reward[2][1],next_observation[2][1],done[2][1])


    def get_ego_policies(self,observation):
        policies = []
        vals = []
        for ag in self.ego_agents_list:
            t_obs = torch.tensor(observation).float()
            if ag.degree > 1:
                feat_actions = torch.stack([t_obs[action].flatten() for action in ag.all_actions]).float()
            else:
                feat_actions = t_obs
            obs = torch.mean(t_obs,axis=0)
            policy,val = ag.policy.get_policy(obs,feat_actions)
            policies.append(policy)
            vals.append(val)
        return policies,vals


    def save_policy_if_best(self, best_rew, rew, folder='.', filename=''):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_tmp = "{}-{}".format(filename, self.name) if filename is not '' else "{}".format(self.name)
        for ind, ag in enumerate(self.agents[:2]):
            if isinstance(ag, TrainableAgent):
                best_rew[ind] = ag.save_policy_if_best(best_rew[ind], rew[ind], folder=folder, filename=filename_tmp)
            else:
                best_rew[ind] = ag.get_best_rew(best_rew[ind], rew[ind])
        return best_rew