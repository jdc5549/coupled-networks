import marl
from . import TrainableAgent, MATrainable
from ..policy import QPolicy,QCriticPolicy,MinimaxQCriticPolicy,MinimaxQTablePolicy
from ..model import MultiQTable
from marl.tools import gymSpace2dim,get_combinatorial_actions
from ..experience import ReplayMemory

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class QAgent(TrainableAgent):
    """
    The class of trainable agent using Qvalue-based methods
    
    :param qmodel: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param batch_size: (int) The size of a batch
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    def __init__(self, qmodel, observation_space, action_space, experience="ReplayMemory-1", exploration="EpsGreedy", gamma=0.99, lr=0.1, batch_size=1, target_update_freq=None, name="QAgent"):
        super(QAgent, self).__init__(policy=QPolicy(model=qmodel, observation_space=observation_space, action_space=action_space), observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, name=name)
        self.off_policy = target_update_freq is not None
        self.target_update_freq = target_update_freq
                
        if self.off_policy:
            self.target_policy = copy.deepcopy(self.policy)
        
    def update_model(self, t):
        """
        Update the model.
        
        :param t: (int) The current timestep
        """
        if len(self.experience) < self.batch_size:
            return
        # Get changing policy
        curr_policy = self.target_policy if self.off_policy else self.policy
        # Get batch of experience
        #if isinstance(self, MATrainable):
        #     #ind = self.experience.sample_index(self.batch_size)
        #     #batch = self.mas.experience.get_transition(len(self.mas.experience) - np.array(ind)-1)
        # else:
        batch = self.experience.sample(self.batch_size)
        # Compute target r_t + gamma*max_a Q(s_t+1, a)
        target_value = self.target(curr_policy.Q, batch)
        
        # Compute current value Q(s_t, a_t)
        #print(batch.observation)
        #print(batch.action)
        curr_value = self.value(batch.observation, batch.action)

        # Update Q values
        self.update_q(curr_value, target_value, batch)
        
        if self.off_policy and t % self.target_update_freq==0:
            self.update_target_model()
            
    
    def target(self, Q, batch):
        """
        Compute the target value.
        
        :param Q: (Model or torch.nn.Module) The model of the Q-value
        :param batch: (list) A list composed of the different information about the batch required
        """
        raise NotImplementedError
    
    def value(self, observation, action):
        """
        Compute the value.
        
        :param observation: The observation
        :param action: The action
        """
        raise NotImplementedError
    
    def update_q(self, curr_value, target_value, batch):
        """
        Update the Q value.
        
        :param curr_value: (torch.Tensor) The current value 
        :param target_value: (torch.Tensor) The target value
        :param batch: (list) A list composed of the different information about the batch required
        """
        raise NotImplementedError    
    
    def update_target_model(self):
        """
        Update the target model.
        """
        raise NotImplementedError
            
class QTableAgent(QAgent):
    """
    The class of trainable agent using  Q-table to model the  function Q 
    
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, observation_space, action_space, exploration="EpsGreedy", gamma=0.99, lr=0.1, target_update_freq=None, name="QTableAgent"):
        super(QTableAgent, self).__init__(qmodel="QTable", observation_space=observation_space, action_space=action_space, experience="ReplayMemory-1", exploration=exploration, gamma=gamma, lr=lr, batch_size=1, target_update_freq=target_update_freq, name=name)
        
    def update_q(self, curr_value, target_value, batch):
        self.policy.Q.q_table[batch.observation, batch.action] = curr_value + self.lr * (target_value - curr_value)
        
    def update_target_model(self):
        self.target_policy = copy.deepcopy(self.policy)
        
    def target(self, Q, batch):
        next_obs  = batch.next_observation
        next_action_value = Q(next_obs).max(1).values.float()
        rew = torch.tensor(batch.reward).float()
        not_dones = 1.- torch.tensor(batch.done_flag).float()
        
        target_value = rew + not_dones * self.gamma * next_action_value
        return target_value
        
    def value(self, observation, action):
        return self.policy.Q(observation, action)
    
class MinimaxQAgent(QAgent, MATrainable):
    """
    The class of trainable agent using  minimax-Q-table algorithm 
    
    :param observation_space: (gym.Spaces) The observation space
    :param my_action_space: (gym.Spaces) My action space
    :param other_action_space: (gym.Spaces) The action space of the other agent
    :param index: (int) The position of the agent in the list of agent
    :param mas: (marl.agent.MAS) The multi-agent system corresponding to the agent
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, observation_space, my_action_space, other_action_space, index=None, mas=None,act_degree=1, batch_size=100,experience=None,exploration="EpsGreedy", gamma=0.99, lr=0.1, target_update_freq=None, name="MinimaxQAgent",model=None,train=True):
        self.qmodel = MultiQTable(gymSpace2dim(observation_space), [gymSpace2dim(my_action_space), gymSpace2dim(other_action_space)])
        QAgent.__init__(self, qmodel=self.qmodel, observation_space=observation_space, action_space=my_action_space, experience="ReplayMemory-1", exploration=exploration, gamma=gamma, lr=lr, batch_size=1, target_update_freq=target_update_freq, name=name)      
        self.degree = act_degree
        self.all_actions = get_combinatorial_actions(gymSpace2dim(my_action_space),self.degree)
        self.policy = MinimaxQTablePolicy(self.qmodel,action_space=my_action_space,observation_space=observation_space,player=index,all_actions=self.all_actions,act_degree=self.degree)
        MATrainable.__init__(self, mas, index)
        if model is not None:
            self.policy.load(model)
        self.train = train
        
    def update_q(self, curr_value, target_value, batch):
        if len(batch.action[0]) > 2:
            raise Exception("The number of agents should not exceed 2.")
        self.policy.Q.q_table[:, batch.action[0][self.index], batch.action[0][1-self.index]]  = curr_value + self.lr * (target_value - curr_value)

    def update_target_model(self):
        self.target_policy = copy.deepcopy(self.policy)
        
    def target(self, Q, joint_batch):
        rew = torch.tensor(joint_batch.reward).float()
        target_value = torch.tensor([joint_batch.reward[i][self.index] for i in range(len(joint_batch.reward))])
        return target_value.unsqueeze(1).detach()
        
    def value(self, observation, action):
        return self.policy.Q.q_table[:, action[0][self.index], action[0][1-self.index]]
    
class DQNAgent(QAgent):
    """
    The class of trainable agent using a neural network to model the  function Q
    
    :param qmodel: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param batch_size: (int) The size of a batch
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, qmodel, observation_space, action_space, experience="ReplayMemory-10000", exploration="EpsGreedy", gamma=0.99, lr=0.1,batch_size=32, tau=1., target_update_freq=1000,name="DQNAgent"):
        super(DQNAgent, self).__init__(qmodel=qmodel, observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, target_update_freq=target_update_freq, name=name)
        self.criterion = nn.SmoothL1Loss() # Huber criterion
        self.optimizer = optim.Adam(self.policy.Q.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=500e6,gamma=0.1)
        self.tau = tau
        if self.off_policy:
            self.target_policy.Q.eval()
            
    def update_q(self, curr_value, target_value, batch):
        self.optimizer.zero_grad()
        loss = self.criterion(curr_value, target_value)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if self.lr != self.scheduler.get_last_lr()[0]:
            print('learning rate:',self.scheduler.get_last_lr()[0])
        self.lr = self.scheduler.get_last_lr()[0]
    
    def write_logs(self, t):
        if t%50 == 0:
            b = self.experience.sample(self.batch_size)
        obs_state = torch.FloatTensor(b.observation)
        with torch.no_grad():
            q = self.target_policy.Q(obs_state)
            self.writer.add_scalar('Q-value/q-value', q.mean().item(), t)

    def update_target_model(self):
        if self.tau ==1:
            self.hard_update()
        else:
            self.soft_update(self.tau)
        
    def hard_update(self):
        self.target_policy.Q.load_state_dict(self.policy.Q.state_dict())

    def soft_update(self, tau):
        for target_param, local_param in zip(self.target_policy.Q.parameters(), self.policy.Q.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    
    def target(self, Q, batch):
        next_obs  = torch.tensor(batch.next_observation).float()
        next_action_values = Q(next_obs).max(1).values.float()
        rew = torch.tensor(batch.reward).float()
        not_dones = 1.- torch.tensor(batch.done_flag).float()
        target_value = (rew + not_dones * self.gamma * next_action_values).unsqueeze(1)
        return target_value.detach()
        
    def value(self, observation, action):
        t_action = torch.tensor(action).long().unsqueeze(1)
        t_observation = torch.tensor(observation).float()
        return self.policy.Q(t_observation).gather(1, t_action)

class DQNCriticAgent(DQNAgent):
    """
    The class of trainable agent using a neural network to model the  function Q
    
    :param qmodel: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param batch_size: (int) The size of a batch
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, qmodel,observation_space, action_space,experience="ReplayMemory-10000", exploration="EpsGreedy",gamma=0.99, lr=0.1,  batch_size=32, tau=1., target_update_freq=1000, name="DQNCriticAgent",train=True,model=None,act_degree=1):
        super(DQNCriticAgent, self).__init__(qmodel=qmodel, action_space=action_space,observation_space=observation_space,experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, target_update_freq=target_update_freq, name=name)
        self.degree = act_degree
        self.all_actions = get_combinatorial_actions(gymSpace2dim(self.observation_space)[0],self.degree)
        self.policy = QCriticPolicy(qmodel,action_space=action_space,observation_space=observation_space,all_actions=self.all_actions)
        if model is not None:
            self.policy.load(model)
        self.train = train

    def target(self, Q, batch): 
        rew = torch.tensor(batch.reward).float()
        target_value = rew    
        return target_value.unsqueeze(1).detach()
        
    def value(self, observation, action):
        t_observation = torch.stack([torch.tensor(obs).float() for obs in observation])
        if type(action[0]) == list:
            num_samples = int(len(action[0])/self.degree)
        else:
            num_samples = 1
            action = [self.all_actions[a] for a in action]
        feat_actions = []
        for i,a in enumerate(action):
            acts = t_observation[i,a]
            if self.degree > 1:
                for j in range(0,len(acts),self.degree):
                    feat_actions.append(torch.cat([acts[k] for k in range(j,j+self.degree)]))
            else:
                feat_actions.append(acts)
        if num_samples == 1:
            feat_actions = torch.stack(feat_actions)
        else:
            feat_actions = torch.cat(feat_actions)
        t_observation = torch.sum(t_observation,dim=1)
        t_observation = t_observation.repeat([num_samples,1])
        if num_samples > 1:
            q = self.policy.Q(t_observation,feat_actions)
            value = torch.zeros(self.batch_size)
            for i in range(self.batch_size):
                avg_q = torch.zeros(num_samples)
                for j in range(i*num_samples,(i+1)*num_samples):
                    avg_q[j-i*num_samples] = q[j]
                value[i] = torch.mean(avg_q)
            return value.unsqueeze(1)
        else:
            return self.policy.Q(t_observation,feat_actions)
class MinimaxDQNCriticAgent(DQNAgent,MATrainable):
    """
    The class of trainable agent using a neural network to model the  function Q
    
    :param qmodel: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameters
    :param lr: (float) The learning rate
    :param batch_size: (int) The size of a batch
    :param target_update_freq: (int) The update frequency of the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, qmodel,observation_space, my_action_space, other_action_space,act_degree=1, index=None, mas=None,experience="ReplayMemory-10000", exploration="EpsGreedy", gamma=0.99, lr=0.1,  batch_size=32, tau=1., target_update_freq=1000, name="MultiDQNCriticAgent",train=True,model=None):
        super(MinimaxDQNCriticAgent, self).__init__(qmodel=qmodel, action_space=my_action_space,observation_space=observation_space,experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, target_update_freq=target_update_freq, name=name)
        self.degree = act_degree
        self.all_actions = get_combinatorial_actions(gymSpace2dim(self.observation_space)[0],self.degree)
        self.policy = MinimaxQCriticPolicy(qmodel,action_space=my_action_space,observation_space=observation_space,player=index,all_actions = self.all_actions,act_degree=self.degree)
        MATrainable.__init__(self, mas, index)
        if model is not None:
            self.policy.load(model)
        self.train = train
    def target(self, Q, joint_batch):
        # print(joint_batch.action)
        #print(joint_batch.reward)
        # exit() 
        my_rew = torch.tensor([joint_batch.reward[i][self.index] for i in range(len(joint_batch.reward))])
        target_value = my_rew
        return target_value.unsqueeze(1).detach()
        
    def value(self, observation, actions):
        t_observation = torch.stack([torch.tensor(obs[self.index]).float() for obs in observation]) #same for both players so no need to distinguish
        p1_actions = [a[0] for a in actions]
        if type(p1_actions[0]) == list:
            num_samples = int(len(p1_actions[0])/self.degree)
        else:
            num_samples = 1
            p1_actions = [self.all_actions[a] for a in p1_actions]
        p1_feat_actions = []
        for i,a in enumerate(p1_actions):
            acts = t_observation[i,a]
            if self.degree > 1:
                for j in range(0,len(acts),self.degree):
                    p1_feat_actions.append(torch.cat([acts[k] for k in range(j,j+self.degree)]))
            else:
                p1_feat_actions.append(acts)
        if num_samples == 1:
            t_action_p1 = torch.stack(p1_feat_actions)
        else:
            t_action_p1 = torch.cat(p1_feat_actions)
        p2_actions = [a[1] for a in actions]
        if type(p2_actions[0]) == list:
            num_samples = int(len(p2_actions[0])/self.degree)
        else:
            num_samples = 1
            p2_actions = [self.all_actions[a] for a in p2_actions]
        p2_feat_actions = []
        for i,a in enumerate(p2_actions):
            acts = t_observation[i,a]
            if self.degree > 1:
                for j in range(0,len(acts),self.degree):
                    p2_feat_actions.append(torch.cat([acts[k] for k in range(j,j+self.degree)]))
            else:
                p2_feat_actions.append(acts)
        if num_samples == 1:
            t_action_p2 = torch.stack(p2_feat_actions)
        else:
            t_action_p2 = torch.cat(p2_feat_actions)
        t_observation = torch.sum(t_observation,dim=1)
        t_observation = t_observation.repeat([num_samples,1])
        if num_samples > 1:
            q = self.policy.Q(t_observation,t_action_p1,t_action_p2)
            value = torch.zeros(self.batch_size)
            for i in range(self.batch_size):
                avg_q = torch.zeros(num_samples)
                for j in range(i*num_samples,(i+1)*num_samples):
                    avg_q[j-i*num_samples] = q[j]
                value[i] = torch.mean(avg_q)
            return value.unsqueeze(1)
        else:
            return self.policy.Q(t_observation,t_action_p1,t_action_p2)

class ContinuousDQNAgent(DQNAgent):
    """
    The class of trainable agent using a neural network to model the  function Q
    
    :param qmodel: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param gamma: (float) The training parameterstimestep_initof the target model  
    :param name: (str) The name of the agent      
    """
    
    def __init__(self, qmodel, actor_policy, observation_space, action_space, experience="ReplayMemory-10000", exploration="EpsGreedy", gamma=0.99, lr=0.0005,  batch_size=32, target_update_freq=1000, name="DQNAgent"):
        super(ContinuousDQNAgent, self).__init__(qmodel=qmodel, observation_space=observation_space, action_space=action_space, experience=experience, exploration=exploration, gamma=gamma, lr=lr, batch_size=batch_size, target_update_freq=target_update_freq, name=name)
        self.actor_policy = actor_policy
        if model is not None:
            self.policy.load(model)
        self.train = train

    def target(self, Q, batch):
        next_obs  = torch.tensor(batch.next_observation).float()
        next_action =  torch.tensor(self.actor_policy(next_obs)).float()
        next_state_action_values = Q(next_obs, next_action)
        rew = torch.tensor(batch.reward).float()
        not_dones = 1.- torch.tensor(batch.done_flag).float()
        target_value = (rew + not_dones * self.gamma * next_state_action_values).unsqueeze(1)
        return target_value.detach()
        
    def value(self, observation, action):
        obs  = torch.tensor(observation).float()
        action =  torch.tensor(action).float()
        return self.policy.Q(obs, action)