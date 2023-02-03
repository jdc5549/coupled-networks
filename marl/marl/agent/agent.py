import marl

from marl.tools import ClassSpec, _std_repr, is_done, reset_logging,ncr,get_implied_policy,get_combinatorial_actions
from marl.policy.policy import Policy
from marl.exploration import ExplorationProcess
from marl.experience import ReplayMemory, PrioritizedReplayMemory

import os
import sys
import time
import math
import random
import copy
import logging
import numpy as np
from datetime import datetime
from scipy.stats import entropy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



class Agent(object):
    """
    The class of generic agent.
    
    :param policy: (Policy) The policy of the agent
    :param name: (str) The name of the agent      
    """
    
    agents = {}
    
    counter = 0
    
    def __init__(self, policy, name="UnknownAgent"):
        Agent.counter +=1
        
        self.id = Agent.counter
        
        self.name = name
        self.policy = marl.policy.make(policy)
        self.last_policies = None

    def action(self, observation,num_actions=1):
        """
        Return the action given an observation  
        :param observation: The observation
        """
        return self.policy(observation,num_actions=num_actions)
    
    def greedy_action(self, observation,num_actions=1):
        """
        Return the greedy action given an observation  
        :param observation: The observation
        """
        return Agent.action(self, observation,num_actions=num_actions)
    
    def reset(self):
        pass
    
    def worst_err(self):
        return np.inf
    
    # def get_best_rew(self, rew1, rew2):
    #     return rew2 if rew1 < rew2 else rew1
        
    def test(self, envs, nb_episodes=4, max_num_step=1, render=True, multi_proc=False,time_laps=0.,nashEQ_policies=None,utils=None,exploiters=False):
        """
        Test a model.
        
        :param env: (Gym) The environment
        :param nb_episodes: (int) The number of episodes to test
        :param max_num_step: (int) The maximum number a step before stopping an episode
        :param render: (bool) Whether to visualize the test or not (using render function of the environment)
        """
        mean_rewards = np.array([])
        sum_rewards = np.array([])
        nash_eq_divergences = []
        last_policy_divergences = []
        curr_policies = []
        exploitabilities = []
        exploiter_rewards = []
        errs = []
        s_errs = []
        #all_actions = get_combinatorial_actions(envs[0].net_size,envs[0].num_nodes_attacked) #delete this after debugging
        for i,env in enumerate(envs):
            policy_i = []
            all_actions = get_combinatorial_actions(env.net_size,env.num_nodes_attacked)
            for episode in range(nb_episodes):
                observation = env.reset(fid=i)
                done = False
                if render:
                    env.render()
                    time.sleep(time_laps)
                for step in range(max_num_step):
                    if(multi_proc):
                        obs_reshaped = np.reshape(observation,(observation.shape[1],observation.shape[0],observation.shape[2],observation.shape[3]))
                        action = self.greedy_action(obs_reshaped,num_actions=env.get_attr('num_nodes_attacked')[0])
                    else:
                        num_samples = int(env.num_nodes_attacked/env.degree)
                        action = self.greedy_action(observation,num_actions=num_samples)
                        #if not exploiters:
                        policy,Q_val = self.get_agent_policies(observation[0])
                        #if nashEQ_policies is not None:
                        if exploiters:
                            if self.exploited == 'NN':
                                ego_policy = self.get_ego_policies(observation[0])
                            elif self.exploited == 'Random':
                                ego_policy = [np.ones(len(policy[0]))/len(policy[0]),np.ones(len(policy[0]))/len(policy[0])]
                            elif self.exploited == 'Heuristic':
                                ego_policy = [np.zeros(ncr(len(policy[0]),num_samples)),np.zeros(ncr(len(policy[0]),num_samples))]
                                action[0][0].sort()
                                action[0][1].sort()
                                ego_policy[0][all_actions.index(action[0][0])] = 1
                                ego_policy[1][all_actions.index(action[0][1])] = 1
                        else:
                            ego_policy = policy
                        if num_samples > 1 and self.exploited != 'Heuristic':
                            impl_policy = np.zeros([len(ego_policy),ncr(len(ego_policy[0]),num_samples)])
                            for i in range(len(ego_policy)):
                                impl_policy[i,:] = get_implied_policy(ego_policy[i],env.num_nodes_attacked)
                            impl_Q = np.zeros([len(all_actions),len(all_actions)])
                            for i,a in enumerate(all_actions):
                                for j,d in enumerate(all_actions):
                                    impl_Q[i,j] = 0
                                    for a_ in a:
                                        for d_ in d:
                                            impl_Q[i][j] += Q_val[0][a_][d_]/(2*len(a))
                            impl_Q = [impl_Q, -impl_Q]
                        else:
                            impl_policy = np.asarray(ego_policy)
                            impl_Q = Q_val
                        if self.last_policies is not None and not exploiters:
                            kl = entropy(np.mean(self.last_policies,axis=1)[i].flatten(),np.array(ego_policy).flatten())
                            last_policy_divergences.append(kl)
                        else:
                            last_policy_divergences.append(0)
                        policy_i.append(ego_policy)
                        #print(ego_policy)
                        #print(nashEQ_policies[i])
                        if nashEQ_policies is not None:
                            nashEQ_policy = nashEQ_policies[i]
                            kl = entropy(nashEQ_policy.flatten(),impl_policy.flatten())
                            # print('policy: ',impl_policy)
                            # print('nash policy: ',nashEQ_policy)
                            # print('Q: ',Q_val[0])
                            # print('util: ',utils[i])
                            nash_eq_divergences.append(kl)
                            if len(utils) > 0:
                                err_mat = [impl_Q[0]-utils[i],impl_Q[1]+utils[i]]
                                #print(err_mat[0])
                                #print(err_mat[1])
                                err = []
                                #err0 = []
                                #err1 = []
                                for i, erri in enumerate(err_mat):
                                    for j, errj in enumerate(erri):
                                        for k,errk in enumerate(errj):
                                            abs_errj = np.abs(errk)
                                            # if i != j:
                                            #     err1.append(abs_errj)
                                            # else:
                                            #     err0.append(abs_errj)
                                            err.append(abs_errj)
                                #print(f'0 error: {np.mean(err0)}')
                                #print(f'1 error: {np.mean(err1)}')
                                errs.append(np.mean(err))


                    if render:
                        print("Step {} - Action {}".format(step, action))
                    if(multi_proc):
                        action = self.action(obs_reshaped)
                        orig = np.asarray(action)
                        action = np.zeros((orig.shape[1],orig.shape[0],orig.shape[2]))
                        for i in range(orig.shape[1]):
                            for j in range(orig.shape[0]):
                                action[i][j] = orig[j][i]
                        obs2, reward, done, _ = env.step(action)
                        sum_r = reward[:,0]
                    else:
                        if exploiters:
                            rew = []
                            for act in action:
                                obs_ = env.reset()
                                _,rew_,done,_ = env.step(act)
                                rew.append(rew_)
                            # print(rew)
                            # print(rew[0][0])
                            # print(rew[1][0])
                            # print(rew[2][1])
                            sum_r = np.array([rew[0][0]]) if step==0 else np.add(sum_r, rew[0][0])
                            exploitability = (rew[1][0]-rew[0][0]) + (rew[2][1]-rew[0][1])
                            exploiter_rewards.append([rew[1][0],rew[2][1]])
                            exploitabilities.append(exploitability)
                        else:
                            #print(f'Test Env capacity {env.scm.capacity}')
                            #exit()
                            obs2, reward, done, _ = env.step(action)
                            sum_r = np.array([reward[0]]) if step==0 else np.add(sum_r, reward[0])
                            #print(Q_val[0])
                            # print('Q_val: ',Q_val[0][all_actions.index(action[0])][all_actions.index(action[1])])
                            # print('Value function: ',self.agents[0].value([observation],[action]))
                            # exit()
                            if type(action[0]) == int:
                                a0 = action[0]
                                a1 = action[1]
                            else:
                                a0 = list(set(action[0]))
                                a1 = list(set(action[1]))
                            s_errs.append(np.abs(reward[0] - impl_Q[0][all_actions.index(a0)][all_actions.index(a1)]))
                    if render:
                        env.render()
                        time.sleep(time_laps)
                    if is_done(done):
                        break
                sum_rewards = sum_r if sum_r == np.array([]) else np.append(sum_rewards, sum_r, axis=0)
            #mean_policy = np.mean(policy_i,axis=0)
            curr_policies.append(policy_i)
        nash_eq_divergences = np.asarray(nash_eq_divergences)
        #print('nash_eq_divergences:',nash_eq_divergences)
        # print('util errs: ',errs)
        # print('s_errs',s_errs)

        last_policy_divergences = np.asarray(last_policy_divergences)
        curr_policies = np.asarray(curr_policies)
        exploitabilities = np.asarray(exploitabilities)
        exploiter_rewards = np.asarray(exploiter_rewards)
        #print('Nash Policies:',nashEQ_policies)
        #print('Current Policies:',curr_policies)
        # print(sum_r)
        # print(np.mean(exploiter_rewards,axis=0))
        #print('Defender Exploiter',np.mean(curr_policies,axis=1)[0][0])
        #print('Attacker Exploiter',np.mean(curr_policies,axis=1)[0][1])
        self.last_policies = curr_policies
        if render:
            env.close()
        return_dict = {'last_kl_div' :[last_policy_divergences, last_policy_divergences.mean(axis=0), last_policy_divergences.std(axis=0)],
            'util_mse': [errs, np.mean(errs),np.std(errs)],
            'sample_mse': [s_errs, np.mean(s_errs),np.std(s_errs)],
            'agent_rewards' : [sum_rewards, sum_rewards.mean(axis=0), sum_rewards.std(axis=0)],
            'policies': curr_policies}
        if exploiters:
            return_dict['exploitability'] = [exploitabilities, exploitabilities.mean(axis=0), exploitabilities.std(axis=0)]
            return_dict['exploiter_rewards'] = [exploiter_rewards,np.mean(exploiter_rewards[:,0]),np.mean(exploiter_rewards[:,1])]
        if nashEQ_policies is not None:
            return_dict['nash_kl_div'] = [nash_eq_divergences, np.sort(nash_eq_divergences)[:3].mean(axis=0), np.sort(nash_eq_divergences[:3]).std(axis=0)] #[mean_rewards, mean_rewards.mean(axis=0), mean_rewards.std(axis=0)] ,
        return return_dict
    
    def __repr__(self):
        return _std_repr(self)
    
    @classmethod
    def make(cls, id, **kwargs):
        if isinstance(id, cls):
            return id
        else:
            return Agent.agents[id].make(**kwargs)
    
    @classmethod
    def register(cls, id, entry_point, **kwargs):
        if id in Agent.agents.keys():
            raise Exception('Cannot re-register id: {}'.format(id))
        Agent.agents[id] = ClassSpec(id, entry_point, **kwargs)
        
    @classmethod
    def available(cls):
        return Agent.agents.keys()
        
class TrainableAgent(Agent):
    """
    The class of trainable agent.
    
    :param policy: (Policy) The policy 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    :param experience: (Experience) The experience memory data structure
    :param exploration: (Exploration) The exploration process 
    :param lr: (float) The learning rate
    :param gamma, batch_size: (float) The training parameters
    :param name: (str) The name of the agent      
    """
         
    def __init__(self, policy, observation_space=None, action_space=None, model=None, experience="ReplayMemory-10000", exploration="EpsGreedy", gamma=0.99, lr=0.001, batch_size=32, name="TrainableAgent", log_dir='logs',train=True):
        Agent.__init__(self, policy=marl.policy.make(policy, model=model, observation_space=observation_space, action_space=action_space), name=name)
        
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Create policy, exploration and experience
        self.experience = marl.experience.make(experience)
        self.exploration = marl.exploration.make(exploration)
        
        assert self.experience.capacity >= self.batch_size
        
        # self.log_dir = log_dir
        # self.init_writer(log_dir)    
    
    @property
    def observation_space(self):
        return self.policy.observation_space
    
    @property
    def action_space(self):
        return self.policy.action_space
    
    
    # def init_writer(self, log_dir):
    #     self.writer = SummaryWriter(os.path.join(log_dir, self.name))
        
    def store_experience(self, *args):
        """
        Store a transition in the experience buffer.
        """
        experience = [arg for arg in args]
        obs = experience[0]
        actions = experience[1]
        #print('Store experience reward:',args[2])
        if isinstance(self.experience, ReplayMemory):
            # if type(actions) == list:
            #     for act in actions:
            #         exp = experience.copy()
            #         exp[1] = act
            #         self.experience.push(*exp)
            # elif len(obs.shape) > 2: #multiprocessing
            #     for i in range(obs.shape[0]):
            #         exp_i = [exp[i] for exp in experience]
            #         for act in actions[i]:
            #             exp = exp_i.copy()
            #             exp[1] = act
            #             self.experience.push(*exp)
            # else: 
            self.experience.push(*args)
        elif isinstance(self.experience, PrioritizedReplayMemory):
            self.experience.push_transition(*args)
        
    def update_model(self, t):
        """
        Update the model.
        """
        raise NotImplementedError

    def reset_exploration(self, nb_timesteps):
        """
        Reset the exploration process. 
        """
        self.exploration.reset(nb_timesteps)
    
    def update_exploration(self, t):
        """
        Update the exploration process.
        """
        self.exploration.update(t)
        
    def action(self, observation):
        """
        Return an action given an observation (action in selected according to the exploration process).
        
        :param observation: The observation
        """
        return self.exploration(self.policy, observation)
        
    def save_policy(self, folder='.',filename='', timestep=None):
        """
        Save the policy in a file called '<filename>-<agent_name>-<timestep>'.
        
        :param filename: (str) A specific name for the file (ex: 'test2')
        :param timestep: (int) The current timestep  
        """
        folder += '/{}/{}'.format(filename, self.name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_tmp = "{}".format(filename) if filename is not '' else "{}".format(self.name)
        filename_tmp = "{}".format(filename_tmp) if timestep is None else "{}-{}".format(filename_tmp, timestep)
        
        filename_tmp = os.path.join(folder, filename_tmp)
        self.policy.save(filename_tmp)
    
    def save_policy_if_best(self, best_err, err, folder=".", filename=''):
        if best_err < err:
            logging.info("#> {} - New Best Test Error ({}) - Save Model\n".format(self.name, rew))
            filename_tmp = "{}-{}".format("best", filename) if filename is not '' else "{}".format("best")
            self.save_policy(folder=folder, filename=filename_tmp)
            return err
        return best_err
        
    def save_all(self):
        pass
    
    def learn(self, env, nb_timesteps, max_num_step=100, test_freq=1000, save_freq=1000, save_folder="models/marl",test_envs=None, exploiters=False,render=False, multi_proc=False,time_laps=0., verbose=1, timestep_init=0, log_file=None):
        """
        Start the learning part.
        
        :param env: (Gym) The environment
        :param nb_timesteps: (int) The total duration (in number of steps)
        :param max_num_step: (int) The maximum number a step before stopping episode
        :param test_freq: (int) The frequency of testing model
        :param save_freq: (int) The frequency of saving model
        """
        assert timestep_init >=0, "Initial timestep must be upper or equal than 0"
        assert timestep_init < nb_timesteps, "Initial timestep must be lower than the total number of timesteps"
        
        start_time = datetime.now()
        logging.basicConfig()
        
        reset_logging()
        if log_file is None:
            logging.basicConfig(stream=sys.stdout, format='%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(filename=os.path.join(self.log_dir,log_file), format='%(message)s', level=logging.INFO)

        logging.info("#> Start learning process.\n|\tDate : {}".format(start_time.strftime("%d/%m/%Y %H:%M:%S")))
        if test_envs is None:
            test_envs = [env]
        timestep = timestep_init
        episode = 0
        best_err = self.worst_err()
        test = False
        rewards = []
        cmses = []
        self.reset_exploration(nb_timesteps)
        while timestep < nb_timesteps:
            #tic = time.perf_counter()
            eps = self.update_exploration(timestep)
            episode +=1
            if exploiters:
                fid = random.choice([i for i in range(len(env.filename))])
                obs = env.reset(fid=fid)
            else:
                obs = env.reset()
            if(multi_proc):
                obs_reshaped = np.reshape(obs,(obs.shape[1],obs.shape[0],obs.shape[2],obs.shape[3]))
            done = False
            if render:
                time.sleep(time_laps)
            for _ in range(timestep, timestep + max_num_step):
                if(multi_proc):
                    action = self.action(obs_reshaped)
                    orig = np.asarray(action)
                    action = np.zeros((orig.shape[1],orig.shape[0],orig.shape[2]))
                    for i in range(orig.shape[1]):
                        for j in range(orig.shape[0]):
                            action[i][j] = orig[j][i]
                    obs2, rew, done, _ = env.step(action)
                    obs2_reshaped = np.reshape(obs2,(obs2.shape[1],obs2.shape[0],obs2.shape[2],obs2.shape[3]))
                    self.store_experience(obs_reshaped, orig, rew.T,obs2_reshaped, done.T)
                    for r in rew.T[0]:
                        rewards.append(r)
                else:
                    action = self.action(obs,num_actions=int(env.num_nodes_attacked/env.degree))
                    if exploiters:
                        obs2 = []
                        rew = []
                        done = []
                        obs = []
                        for act in action:
                            obs_ = env.reset(fid=fid)
                            obs2_, rew_, done_, _ = env.step(act)
                            obs2.append(obs2_)
                            rew.append(rew_)
                            done.append(done_)
                            obs.append(obs_)
                        rewards.append([rew[0][0],rew[1][0],rew[2][1]])
                    else:
                        #print(f'Train Env capacity {env.scm.capacity}')
                        obs2, rew, done, info = env.step(action)
                        # all_actions = get_combinatorial_actions(env.net_size,env.num_nodes_attacked) #delete this after debugging
                        # a1 = all_actions.index(sorted(action[0]) if type(action[0]) == list else action[0])
                        # a2 = all_actions.index(sorted(action[1]) if type(action[1]) == list else action[1])

                        # diff = np.abs(self.utils[env.fid][a1][a2]-rew[0])
                        # if diff > 0:
                        #     print(f'Environment: {env.fid}')
                        #     print(f'Action: {action}')
                        #     print(f'Reward: {rew[0]}')
                        #     print(f'Utility: {self.utils[env.fid][a1][a2]}')
                        #     exit()
                        rewards.append(rew[0])
                    #oc_bstore = time.perf_counter()
                    #print(f'Time to store experience from start of learn: {toc_bstore-tic}')
                    self.store_experience(obs, action, rew, obs2, done,info)
                    #toc_astore = time.perf_counter()
                    #print(f'Store experience time: {toc_astore-toc_bstore}')
                #obs = obs2

                #toc_bupdate = time.perf_counter()
                #print(f'Time to store experience from start of learn: {toc_bupdate-tic}')
                critic_mse = self.update_model(timestep)
                #toc_aupdate = time.perf_counter()
                # if timestep > self.agents[0].batch_size:
                #     print(f'Update model time: {toc_aupdate-toc_bupdate}')               
                if critic_mse is not None:
                    for mse in critic_mse:
                        if type(mse) == list:
                            for e in mse:
                                cmses.append(e)
                        else:
                            if not math.isnan(mse):
                                cmses.append(mse.detach().cpu().numpy())
                if(multi_proc):
                    timestep += obs.shape[0]
                else:
                    timestep+=1
                if render:
                    env.render()
                    time.sleep(time_laps)
                
                self.writer.add_scalar("Hyperparameters/Epsilon",eps,timestep)
                self.writer.add_scalar("Hyperparameters/Learning_Rate",self.agents[0].lr,timestep)
                # Save the model
                if timestep % save_freq == 0:
                    if exploiters:
                        mean_rew_ego = sum([r[0] for r in rewards]) / len(rewards)
                        mean_rew_def_expltr = sum([r[1] for r in rewards]) / len(rewards)
                        mean_rew_atk_expltr = sum([r[2] for r in rewards]) / len(rewards)
                        logging.info("#> Step {}/{} --- Mean Reward Ego: {} --- Mean Reward Def Expltr {} --- Mean Reward Atk Expltr {} \n".format(timestep, nb_timesteps,mean_rew_ego,mean_rew_def_expltr,mean_rew_atk_expltr))
                    else:
                        mean_rew = sum(rewards)/len(rewards)
                        if len(cmses) > 0:
                            mean_cmse = sum(cmses)/len(cmses) if len(cmses) > 0 else np.NAN
                            #self.writer.add_scalar("Critic/critic_error",mean_cmse,timestep)
                            cmses = []
                            logging.info("#> Step {}/{} --- Mean Reward: {} --- Critic Training Error: {} --- Epsilon: {}\n".format(timestep, nb_timesteps,mean_rew,mean_cmse,eps))
                            self.writer.add_scalar("Critic/Critic_Train_err", mean_cmse, timestep)
                        else:
                            logging.info("#> Step {}/{} --- Mean Reward: {}\n".format(timestep, nb_timesteps,mean_rew))

                    self.save_policy(timestep=timestep, folder=save_folder)
                    #self.writer.add_scalar("Reward/mean_reward",mean_rew,timestep)
                    rewards = []
                
                # Test the model
                if timestep % test_freq == 0:
                    test = True
                if is_done(done):
                    break
            #toc = time.perf_counter()
            #print(f"Episode {episode} completed in {toc - tic:0.4f} seconds")
            if test:
                print("--------------------------------------------------------Testing---------------------------------------------------------------------------")
                res_test = self.test(test_envs, nb_episodes=1, max_num_step=max_num_step, render=False,multi_proc=multi_proc,nashEQ_policies=self.nash_policies,utils=self.utils,exploiters=exploiters)
                _, last_kl_m, last_kl_std = res_test['last_kl_div']
                _, s_m_mses, s_std_mses = res_test['util_mse']
                _, s_m_rews, s_std_rews = res_test['agent_rewards']

                if exploiters:
                    _,explt1_m,explt2_m = res_test['exploiter_rewards']
                    self.writer.add_scalar("Reward/exploiter1_reward",sum(explt1_m)/len(explt1_m) if isinstance(explt1_m, list) else explt1_m, timestep-test_freq)
                    self.writer.add_scalar("Reward/exploiter2_reward",sum(explt2_m)/len(explt2_m) if isinstance(explt2_m, list) else explt2_m, timestep-test_freq)
                    _,explt_m,explt_std = res_test['exploitability']
                    self.writer.add_scalar("Reward/exploitability",sum(explt_m)/len(explt_m) if isinstance(explt_m, list) else explt_m, timestep-test_freq)
                else:
                    self.writer.add_scalar("Reward/Attacker_mean_rew", sum(s_m_rews)/len(s_m_rews) if isinstance(s_m_rews, list) else s_m_rews, timestep)
                    self.writer.add_scalar("Critic/Critic_Test_err", sum(s_m_mses)/len(s_m_mses) if isinstance(s_m_mses, list) else s_m_mses, timestep)
                    if not math.isnan(last_kl_m):
                        self.writer.add_scalar("Policy/last_kl_div", sum(last_kl_m)/len(last_kl_m) if isinstance(last_kl_m, list) else last_kl_m, timestep)
                if self.nash_policies is not None:
                   _, nash_kl_m, nash_kl_std = res_test['nash_kl_div'] 
                   self.writer.add_scalar("Policy/nash_kl_div", sum(nash_kl_m)/len(nash_kl_m) if isinstance(nash_kl_m, list) else nash_kl_m, timestep)
                duration = datetime.now() - start_time
                if verbose == 2:
                    log = "#> Step {}/{} (ep {}) - {}\n\
                        |\tCritic Test Error {} / Dev {}\n\
                        |\tAttacker Mean Reward {} / Dev {}\n\
                        |\tLast Kl Divergence {} / Dev {}) \n".format(timestep, 
                                                            nb_timesteps, 
                                                            episode, 
                                                            duration,
                                                            np.format_float_scientific(s_m_mses, precision=4), 
                                                            np.around(s_std_mses, decimals=4), 
                                                            np.around(s_m_rews, decimals=4), 
                                                            np.around(s_std_rews, decimals=4), 
                                                            np.around(last_kl_m, decimals=4), 
                                                            np.around(last_kl_std, decimals=4))
                    if self.nash_policies is not None:
                        log += "|\t Nash Kl Divergence {} / Dev {}) \n".format(np.format_float_scientific(nash_kl_m,precision=4),np.around(nash_kl_std,decimals=4))
                    if exploiters:
                        log += "|\t Exploitability {} / Dev {}) \n".format(np.around(explt_m,decimals=4),np.around(explt_std,decimals=4))
                        log += self.training_log(verbose)
                else:
                    if self.nash_policies is not None:
                        log = "#> Step {}/{} (ep {}) - {}\n\
                        |\tNash Kl Divergence {}\n\
                        |\tLast Kl Divergence {}\n\
                        |\tAttacker Mean Reward {}\n".format(timestep, 
                                                            nb_timesteps, 
                                                            episode, 
                                                            duration,
                                                            np.around(nash_kl_m,decimals=4),
                                                            np.around(last_kl_m, decimals=4), 
                                                            np.around(s_m_rews, decimals=4))
                    else:
                        log = "#> Step {}/{} (ep {}) - {}\n\
                            |\tLast Kl Divergence {}\n\
                            |\tAttacker Mean Reward {}\n".format(timestep, 
                                                                nb_timesteps, 
                                                                episode, 
                                                                duration,
                                                                np.around(last_kl_m, decimals=4), 
                                                                np.around(s_m_rews, decimals=4))
                logging.info(log)
                best_err = self.save_policy_if_best(best_err, s_m_mses, folder=save_folder)
                test = False
                
        logging.info("#> End of learning process !")
    
    
    def training_log(self, verbose):
        if verbose >= 2:
            log = "#> {}\n\
                    |\tExperience : {}\n\
                    |\tExploration : {}\n".format(self.name, self.experience, self.exploration)
            return log

class MATrainable(object):
    def __init__(self, mas, index):    
        self.mas = mas
        self.index = index
    
    def set_mas(self, mas):
        self.mas = mas
        for ind, ag in enumerate(self.mas.agents):
            if ag.id == self.id:
                self.index = ind
        
def register(id, entry_point, **kwargs):
    Agent.register(id, entry_point, **kwargs)
    
def make(id, **kwargs):
    return Agent.make(id, **kwargs)
    
def available():
    return Agent.available()

