from . import MATrainable,MinimaxDQNCriticAgent
from ..policy import QPolicy,QCriticPolicy,MinimaxQCriticPolicy,MinimaxQTablePolicy
from ..model import MultiQTable
from marl.tools import gymSpace2dim,get_combinatorial_actions
from marl.experience import make
from ..experience import ReplayMemory, transition_tuple
from cascade_cfa import Counterfactual_Cascade_Fns

import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

class CFA_MinimaxDQNCriticAgent(MinimaxDQNCriticAgent,MATrainable):
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
	
	def __init__(self, qmodel,observation_space, my_action_space, other_action_space,fact_experience,cfact_experience,env,training_epochs,topo_eps,act_degree=1, index=None, mas=None, exploration="EpsGreedy", gamma=0.99, lr=0.1,sched_step=100e3,sched_gamma=0.1,batch_size=32, tau=1., target_update_freq=1000, name="MultiDQNCriticAgent",train=True,model=None):
		self.topo_eps = topo_eps
		self.episode = 0
		self.topo = 0
		self.last_cfact_len = 0
		super(CFA_MinimaxDQNCriticAgent, self).__init__(qmodel=qmodel, my_action_space=my_action_space,other_action_space=other_action_space,observation_space=observation_space,act_degree=act_degree,exploration=exploration, gamma=gamma, lr=lr,sched_step=sched_step,sched_gamma=sched_gamma, batch_size=batch_size, target_update_freq=target_update_freq, name=name)
		#fact_experience=fact_experience,cfact_experience=cfact_experience,
		# Create separate fact and cfact experience buffers
		MATrainable.__init__(self, mas, index)
		self.max_epochs = 100
		self.policy = MinimaxQCriticPolicy(qmodel,action_space=my_action_space,observation_space=observation_space,player=index,all_actions=self.all_actions,act_degree=self.degree)
		self.fact_experience = make(fact_experience)
		if cfact_experience is not None:
			self.cfact_experience = make(cfact_experience)
		self.env = env
		self.cfa_cascade_fns = Counterfactual_Cascade_Fns(self.env)
		self.fail_components = []
		self.init_fails = []
		assert self.fact_experience.capacity >= self.batch_size
		if hasattr(self,'cfact_experience'):
			assert self.cfact_experience.capacity >= self.batch_size

	def store_experience(self,*args,f='fact'):
		"""
		Store a transition in the experience buffer.
		"""
		if f == 'fact':
			buffer = self.fact_experience
		elif f == 'cfact':
			buffer = self.cfact_experience
		else:
			print('Unrecognized Buffer.')

		experience = [arg for arg in args]
		# if f == 'fact':
		# 	print(f'Fact experience: {[experience[1],experience[2],experience[5]]}')
		# else:
		# 	print(f'Cfact experience: {[experience[1],experience[2],experience[5]]}')
		#obs = experience[0]
		#actions = experience[1]
		#print('Store experience reward:',args[2])
		if isinstance(buffer, ReplayMemory):
			buffer.push(*args)
		elif isinstance(buffer, PrioritizedReplayMemory):
			buffer.push_transition(*args)
		if f == 'fact' and hasattr(self,'cfact_experience'):
			#obs, action, rew, obs2, done,info = buffer.sample(batch_size=sample_batch_size)
			#print(f'Topo: {self.topo}')
			#print(f'Episode: {self.episode}')
			#print(f'SCM Edges: {self.env.scm.G.edges()}')
			_,_,_,_,_,info = experience
			fail_set = info['fail_set']
			init_fail = info['init_fail']
			#print('Info Edges: ', info['edges'])
			#print('action:',experience[1])
			if self.cfa_cascade_fns.casc_type == 'threshold':
				sub = self.env.scm.G.subgraph(fail_set)
				fail_components = {}
				for c in nx.connected_components(sub):
					c_init = [n for n in c if n in init_fail]
					if len(c_init) > 1:
						continue
					fail_components[c_init[0]] = list(c)
					# for i in c_init:
					# 	fail_components[i] = list(c)
			else:
				fail_components = {-1: []}
				for f in fail_set:
					if f in init_fail: 
						fail_components[f] = [f]
					else:
						fail_components[-1].append(f)
			# for k in fail_components.keys():
			# 	if not set(fail_components[k]).intersection(init_fail):
			# 		print('Error: No part of the initial failure is in this connected component.')
			# 		print('CC: ',fc)
			# 		print('fail_set: ', fail_set)	
			# 		print('init fail: ',init_fail)
			# 		print('thresholds: ',self.env.scm.thresholds)
			# 		nx.draw(self.env.scm.G,with_labels=True)
			# 		plt.draw()
			# 		plt.show()
			# 		exit()	 
			#print('next CCs: ',CCs)
			if self.fact_experience.__len__()-self.topo*self.topo_eps > 1:
				self.gen_cfs(fail_components,init_fail,experience)
			self.fail_components.append(fail_components)
			self.init_fails.append(init_fail)
			self.episode += 1
			if self.episode >= self.topo_eps:
				self.fail_components = []
				self.init_fails = []
				self.episode = 0
				self.topo += 1
			#print('self.fail_components: ',self.fail_components[-1])
			#print('self.init_fails: ',self.init_fails)

	def gen_cfs(self,new_failure_component,init_fail_new,experience):
		#print(f'Start indx = {self.topo*self.topo_eps}')
		#print(f'End indx: {self.fact_experience.__len__()-1}')
		fac_actions = self.fact_experience.get_transition([i for i in range(self.topo*self.topo_eps,self.fact_experience.__len__()-1)]).action
		fac_info = self.fact_experience.get_transition([i for i in range(self.topo*self.topo_eps,self.fact_experience.__len__()-1)]).info
		# print('fac actions: ',fac_actions)
		# print('old_CCs:',self.CCs)
		# print('new_CCs:',CCs_new)
		cfac_count = 0
		# print('New CC: ',CCs_new)
		# print('Old CC count: ',len(self.CCs))	
		for i,a in enumerate(fac_actions):
			num_inits = len(new_failure_component.keys()) + len(self.fail_components[i].keys()) - 2   #sum([1 for n in cc1 if n in init_fail_new or n in self.init_fails[i]])
			if num_inits > len(experience[1][0])+len(experience[1][1]):
				continue
			indep_sets = self.cfa_cascade_fns.check_failure_independence(new_failure_component,self.fail_components[i])
			for idp in indep_sets:
				cfac_atk_action = idp.copy()
				# fac_atk_actions = [n for n in experience[1][0] if (n in idp and n not in experience[1][1])] + [n for n in list(a[0]) if (n in cc2 and n not in list(a[1]))]
				# #if len(fac_atk_actions) < len(experience[1][0]):
				# #	continue
				# random.shuffle(fac_atk_actions)
				# cfac_atk_action = []
				# for atk in fac_atk_actions:
				# 	if atk not in cfac_atk_action:
				# 		cfac_atk_action.append(atk)
				# 	if len(cfac_atk_action) >= len(list(a[0])):
				# 		break
				all_def_actions = experience[1][1] + list(a[1])
				random.shuffle(all_def_actions)
				cfac_def_action = []
				for d in all_def_actions:
					if d not in cfac_def_action and d not in cfac_atk_action:
						cfac_def_action.append(d)
					if len(cfac_def_action) >= len(list(a[1])):
						break
				if len(cfac_def_action) < len(list(a[1])):
					d = np.random.choice([a for a in range(self.env.scm. G.number_of_nodes()) if a not in cfac_atk_action])
					cfac_def_action.append(d)
				cfac_action = [cfac_atk_action,cfac_def_action]
				cfac_init_fail = cfac_atk_action #[n for n in cfac_atk_action if n not in cfac_def_action]
				for f in idp:
					if f in new_failure_component.keys():
						def_1 = f
					if f in self.fail_components[i].keys():
						def_2 = f
				#TODO: Account for case where init fail from one component is a cascaded fail in the other
				# def_1 = cc1 if any(n in cc1 for n in cfac_init_fail) else []
				# def_2 = cc2 if any(n in cc2 for n in cfac_init_fail) else []
				counterfac_casc_fail = new_failure_component[def_1] + self.fail_components[i][def_2]
				#counterfac_casc_fail = new_failure_component[cfac_init_fail] + 
				fac_cfac_casc_fail = self.env.scm.check_cascading_failure(cfac_init_fail)
				self.env.scm.reset()
				#self.env.scm.reset()
				#Check that cfac is valid
				if set(fac_cfac_casc_fail) != set(counterfac_casc_fail):
					print('action1: ', experience[1])
					print('action2: ', [list(a[0]),list(a[1])])
					print('init1: ',init_fail_new)
					print('init2: ', self.init_fails[i])
					print(f'orig_failset1: ', experience[-1]['fail_set'])
					print('orig_failset2: ', fac_info[i]['fail_set'])
					print('cfac_init: ', cfac_init_fail)
					print('comp1: ', new_failure_component)
					print('comp2: ', self.fail_components[i])
					if self.cfa_cascade_fns.casc_type == 'shortPath':
						print('Factual Fail Set 1: ', self.env.scm.check_cascading_failure(init_fail_new))
						self.env.scm.reset()
						print('Factual Fail Set 2: ', self.env.scm.check_cascading_failure(self.init_fails[i]))
						self.env.scm.reset()
					print('cfac_actions: ',cfac_action)
					print('cfac_init_fail: ', cfac_init_fail)
					print('def_1: ', def_1)
					print('def_2: ', def_2)
					print('counterfac_casc_fail: ', counterfac_casc_fail)
					print('fac_cfac_casc_fail: ',fac_cfac_casc_fail)
					exit()
				# if cfac_init_fail == init_fail_new or cfac_init_fail == self.init_fails[i]:
					# 	continue
				r = len(counterfac_casc_fail)/self.env.scm.G.number_of_nodes()
				reward = [r,-r]
				if r == 0 and len(counterfac_casc_fail) > 0:
					print('Reward: ',r)
					print('Num node: ',self.env.scm.G.number_of_nodes())
					print('Old init: ', self.init_fails[i])
					print('New init: ',init_fail_new)
					print('CC1: ', cc1)
					print('CC2: ',cc2)
					print("Cfac Casc: ", counterfac_casc_fail)
					print('Cfac Init:',fcac_init_fail)
					exit()		
				done = [True,True]
				info = {'init_fail':cfac_init_fail,'fail_set':counterfac_casc_fail}
				#print('cfac def action: ', cfac_def_action)
				# if len(cfac_init_fail) > 2:
				# 	#print('num_inits:', num_inits)
				# 	print('cc1:',cc1)
				# 	print('cc2:',cc2)
				# 	print('init fail new: ', init_fail_new)
				# 	print('init fail old: ', self.init_fails[i])
				# 	print('cfac action:',cfac_action)
				# # 	exit()
				self.store_experience(experience[0],cfac_action, reward, experience[3],done,info,f='cfact')
				cfac_count += 1
	#print('cfac_count: ', cfac_count)

	def update_model(self, t):
		"""
		Update the model.
		
		:param t: (int) The current timestep
		"""
		debug_print_step = 500

		if not hasattr(self,'cfact_experience'):
			if len(self.fact_experience) < self.batch_size:
				return np.NAN
			mr = 1
			epochs = 1
		else:
			mr = len(self.fact_experience)/(len(self.cfact_experience)+len(self.fact_experience)) #proportionate mixing of factual and cfactual experiences
			mr = max([mr,1/self.batch_size])
			if len(self.fact_experience) < mr*self.batch_size or len(self.cfact_experience) < (1-mr)*self.batch_size:
				return np.NAN
			diff = len(self.cfact_experience) - self.last_cfact_len 
			epochs = diff + 1

			self.last_cfact_len = len(self.cfact_experience)
		# Get changing policy
		curr_policy = self.target_policy if self.off_policy else self.policy
		# Get batch of experience
		#if isinstance(self, MATrainable):
		#	 #ind = self.experience.sample_index(self.batch_size)
		#	 #batch = self.mas.experience.get_transition(len(self.mas.experience) - np.array(ind)-1)
		# else
		critic_mse = []
		# if epochs > 1:
		# 	print(f'Num epochs {epochs}')
		# 	print(f'Diff: {diff}')
		# if len(self.cfact_experience) > 0 and t % debug_print_step == 0:
		# 	print(f'Cfact Buffer Length {len(self.cfact_experience)}. Training for {epochs} epochs.')
		# 	print(f'Fact Buffer Length {len(self.fact_experience)}')

		for i in range(epochs):
			if mr >= 1:
				batch = self.fact_experience.sample(self.batch_size)
			elif mr < 1/self.batch_size:
				batch = self.cfact_experience.sample(self.batch_size)
			else:
				# print('mixing ratio: ', mr)
				#print('len fac buffer: ',len(self.fact_experience))
				#print('len cfac buffer: ',len(self.cfact_experience))
				# print('------------------------------------\n')
				fac_batch = self.fact_experience.sample(int(self.batch_size*mr))
				cfac_batch = self.cfact_experience.sample(self.batch_size - int(self.batch_size*mr))				   
				batch = {'observation':[], 'action':[], 'reward':[], 'next_observation':[], 'done_flag':[],'info':{'init_fail': [], 'fail_set':[]}}
				for key in batch.keys():
					if key == 'info': 
						 batch['info']['init_fail'] += getattr(fac_batch,key)['init_fail']
						 batch['info']['init_fail'] += getattr(cfac_batch,key)['init_fail']
						 batch['info']['fail_set'] += getattr(fac_batch,key)['fail_set']
						 batch['info']['fail_set'] += getattr(cfac_batch,key)['fail_set']
					else:
						batch[key] += getattr(fac_batch,key)
						batch[key] += getattr(cfac_batch,key)
				batch = transition_tuple['FFTransition'](**batch)
			#if len(batch.observation)%self.batch_size != 0:
				# print('mixing ratio: ', mr)
				# print('len fac buffer: ',len(self.fact_experience))
				# print('len cfac buffer: ',len(self.cfact_experience))
				# print('len fac batch: ',len(fac_batch))
				# print('len cfac batch: ',len(cfac_batch))
			# unique_obs = []
			# for obs in batch.observation:
			# 	if not any([np.array_equal(obs,u_obs) for u_obs in unique_obs]):
			# 		unique_obs.append(obs)		
			# if t % debug_print_step == 0:	
			# 	print(f'Unique Obs in batch (t = {t}): {len(unique_obs)}')
			# unique_acts = []
			# unique_exp = []
			# for i,act in enumerate(batch.action):
			# 	if not any([act == u_act for u_act in unique_acts]):
			# 		unique_acts.append(act)
			# 		unique_exp.append({'action': act, 'reward': batch.reward[i],'init_fail': batch.info['init_fail'][i], 'fail_set':batch.info['fail_set'][i]})
			# 	else:
			# 		for exp in unique_exp:
			# 			if exp['action'] == act:
			# 				if batch.reward[i] != exp['reward']:
			# 					print(f'Reward for same actions does not match! A1: {act}, A2: {exp["action"]}, R1: {batch.reward[i]}, R2: {exp["reward"]}')
			# 					exit()
			# 				if set(batch.info["init_fail"][i]) != set(exp["init_fail"]):
			# 					print(f'Init fail for same actions does not match! A1: {act}, A2: {exp["action"]}, F1: {batch.info["init_fail"][i]}, F2: {exp["init_fail"]}')
			# 					exit()
			# 				if set(batch.info['fail_set'][i]) != set(exp['fail_set']):
			# 					print(f'Fail_set for same actions does not match! A1: {act}, A2: {exp["action"]}, F1: {batch.info["fail_set"][i]}, F2: {exp["fail_set"]}')
			# 					exit()

			# dup_ct = len(batch.action) - len(unique_acts)
			# if t % debug_print_step == 0:	
			# 	print(f'Duplicate actions in batch (t = {t}): {dup_ct}')

			# Compute target r_t + gamma*max_a Q(s_t+1, a)
			# idxs = []
			# for j,a in enumerate(batch.action):
			# 	if a[0] == a[1]:
			# 		idxs.append(j)
			target_value = self.target(curr_policy.Q, batch).float()
			# # Compute current value Q(s_t, a_t)
			#tic = time.perf_counter()
			curr_value = self.value(batch.observation, batch.action)
			#toc = time.perf_counter()
			#print(f'Calculate Value time: {toc-tic}')
			#print(f'Reward: {target_value[0].detach().numpy()}, Pred: {curr_value[0].detach().numpy()}')
			# # Update Q values
			sched_step = (i >= epochs - 1)
			self.update_q(curr_value, target_value, batch,sched_step)
			if self.off_policy and t % self.target_update_freq==0:
				self.update_target_model()
			critic_mse.append(torch.norm(target_value - curr_value,p=1).detach().cpu().numpy()/target_value.shape[0])
		#print(f'Average time per epoch {total/epochs}')
		#print(f'Batch norm time per epoch {total/(epochs*self.batch_size)}')

		return critic_mse