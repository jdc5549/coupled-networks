import networkx as nx
#from generate_networks import create_random_nets
from scm import SCM
import numpy as np
import matplotlib.pyplot as plt

class Counterfactual_Cascade_Fns():
	def __init__(self,env):
		self.env = env
		if self.env.scm.cascade_type == 'shortPath':
			self.deltaL = np.zeros([self.env.scm.G.number_of_nodes(),self.env.scm.G.number_of_nodes()])
			self.L = np.zeros(self.deltaL.shape[0])
		self.casc_type = self.env.scm.cascade_type

	def generate_data(self,num2gen,fail_size):
		initial_failures = []
		failure_sets = []
		for i in range(num2gen):
			initial_failure = sorted(list(np.random.choice([j for j in range(0,self.env.scm.G.number_of_nodes())],size=fail_size)))
			while len(initial_failure) > len(set(initial_failure)):
				initial_failure = sorted(list(np.random.choice([j for j in range(0,self.env.scm.G.number_of_nodes())],size=fail_size)))
			initial_failures.append(initial_failure)
			failure_set = self.env.scm.check_cascading_failure(initial_failures=initial_failure)
			failure_sets.append(failure_set)
			self.env.scm.reset()
		return initial_failures, failure_sets

	def check_failure_independence(self,f1,f2):
		f1_inits = list(f1.keys())
		f2_inits = list(f2.keys())
		indep_sets = []
		if self.casc_type == 'threshold':
			for i,k1 in enumerate(f1_inits):
				if k1 in f2_inits: continue
				for j,k2 in enumerate(f2_inits):
					if k2 in f1_inits: continue
					f1_init = f1_inits[i]
					f2_init = f2_inits[j]
					f1_fail_set = f1[f1_init]
					f2_fail_set = f2[f2_init]
					#if one cascade is subset of the other no info can be gained from combining them
					if set(f1_fail_set).issubset(set(f2_fail_set)) or set(f2_fail_set).issubset(set(f1_fail_set)):
						return False
					#check independence through neighbors
					thresh_copy = self.env.scm.thresholds.copy()
					all_fail_nodes = list(set(f1_fail_set) | set(f2_fail_set)) #merge lists without repeating elements 
					casc_thresh = False
					for node in all_fail_nodes:
						for n in self.env.scm.G[node]: 
							if n not in all_fail_nodes: 
								thresh_copy[n] -= 1/len(self.env.scm.G[n])    #decrement thresh of neighbor nodes
								if thresh_copy[n] <= 0:
									casc_thresh = True
					if not casc_thresh:
						indep_sets.append([k1,k2])
		elif self.casc_type == 'shortPath':
			f1_inits.remove(-1)
			f2_inits.remove(-1)
			#print(f1_inits)
			#print(f2_inits)
			for i,k1 in enumerate(f1_inits):
				if k1 in f2_inits: continue
				for j,k2 in enumerate(f2_inits):
					if k2 in f1_inits: continue
					f1_init = f1_inits[i]
					f2_init = f2_inits[j]
					f1_fail_set = f1[f1_init]
					f2_fail_set = f2[f2_init]
					unsure_fail_set = f1[-1] + f2[-1]
					l0 = self.env.scm.loads
					for n in [f1_init,f2_init]:
						solo_fail_set = self.env.scm.check_cascading_failure([n])
						self.env.scm.reset()
						if np.array_equal(self.deltaL[n],np.zeros_like(self.deltaL[n])):
							self.L[n] = np.sum([l0[f] for f in solo_fail_set])-len(solo_fail_set)*(len(l0)-1)
							lf = self.env.scm.loads
							self.deltaL[n] = lf - l0
						found_fails = [v for v in solo_fail_set if v in unsure_fail_set]
						for f in found_fails:
							if f in f1[-1] and n == f1_init: 
								f1[f1_init].append(f)
								f1[-1].remove(f)
							if f in f2[-1] and n == f2_init: 
								f2[f2_init].append(f)
								f2[-1].remove(f)
					if set(f2_fail_set).issubset(set(f1_fail_set)) or set(f1_fail_set).issubset(set(f2_fail_set)):
						#print(f'One of these should be a subset of the other: {f1_fail_set},{f2_fail_set}')
						continue
					if any([f not in f1[f1_init]+f2[f2_init] for f in unsure_fail_set]):
						#print(f'element in {unsure_fail_set} not found in {f1[f1_init]+f2[f2_init]}')
						#print(f'f1: {f1}, f2: {f2}')
						continue

					capacity = self.env.scm.capacity
					indep = True
					for n in self.env.scm.G.nodes():
						if n not in f1_fail_set or f2_fail_set: 
							if capacity[n] < l0[n] + self.deltaL[f1_init,n] + self.L[f2_init] or capacity[n] < l0[n] + self.deltaL[f2_init,n] + self.L[f1_init]:
								indep = False
					if indep:
						indep_sets.append([k1,k2])
		else: 
			print(f'Error: Cascade Type {self.casc_type} is not recognized')
			exit()
		for idp in indep_sets:
			if len(idp) > 2:
				print(idp)
				exit()
			if len(set(idp)) > len(idp):
				print(idp)
				exit()
		return indep_sets


if __name__ == '__main__':
	from couplednetworks_gym_cfa import SimpleCascadeEnv, create_random_nets
	num_nodes = 100
	env = SimpleCascadeEnv(num_nodes,0.1,0.1,'SF',degree=1,cascade_type='shortPath')
	comm_net,pow_net = create_random_nets('',num_nodes,num2gen=1,show=False)
	ccf = Counterfactual_Cascade_Fns(env)

	casc2gen = 1
	initial_failures, failure_sets = ccf.generate_data(casc2gen,2)
	CCs = ccf.get_failure_component([[0,1]])
	counterfac_init_fails = []
	counterfac_casc_fails = []
	fac_cfac_casc_fails = []
	print(initial_failures)
	print('\n')
	print(CCs)
	print('\n')
	exit()

	for i,f1 in enumerate(CCs):
		for j,f2 in enumerate(CCs):
			if i>=j:
				continue
			for k,cc1 in enumerate(f1):
				for m,cc2 in enumerate(f2):
					extra = 0
					if check_component_independence(cc1,cc2,scm):
						# print(i)
						# print(j)
						cfail = []
						for f in initial_failures[i]:
							if f in cc1:
								cfail.append(f)
						for f in initial_failures[j]:
							if f in cc2 and f not in cc1:
								cfail.append(f)
						counterfac_init_fails.append(sorted(cfail))
						counterfac_casc_fails.append(sorted(list(set(cc1) | set(cc2))))
						fcl = len(cc1+cc2)
						scl = len(set(cc1+cc2))

						if fcl > scl:
							extra +=1
						fac_cfac_casc_fails.append(sorted(scm.check_cascading_failure(initial_failures=cfail)))
						scm.reset()
						if counterfac_casc_fails[-1] != fac_cfac_casc_fails[-1]:
							print('Initial: ', sorted(cfail))
							print('Cascade 1: ',sorted(cc1))
							print('Cascade 2: ',sorted(cc2))
							print(f'Full Combined Length {fcl}: {cc1 + cc2}')
							print(f'Set Combined Length {scl}: {set(cc1 + cc2)}')
							print('CFactual: ', counterfac_casc_fails[-1])
							print('Factual: ', fac_cfac_casc_fails[-1])
							nx.draw(scm.G,with_labels=True)
							plt.draw()
							plt.show()
	print(f'Total of {len(counterfac_casc_fails)} examples generated from {num2gen} factual examples')
	print(f'{extra} Extra samples by allowing overlap in factual samples')
	# for i,f in enumerate(counterfac_init_fails):
	# 	print('Initial: ',f,'\n')
	# 	print('Factual: ',fac_cfac_casc_fails[i],'\n')
	# 	print('CFactual: ',counterfac_casc_fails[i],'\n')
	# 	print('--------------------------')

	# nx.draw(scm.G,with_labels=True)
	# plt.draw()
	# plt.show()
	# exit()