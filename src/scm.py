import networkx as nx
import generate_networks as gn
import numpy as np
import matplotlib.pyplot as plt

class SCM():
	def __init__(self,G, cascade_type='threshold',comm_net=None,thresholds=None):
		self.G = G
		self.cascade_type = cascade_type
		self.comm_net = comm_net
		#SCM is a directed graph with 2 nodes for every node in G. Nodes 1-N represent state of nodes in G at time 0, and nodes N+1-2N represent state of nodes at time 1.
		self.SCM = nx.DiGraph()

		if cascade_type == 'threshold' or 'shortPath':
			#Add thresholds to the nodes in G
			if thresholds is None:
				self.thresholds = []
				for node in self.G.nodes():
					if cascade_type == 'threshold':
						thresh = 1/len(self.G[node])*np.random.choice([i for i in range(1,len(self.G[node])+1)])
					else: #cascade_type == 'shortPath':
						thresh = np.random.normal(1,0.5)
					self.thresholds.append(thresh)
			else:
				self.thresholds = thresholds
			for i in range(2*self.G.number_of_nodes()):
				self.SCM.add_node(i,state=1)
			for node in self.G.nodes():
				self.SCM.add_edge(node, node+self.G.number_of_nodes()) #always an edge from time 0 to time 1 for the same node
				for n in self.G[node]:
					self.SCM.add_edge(node,n+self.G.number_of_nodes())
		elif cascade_type == 'coupled':
			for i in range(2*self.G.number_of_nodes()):
				self.SCM.add_node(f'P{i}',state=1)
			for i in range(2*self.comm_net.number_of_nodes()):
				self.SCM.add_node(f'C{i}',state=1)
			for node in self.G.nodes():
				self.SCM.add_edge(f'P{node}',f'P{node+self.G.number_of_nodes()}')
				self.SCM.add_edge(f'P{node}',f'C{node+self.G.number_of_nodes()}')
				for n in self.G[node]:
					self.SCM.add_edge(f'P{node}',f'P{n+self.G.number_of_nodes()}')
			for node in self.comm_net.nodes():
				self.SCM.add_edge(f'C{node}',f'C{node+self.G.number_of_nodes()}')
				self.SCM.add_edge(f'C{node}',f'P{node+self.G.number_of_nodes()}')
				for n in self.comm_net[node]:
					self.SCM.add_edge(f'C{node}',f'C{n+self.G.number_of_nodes()}')		
		elif cascade_type == 'electric_coupled':
			print('Electric coupled networks not yet implemented. Exiting.')
		else:
			print('Error: Cascade Type not recognized. Exiting.')
		self.reset()

	def reset(self):
		for i in range(len(self.SCM.nodes())):
			self.SCM.nodes()[i]['state'] = 1
		if self.cascade_type == 'shortPath':
			self.loads = np.zeros(self.G.number_of_nodes())
			paths = dict(nx.all_pairs_shortest_path(self.G))
			for s in paths:
				for t in paths[s]:
					for n in paths[s][t]:
						if s < t:
							self.loads[n] += 1	
			self.capacity = [int(np.round(self.loads[n]*max([(1+self.thresholds[n]),1]))) for n in range(self.G.number_of_nodes())]

	def check_cascading_failure(self,initial_failures):
		for n in initial_failures:
			if self.cascade_type == 'coupled':
				self.SCM.nodes()[f'P{n}']['state'] = 0
				self.SCM.nodes()[f'P{n+self.G.number_of_nodes()}']['state'] = 0

			else:
				self.SCM.nodes()[n]['state'] = 0
				self.SCM.nodes()[n+self.G.number_of_nodes()]['state'] = 0
		x = 0
		ee = 1000
		steady_state = False
		failure_set = []
		pow_copy = self.G.copy()
		if self.cascade_type == 'coupled':
			comm_copy = self.comm_net.copy()
		if self.cascade_type == 'shortPath':
			#Check initial cascades
			for n in range(0, self.G.number_of_nodes()):
				if self.SCM.nodes()[n]['state'] == 0:
					if n not in failure_set:
						failure_set.append(n) 
						pow_copy.remove_node(n)

		while not steady_state and x < ee:
			if self.cascade_type == 'threshold':
				#Check immediate cascades
				for n in range(0, self.G.number_of_nodes()):
					if self.SCM.nodes()[n]['state'] == 0:
						self.SCM.nodes()[n+self.G.number_of_nodes()]['state'] = 0
						if n not in failure_set:
							failure_set.append(n) 
					neighbor_states = [self.SCM.nodes()[v]['state'] for v in self.G[n]]
					if (1-np.mean(neighbor_states)) > self.thresholds[n]:
						self.SCM.nodes()[n+self.G.number_of_nodes()]['state'] = 0
						if n not in failure_set:
							failure_set.append(n)
				#Check if steady state has been reached
				steady_state = True
				for n in range(0, self.G.number_of_nodes()):
					if self.SCM.nodes()[n]['state'] != self.SCM.nodes()[n+self.G.number_of_nodes()]['state']:
						steady_state = False
						self.SCM.nodes()[n]['state'] = self.SCM.nodes()[n+self.G.number_of_nodes()]['state']
				x += 1
				#debugging
				# if steady_state:
				# 	for n in failure_set:
				# 		neighbors_n = [v for v in self.G[n]]
				# 		print(f'Neighbors of {n}: {neighbors_n}')
				# 		if not (set(neighbors_n).union(set([n]))).intersection(set(failure_set)):
				# 			print(f'fail_set: {failure_set}')
				# 			exit()
				# 		nx.draw(self.G,with_labels=True)
				# 		plt.draw()
				# 		plt.show()
			elif self.cascade_type == 'shortPath':
				#redistribute loads and fail any that exceed capacity
				self.loads = np.zeros(self.G.number_of_nodes())
				paths = dict(nx.all_pairs_shortest_path(pow_copy))
				for s in paths:
					for t in paths[s]:
						for n in paths[s][t]:
							if s < t:
								self.loads[n] += 1	
				sub = self.G.subgraph([n for n in self.G.nodes() if n not in failure_set])
				ccs = sorted(nx.connected_components(sub),key=len,reverse=True)
				Gcc = self.G.subgraph(ccs[0])
				for n in range(0, self.G.number_of_nodes()):
					if self.loads[n] > self.capacity[n] or n not in Gcc.nodes():
						self.SCM.nodes()[n+self.G.number_of_nodes()]['state'] = 0
						if n not in failure_set:
							failure_set.append(n) 
							pow_copy.remove_node(n)

				#Check if steady state has been reached
				steady_state = True
				for n in range(0, self.G.number_of_nodes()):
					if self.SCM.nodes()[n]['state'] != self.SCM.nodes()[n+self.G.number_of_nodes()]['state']:
						steady_state = False
						self.SCM.nodes()[n]['state'] = self.SCM.nodes()[n+self.G.number_of_nodes()]['state']
				x += 1
			elif self.cascade_type == 'coupled':
				#Check immediate cascades
				for n in range(0, self.G.number_of_nodes()):
					if self.SCM.nodes()[f'P{n}']['state'] == 0 or self.SCM.nodes()[f'C{n}']['state'] == 0:
						self.SCM.nodes()[f'P{n+self.G.number_of_nodes()}']['state'] = 0
						self.SCM.nodes()[f'C{n+self.G.number_of_nodes()}']['state'] = 0
						if n not in failure_set:
							failure_set.append(n) 
							pow_copy.remove_node(n)
							comm_copy.remove_node(n)

				#Remove any edges in copy networks that don't meet coupled path requirement
				for edge in pow_copy.edges():
					if not nx.has_path(comm_copy,edge[0],edge[1]):
						pow_copy.remove_edge(edge[0],edge[1])
				for edge in comm_copy.edges():
					if not nx.has_path(pow_copy,edge[0],edge[1]):
						comm_copy.remove_edge(edge[0],edge[1])

				#Remove any nodes that are not in largest connected component of their network
				for n in range(0, self.G.number_of_nodes()):
					if n not in max(nx.connected_components(pow_copy),key=len) or n not in max(nx.connected_components(comm_copy),key=len):
						if n not in failure_set:
							self.SCM.nodes()[f'P{n+self.G.number_of_nodes()}']['state'] = 0
							self.SCM.nodes()[f'C{n+self.G.number_of_nodes()}']['state'] = 0
							failure_set.append(n)
							pow_copy.remove_node(n)
							comm_copy.remove_node(n)
						
				#Check if steady state has been reached
				steady_state = True
				for n in range(0, self.G.number_of_nodes()):
					if self.SCM.nodes()[f'P{n}']['state'] != self.SCM.nodes()[f'P{n+self.G.number_of_nodes()}']['state']:
						steady_state = False
						self.SCM.nodes()[f'P{n}']['state'] = self.SCM.nodes()[f'P{n+self.G.number_of_nodes()}']['state']

					if self.SCM.nodes()[f'C{n}']['state'] != self.SCM.nodes()[f'C{n+self.G.number_of_nodes()}']['state']:
						steady_state = False
						self.SCM.nodes()[f'C{n}']['state'] = self.SCM.nodes()[f'C{n+self.G.number_of_nodes()}']['state']

				x += 1
			else:
				print(f'Cascade type {self.cascade_type} not supported')
				exit()
		if x >= ee:
			print(f'Steady State not reached. Returning with failure set {failure_set}')
		return failure_set

	def show_graph(self,G):
		import matplotlib.pyplot as plt
		nx.draw(G,with_labels=True)
		plt.draw()
		plt.show()

if __name__ == '__main__':
	num_nodes = 10
	num_attacked = 1

	for i in range(1000):
		comm_net,pow_net = gn.create_random_nets('',num_nodes,num2gen=1,show=False)
		scm = SCM(pow_net,cascade_type='shortPath')
		for j in range(100):
			initial_failures = np.random.choice([i for i in range(num_nodes)],size=num_attacked,replace=False)
			num_fails = []
			for k in range(10):
				failure_set = scm.check_cascading_failure(initial_failures=initial_failures)
				scm.reset()
				num_fails.append(len(failure_set))
			if np.std(num_fails) > 0:
				print(initial_failures)
				print(np.std(num_fails))
				exit()
	print('No variance in scm results.')