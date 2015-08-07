#!/usr/bin/env python

import multiprocessing as mlt
import math
import random
import os
import sys
import json
import time
import networkx as nx
from collections import defaultdict
import csv
import subprocess
import shutil
import argparse
import glob
from operator import itemgetter
import bz2
import codecs

# Run like: python cfs_comms_model.py config-cfs.json 0

parser = argparse.ArgumentParser(description='Coupled Network Model')
parser.add_argument('config_file', metavar='configuration_file', type=str, nargs=1, help='name and location of config.json file')
parser.add_argument('r_num', metavar='replicate_number', type=int, nargs=1, help='replicate number if running in batch mode')
args = parser.parse_args()
config_file = args.config_file[0]
r_num = args.r_num[0] 

NUM_PROCS = mlt.cpu_count()
print "Number of processors " + str(NUM_PROCS)
config = json.load(open(config_file))  # read in the input configuration
hostOS = config['hostOS']
targeted = config['targeted']
pythonLocation = config['pythonLocation']
commModel = config['commModel']
generateEachRun = config['generateEachRun']
networkType = config['networkType']
relpath = commModel.split('source/couplednetworks.py')[0]  # should give "coupled-networks/" root folder
randomRewireProb = config['randomRewireProb']
shuffleNetworks = config['shuffleNetworks']
n = config['n']
runs = config['runs']
verbose = config['verbose']
pValues = config['pValues']	 
pMin = config['pMin']
pMax = config['pMax']
logAllPValues = config['logAllPValues']
gCThreshold = config['gCThreshold']
gridGCThreshold = config['gridGCThreshold']
outagesFromFile = config['outagesFromFile']
findScalingExponent = config['findScalingExponent']
minOutagePoint = config['minOutagePoint']
maxOutagePoint = config['maxOutagePoint']
numOutagesInFile = config['numOutagesInFile']
networkFromFile = config['networkFromFile']
inverseStepSize = config['inverseStepSize']
batchMode = config['batchMode']
commNetworkLocation = config['commNetworkLocation']
powerNetworkLocation = config['powerNetworkLocation']
degOfCoupling = config['degOfCoupling']
hpc = config['hpc']
numOutagesInCouplingFile = config['numOutagesInCouplingFile']
minCouplingPoint = config['minCouplingPoint']
maxCouplingPoint = config['maxCouplingPoint']
couplingFromFile = config['couplingFromFile']
inverseStepSizeForCoupling = config['inverseStepSizeForCoupling']
outputRemovedNodesToFile = config['outputRemovedNodesToFile']

shutil.copy(config_file,relpath + 'source/compiled-matlab/config.json')  # copy the config over so Matlab uses the same one

nodes = range(1,n+1,1)

if hpc is "HIVE":
	r_num = r_num + 1  # incremented by 1 to use with condor process variable which starts at 0
	print "HIVE r_num: " + str(r_num)

threshold = gCThreshold * n
if gCThreshold == 0:
	threshold = 1
showPlot = config['showPlot']
debug = config['debug']
if debug is True:
	debug = 1
else:
	debug = 0
pRange = pMax - pMin
home = os.path.expanduser("~")
allY = []

# Only import matplotlib if you want to generate plots. It uses about 20MB RAM.
if showPlot is True:
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		raise "matplotlib not found, can't show plots"
		sys.exit(-1)


def configType(shuffleNetworks, generateEachRun):
	gen = "0"
	shuffle = "0"
	if generateEachRun is True:
		gen = "1"
	if shuffleNetworks is True:
		shuffle = "1"
	return gen + shuffle

networkLabel = (networkType + '{nodes}nodes_{rw}rw_{thresh}thresh_{q}q_{sruns}Avg'.format(nodes=n, sruns=runs, rw=randomRewireProb, thresh=threshold, q=degOfCoupling) + 
	'_Config_%s' % configType(shuffleNetworks, generateEachRun) + "_")  # this gets put in the file name


def attackNetwork(run,networks,processId):
	global nodes
	print "Run number " + str(run+1) + " of " + str(runs) + "."
	runstart = time.time()
	# where MATLAB writes the blackout csv file
	blackout_filename = '/tmp/blackout' + processId + '.csv'
	y = []

	# Create the networks 
	if generateEachRun is True:
		[networkA, networkB] = createNetworks(networkType)
	else: 
		networkA = networks[0]
		# networkB = networks[1]

	# where the compiled MATLAB is
	if hostOS == 0:  # Mac
		appname = relpath + 'source/compiled-matlab/cmp_dcsimsep/for_redistribution_files_only/cmp_dcsimsep.app/Contents/MacOS/cmp_dcsimsep'
	else: 
		appname = relpath + 'source/compiled-matlab/cmp_dcsimsep/src/cmp_dcsimsep'  # Linux

	for i in range(0,pValues+1,1):
		gc_size = 0.0
		p = i/float(pValues)
		p = p/(1/float(pRange))+pMin
		if p == 0.0:
			y.append([p,(0,0.0,1.0)])  # if p=0 you're going to remove all nodes and you know it will fail so don't bother simulating
			continue
		elif p == 1.0:
			y.append([p,(1,1.0,0.0)])  # if p=1 you won't remove any nodes and success is guaranteed so don't bother simulating
			continue
		randomRemovalFraction=1-p  # Fraction of nodes to remove
		# print "randomRemovalFraction " + str(randomRemovalFraction)
		numNodesAttacked = int(math.floor(randomRemovalFraction * n))
		# print "numNodesAttacked " + str(numNodesAttacked)
		# if targeted is True and outagesFromFile is False:
			# TODO, not implemented yet
			# nodesAttacked = targetedNodes(networkA,numNodesAttacked)
		if targeted is False and outagesFromFile is True and batchMode is False and couplingFromFile is True:
			nodesAttacked = getNodesFromFile(run+1,p,networkA)
			coupledNodes = getCoupledNodesFromFile(run+1,degOfCoupling,networkA)
		elif targeted is False and outagesFromFile is True and batchMode is True and couplingFromFile is True:
			nodesAttacked = getNodesFromFile(r_num,p,networkA)
			coupledNodes = getCoupledNodesFromFile(r_num,degOfCoupling,networkA)
		elif targeted is False and outagesFromFile is False and batchMode is True and couplingFromFile is False:
			number_coupled = int(math.floor(degOfCoupling * n))
			coupledNodes = random.sample(nodes,number_coupled)
			print "numNodesAttacked "  +  str(numNodesAttacked)
			nodesAttacked = random.sample(networkA.nodes(),numNodesAttacked)
		else:
			sys.stderr.write("Unknown configuration of inputs: targeted " + str(targeted) + ", outagesFromFile " + str(outagesFromFile) + ", and couplingFromFile " + str(couplingFromFile) )
			sys.exit(-1)

		blackout = False

		bus_outage_filename = '/tmp/bus_outage_' + processId + '.csv'
		with open(bus_outage_filename, 'wb') as f:
			writer = csv.writer(f)
			writer.writerow(['bus'])
			for row in nodesAttacked:
				writer.writerow([row])

		coupled_node_filename = '/tmp/coupled_nodes_' + processId + '.csv'
		with open(coupled_node_filename, 'wb') as f:
			writer = csv.writer(f)
			writer.writerow(['node'])
			for row in coupledNodes:
				writer.writerow([row])

		try:
			if len(nodesAttacked) != 0 and len(coupledNodes) == 0:
				subprocess.call([appname, processId, '[-1]', "[]", config_file])  # [-1] for nodesAttacked and coupledNodes to tell cmp_dcsimsep.m to read from file
			elif len(nodesAttacked) == 0 and len(coupledNodes) != 0:
				subprocess.call([appname, processId, "[]", '[-1]', config_file])  # [-1] for nodesAttacked and coupledNodes to tell cmp_dcsimsep.m to read from file    
			elif len(nodesAttacked) == 0 and len(coupledNodes) == 0:
				subprocess.call([appname, processId, "[]", "[]", config_file])  # [-1] for nodesAttacked and coupledNodes to tell cmp_dcsimsep.m to read from file                 	        
			else:
				subprocess.call([appname, processId, '[-1]', '[-1]', config_file])  # [-1] for nodesAttacked and coupledNodes to tell cmp_dcsimsep.m to read from file
		except:
			print "Bad call to compiled MATLAB. Appname: " + str(appname) + ", Process ID: " + str(processId) + ", config file: " + str(config_file)
			break

		try:
			with open(blackout_filename, 'rb') as f:
				reader = csv.DictReader(f)
				for item in reader:
					if int(item['blackout']) == 0:
						blackout = False
					else:
						blackout = True
						if verbose is True: print ">>>>>>>>> Blackout!!! <<<<<<<<<<<"
					gc_size = float(item['GC_size'])
					mw_lost = float(item['MW_lost'])
		except:
			blackout = True
			gc_size = 0.0
			mw_lost = 1.0
			print "CSV reader error for blackout file: " + str(blackout_filename) + " in run: " + str(run) + " at: " + str(p)

		if outputRemovedNodesToFile is True:
			write_removed_nodes_to_file(processId,p)

		if blackout is False:
			y.append([p,(1,gc_size,mw_lost)])
		else:
			y.append([p,(0,gc_size,mw_lost)])
	# print "y is " + str(y)
	print "Run time was " + str(time.time() - runstart) + " seconds"
	return y


def makeComms(networkB,copy):
	if randomRewireProb == -1:
		degree_sequence=sorted(nx.degree(networkB).values(),reverse=True)
		networkA=nx.configuration_model(degree_sequence)
	else:
		if copy is True:
			networkA=networkB.copy()
		else:
			networkA = networkB
		nodes = []
		targets = []
		for i,j in networkA.edges():
			nodes.append(i)
			targets.append(j)

		# rewire edges from each node, adapted from NetworkX W/S graph generator
		# http://networkx.github.io/documentation/latest/_modules/networkx/generators/random_graphs.html#watts_strogatz_graph
		# no self loops or multiple edges allowed
		for u,v in networkA.edges(): 
			if random.random() < randomRewireProb:
				w = random.choice(nodes)
				# Enforce no self-loops or multiple edges
				while w == u or networkA.has_edge(u, w): 
					w = random.choice(nodes)
				# print "node: " + str(u) + ", target: " + str(v)
				networkA.remove_edge(u,v)  
				networkA.add_edge(u,w)
	if copy is True:
		mapping=dict(zip(networkA.nodes(),range(1,2384)))  # 2384 for Polish, 4942 for western. relabel the nodes to start at 1 like networkA
		networkA=nx.relabel_nodes(networkA,mapping)
	return networkA


def getNodesFromFile(run, p_point, networkA):
	print "r_num: " + str(r_num)
	random.seed(r_num)  # Set the seed to the run number to get consistent random draws each time that run is done
	nodesAttacked = []
	file_name = relpath + 'data/node-removals/bus_outage_' + str(run) + '.csv.bz2'
	p_calc = numOutagesInFile - inverseStepSize*(p_point-minOutagePoint)
	p_column = int(round(p_calc))  # 50-10000*(0.995-0.75)/50 = 1, round first otherwise you can get unexpected results
	# print "p_column " + str(p_column) + " from file: " + file_name
	if p_column > numOutagesInFile or p_column == 0  or n != 2383:
		if verbose is True: print "No outages found for p_point " + str(p_point) + " at column " + str(p_column) + " creating outage instead"
		randomRemovalFraction=1-p_point	 # Fraction of nodes to remove
		numNodesAttacked = int(math.floor(randomRemovalFraction * n))
		nodesAttacked = random.sample(networkA.nodes(),numNodesAttacked)
		if verbose is True: print "size nodesAttacked list " + str(len(nodesAttacked)) + " for run " + str(run) + ", p_point " + str(p_point) 
		return nodesAttacked
	try:
		f = bz2.BZ2File(file_name,mode='r')
		c = codecs.iterdecode(f, "utf-8")
		data = csv.DictReader(c)
		# with open(file_name, 'rb') as f:
		# 	data = csv.DictReader(f)
		for next_row in data:
			item = int(next_row[str(p_column)])
			if item != 0:
				nodesAttacked.append(item)
		if verbose is True: print "size nodesAttacked list " + str(len(nodesAttacked)) + " for run " + str(run) + ", p_point " + str(p_point) + ", p_column " + str(p_column)
	except Exception, e:
				print "Node outage read error: " + str(e)
				raise
	# print "nodesAttacked " + str(nodesAttacked) + " run " + str(run) + ", p_point " + str(p_point) 
	return nodesAttacked


def getCoupledNodesFromFile(run, q_point, networkA):
	'''
	q is defined in the opposite direction of p so at a q of 0.99 we want to return a list of 99% of 
	the nodes in a network. This is the reason for taking 1-q_point.
	'''
	random.seed(r_num)  # Set the seed to the run number to get consistent random draws each time that run is done
	q_point = 1-q_point  
	coupledNodes = []
	comm_file_name = relpath + 'data/coupled-nodes/coupled_node_' + str(run) + '.csv.bz2'
	p_calc = numOutagesInCouplingFile - inverseStepSizeForCoupling*(q_point-minCouplingPoint)
	p_column = int(round(p_calc)) - 1  # 100-100*(0.99-0.01) - 1 = 1, round first otherwise you can get unexpected results
	# print "p_column " + str(p_column) + " from file: " + file_name
	if p_column > numOutagesInCouplingFile or p_column == 0 or n != 2383:
		if verbose is True: print "No outages found for q_point " + str(1-q_point) + " at column " + str(p_column) + " for " + str(n) + " nodes, creating outage instead"
		randomRemovalFraction=1-q_point	 # Fraction of nodes to remove
		numcoupledNodes = int(math.floor(randomRemovalFraction * n))
		coupledNodes = random.sample(networkA.nodes(),numcoupledNodes)
		if verbose is True: print "size coupledNodes list " + str(len(coupledNodes)) + " for run " + str(run) + ", q_point " + str(1-q_point) 
		return coupledNodes
	try:
		f = bz2.BZ2File(comm_file_name,mode='r')
		c = codecs.iterdecode(f, "utf-8")
		comm_data = csv.DictReader(c)
		# with open(comm_file_name, 'rb') as f:
		# 	comm_data = csv.DictReader(f)
		for next_row in comm_data:
			item = int(next_row[str(p_column)])
			if item != 0:
				coupledNodes.append(item)
		if verbose is True: print "size coupledNodes list " + str(len(coupledNodes)) + " for run " + str(run) + ", q_point " + str(1-q_point) + ", p_column " + str(p_column)
	except Exception, e:
				print "Coupled node outage read error: " + str(e)
				raise
	# print "coupledNodes " + str(coupledNodes) + " run " + str(run) + ", q_point " + str(q_point) 
	return coupledNodes


def getNetworkFromFile(networkName):
	path = relpath + networkName
	try:
		network=nx.read_edgelist(path, nodetype=int)
		if 0 in network.nodes():
			max_nodes = len(network.nodes()) + 1
			mapping=dict(zip(network.nodes(),range(1,max_nodes)))  # renumber the nodes to start at 1 for MATLAB
			network=nx.relabel_nodes(network,mapping)
	except Exception, e:
		print "Error Unknown network type for " + networkName + " " + str(e)
		raise

	return network


def write_removed_nodes_to_file(processId,p):
	grid_status = []
	comm_status = []
	node_bus_list = []
	grid_status_filename = '/tmp/grid_status_' + str(processId) + '.csv'
	comm_status_filename = '/tmp/comm_status_' + str(processId) + '.csv'
	grid_status_output_filename = '../output/grid_status_' + str(r_num) + 'run_' + str(p) + 'p.csv'
	comm_status_output_filename = '../output/comm_status_' + str(r_num) + 'run_' + str(p) + 'p.csv'
	write_by_column = False

	if write_by_column is True:
		with open(grid_status_filename, 'rb') as f:
			reader = csv.DictReader(f)
			for item in reader:
				if int(item['status']) == 0:
					grid_status.append(int(item['bus']))

		with open(comm_status_filename, 'rb') as f:
			reader = csv.DictReader(f)
			for item in reader:
				if int(item['status']) == 0:
					comm_status.append(int(item['node']))

		with open(grid_status_output_filename, 'wb') as f:
			writer = csv.writer(f)
			writer.writerow([str(p)])
			for row in grid_status:
				writer.writerow([row])

		with open(comm_status_output_filename, 'wb') as f:
			writer = csv.writer(f)
			writer.writerow([str(p)])
			for row in comm_status:
				writer.writerow([row])
	else:
		with open(grid_status_filename, 'rb') as f:
			reader = csv.DictReader(f)
			for item in reader:
				node_bus_list.append(int(item['bus']))
				grid_status.append(int(item['status']))

		with open(comm_status_filename, 'rb') as f:
			reader = csv.DictReader(f)
			for item in reader:
				comm_status.append(int(item['status']))

		with open(grid_status_output_filename, 'wb') as f:
			writer = csv.writer(f)
			writer.writerow(node_bus_list)
			writer.writerow(grid_status)

		with open(comm_status_output_filename, 'wb') as f:
			writer = csv.writer(f)
			writer.writerow(node_bus_list)
			writer.writerow(comm_status)


def createNetworks(networkType):
	if networkFromFile is True:
		networkA = getNetworkFromFile(commNetworkLocation)
		networkB = getNetworkFromFile(powerNetworkLocation)
	else:
		sys.stderr.write("networkFromFile cannot be false. You must use pre-created networks when running with the CFS.")
		sys.exit(-1)

	# Order the nodes of the networks randomly
	if shuffleNetworks is True:
		randomlist=range(1,n+1)  # node must names start at 1 for MATLAB, remapping to start at 1 done above
		random.shuffle(randomlist)
		mapping=dict(zip(networkA.nodes(),randomlist))
		networkA=nx.relabel_nodes(networkA,mapping)
		random.shuffle(randomlist)
		mapping=dict(zip(networkB.nodes(),randomlist))
		networkB=nx.relabel_nodes(networkB,mapping)
		del mapping
		del randomlist

	return [networkA,networkB]


def logResult(result):
	allY.extend(result)


def writeOutput(x, p_inf, gc_size, mw_lost, run):
	if logAllPValues is False and run != runs:
		return

	# transform the data to enable writing to csv
	output = []
	output.append(x)
	output.append(p_inf)
	output.append(gc_size)
	output.append(mw_lost)
	output = zip(*output)
	output.sort(key=itemgetter(0))
	numRun = ''
	if run == runs:
		numRun = 'run_final_'
	else:
		numRun = 'run_' + str(run) + '_' 

	# write the output to file
	filename = str(r_num) + "_run_" + networkLabel + str(n) + 'nodes_' + numRun + 'rewire' + str(randomRewireProb) + '_GCThresh' + str(gCThreshold) + '_GridGCThresh' + str(gridGCThreshold) + '.csv'
	path = relpath + "output/" + filename
	completeName = os.path.abspath(path)
	with open(completeName, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(['p','Pinf','GC_size','MW_lost'])
		for row in output:
			writer.writerow(row)


def cleanup(processId):
	# print "going to cleanup processId: " + str(processId)
	for f in glob.glob("/tmp/*" + processId +".csv"):  # tmp files no longer needed so delete them
		# print "removing: " + str(f)
		os.remove(f)


def main():
	networks = []
	global allY  # Tell Python interpreter that allY is a global variable
	processId = str(os.getpid())

	if batchMode is False:  # Send each run to as many processors as there are to run in parallel
		manager = mlt.Manager()
		if generateEachRun is False:
			# allow the generated networks to be shared by all processes/threads
			networks = manager.list(createNetworks(networkType))
		pool = mlt.Pool()
		for i in range(0,runs,1):
			pool.apply_async(attackNetwork, args=(i, networks, processId, ), callback=logResult)

		pool.close()
		pool.join()
	else:
		networks = createNetworks(networkType)
		allY = attackNetwork(0, networks, processId)

	d = defaultdict(list)
	print "allY: " + str(allY)
	for key, value in allY:
		d[key].append(value)
	# average = [float(sum(value)) / len(value) for key, value in d.items()]
	p_inf = [value[0][0] for key, value in d.items()]
	gc_size = [value[0][1] for key, value in d.items()]
	mw_lost = [value[0][2] for key, value in d.items()]
	x = [key for key, value in d.items()]

	print "x is: " + str(x)
	print "p_inf is: " + str(p_inf)
	print "gc_size is: " + str(gc_size)
	print "mw_lost is: " + str(mw_lost)

	writeOutput(x,p_inf,gc_size,mw_lost,runs)

	cleanup(processId)

	if showPlot is True:
		#plot the result
		plt.plot(x,p_inf,'bo')  # 'rx' for red x 'g+' for green + marker
		# show the plot
		plt.show()	

if __name__ == '__main__':
	try:
		main()
	except Exception as e:
		print e
		raise
