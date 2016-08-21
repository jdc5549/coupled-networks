#!/usr/bin/env python
"""
Coupled network cascading failure simulator.

Copyright 2016 The MITRE Corporation and The University of Vermont

Run like:
python couplednetworks.py -1 -1 -1 config.json 1

The program takes four required arguments and one optional argument (```r_num```):
    *   ```mpid``` - MATLAB process id
    *   ```cfs_iter``` - Iteration number from MATLAB
    *   ```percent_removed``` - Percent of nodes removed for this model run.
    *   ```config``` - Name and location of the configuration file.
    *   ```r_num``` - Optional - Replicate number to use when running from an HPC. ```r_num``` determines which outage file to use.

All arguments, aside from config.json, can be set to -1 if running without the cascading failure simulator (CFS, MATLAB model)
"""

import argparse
import bz2
import codecs
import csv
import json
import logging
import math
import multiprocessing as mlt
import os
import random
import sys
import time

from collections import defaultdict, deque
from operator import itemgetter

import networkx as nx
from networkx.readwrite import json_graph

if sys.version_info < (2, 6):
    raise "must use Python 2.7 or greater"
    sys.exit(-1)

"""
Grab the command line arguments.
"""
parser = argparse.ArgumentParser(description='Coupled Network Model')
parser.add_argument('mpid', metavar='MATLAB_PID', type=int, nargs=1, help='process ID from MATLAB, ' +
    'if running without MATLAB use -1 as a command line argument')
parser.add_argument('cfs_iter', metavar='CFS_iteration', type=int, nargs=1, help='iteration from MATLAB, ' +
    'if running without MATLAB use -1 as a command line argument')
parser.add_argument('percent_removed', metavar='percent_removed', type=float, nargs=1, help='Percent of nodes removed ' +
    'if running without MATLAB use -1 as a command line argument')
parser.add_argument('config_name', metavar='config_name', type=str, nargs=1, help='Name and location of config.json file')
parser.add_argument('r_num', default=-1, metavar='replicate_number', type=int, nargs='?', help='Optional replicate number if running in batch mode')
args = parser.parse_args()
mpid = args.mpid[0]
cfs_iter = args.cfs_iter[0]
percent_removed = args.percent_removed[0]
config_name = args.config_name[0]
r_num = args.r_num

"""
Read from the configuration file.
"""
NUM_PROCS = mlt.cpu_count()
config = json.load(open(config_name))
k = config['k']  # average degree, <k>, must be a float, 4.0 is what Buldrev used
n = config['n']  # number of nodes, 50000 is what Buldyrev used, 2383 for Polish grid
runs = config['runs']  # number of runs to make
p_values = config['p_values']    # how many increments of p
p_min = config['p_min']
p_max = config['p_max']
start_with_comms = config['start_with_comms']
# the fraction of the number of components in the giant component under which failure is declared
# for replication this should be 0 which gets changed to a threshold of 1 node
gc_threshold = config['gc_threshold']
grid_gc_threshold = config['grid_gc_threshold']  # the fraction of the number of components in the giant component under which blackout is declared
output_gc_size = config['output_gc_size']

outages_from_file = config['outages_from_file']
min_outage_point = config['min_outage_point']
num_outages_in_file = config['num_outages_in_file']
coupling_from_file = config['coupling_from_file']
num_couplings_in_file = config['num_couplings_in_file']
min_coupling_point = config['min_coupling_point']
deg_of_coupling = config['deg_of_coupling']  # Fraction of nodes that are coupled between networks. Used if coupling_from_file is false.
inverse_step_size = config['inverse_step_size']
inverse_step_size_for_coupling = config['inverse_step_size_for_coupling']

targeted = config['targeted']  # If this is true the highest degree nodes are removed first
output_removed_nodes_to_DB = config['output_removed_nodes_to_DB']  # For visualization, this writes a csv file with all the node removals at each p-value
output_removed_nodes = config['output_removed_nodes']
# Output files at each p-value or just a file at the end. This should probably be false...
# unless you're concerned that you may need to restart a simulation mid run
log_all_p_values = config['log_all_p_values']
write_networks_out = config['write_networks_out']  # writes a json formatted copy of the networks to file for each run. Caution, will produce lots of files
# if random_rewire_prob is -1, base the comms network on a configuration model...
# otherwise the comms will be randomly rewired with this probability
# For scale free replication lambda/gamma = ~3 when random_rewire_prob = 0.16...
# and ~2.7 when random_rewire_prob = 0.31 and ~2.3 with rewireProb = 0.57
random_rewire_prob = config['random_rewire_prob']  # Only used if network_from_file is false
network_type = config['network_type']    # If generating neworks this must be either 'SF', 'RR', 'ER', 'CFS-SW', and others TBD in create_networks(). Otherwise it can be any string wanted in the output file name.
show_plot = config['show_plot']  # Create a plot of p vs. p_inf
# the fraction of the number of components in the giant component under which failure is declared
# for replication this should be 0 which gets changed to a threshold of 1 node
gc_threshold = config['gc_threshold']
grid_gc_threshold = config['grid_gc_threshold']  # the fraction of the number of components in the giant component under which blackout is declared
shuffle_networks = config['shuffle_networks']  # randomly renumber the nodes on the networks, for replication this should be True
generate_each_run = config['generate_each_run']  # generate a new network for each run, True, or keep the same for all, False
hpc = config['hpc']
find_scaling_exponent = config['find_scaling_exponent']
output_result_to_DB = config['output_result_to_DB']
batch_mode = config['batch_mode']
betweenness = config['betweenness']
log_level = config['log_level']
verbose = config['verbose']

network_from_file = config['network_from_file']
comm_network_location = config['comm_network_location']
power_network_location = config['power_network_location']

# Options for running with the physics based cascading failure simulator (CFS).
real = config['real']  # set to true when using CFS and not just topo networks
comm_model = config['comm_model']  # Used by MATLAB to find the path to this file.
relpath = comm_model.split('src/couplednetworks.py')[0]


"""
Perform other setup related tasks
"""
# Set up logging.
logger = logging.getLogger('couplednetworks')
logger.setLevel(log_level)  # 10 is debug, 20 is info, 30 is warning, 40 is error
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

ep = k / (n - 1)  # probability for edge creation
# print "Probability for edge creation: " + str(ep)
nodes = range(1, n + 1, 1)
p_range = p_max - p_min
home = os.path.expanduser("~")
threshold = gc_threshold * n
if int(threshold) == 0:
    threshold = 1
# if reading networks from file critical node will be set to the max betweenness value
# pre-calculed in config.json for that network
critical_node = -1

network_filename = '/tmp/comms_net_' + str(mpid) + '.edgelist'

if hpc == "HIVE":
    r_num = r_num + 1  # incremented by 1 to use with condor process variable which starts at 0

if r_num is -1 and real is False and batch_mode is True:
    sys.stderr.write("Trying to run in batch mode but run number not provided as an argument.")
    sys.exit(-1)
# Change the internal state of the random generator for each run
if batch_mode is False:
    newstate = random.randint(2, n * 2)
    random.jumpahead(newstate)
else:
    random.seed(r_num)
    newstate = r_num


def config_type(shuffle_networks, generate_each_run):
    """Create a configuration type indicator for the output file name."""
    gen = "0"
    shuffle = "0"
    if generate_each_run is True:
        gen = "1"
    if shuffle_networks is True:
        shuffle = "1"
    return gen + shuffle


network_label = (network_type + '{nodes}nodes_{rw}rw_{thresh}thresh_{q}q_{sruns}Avg_GCThresh{gc_threshold}_GridGCThresh{grid_gc_threshold}'.format(
    nodes=n, sruns=runs, rw=random_rewire_prob, thresh=threshold, q=deg_of_coupling, gc_threshold=gc_threshold, grid_gc_threshold=grid_gc_threshold) +
    '_Config_%s' % config_type(shuffle_networks, generate_each_run) + "_")  # this gets put in the file names

if verbose is True:
    logger.info("Network type: " + network_label + ", comms threshold: " + str(gc_threshold) +
    ", grid threshold: " + str(grid_gc_threshold) + ", coupling factor: " + str(deg_of_coupling))
    # print "Number of processors " + str(NUM_PROCS)

# Only import matplotlib if you want to generate plots. It uses about 20MB RAM.
if show_plot is True:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise "matplotlib not found, can't show plots"
        sys.exit(-1)

# Optional modules
if output_result_to_DB is True or output_removed_nodes_to_DB is True or find_scaling_exponent is True:
    try:
        import powerlaw
        from pymongo import Connection
        from pymongo.errors import ConnectionFailure
    except ImportError:
        print "Missing powerlaw and/or pymongo modules."

# x holds the p values used
x = []
# allY holds whether the giant mutually connected component exists for each p value as:
# [p,exists?] e.g. [0.01, 1] for p=0.01 and the GMCC existing
allY = []


def get_coupled_nodes_from_file(run, q_point, pid):
    """
    Read in the coupled nodes file that matches run and q_point.

    q is defined in the opposite direction of p so at a q of 0.99 we want to return a list of 99% of
    the nodes in a network. This is the reason for taking 1-q_point.
    """
    coupled_nodes = []
    if pid == -1:
        random.seed(r_num)  # Set the seed to the run number to get consistent random draws each time that run is done
        q_point = 1 - q_point
        comm_file_name = relpath + 'data/coupled-nodes/coupled_node_' + str(run) + '.csv.bz2'
        p_calc = num_couplings_in_file - inverse_step_size_for_coupling * (q_point - min_coupling_point)
        p_column = int(round(p_calc)) - 1  # 100-100*(0.99-0.01) - 1 = 1, round first otherwise you can get unexpected results
        if p_column > num_couplings_in_file or p_column == 100 or n != 2383 or p_column == 0:
            random_removal_fraction = 1 - q_point    # Fraction of nodes to remove
            num_coupled_nodes = int(math.floor(random_removal_fraction * n))
            logger.warning("No coupling found for q_point " + str(q_point) + " at column " + str(p_column) +
            " for " + str(n) + " nodes, creating coupling instead of " + str(num_coupled_nodes) + " coupled nodes")
            coupled_nodes = random.sample(nodes, num_coupled_nodes)
            # logger.info("size coupled_nodes list " + str(len(coupled_nodes)) + " for run " + str(run) + ", q_point " + str(q_point))
            return coupled_nodes
        try:
            f = bz2.BZ2File(comm_file_name, mode='r')
            c = codecs.iterdecode(f, "utf-8")
            comm_data = csv.DictReader(c)
            # with open(comm_file_name, 'rb') as f:
            #   comm_data = csv.DictReader(f)
            for next_row in comm_data:
                item = int(next_row[str(p_column)])
                if item != 0:
                    coupled_nodes.append(item)
            logger.debug("size coupled_nodes list " + str(len(coupled_nodes)) + " for run " + str(run) + ", q_point " + str(1 - q_point) + ", p_column " + str(p_column))
        except Exception, e:
            print "Coupled node outage read error: " + str(e)
            raise
        # print "coupled_nodes " + str(coupled_nodes) + " run " + str(run) + ", q_point " + str(q_point)
    else:  # getting called by something else so read the coupled node file created by the other process
        coupled_node_filename = '/tmp/coupled_nodes_' + str(pid) + '.csv'
        try:
            with open(coupled_node_filename, 'rb') as f:
                coupled_data = csv.DictReader(f)
                for next_row in coupled_data:
                    coupled_nodes.append(int(next_row['node']))
                logger.debug("size coupled_nodes list " + str(len(coupled_nodes)) + " for run " + str(run) + ", q_point " + str(1 - q_point))
        except Exception, e:
                    print "Coupled node read error: " + str(e)
                    raise
    return coupled_nodes

coupled_nodes = []
if coupling_from_file is True and real is False and batch_mode is True:
    coupled_nodes = get_coupled_nodes_from_file(r_num, deg_of_coupling, -1)
elif coupling_from_file is True and real is False and batch_mode is False:
    coupled_nodes = get_coupled_nodes_from_file(r_num + 1, deg_of_coupling, -1)
elif coupling_from_file is False and real is False:
    number_coupled = int(math.floor(deg_of_coupling * n))
    coupled_nodes = random.sample(nodes, number_coupled)  # nodes that are connected between networks
else:
    coupled_nodes = get_coupled_nodes_from_file(-1, -1, mpid)  # if called by CFS read coupled nodes from file


def remove_links(network_a, network_b, swap_networks, iteration):
    """
    Remove links in network_b per the process given in Buldyrev 2010, Catastrophic cascade of failures in interdependent networks.

    "Each node in network A depends on one and only one node in network B, and
    vice versa. One node from network A is removed ('attack'). b, Stage 1: a
    dependent node in network B is also eliminated and network A breaks into
    three a1-clusters, namely a11, a12 and a13. c, Stage 2: B-links that link
    sets of B-nodes connected to separate a1-clusters are eliminated and network B
    breaks into four b2-clusters, namely b21, b22, b23 and b24. d, Stage 3:
    A-links that link sets of A-nodes connected to separate b2-clusters are
    eliminated and network A breaks into four a3- clusters, namely a31, a32, a33
    and a34. These coincide with the clusters b21, b22, b23 and b24, and no
    further link elimination and network breaking occurs. Therefore, each
    connected b2-cluster/a3-cluster pair is a mutually connected cluster and the
    clusters b24 and a34, which are the largest among them, constitute the giant
    mutually connected component."
    """
    # Get the connected components of each network.
    gmcca = sorted(nx.connected_components(network_a), key=len, reverse=True)
    gmccb = sorted(nx.connected_components(network_b), key=len, reverse=True)

    # Find which network has the fewest number of connected component subgraphs
    num_subnets = len(gmcca) if len(gmcca) < len(gmccb) else len(gmccb)

    identical_subnets = 0
    swap_again = swap_networks
    nodes_to_remove = []

    if num_subnets == 0:
        swap_again = 0
        return [swap_again, nodes_to_remove]
    elif len(gmcca[0]) < threshold or len(gmccb[0]) < threshold:  # check against threshold
        swap_again = 0
        return [swap_again, nodes_to_remove]

    num_subnets = 1  # Only look at the giant component, remove this to look at all subgraphs
    for j in range(0, num_subnets, 1):
        # Make the nodes sets so you can do .difference() on them
        set_a = set(gmcca[j])
        set_b = set(gmccb[j])
        # Find the elements in B but not in A
        out_nodes_b = set_b.difference(set_a)
        if output_removed_nodes is True:
            if len(out_nodes_b) > 0:
                for node in out_nodes_b:
                    if node in coupled_nodes:
                        nodes_to_remove.extend([node])
                logger.debug("Subnet " + str(j + 1) + " of " + str(num_subnets) +
                ". Total nodes to remove: " + str(len(nodes_to_remove)))
        logger.debug("Number elements in B but not in A " + str(len(out_nodes_b)))
        if len(out_nodes_b) == 0:
            logger.debug("No differences in subnet " + str(j + 1))
            identical_subnets += 1
            # Check to see if the nodes in each subgraph are the same
            # If they are then no more link removal is needed and we can return
            if identical_subnets == num_subnets:
                swap_again = 0
                logger.debug("All the same" + ", iteration: " + str(iteration) + ", gmcca size " +
                    str(len(gmcca[0])) + " gmccb size " + str(len(gmccb[0])))
                return [swap_again, nodes_to_remove]
        for k in range(0, len(out_nodes_b), 1):
            # For each node that's not in this MCC subnet of A and B networks
            node_in_subgraph = list(out_nodes_b)[k]
            logger.debug("Node that's different in subgraph k:" + str(k) + " is: " +
            str(node_in_subgraph) + ", iteration: " + str(iteration))
            # Find their neighbors
            node_in_subgraph_neighbors = nx.neighbors(network_b, node_in_subgraph)
            for m in range(0, len(node_in_subgraph_neighbors), 1):
                if node_in_subgraph in coupled_nodes:  # check that this node is coupled
                    # Remove all the edges going to those neighbor nodes
                    network_b.remove_edges_from([(node_in_subgraph, node_in_subgraph_neighbors[m])])
                    logger.debug("Removing edge " + str(node_in_subgraph) + "," + str(node_in_subgraph_neighbors[m]))
        # At the last subnet increment swap_again so that the networks get swapped next time
        if j == num_subnets - 1:
            swap_again += 1
    logger.debug("Original number of nodes to remove: " + str(len(nodes_to_remove)))
    solitary_nodes = [n for n, d in network_b.degree_iter() if d == 0]  # nodes with 0 degree
    nodes_to_remove.extend(solitary_nodes)
    if logger.isEnabledFor(logging.DEBUG):
        nodes_to_remove = set(nodes_to_remove)
        nodes_to_remove = list(nodes_to_remove)
        print "Final number of nodes to remove: " + str(len(nodes_to_remove))
        print "swap_again " + str(swap_again) + " netA size " + str(len(network_a.nodes())) + " netB size " + str(len(network_b.nodes()))
        print "gmcca size " + str(len(gmcca)) + " gmccb size " + str(len(gmccb))
    return [swap_again, nodes_to_remove]


def attack_network(run, networks):
    """Create the networks and run the attack and cascade sequence on them."""
    logger.debug("\t\tRun number " + str(run) + " of " + str(runs) + ", iteration: " + str(cfs_iter))

    runstart = time.time()
    y = []

    # Create the networks
    if generate_each_run is True:
        [network_a, network_b] = create_networks(network_type)
    else:
        network_a = networks[0]
        network_b = networks[1]

    dbh = -1  # initialize the database handle to something in case it's *not* used

    if write_networks_out == "edgelist" or write_networks_out == "json":
        write_networks(run, network_a, network_b, write_networks_out)

    # Setup the database if outputing to it
    if output_removed_nodes_to_DB is True or output_result_to_DB is True:
        try:  # Connect to MongoDB
            c = Connection(host="localhost", port=27017)
        except ConnectionFailure, e:
            sys.stderr.write("Could not connect to MongoDB: %s" % e)
            # sys.exit(1)
        dbh = c["runs"]  # sets the db to "runs"
        assert dbh.connection == c

    check_for_failure(network_a, network_b, dbh, run, y)

    if real is False:  # Capture the result if not running from MATLAB
        d = defaultdict(list)
        p_half = defaultdict(int)
        for key, value in y:
            d[key].append(value)
            if value > 0.5:
                p_half[key] += 1
            else:
                p_half[key] += 0
        average = [float(sum(value)) / len(value) for key, value in d.items()]
        average_p_half = [float(value) / runs for key, value in p_half.items()]
        x = [key for key, value in d.items()]
        write_output(x, average, average_p_half, run)

    logger.debug("Comms run time was " + str(time.time() - runstart) + " seconds")
    return y


def check_for_failure(network_a, network_b, dbh, run, y):
    global nodes
    global critical_node
    global coupled_nodes

    if real is True:  # real == True enables the use of the external DC power flow simulator.
        network_a_copy = network_a.copy()  # Comm network
        network_b_copy = network_b.copy()  # Power network
        num_nodes_attacked = int(math.floor(percent_removed * n))
        busessep = []  # bus separations read from MATLAB
        nodes_attacked = []  # if start_with_comms is true then this holds the initial contingency otherwise it gets set to busessep
        coupled_busessep = []  # bus separations attached to coupled nodes
        swap_networks = 1  # Determines which network to start with in remove_links
        nodes = range(1, n + 1, 1)  # for interfacing with MATLAB start at 1
        comm_status_filename = '/tmp/comm_status_' + str(mpid) + '.csv'
        grid_status_filename = '/tmp/grid_status_' + str(mpid) + '.csv'
        # grid_status_filename = '/tmp/grid_status_test.csv'
        # print "\t !!!!!!! Using test grid status file !!!!!!!!!!"

        # Determine which nodes to remove
        if targeted is False and outages_from_file is False:
            nodes_attacked = random.sample(network_a.nodes(), num_nodes_attacked)
        elif targeted is True and outages_from_file is False:
            nodes_attacked = targeted_nodes(network_a, num_nodes_attacked)
        elif targeted is False and outages_from_file is True and start_with_comms is True:
            # this is only used if start_with_comms is true
            nodes_attacked = get_nodes_from_file(run + 1, 1 - percent_removed, network_a)  # TODO fix this
        elif targeted is False and outages_from_file is True and start_with_comms is False:
            nodes_attacked = []  # get nodesAttached from the grid status file
        else:
            sys.stderr.write("Unknown configuration of inputs: targeted and outages_from_file")
            sys.exit(-1)

        # Remove nodes from the networks
        if start_with_comms is True:  # TODO To start with comms in "real" mode the interface needs to be worked out with DCSIMSEP
            logger.debug("Starting with comms, percent_removed " + str(percent_removed) + ", num_nodes_attacked " + str(num_nodes_attacked))
            # remove nodes from the comms network
            network_a_copy.remove_nodes_from(nodes_attacked)
            # network_b_copy is the copy of the grid that's used in MATLAB, remove the bus separations on it for comparing with network_a_copy
            # network_b_copy.remove_nodes_from(nodes_attacked)
        else:
            # read grid status and remove nodes accordingly
            try:
                with open(grid_status_filename, 'rb') as f:
                    try:
                        reader = csv.DictReader(f)
                        for item in reader:
                            try:
                                if int(item['status']) == 0:
                                    bus = int(item['bus'])
                                    busessep.append(bus)
                                    if bus in coupled_nodes:
                                        coupled_busessep.append(bus)  # nodes that will be removed from comms
                            except ValueError:
                                # print "No bus separations"
                                pass
                    except:
                        print "CSV reader error. Check headers in file " + grid_status_filename + " and check for Unix line endings in that file."
            except:
                print "*************** -> Missing grid status file <- ****************"
            # if os.path.isfile(grid_status_filename):  # this is now deleted by the calling process. tmp file no longer needed so delete it
            #   os.remove(grid_status_filename)

            # Remove the nodes on network_b, power grid, to match the state it was in leaving MATLAB
            network_b_copy.remove_nodes_from(busessep)
            # Remove the nodes on network_a, comms, that go to the bus separations on network_b, power
            # and are also coupled
            network_a_copy.remove_nodes_from(coupled_busessep)
            nodes_attacked = busessep
            if logger.isEnabledFor(logging.DEBUG):
                num_nodes_attacked = len(busessep)
                num_coupled_nodes_attacked = len(coupled_busessep)
                print ">>>> Starting with grid, number of bus separations: " + str(num_nodes_attacked) + ". Number coupled separations: " + str(num_coupled_nodes_attacked)
        logger.debug("Subgraphs in network_b: " + str(len(sorted(nx.connected_component_subgraphs(network_b_copy), key=len, reverse=True))) +
            ", subgraphs in network_a: " + str(len(sorted(nx.connected_component_subgraphs(network_a_copy), key=len, reverse=True))))
        logger.debug("***Total nodes out PRE comms removal: " + str(n - len(network_a_copy.nodes())))
        coupled_losses = set(nodes_attacked)
        logger.info("In iteration " + str(cfs_iter) + ", nodes lost: " + str(len(coupled_losses)))

        # Remove the links in the comms network, network A, that no longer connect between networks
        result = remove_links(network_b_copy, network_a_copy, swap_networks, cfs_iter)

        swap_networks = result[0]
        if len(result[1]) == 0:
            if verbose is True:
                print "\t ######## No nodes removed #########"
            pass
        else:
            nodes_attacked.extend(result[1])
            nodes_attacked = set(nodes_attacked)  # remove duplicates
            nodes_attacked = list(nodes_attacked)  # convert back to a list
            coupled_busessep.extend(result[1])
            coupled_busessep = set(coupled_busessep)  # remove duplicates
            coupled_busessep = list(coupled_busessep)  # convert back to a list
            # print ("Additional number of nodes out: " + str(len(result[1])) +
            #   ", total nodes out: " + str(len(nodes_attacked)))
        del result  # free up memory

        if logger.isEnabledFor(logging.DEBUG):
            network_a_copy.remove_nodes_from(coupled_busessep)  # TODO figure out where coupling is managed for power/comms
            nodes_out_post = n - len(network_a_copy.nodes())
            print ">>>Total nodes out POST comms removal: " + str(nodes_out_post)

        with open(comm_status_filename, 'w') as f:
            try:
                writer = csv.writer(f)
                writer.writerow(['node', 'status'])  # Header
                status = 0
                if critical_node != -1:
                    nodes_attacked = critical_node_connection(network_a_copy, coupled_busessep)
                for item in nodes:
                    if item not in nodes_attacked:
                        status = 1
                    else:
                        status = 0
                    writer.writerow([item, status])
            except:
                print "CSV writer error"
    else:  # If real is false then only simulate topological cascades.
        for i in range(0, p_values + 1, 1):
            # innerrunstart = time.time()
            # copy the networks so that all node removals are done on the same network layout
            # new network layouts will be generated for each run if generate_each_run is true
            network_a_copy = network_a.copy()
            network_b_copy = network_b.copy()
            coupled_nodes_attacked = []

            p = i / float(p_values)
            p = p / (1 / float(p_range)) + p_min
            random_removal_fraction = 1 - p  # Fraction of nodes to remove
            num_nodes_attacked = int(math.floor(random_removal_fraction * n))
            # print "p " + str(p) + ", num_nodes_attacked " + str(num_nodes_attacked)

            if targeted is False and outages_from_file is False:
                nodes_attacked = random.sample(network_a.nodes(), num_nodes_attacked)
            elif targeted is True and outages_from_file is False:
                nodes_attacked = targeted_nodes(network_a, num_nodes_attacked)
            elif targeted is False and outages_from_file is True and batch_mode is True:
                nodes_attacked = get_nodes_from_file(run, p, network_a_copy)
            elif targeted is False and outages_from_file is True and batch_mode is False:
                nodes_attacked = get_nodes_from_file(run + 1, p, network_a_copy)
            else:
                sys.stderr.write("Unknown configuration of inputs: targeted and outages_from_file")
                sys.exit(-1)

            # Track the initial nodes attacked if output_result_to_DB is true.
            initial_nodes_attacked = []
            # if output_result_to_DB is True or verbose is True:
            initial_nodes_attacked.extend(nodes_attacked)

            for node in nodes_attacked:
                    if node in coupled_nodes:
                        coupled_nodes_attacked.extend([node])

            # If the critical node has been attacked then all nodes in network_a can't communicate
            if critical_node in coupled_nodes_attacked and start_with_comms is True:
                network_a_copy.remove_nodes_from(nodes)
                if output_removed_nodes_to_DB is True:
                    write_removed_nodes(run, p, 1, nodes_attacked, dbh, "A", network_a_copy)
            else:  # Proceed normally
                # print "<><><><>Starting, iteration " + str(i) + ", " + str(len(nodes_attacked)) + " nodes removed<><><><>"
                # remove the attacked nodes from both networks
                if start_with_comms is True:
                    network_a_copy.remove_nodes_from(nodes_attacked)
                    network_b_copy.remove_nodes_from(coupled_nodes_attacked)
                else:
                    network_a_copy.remove_nodes_from(coupled_nodes_attacked)
                    network_b_copy.remove_nodes_from(nodes_attacked)

                logger.debug("\t>>>> Number of nodes attacked: " + str(num_nodes_attacked) +
                    ", fraction: " + str(random_removal_fraction) + ". Of these " +
                    str(len(coupled_nodes_attacked)) + " are coupled nodes")
                logger.debug("Starting number of edges after attack for i: " + str(i) +
                    " netA: " + str(len(network_a_copy.edges())) + " netB: " + str(len(network_b_copy.edges())))
                # Track whether the subnets are the same, if they're not we need to
                # swap them and run remove_links again.
                swap_networks = 1
                if output_removed_nodes_to_DB is True:
                    write_removed_nodes(run, p, swap_networks, nodes_attacked, dbh, "A", network_a_copy)
                    # print "Writing removed nodes: " + str(output_removed_nodes_to_DB)
                edge_comparison = deque()
                while swap_networks != 0 and len(network_a_copy.nodes()) > 0:
                    if swap_networks % 2 == 1:
                        result = remove_links(network_a_copy, network_b_copy, swap_networks, i)
                        swap_networks = result[0]
                        logger.debug("Number of edges after attack for i: " + str(i) +
                            " at swap: " + str(swap_networks) + ", netA: " + str(len(network_a_copy.edges())) +
                            ", netB: " + str(len(network_b_copy.edges())))
                        if swap_networks != 0 and output_removed_nodes_to_DB is True:
                            new_nodes_attacked = list(set(result[1]) - set(nodes_attacked))  # get only the new nodes removed
                            write_removed_nodes(run, p, swap_networks, new_nodes_attacked, dbh, "A", network_a_copy)
                        nodes_attacked.extend(result[1])
                        nodes_attacked = set(nodes_attacked)  # remove duplicates
                        nodes_attacked = list(nodes_attacked)  # convert back to a list
                        coupled_nodes_attacked.extend(result[1])
                        coupled_nodes_attacked = set(coupled_nodes_attacked)  # remove duplicates
                        coupled_nodes_attacked = list(coupled_nodes_attacked)  # convert back to a list
                        logger.debug("Number of nodes lost at swap " + str(swap_networks) + ": " + str(len(nodes_attacked)))
                    else:
                        result = remove_links(network_b_copy, network_a_copy, swap_networks, i)
                        swap_networks = result[0]
                        logger.debug("Number of edges after attack for i: " + str(i) +
                            " at swap: " + str(swap_networks) + ", netA: " + str(len(network_a_copy.edges())) +
                            ", netB: " + str(len(network_b_copy.edges())))
                        if swap_networks != 0 and output_removed_nodes_to_DB is True:
                            new_nodes_attacked = list(set(result[1]) - set(nodes_attacked))  # get only the new nodes removed
                            write_removed_nodes(run, p, swap_networks, new_nodes_attacked, dbh, "B", network_b_copy)
                        nodes_attacked.extend(result[1])
                        nodes_attacked = set(nodes_attacked)  # remove duplicates
                        nodes_attacked = list(nodes_attacked)  # convert back to a list
                        coupled_nodes_attacked.extend(result[1])
                        coupled_nodes_attacked = set(coupled_nodes_attacked)  # remove duplicates
                        coupled_nodes_attacked = list(coupled_nodes_attacked)  # convert back to a list
                        logger.debug("Number of nodes lost at swap " + str(swap_networks) + ": " + str(len(nodes_attacked)))

                    num_net_a_edges = len(network_a_copy.edges())
                    num_net_b_edges = len(network_b_copy.edges())
                    edge_comparison.appendleft((num_net_a_edges, num_net_b_edges))
                    if (len(edge_comparison) > 1):
                        if edge_comparison[0] == edge_comparison[1]:
                            logger.debug("No more edges to remove, breaking loop")
                            break
                if start_with_comms is True:  # Remove nodes_attacked from network_a, coupled_nodes_attacked from network_b
                    # if using a critical node check for connection to it
                    if critical_node is not -1:
                        nodes_attacked = critical_node_connection(network_a_copy, nodes_attacked)
                    network_a_copy.remove_nodes_from(nodes_attacked)
                    network_b_copy.remove_nodes_from(coupled_nodes_attacked)
                else:  # Remove coupled_nodes_attacked from network_a, nodes_attacked from network_b
                    if critical_node is not -1:
                        coupled_nodes_attacked = critical_node_connection(network_a_copy, coupled_nodes_attacked)
                    network_a_copy.remove_nodes_from(coupled_nodes_attacked)
                    network_b_copy.remove_nodes_from(nodes_attacked)

            coupled_losses = set(nodes_attacked) - set(initial_nodes_attacked)
            logger.info("At p: " + str(p) + ", initial nodes lost on network_a: " + str(len(initial_nodes_attacked)) + ", nodes lost after link removal and coupling on network_b: " + str(len(coupled_nodes_attacked)) +
                ", additional nodes lost due to cascades on both networks: " + str(len(coupled_losses)))

            logger.debug("Final number of edges after attack for i: " + str(i) +
                ", netA: " + str(len(network_a_copy.edges())) + ", netB: " + str(len(network_b_copy.edges())))
            logger.debug("Final number of nodes after attack for i: " + str(i) +
                ", netA: " + str(len(network_a_copy.nodes())) + ", netB: " + str(len(network_b_copy.nodes())))

            if start_with_comms is True:
                conn_comp = sorted(nx.connected_components(network_a_copy), key=len, reverse=True)
            else:
                conn_comp = sorted(nx.connected_components(network_b_copy), key=len, reverse=True)
            conn_comp0 = []  # giant component
            if not conn_comp:  # check if conn_comp is empty
                giant_comp_size = 0
            else:
                conn_comp0 = conn_comp[0]  # conn_comp of the new network
                giant_comp_size = len(conn_comp0)
            logger.debug("Number of CC sub-graphs: " + str(len(conn_comp)) + ", Nodes in giant component: " + str(giant_comp_size))

            # gmccb is only used for verification
            # gmccb = nx.connected_component_subgraphs(network_b_copy)
            # if not gmccb:  # check if gmccb is empty
            #   gmccbSize = 0
            # else:
            #   gmccb0= gmccb[0]  # gmccb of the new network
            #   gmccbSize = len(gmccb0)
            # print "Nodes in conn_comp: " + str(giant_comp_size) + ", nodes in gmccb: " + str(gmccbSize) + ". Blackout? " + str(blackout)
            # does the giant MCC exist and is its size greater than the threshold?

            result = 1  # for passing to write_result

            if output_gc_size is True:
                y.append([p, (float(giant_comp_size) / n)])
            else:
                if giant_comp_size > threshold:
                    y.append([p, 1])
                    # y=[p,1]
                else:
                    result = 0
                    y.append([p, 0])
                    # y=[p,0]

            if output_result_to_DB is True:
                write_result(run, p, nodes_attacked, initial_nodes_attacked, result, dbh)
            # print "y is: " + str(y)
            # print "Inner run time was " + str(time.time() - innerrunstart) + " seconds"


def log_result(result):
    allY.extend(result)


def critical_node_connection(network, nodes_attacked):
    original_num = len(nodes_attacked)
    if start_with_comms is True:
        # Find out if the critical node has been attacked, if so then no one can reach it
        # and all nodes should be in the nodes_attacked list
        if critical_node in nodes_attacked:
            logger.debug(">>>>>> Critical node found in nodes from file <<<<<<")
            return nodes
    # remove nodes attacked from network_a
    network.remove_nodes_from(nodes_attacked)
    # ccsg = nx.connected_component_subgraphs(network)
    ccsg = sorted(nx.connected_components(network), key=len, reverse=True)
    unconnected_nodes = []
    for i in range(0, len(ccsg), 1):
        # if critical_node not in ccsg[i].nodes():
        if critical_node not in ccsg[i]:
            # unconnected_nodes.extend(ccsg[i].nodes())
            unconnected_nodes.extend(ccsg[i])
        else:
            if i is not 0:
                logger.debug("\t >>>>>> Critical node in subgraph " + str(i))
    nodes_attacked.extend(unconnected_nodes)
    nodes_attacked = set(nodes_attacked)  # remove duplicates
    nodes_attacked = list(nodes_attacked)  # convert back to a list
    logger.debug("Adding nodes connected to critical node to nodes_attacked list, size original " +
    str(original_num) + ", final size: " + str(len(nodes_attacked)))
    return nodes_attacked


def write_networks(run, network_a, network_b, write_networks_out):
    if write_networks_out == "json":
        data = json_graph.node_link_data(network_a)
        filename = 'network_a_' + network_label + '_run' + str(run) + '.json'
        path = relpath + "output/" + filename
        complete_name = os.path.abspath(path)
        with open(complete_name, 'w') as f:
            json.dump(data, f)
        data = json_graph.node_link_data(network_b)
        filename = 'network_b_' + network_label + '_run' + str(run) + '.json'
        path = relpath + "output/" + filename
        complete_name = os.path.abspath(path)
        with open(complete_name, 'w') as f:
            json.dump(data, f)
    elif write_networks_out == "edgelist":
        edgelist_filename = 'network_a_' + network_label + '_run' + str(run) + '.edgelist'
        edgelist_path = relpath + "output/" + edgelist_filename
        edgelist_complete_path = os.path.abspath(edgelist_path)
        nx.write_edgelist(network_a, edgelist_complete_path)
        edgelist_filename = 'network_b_' + network_label + '_run' + str(run) + '.edgelist'
        edgelist_path = relpath + "output/" + edgelist_filename
        edgelist_complete_path = os.path.abspath(edgelist_path)
        nx.write_edgelist(network_b, edgelist_complete_path)


def write_result(run, p, nodes_attacked, initial_nodes_attacked, result, dbh):
    # Write the final list of nodes attacked and whether it caused a blackout
    experiment = network_type + "_result_" + str(n)  # This is the collection name
    nodes_remaining = list(set(nodes) - set(nodes_attacked))

    nodes_removed = {
        "run": run,
        "network": network_type,
        "initial_nodes_attacked": initial_nodes_attacked,
        "nodes_remaining_at_end": nodes_remaining,
        "all_nodes_attacked": nodes_attacked,
        "p": p,
        "size": len(nodes_attacked),
        "no_blackout": result,
        "config": config
    }
    # print str(nodes_removed)
    dbh[experiment].insert(nodes_removed, safe=True)  # sets the collection to experiment


def write_removed_nodes(run, p, swap_networks, nodes_attacked, dbh, network_name, network):
    filename = 'nodesRemoved_' + network_label + '_run' + str(run) + '.csv'  # + '_p_%.2f'%p + '.csv'
    path = relpath + "output/" + filename
    complete_name = os.path.abspath(path)
    # print(nodes_attacked)
    write_to_csv = False
    if write_to_csv is True:
        try:
            with open(complete_name):
                with open(complete_name, 'a') as f:  # append to the file if it already exists
                    writer = csv.writer(f)
                    # create a list of tuples and append that to the file
                    for row in [(swap_networks, nodes_attacked[i], p) for i in range(len(nodes_attacked))]:
                        writer.writerow(row)
                    pass
        except IOError:  # create the file and write the header if it doesn't exist yet
            with open(complete_name, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Step', 'Node', 'p'])
                for row in [(swap_networks, nodes_attacked[i], p) for i in range(len(nodes_attacked))]:
                    writer.writerow(row)

    experiment = network_type + "_" + str(n)  # This is the collection name

    nodes_remaining = list(set(nodes) - set(nodes_attacked))
    nodes_removed = {
        "run": run,
        "network": network_name,
        "step": swap_networks,
        "nodes": nodes_remaining,
        "nodes_attacked": nodes_attacked,
        "p": p,
        "size": len(nodes_attacked)
    }

    # If it's the first run put in the config
    if run == 0 and swap_networks == 1:
        nodes_removed["initial_network"] = json_graph.node_link_data(network)
        nodes_removed["config"] = config

    dbh[experiment].insert(nodes_removed, safe=True)  # sets the collection to experiment


def targeted_nodes(network_a, num_nodes_attacked):
    degree_sequence = nx.degree(network_a).values()
    node_by_deg = sorted(zip(degree_sequence, network_a), reverse=True)
    nodes_attacked = []

    for i in range(0, num_nodes_attacked, 1):
        nodes_attacked.append(node_by_deg[i][1])

    return nodes_attacked


def get_nodes_from_file(run, p_point, network_a):
    nodes_attacked = []
    file_name = relpath + 'data/node-removals/bus_outage_' + str(run) + '.csv.bz2'
    # print file_name
    p_calc = num_outages_in_file - inverse_step_size * (p_point - min_outage_point)
    p_column = int(round(p_calc))  # 50-10000*(0.995-0.75)/50 = 1, round first otherwise you can get unexpected results
    # print "p_column " + str(p_column)
    if p_column > num_outages_in_file or p_column == 0:
        if verbose is True:
            print "No outages found for p_point " + str(p_point) + " at column " + str(p_column) + " creating outage instead"
        random_removal_fraction = 1 - p_point    # Fraction of nodes to remove
        num_nodes_attacked = int(math.floor(random_removal_fraction * n))
        nodes_attacked = random.sample(network_a.nodes(), num_nodes_attacked)
        if verbose is True:
            print "size nodes_attacked list " + str(len(nodes_attacked)) + " for run " + str(run) + ", p_point " + str(p_point)
        return nodes_attacked
    try:
        f = bz2.BZ2File(file_name, mode='r')
        c = codecs.iterdecode(f, "utf-8")
        data = csv.DictReader(c)
        # with open(file_name, 'rb') as f:
        #   data = csv.DictReader(f)
        for next_row in data:
            item = int(next_row[str(p_column)])
            if item != 0:
                nodes_attacked.append(item)
        if verbose is True:
            print "size nodes_attacked list " + str(len(nodes_attacked)) + " for run " + str(run) + ", p_point " + str(p_point) + ", p_column " + str(p_column)
    except Exception, e:
                print "Node outage read error: " + str(e)
                raise
    # print "nodes_attacked " + str(nodes_attacked) + " run " + str(run) + ", p_point " + str(p_point)
    return nodes_attacked


def get_network_from_file(network_name):
    path = relpath + network_name
    try:
        network = nx.read_edgelist(path, nodetype=int)
        if 0 in network.nodes():
            max_nodes = len(network.nodes()) + 1
            mapping = dict(zip(network.nodes(), range(1, max_nodes)))  # renumber the nodes to start at 1 for MATLAB
            network = nx.relabel_nodes(network, mapping)
    except Exception, e:
        print "Error Unknown network type for " + network_name + " " + str(e)
        raise

    return network


def make_comms(network_b, copy):
    global n
    if random_rewire_prob == -1:
        degree_sequence = sorted(nx.degree(network_b).values(), reverse=True)
        network_a = nx.configuration_model(degree_sequence)
    else:
        if copy is True:
            network_a = network_b.copy()
        else:
            network_a = network_b
        nodes = []
        targets = []
        for i, j in network_a.edges():
            nodes.append(i)
            targets.append(j)

        # rewire edges from each node, adapted from NetworkX W/S graph generator
        # http://networkx.github.io/documentation/latest/_modules/networkx/generators/random_graphs.html#watts_strogatz_graph
        # no self loops or multiple edges allowed
        for u, v in network_a.edges():
            if random.random() < random_rewire_prob:
                w = random.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or network_a.has_edge(u, w):
                    w = random.choice(nodes)
                # print "node: " + str(u) + ", target: " + str(v)
                network_a.remove_edge(u, v)
                network_a.add_edge(u, w)
    if copy is True:
        mapping = dict(zip(network_a.nodes(), range(1, n + 1)))  # 2384 for Polish, 4942 for western. relabel the nodes to start at 1 like network_a
        network_a = nx.relabel_nodes(network_a, mapping)
    return network_a


def create_networks(network_type):
    global critical_node
    if network_from_file is True:
        network_a = get_network_from_file(comm_network_location)
        # network_a.name = comm_network_location
        network_b = get_network_from_file(power_network_location)
        # network_b.name = power_network_location
        try:
            critical_node = betweenness[comm_network_location.split('/')[-1]]  # get just the filename from comm_network_location
        except KeyError:
            logger.warning("Betweenness not found for network " + comm_network_location.split('/')[-1] +
                " not setting critical node")
            critical_node = -1
    else:
        if network_type == 'ER':
            # Erdos-Renyi random graphs
            network_a = nx.gnp_random_graph(n, ep)
            network_b = nx.gnp_random_graph(n, ep)
        elif network_type == 'RR':
            # random regular graphs
            network_a = nx.random_regular_graph(int(k), n)
            network_b = nx.random_regular_graph(int(k), n)
        elif network_type == 'SF':
            # Scale free networks
            # m==2 gives <k>==4, for this lambda/gamma is always 3
            network_a = nx.barabasi_albert_graph(n, 2)
            network_b = nx.barabasi_albert_graph(n, 2)
            if random_rewire_prob != -1:
                network_a = make_comms(network_a, False)
                network_b = make_comms(network_b, False)
        elif network_type == 'Lattice':
            l = math.sqrt(n)
            if(n % l != 0):
                print "Number of nodes, " + str(n) + ", not square (i.e. sqrt(n) has a remainder) for lattice. Adjust n and retry."
                raise
                sys.exit(-1)
            l = int(l)
            network_a = nx.grid_2d_graph(l, l, periodic=True)
            network_b = nx.grid_2d_graph(l, l, periodic=True)
        elif network_type == 'CFS-SW':
            '''network_b is a topological representation of the power network. network_a
            is a generated  communication network created through either a configuration
            model of the power grid or by randomly rewiring the power grid topology.
            '''
            path = relpath + "data/power-grid/case2383wp.edgelist"
            network_b = nx.read_edgelist(path, nodetype=int)
            mapping = dict(zip(network_b.nodes(), range(1, n + 1)))  # renumber the nodes to start at 1 for MATLAB
            network_b = nx.relabel_nodes(network_b, mapping)
            '''make the comms network'''
            if cfs_iter == 1 or real is False:
                if verbose is True:
                    print "^^^^^ Making comm network ^^^^^^"
                network_a = make_comms(network_b, True)
            else:
                try:
                    if verbose is True:
                        print "reading from comm network at: " + network_filename
                    network_a = nx.read_edgelist(network_filename, nodetype=int)
                except Exception as e:
                    print e
                    print "Read edgelist error"
                    raise
        else:
            print 'Invalid network type: ' + network_type
            return []

    # Order the nodes of the networks randomly
    if real is False and shuffle_networks is True:
        randomlist = range(1, n + 1)  # node must names start at 1 for MATLAB, remapping to start at 1 done above
        random.shuffle(randomlist)
        mapping = dict(zip(network_a.nodes(), randomlist))
        network_a = nx.relabel_nodes(network_a, mapping)
        random.shuffle(randomlist)
        mapping = dict(zip(network_b.nodes(), randomlist))
        network_b = nx.relabel_nodes(network_b, mapping)
        del mapping
        del randomlist
    elif real is True and shuffle_networks is True and verbose is True:
        print "\t ******* -> Not shuffling networks when using a real grid with CFS <- ********"
    if generate_each_run is False and find_scaling_exponent is True:
        degseq = sorted(nx.degree(network_a).values(), reverse=False)
        fit = powerlaw.Fit(degseq, xmin=2.0)
        if verbose is True:
            print "Scaling exponent network_a " + str(fit.power_law.alpha)
        degseq = sorted(nx.degree(network_b).values(), reverse=False)
        fit = powerlaw.Fit(degseq, xmin=2.0)
        if verbose is True:
            print "Scaling exponent network_b " + str(fit.power_law.alpha)

    return [network_a, network_b]


def main():
    # Setup multi-core parallel runs if not running on an HPC
    manager = mlt.Manager()
    networks = []
    if generate_each_run is False and real is False:
        # allow the generated networks to be shared by all processes/threads
        networks = manager.list(create_networks(network_type))
    else:
        networks = create_networks(network_type)
    if real is False and batch_mode is False:
        runstart = time.time()
        pool = mlt.Pool()
        for i in range(0, runs, 1):
            pool.apply_async(attack_network, args=(i, networks, ), callback=log_result)

        pool.close()
        pool.join()
        d = defaultdict(list)
        p_half = defaultdict(int)
        for key, value in allY:
            d[key].append(value)
            if value > 0.5:
                p_half[key] += 1
            else:
                p_half[key] += 0
        average = [float(sum(value)) / len(value) for key, value in d.items()]
        average_p_half = [float(value) / runs for key, value in p_half.items()]
        x = [key for key, value in d.items()]
        write_output(x, average, average_p_half, runs)

        print "Run time was " + str(time.time() - runstart) + " seconds"
    elif real is False and batch_mode is True:
        runstart = time.time()
        y = attack_network(r_num, networks)

        d = defaultdict(list)
        p_half = defaultdict(int)
        for key, value in y:
            d[key].append(value)
            if value > 0.5:
                p_half[key] += 1
            else:
                p_half[key] += 0
        average = [float(sum(value)) / len(value) for key, value in d.items()]
        average_p_half = [float(value) / runs for key, value in p_half.items()]
        x = [key for key, value in d.items()]

        write_output(x, average, average_p_half, runs)

        print "Run time was " + str(time.time() - runstart) + " seconds"

    else:  # This is being called by MATLAB
        attack_network(0, networks)  # set multi-run to 0 so it never writes to output

    if show_plot is True:
            # plot the result
            plt.plot(x, average, 'bo')  # 'rx' for red x 'g+' for green + marker
            # show the plot
            plt.show()


def write_output(x, average, average_p_half, run):
    """Write the output to file."""
    if log_all_p_values is False and run != runs:
        return

    # transform the data to enable writing to csv
    output = []
    output.append(x)
    output.append(average)
    output.append(average_p_half)
    output = zip(* output)
    output.sort(key=itemgetter(0))  # sort on x (0)

    num_run = ''
    if run == runs:
        num_run = 'run_final'
    else:
        num_run = 'run_' + str(run)

    # write the output to file
    filename = network_label + num_run + '.csv'
    if batch_mode is True:
        filename = str(r_num) + "_run_" + filename
    path = relpath + "output/" + filename
    complete_name = os.path.abspath(path)
    with open(complete_name, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['p', 'Pinf', 'p0.5'])
        for row in output:
            writer.writerow(row)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print e
        raise
