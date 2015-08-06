# Coupled network model notes

## Model description

### Coupled Topology Model

Two networks are created based on the following parameters in config.json: ```networkFromFile```, ```network_type```, ```k```, ```n```, and ```random_rewire_prob```. If ```networkFromFile``` is true then the networks are read from comm_network_location and power_network_location - ```network_type```, k, n, and ```random_rewire_prob``` are used only in output file names and do not impact the model in this case. Otherwise networks are generated based on ```network_type```, k, n, and ```random_rewire_prob``` as described below.

After the networks are created, an attack occurs that removes nodes on the networks. The nodes attacked are determined by the parameters ```outages_from_file``` and ```real```. If ```outages_from_file``` is true then the nodes attacked at each ```p_value``` are read from bus_outages_n.csv where n is the replicate number ```r_num```. Each bus_outage file contains 100 random, pre-generated sets of node removals for every ```p_value``` from 0.5 to 0.955 in steps of 0.005. 

The two networks are connected by nodes at a rate of ```1:q```, ```q``` in ```[0,1]```. ```q``` is set by ```deg_of_coupling```. ```q==1``` implies that each node in one network is connected to exactly one other node in the other network. At ```q==0``` there are no connections between networks while ```0<q<1``` leads to a fraction of nodes are connected across networks.

Cascading outages on coupled networks are simulated in a process similar to Bulydrev et al., 2011. The node removals after an attack may leave a network, network A, with some nodes that no longer have links to nodes in network A. When a node no longer has links within its own network we look to its coupled node in its interacting network, network B, and remove all of the links leaving this coupled node. Now network B may have orphaned nodes and the process repeats in the other direction until there are no more orphaned nodes in either network. After the cascade stops we check the size, i.e. number of nodes, of the giant component of the network that was initially attacked, in this case network A. If the size of the giant component of network A is greater than our threshold then the networks are deemed to have survived the attack. 

### Smart Grid Model

The smart grid model uses a physics-based DC power flow cascading failure simulator (CFS), ```dcsimsep.m```, to provide a more realistic model of failures in the power grid. This mode is activated by setting ```real``` to true. Cascades occur across networks occur in a similar manner to the cooupled topological model with the differences detailed in the accompanying paper [Reducing Cascading Failure Risk by Increasing Infrastructure Network Interdependency](http://arxiv.org/abs/1410.6836). If ```real``` is true then the ```couplednetworks``` is being called by the CFS and node outages are read from a file created by the CFS model. 

```cn_runner.m``` is a wraper for running from an HPC cluster or from a workstation - set by ```batch_mode```, true for cluster and false for workstation. It creates a network based on the config.json settings but the network is only used to determine which nodes to remove in the case that the ```outages_from_file``` is false or the p_value called for is not found in the ```outages_from_file``` csv. Once the nodes to remove are determined cmp_dcsimsep is called.

## Running couplednetworks.py

Expects to be run from the command line from the source directory like so:

	machine:source user$python couplednetworks.py -1 -1 -1 config.json

The program takes four required arguments and one optional argument (```r_num```):

*	```mpid``` - MATLAB process id

*	```cfs_iter``` - Iteration number from MATLAB

*	```percent_removed``` - Percent of nodes removed for this model run.

*	```config``` - Name and location of the configuration file.

*	```r_num``` - Optional - Replicate number to use when running from an HPC. ```r_num``` determines which outage file to use.

All arguments, aside from config.json, can be set to -1 if running without the cascading failure simulator (CFS, MATLAB model)

## Parameters in config.json

*	```k```
	*	Average degree as a float. Used to determine ep, probability of edge creation, a parameter to generate Erdos-Renyi random graphs. ep = k / n-1. This is cast to an integer for and must be even RR graphs. Ignored if using networks from file.
*	```n```
	*	Number of nodes. 2383 for Polish grid.
*	```runs```
	*	Number of times to run the model if not running in batch mode. For batch mode set this to 1.
*	```p_values, p_min, p_max```
	*	Number of distinct percent-removals, ```p_values```, to do across ```[p_min p_max]``` in each run. 
*	```start_with_comms```
	*	Starts the attack on the communication network, network A, instead of the grid, network B.
*	```gc_threshold```
	*	The fraction of the number of components in the giant component of the communications network, network A, under which failure is declared. For replication of Bulydrev this should be 0 which gets changed to a threshold of 1 node. 
*	```grid_gc_threshold```
	*	The fraction of the number of components in the giant component in the grid, network B, under which blackout is declared.
*	```output_gc_size```
	*	Instead of checking for failure against a threshold the size of the giant component can be output if this parameter is true.
*	```outages_from_file, min_outage_point, max_outage_point, num_outages_in_file```
	*	If true, reads the outages from coupled-networks/data/node-removals/ instead generating them as the model runs. min_outage_point, max_outage_point, num_outages_in_file determine which outage file to read from.
*	```inverse_step_size```
	*	1/(p2-p1) where p is the percent of nodes/branches removed. If the step size is 0.005 inverse_step_size is 200. Used to find the proper node/branch outage file.
*	```num_outages_in_coupling_file, min_coupling_point, max_coupling_point, coupling_from_file, inverse_step_size_for_coupling```
	*	For defining how to access the couplings from file in a similar way the outages are accessed.
*	```targeted```
	*	Sorts the network in order of degree sequence and returns a list of nodes to attack starting with the highest degree node.
*	```output_removed_nodes```
	*	When removing links, returns a list of nodes in networkB but not in networkA. When running with CFS, i.e. real == true, this needs to be true as well.
*	```output_removed_nodes_to_DB```
	*	Writes the configuration (config.json), the original networks and the status of the networks at every step of each pValue of each run to a database. Requires pymongo as an import a MongoDB on the system the model is running on and mongod running.
*	```log_all_p_values```
	*	Writes the output to file at the end of each run instead of just at the end of all runs.
*	```write_networks_out```
	*	Specify either "edgelist" or "json" to writes the networks to file. This can be useful to check the networks that are generated if not getting them from file. However, it will produce a lot of files depending on how many runs are being done.
*	```show_plot```
	*	Produce a plot of the output at the end. Requires matplotlib as an import. Set to false if running in batch mode to safe memory.
*	```network_type```
	*	Sets the types of networks to use. If networks are being generated this must be either 'SF' (scale-free), 'RR' (random regular), 'ER' (Erdos-Renyi), 'CFS-SW' (cascading failure simulator/small-world). CFS-SW uses the topology of the Polish grid as the power network and some rewiring of that (set by ```random_rewire_prob```) for the communicaion network. If networks are being read from file then this can be anything and gets placed in the name of the output files.
*	```random_rewire_prob```
	*	Used to rewire generated scale-free and CFS-SW networks. If ```random_rewire_prob``` is -1, base the comms network on a configuration model otherwise the comms will be randomly rewired with this probability. For scale free replication lambda/gamma = ~3 when ```random_rewire_prob``` = 0.16 and ~2.7 when ```random_rewire_prob``` = 0.31 and ~2.3 with rewireProb = 0.57
*	```shuffle_networks```
	*	Randomly renumber the nodes on the networks, for replication this should be True but when using with the CFS it should be false.
*	```generate_each_run```
	*	Generate a new network for each run, True, or keep the same for all, False.
*	```verbose```
	*	If true, prints detailed messages about what's happening as the model is running
*	```debug```
	*	Passed as an argument to the compiled MATLAB telling it whether to print status messages to the console
*	```output_result_to_DB```
	*	Sends the run results to a database
*	```single_net_cascade```
	*	Only applicable in singlenetwork.py. Determines whether outages lead to cascades
*	```find_scaling_exponent```
	*	If the powerlaw module is installed this will print out the scaling exponent of the networks in the model
*	```batch_mode```
	*	Set to true when running from a cluster. Each run is written to its own output file. If false cfs_comms_model will send the runs to as many cores/hyperthreads on a workstation and combine the results at the end.
*	```deg_of_coupling```
	*	The fraction of nodes connected between networks in [0,1].
*	```hpc```
	*	For running in batch mode. If this is set to "HIVE" then the condor process ID, which starts at 0, is incremented by 1 to ensure the correct outage and coupling files are retrieved.
*	```networkFromFile```
	*	Reads the network from coupled-networks/data/networks/generated/ instead of generating it. Uses ```network_type``` and ```n``` to determine which file to read from. 
*	```comm_network_location```
	*	Relative path (from coupled-networks) to where the communication network file is located.
*	```power_network_location```
	*	Relative path (from coupled-networks) to where the power grid file is located.
*	```real```
	*	Set to true if using a CFS model for the power grid. If false then the power grid is only modeled topologically, not physically.
*	```comm_model```
	*	Used by MATLAB to find the comms model on the machine the model is running on.
*	```host_OS```
	*	For the compiled MATLAB this determines where to find cmp_dcsimsep, 0 for Mac, 1 for Linux
*	```optimizer```
	*	Tells cmp_dcsimsep which optimizer to use. Options are 'cplex' and 'mexosi'.
*	```two_way```
	*	Enables the two-way, beneficial, coupling mode in dcsimsep.
*	```two_way_extreme```
	*	Enables the two-way extreme, vulnerable, coupling mode in dcsimsep. In this mode comm outages connected to generators cause the generators to fail.
*	```power_systems_data_location```
	*	Sets the .mat file to use as input for the power grid.
*	```python_location```
	*	Used by MATLAB to know where to find Python on the machine the model is running on.
*	```betweenness```
	*	For pre-defined networks this is the node with the greatest betweenness connectivity for that network and is used as a proxy for a control center. In the comms model, if a node has no path to this node then it is considered failed. 

## mexosi Build/Install notes

### Mac

If running 10.9 changing 10.7 to 10.9 in this link need to be made to your mexopts.sh file.
http://www.mathworks.com/matlabcentral/answers/103904-can-i-use-xcode-5-as-my-c-or-c-compiler-in-matlab-8-1-r2013a-or-matlab-8-2-r2013b

The following also needs to be added to mexopts.sh per:
http://stackoverflow.com/questions/22367516/mex-compile-error-unknown-type-name-char16-t
Add -std=c++11 to CXXFLAGS

Also changed the following:

From: ```MW_SDKROOT='$MW_SDKROOT_TMP' ```

To: ```MW_SDKROOT='/Library/Developer/CommandLineTools'```
