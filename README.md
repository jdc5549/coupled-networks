## Coupled Networks Gym Environment
![](images/coupled-networks.png)

The gym environment created for this work was based off of the simulation created by [Korkali et al., 2017](https://www.nature.com/articles/srep44499). This model accounts for failure propagating from both the power network to the commmunication network and vice versa, as shown in the figure above. This simulation has been modiffied in this work to conform to the Open AI Gym format so that it is suitable for training Reinforcement Learning (RL) agents.

## Multi-Agent Reinforcement Learning
This project focuses on the two player zero-sum security game scenario, in which an attacker and defender agent each select nodes on the coupled network graph to attack and defend respectively based on topological information of the nodes. The attacker is rewarded proportionate to the resulting cascading failure size, and the defender receives the negative of this reward. We use RL to train these agents against each other to learn the optimal strategy of nodes to select. We modify [David Albert's](https://github.com/blavad/marl) framework for Multi-Agent Reinforcement Learning (MARL) to implement the algorithms in our paper. 


## Running Experiments
In the file ```config/config.py``` the fields ```comm_model``` and ```power_systems_data_location``` will need to be modified to match your path to the file from the root directory. Also check that the field ```python_location``` points to the location that python is installed on your machine.

This repository includes the network topologies used in our paper's experiments (under ```data/networks/generated/```), as well as precomputed Nash equilibirum for small-scale experiments (under ```output/nasheqs```). To generate your own network, run the command

```python src/generate_networks.py --num_nodes <number of nodes in network> --num2gen <number of topologies to generate> --net_save_dir <location to save the network files>```  

If you want to also calculate the Nash equilibrium, you can include the arguments ```--nash_eqs_dir``` to specify where to save the calculated EQs, and ```--p``` to specify the fraction of nodes to be attacked/defended (0.1 by default). Note that this is only computationally feasible for small networks and values of ```p```.

The main script for running experiments is ```src/couplednetworks_gym_main.py```. There are a number of arguments that can be used to specify expriment parameters.
*	```train```
	*	Boolean. Specify whether a model is being trained. Only evaluates a given model when false. Default True.
*	```training_steps```
	*	Integer. Number of steps to train the model if we are training. Default 100k.
*	```batch_size```
	*	Integer. Batch size for neural network training. Default 100.
*	```mlp_hidden_size```
	*	Int. Hidden layer size for MLP nets used for RL agent.
*	```test_freq```
	*	Integer. How often the model being trained should be evaluated. The more often it is tested, the slower training will go. Default 10k.
*	```save_freq```
	*	Integer.  How often the model will be saved. Default 10k.
*	```exploiters```
	*	Boolean.  Whether to train exploiters. Useful when problem is too large to compute a Nash EQ.
*	```exploited_type```
	*	String.  What type of agent to train exploiters against. Valid choices are: NN,Random, and Heuristic. Default 'NN'.
*	```p```
	*	Float. Fraction of total nodes to be attacked/defended. Default 0.1.
*	```degree```
	*	Int. Number of nodes selected by the agent policy at a time. Default 1.
*	```net_type```
	*	String. Strategy for network creation. Use "SF" for random scale-free network, and "File" to load the network in "net_file". Default 'SF'
*	```tabular_q```
	*	Boolean. Use tabular Q-learning instead of neural network Q learning. Default False.
*	```nash_eqs_dir```
	*	Str. Directory where Nash EQ benchmarks are stored. If None, Nash EQs are not used. Default None.
*	```test_nets_dir``` 
	*	Str. Directory where network topologies used for testing are stored. Default None.
*	```cn_config```
	*	Str. Location of config file for the coupled network simulator. Default 'config/config.json'
*	```exp_name```
	* 	Str. Name of experiment that will be associated with log and model files. Default 'my_exp'

**Important Note: The config file must have the correct size of the networks in the "n" field for the main script to work**.

### Some example commands
Training agents to attack/defend 1 node in a 10 node network for 100k steps, while using precalculated Nash EQs as benchmark.

```python src/couplednetworks_gym_main.py --train True --training_steps 100000 --test_freq 10000 --save_freq 5000 --test_nets_dir 'data/networks/generated/SF_10n_2.422deg' --exp_name '10c1d1' --p 0.1 --nash_eqs_dir 'output/nasheqs/10n_p10```

Training exploiter agents to exploit the ego agents trained in the previous command.

```python src/couplednetworks_gym_main.py --train True --exploiters True --training_steps 100000 --test_freq 10000 --save_freq 5000 --test_nets_dir 'data/networks/generated/SF_10n_2.422deg' --net_type 'File' --net_file_train_dir 'data/networks/generated/SF_10n_2.422deg' --exp_name '10c1d1_explt' --p 0.1 --ego_model_dir 'models/marl/10c1d1/'```

Evaluating agents

```python src/couplednetworks_gym_main.py --exp_name '10c1d1_eval' --exploiters True --testing_episodes 100 --ego_model_dir 'models/marl/10c1d1/' --exploiter_model_dir 'models/marl/10c1d1_explt/' --test_nets_dir 'data/networks/generated/SF_10n_2.422deg' --net_type 'File' --net_file_train_dir 'data/networks/generated/SF_10n_2.422deg' --nash_eqs_dir 'output/nasheqs/10n_p10'```
