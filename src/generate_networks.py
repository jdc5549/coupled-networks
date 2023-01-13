import pygambit as gambit
import numpy as np
from couplednetworks import create_networks
import networkx as nx
import time
import os

def ncr(n, r):
    import operator as op
    from functools import reduce
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def create_random_nets(save_dir,num_nodes,num2gen=10,gen_threshes=False,show=False):  
    #import random
    #random.seed(np.random.randint(10000))
    for i in range(num2gen):
        [network_a, network_b] = create_networks('SF',num_nodes=num_nodes)
        if save_dir != '':
            f = save_dir + 'net_{}.edgelist'.format(i)
            nx.write_edgelist(network_b,f)
            if gen_threshes:
                thresholds = []
                for node in network_b.nodes():
                    thresh = 1/len(network_b[node])*np.random.choice([i for i in range(1,len(network_b[node])+1)])
                    thresholds.append(thresh)
                ft = save_dir + 'net_{}_thresh.npy'.format(i)
                np.save(ft,np.asarray(thresholds))
                #print(f'Saved to {ft}')

    if show:
        print('Showing one of the generated networks')
        import matplotlib.pyplot as plt
        nx.draw(network_b,with_labels=True)
        plt.draw()
        plt.show()
    return [network_a,network_b]

def create_simple_random_nets():
    random.seed(np.random.randint(0,10000))
    net = nx.Graph()
    nodes = [0,1,2]
    net.add_nodes_from(nodes)
    double_node = random.sample(nodes,1)[0]
    single_nodes = nodes.copy()
    single_nodes.remove(double_node)
    net.add_edges_from([(single_nodes[0],double_node),(double_node,single_nodes[1])])
    return [net,net]

def get_nash_eqs(env):
    num_nodes_attacked = env.num_nodes_attacked
    net_size = env.net_size
    num_actions = ncr(net_size,num_nodes_attacked)
    U = np.zeros([num_actions,num_actions],dtype=gambit.Rational)
    curr_action = [i for i in range(num_nodes_attacked)]
    last_action = [i for i in range(net_size-1,net_size-num_nodes_attacked-1,-1)]
    last_action.reverse()
    all_actions = [curr_action.copy()]
    while curr_action != last_action:
        for i in range(num_nodes_attacked,0,-1):
            if curr_action[i-1] < net_size-(num_nodes_attacked-i+1):
                curr_action[i-1] += 1
                break
            else:
                curr_action[i-1] = curr_action[i-2]+2
                for j in range(i,num_nodes_attacked):
                    curr_action[j] = curr_action[j-1]+1
        all_actions.append(curr_action.copy())
    print('Caculating Ultility Matrix of size {}'.format(num_actions*num_actions))
    for i in range(num_actions):
        for j in range(num_actions):
            node_lists = [all_actions[i],all_actions[j]]
            last_rew = 0
            _,reward,_,_ = env.step(node_lists)
            U[i,j] = gambit.Rational(reward[0])
    #U = U.astype(gambit.Rational)
    print('Caculating Nash Eqs')
    g = gambit.Game.from_arrays(U,-U)
    g.players[0].label = 'Attacker'
    g.players[1].label = 'Defender'
    eqs = gambit.nash.lp_solve(g)
    eqs = np.array(eqs,dtype=float)
    eqs = np.reshape(eqs,(2,num_actions))
    U = np.array(U,dtype=float)
    return eqs, U

if __name__ == '__main__':
    import argparse
    from couplednetworks_gym_main import CoupledNetsEnv2
    from couplednetworks_gym_cfa import SimpleCascadeEnv
    parser = argparse.ArgumentParser(description='Network Generation Args')
    parser.add_argument("--num_nodes",type=int,default=100)
    parser.add_argument("--num2gen",type=int,default=10)
    parser.add_argument("--net_save_dir",default='data/networks/generated/',type=str,help='Directory where the network topologies will be saved.')
    parser.add_argument("--nash_eqs_dir",default=None,type=str,help='Directory where Nash EQ benchmarks will be written. If None (default), then does not calculate Nash EQs.')
    parser.add_argument("--p",default=0.1,type=float,help='If calculating Nash EQs, the percent of nodes to be attacked/defended.')
    parser.add_argument("--env_type", default='SimpleCascadeEnv',help='What type of gym environment should be used to generate the NashEQ utility')
    args = parser.parse_args()

    if args.net_save_dir[-1] != '/':
        args.net_save_dir += '/'
    full_dir = args.net_save_dir + f'SF_{args.num_nodes}n_2.422deg_{args.env_type}_p{args.p}_{args.num2gen}/'
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)
    print(args.env_type)
    if args.env_type == 'SimpleCascadeEnv':
        gen_threshes = True
    else:
        gen_threshes = False
    create_random_nets(full_dir,args.num_nodes,gen_threshes=gen_threshes,num2gen=args.num2gen)
    if args.nash_eqs_dir is not None:
        if args.nash_eqs_dir[-1] != '/':
            args.nash_eqs_dir += '/'
        if not os.path.isdir(args.nash_eqs_dir):
            os.mkdir(args.nash_eqs_dir)
        tic = time.perf_counter()
        files = [f for f in os.listdir(full_dir) if 'thresh' not in f]
        for i,f in enumerate(files):
            if args.env_type == 'SimpleCascadeEnv':
                env = SimpleCascadeEnv(args.num_nodes,args.p,args.p,'File',filename = os.path.join(full_dir,f))
            elif args.env_type == 'CoupledNetsEnv2':
                env = CoupledNetsEnv2(args.num_nodes,args.p,args.p,'File',filename = os.path.join(full_dir,f))
            else:
                print(f'Environment type {args.env_type} is not supported')
                exit()
            eqs,U = get_nash_eqs(env)
            f_eq = args.nash_eqs_dir + f'eq_{i}.npy'
            np.save(f_eq,eqs)
            f_util = args.nash_eqs_dir + f'util_{i}.npy'
            np.save(f_util,U)
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
