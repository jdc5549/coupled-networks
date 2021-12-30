import gambit
import numpy as np
from couplednetworks_gym_main import CoupledNetsEnv2

def create_random_nets(save_dir,num_nodes,show=False):  
    random.seed(np.random.randint(10000))
    for i in range(10):
        [network_a, network_b] = create_networks('SF',num_nodes=num_nodes)
        f = save_dir + 'net_{}.edgelist'.format(i)
        nx.write_edgelist(network_b,f)
    if show:
        print('Showing one of the generated networks')
        import matplotlib.pyplot as plt
        nx.draw(network_b,with_labels=True)
        plt.draw()
        plt.show()

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
    net_size = env.net_b.number_of_nodes()
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
    return eqs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Network Generation Args')
    parser.add_argument("--num_nodes",default='')
    parser.add_argument("--net_save_dir",default='data/networks/generated/',type=str,help='Directory where the network topologies will be saved.')
    parser.add_argument("--nash_eqs_dir",default=None,type=str,help='Directory where Nash EQ benchmarks will be written. If None (default), then does not calculate Nash EQs.')
    parser.add_argument("--p",default=0.1,type=float,help='If calculating Nash EQs, the percent of nodes to be attacked/defended.')

    if args.net_save_dir[-1] != '/':
        args.net_save_dir += '/'
    create_random_nets(args.save_dir,args.num_nodes)
    if args.nash_eqs_dir is not None:
        if args.nash_eqs_dir[-1] != '/':
            args.nash_eqs_dir += '/'
        tic = time.perf_counter()
        for f in os.listdir(args.net_save_dir):
            env = CoupledNetsEnv2(args.p,args.p,'File',filename = os.path.join(args.net_save_dir,f))
            eqs = get_nash_eqs(env)
            f = args.nash_eqs_dir + 'eq_{}'.format(i)
            np.save(f,eqs)
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
