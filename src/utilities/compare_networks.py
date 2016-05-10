import networkx as nx

network_a_location = "/Users/veneman/workspace/rise/coupled-networks/data/networks/generated/network_Polish_2383nodes_2.422deg_0.0rw_12345seed_mat.edgelist"
network_b_location = "/Users/veneman/workspace/rise/coupled-networks/data/networks/generated/network_Polish_2383nodes_2.422deg_0.1rw_12345seed.edgelist"

a = nx.read_edgelist(network_a_location, nodetype=int)
b = nx.read_edgelist(network_b_location, nodetype=int)
edges_a = a.edges_iter()
edges_b = b.edges()
i = 0

for edge in edges_a:
    if(edge in edges_b):
        continue
    reverse_edge = (edge[1],edge[0])
    if(reverse_edge in edges_b):
        continue
    else:
        i = i + 1

print("unmatched edges: " + str(i))