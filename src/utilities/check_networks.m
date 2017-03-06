load '../../data/power-grid/Polish_ps.mat';

% the power grid
% C = psconstants;
nodes1 = ps.bus(:,1);
links1 = ps.branch(:,1:2);
size(links1)

% the comm network
nodes2 = union(links2(:,1),links2(:,2));
links2 = csvread('../../data/networks/generated/network_Polish_2383nodes_2.422deg_0.1rw_12345seed.csv');
size(links2)

% the power grid edgelist
links3 = csvread('../../data/networks/generated/network_Polish_2383nodes_2.422deg_0.0rw_12345seed.csv');
size(links3)

% similarity
n_sim_12 = sum(sum(links1(:,1)==links2(:,1) & links1(:,2)==links2(:,2)))
n_sim_13 = sum(sum(links1(:,1)==links3(:,1) & links1(:,2)==links3(:,2)))
n_sim_23 = sum(sum(links2(:,1)==links3(:,1) & links2(:,2)==links3(:,2)))

