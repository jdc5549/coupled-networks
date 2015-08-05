% Test the two-way coupling model
clear;clc;
%% Initialize
C = psconstants;

%% Read the json file
d = pwd;
config_json = [d '/test_two_way.json'];
opt = psoptions;
opt = read_config_json(opt,config_json);

%% Get p and q from the json file
p_not_failed = opt.json.pMin;
% Degree of coupling
q_deg_of_coupling = opt.json.degOfCoupling;

% load the data
load ../../../data/power-grid/Polish_ps.mat
% Choose some bus outages
n = size(ps.bus,1);
bus_outages = find(rand(n,1)>p_not_failed)';
% Choose which buses are coupled
coupled_nodes = find(rand(n,1)<q_deg_of_coupling)';

%% Write the outage and coupling files
pid = feature('getpid');
bus_outage_file = sprintf('/tmp/bus_outage_%d.csv',pid);
f = fopen(bus_outage_file,'w');
fprintf(f,'bus\n');
fprintf(f,'%d\n',bus_outages);
fclose(f);

coupled_nodes_file = sprintf('/tmp/coupled_node_%d.csv',pid);
f = fopen(coupled_nodes_file,'w');
fprintf(f,'node\n');
fprintf(f,'%d\n',coupled_nodes);
fclose(f);

%% Run two-way coupling model, making some comm nodes fail
disp('Running two-way coupling model');
% Run
[is_bo,GC_size,MW_lost] = cmp_dcsimsep(pid,[-1],[-1],config_json);
fprintf('Blackout: %d, GC size: %.2f %%, Load lost: %.2f MW\n',is_bo,GC_size*100,MW_lost);

