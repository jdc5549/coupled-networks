% Test the one-way coupling model

%% Initialize
randseed(1);
% Set the probability of component operation
p = 0.95;
% load the data
load ../../../data/power-grid/Polish_ps.mat
C = psconstants;
ps.bus(:,C.bu.grid_comm) = 1;
% Choose some bus outages
n = size(ps.bus,1);
%bus_outages = [1 10 20];
bus_outages = find(rand(n,1)>p)';
comm_outages = find(rand(n,1)>p)';

%% Run dcsimsep alone
disp('Running cascading model without control or comm');
opt = psoptions;
% set the ps file location
d = pwd;
opt.ps_file_location = [d '/../../../data/power-grid/Polish_ps.mat'];
% Other options
opt.verbose = false;
opt.sim.stop_threshold = 0.5;
opt.sim.stop_on_sep = 1;
opt.sim.use_control = false;
opt.sim.use_comm_model = false;
% Run
[is_bo,GC_size,MW_lost] = cmp_dcsimsep([],bus_outages,[],opt);
fprintf('Blackout: %d, GC size: %.2f %%, Load lost: %.2f MW\n',is_bo,GC_size*100,MW_lost);

%% Run dcsimsep with emergency control, no comm_outages
disp('Running cascading model with control, no comm outages');
% Other options
opt.sim.use_control = true;
opt.sim.use_comm_model = false;
% Run
[is_bo,GC_size,MW_lost] = cmp_dcsimsep([],bus_outages,[],opt);
fprintf('Blackout: %d, GC size: %.2f %%, Load lost: %.2f MW\n',is_bo,GC_size*100,MW_lost);

%% Run dcsimsep with emergency control, making all comm nodes fail
disp('Running cascading model with control, all comm outages');
% edit some options
opt.sim.use_control = true;
opt.sim.use_comm_model = true;
opt.comm.two_way = false;
% Run
[is_bo,GC_size,MW_lost] = cmp_dcsimsep([],bus_outages,(1:n),opt);
fprintf('Blackout: %d, GC size: %.2f %%, Load lost: %.2f MW\n',is_bo,GC_size*100,MW_lost);

%% Run dcsimsep with emergency control, making some comm nodes fail
disp('Running cascading model with control, some comm outages');
% edit some options
opt.sim.use_control = true;
opt.sim.use_comm_model = true;
opt.comm.two_way = false;
% Run
[is_bo,GC_size,MW_lost] = cmp_dcsimsep([],bus_outages,comm_outages,opt);
fprintf('Blackout: %d, GC size: %.2f %%, Load lost: %.2f MW\n',is_bo,GC_size*100,MW_lost);


